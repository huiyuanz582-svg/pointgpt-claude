# Dynamic PCD Teacher-forced / free-rollout 诊断

独立入口：`tools/diagnose_distill_trajectory.py`。模型、PCD、节点搜索及训练方法不改动。公共推理/指标函数提供默认关闭的执行优化参数，由此审计入口启用；现有 train/val/test 调用保持原执行路径。此脚本只在用户手动执行时加载模型并调用 GPU；开发交付阶段不执行它。

## 复用范围和运行过程

- `runner_distill.load_student_checkpoint`：严格加载 Student 和已训练的 Step Condition，检查 checkpoint 保存的实际训练节点。默认要求 Epoch 5、`[0,10,12,14,16]`、`dynamic_pcd`。
- `builder.load_model`、`runner_distill.freeze_teacher`：加载原始 Teacher，禁用梯度，eval，无 condition。
- `runner_finetune.patch_based_denoise(return_trajectory=True)`：原有 FPS/KNN、固定外层 patches、16 步 Teacher、patch 内更新和融合权重。审计默认将逐步轨迹保留在 GPU 并在 GPU 融合；可显式选择 CPU 存储。Teacher eta=0.3、decay=0.95。不再重复实现 `capture_teacher` 的 16 步循环。
- `runner_distill.infer_student(return_trajectory=True)`：原有真实连续四步 Student rollout，明确传入当前诊断路径。
- `runner_distill.forward_student_interval`：Teacher-forced 适配器按 patch batch 顺序输入缓存的 `T_start`，传入区间 condition 和 `sigma0 * 0.95**start`。外层仍调用原有 `patch_based_denoise`，eta=1，输出为 Student 预测（仅有原有反解/重建操作的浮点舍入）。
- `runner_distill.evaluate_candidate_pcd`：原公式、原 epsilon，未实现新的 PCD。
- 原 `builder.dataset_builder` / `PairedEvalDataset`、`baseline_metric_ops`、`evaluate_baseline_metrics`：沿用配对测试集、clean 单位球归一化、CD、双向 P2M、SOR 和可选 surface projection。

每个 noise/shape 只生成一次 Teacher 轨迹，随后在同一 checkpoint 上依次诊断指定路径。相同 noise/shape 的各路径和两种模式使用相同种子；脚本逐次核对 patch 索引、种子坐标、覆盖数、融合权重、T0，并检查 TF/free 第一阶段最大坐标差不超过 `--first_stage_atol`（默认 `1e-5`，单位为归一化坐标）。不对融合后的整云重新分块或回灌到下一阶段。

这个容差只用于两次独立 float32 前向的输出检查，输入和 patch 对齐仍要求严格相等。模型包含 GPU `index_add_` 累加；固定随机种子并不保证独立前向逐位一致。旧默认 `1e-6` 曾在最大差异 `1.132488e-6` 时中断。旧服务器脚本也可显式传 `--first_stage_atol 1e-5`，不必修改训练代码。容差不会清零差异、替换预测或修改任何指标。新版在通过和失败时都将最大/平均/P95 绝对坐标差、超限坐标数及比例写入 manifest 的 `first_stage_max_abs_differences`；若仍超限，先检查这些记录，不自动继续放宽容差。

所有模型前向和指标均在 `torch.no_grad()` 内，Teacher/Student 均 eval 且冻结；无 optimizer、backward、训练、在线节点搜索或 checkpoint 写入。诊断路径不写回 checkpoint 的推理节点。

## 输入路径和前置检查

在服务器仓库根目录、原训练/测试环境中执行。需要现有 PyTorch/CUDA、PointNet2、Chamfer、PyTorch3D、Open3D 等项目依赖；不添加依赖或配置文件。

必需输入：

1. Epoch 5 Dynamic Student `ckpt-best.pth`，不能给 Teacher、fixed Student 或 Epoch 20 last。默认期待的 epoch/nodes 可显式通过 `--expected_epoch` / `--expected_checkpoint_nodes` 改变，但本次保留默认。
2. 当时训练实际使用的原始 Teacher checkpoint，通常为 `experiments/L_consistency_plus/ckpt-best.pth`。脚本记录两个 checkpoint 的绝对路径、SHA256 和 Student 内记录的 Teacher 路径；历史 checkpoint 未保存 Teacher 文件哈希，因此不能仅靠旧路径证明两份 Teacher 权重相同。
3. Clean：`<clean_root>/<resolution>/<shape>.xyz`。
4. Noisy：`<noisy_path_template.format(resolution=..., noise=...)>/<shape>.xyz`，使用现有 1%/2%/3% 测试文件，不重新加噪。
5. Mesh：`<mesh_root>/test/<shape>.off`。本脚本支持整云 P2M，所以 mesh 必须存在；不会静默用零替代缺失 P2M。
6. 全新的输出目录。目录即使为空，已存在也会拒绝；不要提前 `mkdir` 最末级目录。

默认目录沿用项目结构：

```text
data/ScoreDenoise/PUNet/pointclouds/test/10000_poisson/*.xyz
data/ScoreDenoise/examples/pointclouds/test/PUNet_10000_poisson_0.01/*.xyz
data/ScoreDenoise/examples/pointclouds/test/PUNet_10000_poisson_0.02/*.xyz
data/ScoreDenoise/examples/pointclouds/test/PUNet_10000_poisson_0.03/*.xyz
data/ScoreDenoise/PUNet/meshes/test/*.off
```

同一分辨率所有噪声文件夹必须与 clean 的 shape 名集合完全一致，不取静默交集。`--max_shapes` 从同一个排序后的列表取前 N 个形状。clean/noisy 点数必须一致，行顺序应对应；程序不能自动证明行对应，manifest 明确记录这个假设。patch 到 clean 的 CD 使用 `clean[patch_idx]`，所以行对应对这个指标尤其重要；整云 CD/P2M 不要求这种行对应。

`--noise_levels` 同时决定噪声目录模板和模型收到的 sigma0。sigma0 是 clean 单位球坐标下的每坐标噪声标准差；阶段 sigma 为 `sigma0 * 0.95**start_step`，不再除以归一化 scale。已有文件的“1%”是否使用相同单位需用户确认；manifest 另存配对残差的坐标 RMS、三轴 std，以便核对实际数据。

若 YAML 配置了单一 `TEST_NOISY_PATH`，必须显式传 `--noisy_path_template`，避免把三个噪声等级错误地指向同一目录。数据 CLI 仅改变诊断进程中的配置副本，不写回 YAML。

## Linux 运行命令

以下命令仅供用户执行。先进入服务器仓库根目录并激活原环境，将占位符替换为真实 Linux 路径。模型前向默认仍使用原 YAML 的 `test_patch_batch`；CPU/GPU 执行优化见下节。

### CPU/GPU 执行优化

只改变存储设备、传输、执行批次和重复结果复用，不修改模型、condition、sigma、Teacher/Student 更新式、PCD epsilon、采样、融合权重或指标定义，也不启用 AMP/TF32。GPU 模式按原公式融合整云轨迹，浮点累加顺序可能与 CPU 不同；中间融合结果始终只用于指标，不回灌 rollout。

|参数/行为|默认|用途与边界|
|---|---|---|
|`--trajectory_transfer device`|启用|Teacher/TF/free 的 patch 状态、整云状态、索引、权重和 sigma 元数据常驻输入 GPU；整云轨迹融合和指标输入切片均留在 GPU。|
|`--trajectory_transfer batch` / `step`|显式选择|显存不足时可在新进程中选择 CPU 存储：每个 patch batch 统一回传，或逐步回传。保留原 CPU 融合路径，不自动切换设备或放宽容差。|
|`--metric_patch_batch 32`|32|调用原 PCD 函数的批量接口，仍为每个 patch 分别计算位移 mean/P95；Chamfer 保留逐 patch 调用。GPU 模式无需搬运轨迹输入，仅标量结果每批统一回传，未提前聚合样本。|
|整云指标保留设备|启用|完全关闭后处理时省去 GPU→CPU→GPU 往返；仍调用原 CD 和双向 P2M。SOR/projection 启用时保留原后处理路径。|
|相同最终指标复用|启用|仅 SOR 和 projection 都关闭时，`baseline_postprocessed` 复用 `raw` 的同一组 CD/P2M；两类 CSV 行仍完整输出。|
|`--patch_batch`|YAML（当前为 1）|模型前向批次，独立于指标批次。可在后续获准运行时逐档比较 1/2/4/8，不能仅凭剩余显存直接认定某一批次适合。|

Teacher-forced 在 GPU 模式直接读取原始 Teacher 起点的张量视图，保持节点和 patch 顺序；CPU 模式才批量搬运四个起点。执行参数及实际轨迹设备记录于 manifest 的 `execution` 与 `patch_options`；训练 YAML、checkpoint 及其训练节点均不被修改。公共函数仍默认 CPU 轨迹，只有审计入口默认启用 GPU 模式，原常规测试的 NumPy 导出保持兼容。

50K、146 个 patches、float32 时，三套 patch/整云轨迹坐标共约 62 MiB，另需索引、权重和运算临时空间。GPU 模式不再回传这些轨迹；CPU batch 模式在模型 batch=1 时仍会将 3504 次逐步回传降为 438 次（不含初始状态和融合元数据）。指标的显式结果回传由约 8760 次逐标量读取降为 20 次批量读取（metric batch=32）；原 PCD 有限性检查、对齐检查、日志/manifest 标量读取等仍可能产生同步。这些是代码路径计数，不是实测加速倍数。

后续验证须使用相同 checkpoint/数据/seed/nodes，并保持相同 `--patch_batch`，先对照 CPU/GPU 存储模式的完整轨迹及逐 shape/stage/patch 指标，并确认 manifest 中 `execution.trajectory_device` 符合选择；通过后再单独改变模型 batch。GPU 融合、批量归约和 CUDA 调度可能产生浮点舍入差异，不能承诺逐位相等，也不能自动放宽现有一致性容差。应比较实际耗时、峰值显存和数值差异，而不仅看 GPU 利用率。

当前交付只做静态检查，尚未运行上述 CPU/GPU 数值或速度验证。服务器上旧版脚本不认识新增参数；使用前须同步本次改动的 `tools/diagnose_distill_trajectory.py`、`tools/runner_finetune.py`、`tools/runner_distill.py` 及新文件 `utils/gpu_memory.py`。开发阶段不自动同步或执行服务器任务。

审计入口默认启用 [48 GiB 显存预算](gpu_memory_limit.md)：PyTorch 分配器实际最多 47 GiB，另预留 1 GiB；CUDA OOM 以退出码 86 终止，不自动回传轨迹或重试。该预算按进程/设备生效，不是 Docker 总显存硬隔离；分配器外开销不受硬限制。

设置共用参数（Bash）：

```bash
cd /REPLACE/WITH/SERVER/REPOSITORY

STUDENT_CKPT='/REPLACE/WITH/EPOCH5/ckpt-best.pth'
TEACHER_CKPT='experiments/L_consistency_plus/ckpt-best.pth'
DATA_ROOT='/REPLACE/WITH/ScoreDenoise'
CLEAN_ROOT="$DATA_ROOT/PUNet/pointclouds/test"
NOISY_TEMPLATE="$DATA_ROOT/examples/pointclouds/test/PUNet_{resolution}_{noise:g}"
MESH_ROOT="$DATA_ROOT/PUNet/meshes"
AUDIT_TAG="$(date +%Y%m%d_%H%M%S)_$$"

COMMON=(
  --config cfgs/PointGPT-L/distill_16to4_dynamic.yaml
  --checkpoint "$STUDENT_CKPT"
  --teacher_checkpoint "$TEACHER_CKPT"
  --dataset_root "$DATA_ROOT"
  --clean_root "$CLEAN_ROOT"
  --noisy_path_template "$NOISY_TEMPLATE"
  --mesh_root "$MESH_ROOT"
  --seed 0 --device 0
)
```

推荐先检查一个形状、1% 噪声、checkpoint 原路径（仍包含完整 Teacher 16 步及 Student 4 步，不截断轨迹）：

```bash
CUDA_VISIBLE_DEVICES=0 python -u tools/diagnose_distill_trajectory.py \
  "${COMMON[@]}" \
  --resolution 10000_poisson --noise_levels 0.01 \
  --teacher_nodes 0 10 12 14 16 --max_shapes 1 --patch_batch 1 \
  --output_dir "diagnostics/trajectory_audit/epoch5_smoke_${AUDIT_TAG}"
```

完整主诊断：路径 C、1%/2%/3%，分别覆盖 10K 和 50K，两个独立目录：

```bash
for RES in 10000_poisson 50000_poisson; do
  CUDA_VISIBLE_DEVICES=0 python -u tools/diagnose_distill_trajectory.py \
    "${COMMON[@]}" \
    --resolution "$RES" --noise_levels 0.01 0.02 0.03 \
    --teacher_nodes 0 10 12 14 16 \
    --output_dir "diagnostics/trajectory_audit/epoch5_C_${RES}_${AUDIT_TAG}" || break
done
```

完整路径敏感性诊断：A/B/C/D、1%/2%/3%、10K/50K。它包含上述 C 的主诊断；若直接运行这一组，不必再重复执行上一组：

```bash
for RES in 10000_poisson 50000_poisson; do
  CUDA_VISIBLE_DEVICES=0 python -u tools/diagnose_distill_trajectory.py \
    "${COMMON[@]}" \
    --resolution "$RES" --noise_levels 0.01 0.02 0.03 --all_paths \
    --output_dir "diagnostics/trajectory_audit/epoch5_ABCD_${RES}_${AUDIT_TAG}" || break
done
```

`--all_paths` 依次选择 `[0,4,8,12,16]`、`[0,7,10,13,16]`、`[0,10,12,14,16]`、`[0,13,14,15,16]`。也可重复 `--teacher_nodes ...` 指定任意子集，不能同时传 `--all_paths`。每条路径使用同一个 Epoch 5 checkpoint，是路径敏感性探针，不能作为四个重新训练方法的公平比较。

不传 `--output_dir` 会自动创建带时间戳和随机后缀的独立目录。输入路径参数的相对路径以仓库根目录为基准，checkpoint/config 参数在启动目录解析；推荐始终从仓库根目录运行。

## 指标口径和输出

`stage_metrics.csv` 为长表：每个 noise/path/mode/scope/stage/metric 一行，包含 `mean, median, std, p90, p95, sample_count`。`std` 使用总体标准差（ddof=0）。

|scope / mode|指标|统计样本|
|---|---|---|
|patch / teacher_forced|到对应 Teacher 和配对 clean patch 的 CD；E_imit、D_move、PCD；预测位移范数 mean/P95|每个 patch；不再次按 patch 归一化|
|patch / free_rollout|到 Teacher/clean 的 CD；预测位移范数 mean/P95；E_to_teacher、D_move_teacher、relative_error_teacher_move|同一批 patch|
|patch / teacher|对应 Teacher 目标到 clean patch 的 CD|同一批 patch|
|whole_cloud / 三种模式|到 clean 的 CD、到 mesh 的 P2M；两个 Student 模式另有到对应 Teacher 状态的 CD|每个形状等权|
|free_minus_teacher_forced|同一样本上的 free 误差减 TF 误差，再计算分布统计|patch 或 shape，见 scope|

PCD **不是 CD 的比值**：复用逐点对应的平方 L2 均值 `E_imit / (D_move + 1e-12)`。仅 TF 行的 `PCD` 是当前课程指标。Free 行保留相同 `T_start→T_target` 分母，命名为 `relative_error_teacher_move`，避免误认为训练搜索也使用自由迭代。

位移是 `TF_prediction - T_start` 或 `S_k - S_{k-1}`。先在各 patch 内计算点位移范数的均值、P95，再对这些 patch 级指标报告六项统计；例如 `displacement_norm_p95` 行的 `mean` 是各 patch 的 P95 均值，不能当作全部点合并后的 P95。重叠 patches 作为 patch 样本计数，未去重。

逐阶段整云状态是原固定 patch 轨迹的融合读数，不是每一步重新切整云 patch。最终状态使用原推理函数直接返回的 GPU 融合结果；中间状态按原公式在所选轨迹设备融合（默认 GPU，`step`/`batch` 为 CPU），可能有少量浮点差异。

逐阶段仅计算 raw 状态。最终 CD/P2M 同时给出：

- `raw`：无 SOR、无 surface projection。
- `baseline_postprocessed`：严格采用原 YAML 的 SOR / surface projection 及原测试指标实现。过滤只用于指标，不回灌轨迹。

CD/P2M 均按原协议乘 `1e4`；P2M 仅整云，调用原有双向 mesh 距离，不把单 patch 对整个 mesh 的距离伪装成整云 P2M。最终 score 仍为 `CD + 0.3*P2M`，但诊断脚本不选择或保存 checkpoint。

输出文件：

```text
stage_metrics.csv                 # 所有样本的逐阶段统计
final_metrics.csv                 # 各噪声/路径/模式/后处理的最终统计
per_shape_metrics.csv             # 每个形状最终 CD/P2M/score、输入/输出点数
per_shape_stage_metrics.csv       # 额外：每形状逐阶段统计，可定位 P2M 尾部和 exposure gap
run_manifest.json                # checkpoint/数据/哈希/参数/seed/sigma/实际点数/节点/后处理/git/完成状态
diagnostic_summary.md            # 自动阶段误差、配对 gap、最终结果表和解释边界
run.log                          # stdout/stderr、进度、异常堆栈
```

仅保存 CSV/JSON/Markdown，不新增绘图依赖。Teacher-forced 最终输出是 `Student(T14,14,16)`（路径 C），不代表可独立运行的四步 Student。

## 完成检查和中断处理

成功必须同时满足：

- 进程退出码为 0，`run.log` 末尾有 `COMPLETED`。
- `run_manifest.json` 中 `status=completed`，completed/expected shape-noise-path runs 相等。
- `checkpoint_epoch=5`、`checkpoint_teacher_nodes=[0,10,12,14,16]`；`diagnostic_paths` 为本次要求的路径。
- `integrity` 中 frozen/eval/no_grad、共享 patches、第一阶段一致性、finite 检查通过。
- 全量运行时 `full_dataset=true`；小规模运行不伪称全量。
- 聚合 CSV 非空，整云统计的 `sample_count` 等于本次形状数，patch 统计数为实际 patches 总数。

本脚本不支持断点续跑，不自动跳过已有结果。每形状的逐阶段和最终 CSV 行及时 flush；Ctrl-C/异常保留已写数据，并记录 failed/interrupted。聚合 CSV 和 Markdown 在全部完成后才生成内容，不能把失败目录中的局部数据当作完整统计。SIGKILL/断电来不及记录错误时 manifest 可能停在 running，这也不代表成功。重跑用新目录；不会覆盖旧 CSV 或任何训练结果。

发生错误时请提供完整执行命令、`run.log` 的完整异常堆栈及前后进度、`run_manifest.json`，以及已有 `per_shape_metrics.csv` / `per_shape_stage_metrics.csv`。若在建立输出目录前失败，提供终端异常。无需发送 checkpoint 本体。
