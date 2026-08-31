# 第 0 阶段：教师轨迹与步数冗余审计

本阶段不训练新模型，也不改变教师结构。它用同一个微调 checkpoint 运行当前确定性去噪教师，回答三个问题：

1. 1/2/4/8/15/30 次更新的质量—耗时曲线是什么；
2. 30 次更新中哪些区间仍在有效降低误差，哪些区间可能冗余；
3. 教师每一步的位移、法向/切向分量和局部邻域变化是否适合后续做少步映射蒸馏。

分析入口是 `tools/analyze_teacher_trajectory.py`。它复用正式测试所用的模型、测试集、checkpoint 加载和 `patch_based_denoise`，但不会执行普通 `test()` 中的大量 PLY 可视化。

## 1. 运行前准备

必须在原来的 Linux/CUDA 训练环境中运行，并准备好：

- 与训练 checkpoint 对应的 YAML；
- 微调后的 `ckpt-best.pth`；
- PUNet/ScoreDenoise 的 clean、noisy 和 mesh 数据；
- PyTorch/CUDA、PointNet++、Chamfer CUDA、PyTorch3D、SciPy、Open3D 等项目依赖。

测试分辨率和噪声数据仍由 `cfgs/dataset_configs/ScoreDenoise.yaml` 控制：

- `TEST_RESOLUTION`；
- `TEST_NOISY_DIR` 或 `TEST_NOISY_PATH`；
- `TEST_NOISE`；
- 泛化数据需要时设置 `TEST_CLEAN_PATH` 和 `TEST_MESH_ROOT`。

`--noise_std` 只覆盖教师日程的起始 σ，不会替换已经加载的 noisy 点云，因此不能用它代替上述数据配置来切换 1%/2%/3% 噪声实验。

## 2. 建议先做单样本冒烟

在正式全量运行前，先确认 checkpoint、数据和 CUDA 扩展可以正常加载：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/analyze_teacher_trajectory.py \
  --config cfgs/PointGPT-L/finetune_scoredenoise.yaml \
  --ckpt experiments/finetune_scoredenoise/PointGPT-L/<主实验名>/ckpt-best.pth \
  --run_name smoke_L_10k_1pct \
  --max_shapes 1 \
  --max_steps 2 \
  --budgets 1 2 \
  --timing_shapes 1 \
  --timing_repeats 1
```

冒烟通过的最低标准：

- `manifest.json` 最终状态为 `complete`；
- `summary_by_budget.csv` 同时含 step 0、1、2；
- CD、HD/HD95 和几何统计没有 Inf；
- P2M 有效；启用 P2M 时，mesh 缺失、计算异常或非有限结果都会令运行失败；确实没有 mesh 的数据集必须显式使用 `--no_p2m`；
- `trajectories/<shape>/trajectory.npz` 可正常读取。

## 3. 正式 Stage 0 命令

默认参数已经写入 PointGPT-S/L 的 `finetune_scoredenoise.yaml`：

```yaml
inference_patch_size: 1024  # 普通 val/test、文件夹推理与 Stage 0 共用

stage0_audit:
  { max_steps: 30, budgets: [1, 2, 4, 8, 15, 30], max_shapes: 0,
    save_num_trajectories: 3, timing_shapes: 3, timing_repeats: 3,
    stats_sample_points: 2048, knn_k: 16, normal_confidence_threshold: 0.05,
    consistency_atol: 0.00001,
    seed_ratio: 3, evaluate_sor: True, compute_p2m: True,
    save_patch_trajectory: True, assume_correspondence: True,
    save_xyz_steps: [0, 1, 2, 4, 8, 15, 30] }
```

全量分析命令：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/analyze_teacher_trajectory.py \
  --config cfgs/PointGPT-L/finetune_scoredenoise.yaml \
  --ckpt experiments/finetune_scoredenoise/PointGPT-L/<主实验名>/ckpt-best.pth \
  --run_name L_10k_1pct
```

`max_shapes: 0` 表示遍历全部测试 shape。若希望保存指定的平滑曲面、锐边和薄结构样本，而不是默认保存排序靠前的三个 shape，可使用：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/analyze_teacher_trajectory.py \
  --config cfgs/PointGPT-L/finetune_scoredenoise.yaml \
  --ckpt experiments/finetune_scoredenoise/PointGPT-L/<主实验名>/ckpt-best.pth \
  --run_name L_10k_1pct_representative \
  --trajectory_names <shape_a> <shape_b> <shape_c>
```

CLI 参数优先于 YAML。常用诊断开关如下：

| 参数 | 用途 |
|---|---|
| `--max_shapes N` | 只跑前 N 个测试 shape；0 表示全部 |
| `--max_steps N --budgets ... N` | 临时改变完整教师步数与需要汇总的前缀；budgets 必须包含 max_steps |
| `--timing_shapes N --timing_repeats R` | 控制独立计时的 shape 数和重复次数 |
| `--consistency_atol X` | 同一次捕获内部硬检查容差；独立 replay 超过它只提示和记录 |
| `--save_num_trajectories N` | 未指定名字时保存前 N 个完整轨迹 |
| `--trajectory_names ...` | 按不带扩展名的 shape 名保存代表性轨迹 |
| `--no_patch_trajectory` | NPZ 不写入体积较大的精确 `patch_states`；仍会保存诊断整云轨迹 |
| `--no_sor_metrics` | 只报告 raw，不生成 SOR 质量和耗时行 |
| `--no_p2m` | 明确关闭 P2M；只适用于确实没有 mesh 的诊断 |
| `--non_corresponding_points` | noisy/clean 点顺序不对应时关闭索引对应统计 |
| `--patch_batch N` | 显存不足时降低 patch batch |
| `--output_dir PATH` | 覆盖默认输出位置 |

为防止旧 NPZ 混入新结果，输出目录只允许不存在或为空；同名 `run_name` 已有结果时会直接报错，请为每次实验使用新的名字。

`stage0_audit.max_steps` 在正式运行时必须与 `langevin.num_steps` 一致；如果只是做 2 步冒烟等临时诊断，应显式传 `--max_steps`。外层 patch 大小统一读顶层 `inference_patch_size`，因此普通 val/test、`denoise_folder.py` 与 Stage 0 不会因数据集 `PATCH_SIZE` 的未来改动而静默分叉。

外部数据只有在 noisy 与 clean 保持同点数、同点序时才能使用 `paired_coordinate_rmse`、固定 clean 邻域误差等索引对应统计。不能保证时必须加 `--non_corresponding_points`，相关列会明确写成 NaN，而不是产生看似合理但错误的数值。

## 4. “教师前缀”与 `abl_T1.yaml` 不是一回事

Stage 0 主曲线固定：

```text
同一 checkpoint
step_size = 0.3
decay = 0.95
budget = 1 / 2 / 4 / 8 / 15 / 30
```

每个预算表示当前 30 步教师的前 k 次更新。质量曲线从一次 30 步捕获中的相同 patch 布局和相同日程读取，因此可以判断“30 步里有多少真实冗余”。

`cfgs/PointGPT-L/ablation/abl_T1.yaml` 则使用：

```text
num_steps = 1
step_size = 1.0
```

它表示单步 Tweedie 推理，是论文已有的单步消融基线。它可以单独报告，但不能把 `abl_T1.yaml` 依次改成多种 `num_steps` 后当作教师前缀曲线；当 `step_size=1.0` 时，多步运行很可能过冲，问题定义也已改变。

因此建议保留两类结果：

- Stage 0 教师前缀曲线：固定 `0.3/0.95`，用于决定后续是做 30→4、8→4 还是 8→2；
- Tweedie-1：固定 `num_steps=1, step_size=1.0`，继续作为独立消融基线。

## 5. 轨迹状态的准确含义

当前完整点云推理先固定 FPS/KNN patch，然后每个 patch 独立运行多步教师，最后才融合回完整点云。Stage 0 会保存两种状态：

- `patch_states[T+1, S, patch_size, 3]`：教师真实执行的 patch 内状态，后续构造蒸馏样本时应优先使用；
- `global_states[T+1, N, 3]`：每一步用同一 `patch_idx` 和 `fuse_weights` 得到的诊断整云 readout，用于质量曲线、几何统计和可视化。

`global_states[k]` 不会回灌到第 k+1 步。把它误写成“教师在完整点云上的 Markov 轨迹”会改变当前算法的真实含义。

状态编号同时使用两个字段：

```text
update_step = 0   <=> teacher_t = 30   <=> 原始 noisy
update_step = k   <=> teacher_t = 30-k <=> 完成 k 次更新
update_step = 30  <=> teacher_t = 0    <=> 完整教师输出
```

XYZ 文件名同时包含两种编号，例如：

```text
update_step_004_teacher_t_026.xyz
```

所有逐步动力学都在 raw、SOR 前的固定点数状态上计算。SOR 会删除点，不能用于逐点位移或邻域对应分析。Stage 0 默认另外报告同一步数的 SOR 质量，但不会应用可选的 `surface_projection` 后处理。

## 6. 输出目录

默认输出：

```text
experiments/stage0_teacher_analysis/<run_name>/
├── manifest.json
├── effective_config.yaml
├── progress.json
├── summary_by_budget.csv
├── per_shape_by_budget.csv
├── timing_runs.csv
├── aggregate_per_step.csv
├── per_step_dynamics.csv
└── trajectories/
    └── <shape>/
        ├── metadata.json
        ├── trajectory.npz
        ├── update_step_000_teacher_t_030.xyz
        ├── update_step_001_teacher_t_029.xyz
        └── ...
```

各文件用途：

| 文件 | 内容 |
|---|---|
| `manifest.json` | 命令、Git commit/dirty 状态、checkpoint 元信息、数据路径、日程、设备、运行状态 |
| `effective_config.yaml` | CLI 覆盖后实际使用的完整配置 |
| `progress.json` | 已完成 shape 和最后处理位置；中断时用于确认已有结果，不是自动续跑文件 |
| `summary_by_budget.csv` | 按 update step 和 raw/SOR 聚合的质量、耗时、相对 30 步差距与保留收益 |
| `per_shape_by_budget.csv` | 每个 shape、每个预算的 CD/P2M/HD/HD95 |
| `timing_runs.csv` | 独立无轨迹捕获的每次计时明细 |
| `aggregate_per_step.csv` | 所有 shape 的逐步动力学均值和有效值数量 |
| `per_step_dynamics.csv` | 每个 shape 的 0…30 步位移、法/切分量、误差下降和邻域变化 |
| `trajectory.npz` | 精确 patch 状态、诊断整云状态、patch 索引/权重、σ、归一化信息和法向诊断数据 |

CSV 使用 UTF-8 BOM，便于直接用 Excel 打开。轨迹可能较大；`--no_patch_trajectory` 只减少 NPZ 落盘体积，并不取消捕获诊断整云所需的运行内存。

工具会严格检查三个同一次捕获内部的不变量：初始 patch 必须等于 `noisy[patch_idx]`，`global_states[0]` 必须等于 noisy，CPU 重融合的最后一步必须与同次 GPU 融合输出一致。它们超过 `stage0_audit.consistency_atol`（默认 `1e-5`）会直接失败，防止用错误轨迹继续做蒸馏。

独立计时运行会重新执行 CUDA FPS、scatter 和 index-add。其浮点累加顺序可能不同，误差会随迭代步数累积，因此 fresh replay 与捕获前缀的差异不是硬不变量。工具会把 max/mean/RMS 差异写入 `timing_runs.csv`，超过 `consistency_atol` 时只输出提示；只有出现 NaN/Inf 才中止实验。

## 7. 指标口径

质量表同时保留 `raw` 与 `sor` 两个 variant：

- `cd_x1e4`：正式 CUDA ChamferDistanceL2，与仓库主测试一致；
- `p2m_x1e4`：正式双向 point↔mesh 距离，与仓库主测试一致；
- `hd_euclidean` / `hd95_euclidean`：归一化空间的双向最近邻欧氏 HD/HD95；
- `hd_sq_x1e4` / `hd95_sq_x1e4`：上述距离平方后乘 `1e4`，便于与平方 CD 的量纲对照；
- `cpu_cd_sq_x1e4`：基于 SciPy KDTree 的 CPU 复核值，不替代正式 CUDA CD。

这里的 HD95 固定定义为：把 pred→clean 与 clean→pred 两个方向的最近邻欧氏距离合并后取 95 分位；不是“两个方向各取 P95 后再取最大值”。论文和后续表格应保持这一口径。

逐步动力学包括：

- 单步平均/RMS/P50/P95/最大位移；
- 相对 noisy 的累计位移；
- 基于 clean kNN-PCA 法向的法向绝对位移和切向位移；
- 最近 clean 点误差及相邻两步误差下降量；
- kNN 保留率/变化率；
- 相对 clean 和相邻步骤的局部边长变化。

法向只用于绝对分量和能量比例。局部 PCA 法向没有全局一致朝向，因此不要把 signed normal displacement 跨 shape 平均解释为“向内”或“向外”。

`summary_by_budget.csv` 还给出：

- `*_relative_gap_to_teacher = (M_k - M_30) / |M_30|`；
- `*_retained_gain = (M_noisy - M_k) / (M_noisy - M_30)`。

对越小越好的误差指标，`retained_gain` 接近 1 表示该预算已保留大部分 30 步收益。它可以帮助判断 8 步是否已接近 30 步，但不要把 95% 等经验阈值写成程序成功/失败条件；非单调曲线本身也是重要诊断结果。

## 8. 计时口径

质量轨迹先运行一次，兼作 GPU warmup。随后对前 `timing_shapes` 个 shape、每个预算独立运行 `timing_repeats` 次，并在 CUDA 计时前后同步。

- `denoise_seconds`：patch 构造 + patch 内 rollout + 融合；
- `sor_seconds`：CPU SOR 后处理；
- `end_to_end_seconds`：上述两项之和；
- 不包括 DataLoader 读取、CD/P2M/HD 计算、轨迹 CPU copy、CSV/NPZ/XYZ 写盘和可视化。

step 0 的 raw 基线模型耗时记为 0；step 0 的 SOR 没有独立计时，因此其 latency 留为 NaN，不能解释为零成本后处理。

完整点云被拆成多个 patch batch，因此区分：

- `logical_nfe = budget`：论文质量—速度曲线应报告的逻辑网络步数；
- `actual_forward_calls = budget × patch_batches`：当前实现真实发生的 batched forward 次数。

比较 S/L、10k/50k 或不同方法时，必须固定 GPU、`patch_batch`、`seed_ratio`、`fuse_tau_ratio` 和计时 shape；同时报告 latency 的 mean、median、p95，不能只报告 NFE。

## 9. 推荐实验顺序

1. 单 shape、2 步冒烟；
2. PUNet 10k/1% 全测试集，得到第一条 1/2/4/8/15/30 曲线；
3. 检查代表样本 31 个状态和逐步动力学，确认是否存在末段冗余或过冲；
4. 扩展到 10k/50k × 1%/2%/3%；
5. 分别比较 raw 与 SOR，避免把后处理收益误认为教师轨迹收益；
6. 如果 8 步已接近 30 步，再优先设计 8→2 或 8→4 蒸馏；否则保留 30→4 方案。

不同噪声/分辨率运行必须使用不同 `run_name`，并保留各目录中的 `manifest.json` 和 `effective_config.yaml`。

## 10. 当前本机验证限制

当前 Windows 工作区没有 `data/`、`extensions/`、`pretrainModel/` 或 `experiments/`，系统 Python 也没有安装 PyTorch。因此本机只能完成：

```powershell
python tools/analyze_teacher_trajectory.py --help
python -m py_compile tools/analyze_teacher_trajectory.py utils/trajectory_metrics.py tools/runner_finetune.py utils/p2m_loss.py tests/test_trajectory_metrics.py
python -m unittest discover -s tests -p "test_trajectory_metrics.py" -v
```

当前可用的无 PyTorch 环境已经完成 8 项 NumPy/SciPy 单元测试；本机仍不能证明真实 checkpoint 的 CD/P2M、GPU 轨迹一致性、显存和耗时正确。正式接受该功能前，必须在原训练环境完成单样本冒烟。工具会严格验证同一次捕获的索引和融合不变量，并把独立 replay 的数值漂移作为诊断量记录。
