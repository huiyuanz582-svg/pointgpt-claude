# Teacher10、普通蒸馏、本方法：泛化质量与效率

入口：`tools/benchmark_generalization_comparison.py`。只做评估，不训练、不更新权重。
三个模型在同一套数据、相同归一化、分块与指标实现下逐场景串行运行，每个模型/场景
使用独立进程，同时只在 GPU 上加载一个模型。

## 模型

| 结果标识 | 模型 | 推理步数 | 冻结权重 SHA256 | best epoch |
|---|---|---:|---|---:|
| teacher_10 | 原始教师 | 10 | b02b7a1acfc8b2345cc33cbb7ff99ddd15ae53042fb94a9a1e722d760b4c730d | 95 |
| student_baseline | 普通蒸馏 | 4 | 19b09fd485092c02c208dd59eb834f888035c2da231216ebe1895d6230a92403 | 6 |
| student_local_endpoint | 本方法 | 4 | 94da5dd125e92bec221bb3a9a72b0896ca48eaef86c05bd057516517950e5a12 | 1 |

教师使用 `step_size=0.3, decay=0.95` 连续十步。两个学生使用同一个 `infer_student`
连续四步，教师条件节点 `[0,2,5,7,10]`、学生 eta=1；不把学生当作教师直接截断四步。
默认核对上述三个已归档权重的 SHA256；运行前还检查学生保存的模型配置、步数调度、
best epoch、选模指标及训练角色，并严格加载带有已训练 step condition 的权重。
只要求给出的权重文件，不要求历史训练日志仍在原目录，便于模型目录迁移至 stage2 后评估。
`--expected_checkpoint_sha256` 指教师，另外两个模型分别为
`--expected_vanilla_sha256`、`--expected_method_sha256`；默认即上表身份，不需要额外传参。

## 测试集与 SOR

- 自动发现 `data/other-noise` 下全部 lap/cov 场景，以及 `data/PCNet/noisy` 下全部场景。
- 保留 2.5% 噪声等额外目录，不限定为 1%/2%/3%。先完整核对 noisy/clean/mesh 配对并记录哈希。
- PUNet clean/mesh 根目录自动选择 `data/ScoreDenoise/PUNet` 或 `data/PUNet`；
  PCNet 选择 `data/PCNet` 或 `data/ScoreDenoise/PCNet`。若存在多个候选，使用
  `--punet_root` / `--pcnet_root` 指定。根目录下应有 `pointclouds/test` 与 `meshes/test`。
- lap/cov：每个模型均报告 SOR off/on 两组 CD/P2M；PCNet：每个模型仅报告 SOR off。
  SOR 使用 20 邻居、std_ratio=2.0，所有场景关闭表面投影。
- 两种 SOR 模式复用同一次模型输出；效率场景的 on 质量复用首轮实际计时产生的过滤结果。
- CD/P2M 使用原测试实现，均为原始误差乘 1e4，按完整场景内每云等权平均。
  PUNet 每场景 20 云、PCNet 每场景 10 云以实际完整 clean 集合为准，不硬编码为相同数量。

若数据与 2026-11-23 Teacher4 返回包一致：lap 8 场景、cov 8 场景、PCNet 6 场景。
三模型合计 **66 个模型/场景任务、114 行质量、30 行效率**；逐场景横向比较表 38 行。

## 效率口径（沿用 Teacher4 schema 2）

每类只在 10k/1% 与 50k/3% 测效率：每模型六个效率场景、十行效率。
FP32、patch batch=1、8 CPU 线程，整云预热 2 次，全场景点云计时 3 轮。
其他场景推理一次，只输出质量，耗时与显存列为空。

SOR off 从 GPU 上归一化整云输入计时，到 GPU 去噪输出结束，包含 FPS/KNN、分块、
全部模型迭代与融合。SOR on 从相同起点继续计时，包含 GPU→CPU、CPU SOR、CPU→GPU。
两个终点均同步 CUDA；SOR 之前的同步和峰值读取开销也计入 on 总时间。
排除模型加载、外部传输、归一化/反归一化、指标计算与文件 I/O。

报告秒/云的均值 ± 总体标准差，取全部云 × 三轮的实际总耗时；on 的标准差独立计算，
不把 off 与 SOR 的标准差相加。保留原始毫秒记录、逐轮均值、参数量、allocated/reserved
显存峰值；GPU 显存峰值不含 Open3D 的 CPU 内存。跨模型不复用耗时，不把旧 Teacher4
的效率填入这三个模型。

## Docker 内运行

同步本次文件后，先进入容器内实际项目根目录。下面采用项目相对路径，对应用户给出的
宿主机 `/home/sapi/zhy/pointgpt-claude`；必须保证三个权重与数据均已挂载到容器可见位置。

```bash
OUT_ROOT="experiments/efficiency/generalization_comparison_sor_$(date +%Y%m%d_%H%M%S)_$$"
CUDA_VISIBLE_DEVICES=0 python -u tools/benchmark_generalization_comparison.py \
  --teacher_ckpt experiments/L_consistency_plus/ckpt-best.pth \
  --vanilla_ckpt experiments/input_ablation/stage2/basic_10to4_seed0_20261118_084517_40/train/ckpt-best.pth \
  --method_ckpt experiments/input_ablation/stage2/mixed_10to4_local_endpoint_finetune_1epoch_seed0_20261118_035447_40/train/ckpt-best.pth \
  --data_root data \
  --other_noise_root data/other-noise \
  --pcnet_noisy_root data/PCNet/noisy \
  --output_root "$OUT_ROOT" \
  --device 0 --seed 0 --patch_batch 1 --warmup 2 --repeats 3
```

脚本先打印模型、全部场景及云数，然后开始评估。末尾加 `--plan_only` 只读核对目录、
配对和文件哈希，不反序列化 checkpoint、不导入模型、不执行 CUDA，也不创建结果目录。

本套件为 schema 3。第一次运行必须使用新目录，不能接续旧 Teacher4 的 schema 1/2 目录。
同版中断后，保留相同代码、权重、数据及设置，用原命令加 `--resume`，并将 `--output_root`
固定为原先打印的 `RESULT_ROOT`，不要重新生成时间戳。已完整任务跳过，未完整任务重跑。

只从完整记录重建汇总及 ZIP：

```bash
python tools/benchmark_generalization_comparison.py --report_only /absolute/path/to/RESULT_ROOT
```

## 输出与同步文件

- `comparison.csv`：按数据集/场景/SOR 一行排列三个模型的 CD、P2M、耗时、参数量和显存。
- `summary.csv` / `summary.json` / `summary.md`：全部模型/场景/SOR 的质量与效率长表。
- `efficiency.csv`：三模型合计 30 行实测效率；非效率场景不混入。
- `per_cloud.csv` / `timing_runs.csv`：保留模型、SOR 标签，逐云质量、逐云逐轮总耗时及显存。
- `plan.json`、各任务 manifest/run.log、源码与配置快照；完成后自动生成同名 ZIP。
  ZIP 不含数据或权重二进制。

运行所需更新文件（保留项目相对目录）：

```text
tools/benchmark_generalization_comparison.py
tools/benchmark_teacher4_generalization.py
tools/benchmark_teacher10_six.py
utils/teacher4_generalization.py
utils/denoise_benchmark_report.py
docs/generalization_comparison.md
```

`tests/test_generalization_comparison.py` 为开发用 CPU fixture，覆盖三模型场景网格、
114/30 行汇总、模型标签/步数、学生身份和缺失场景拒绝；本次按用户要求不执行。
语法/差异格式检查不代表 CUDA 评估通过，真实服务器兼容性以运行结果为准。
