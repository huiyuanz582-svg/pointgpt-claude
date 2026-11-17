# Teacher10 / Student4 六项质量与效率比较

入口为 `tools/benchmark_teacher10_six.py`，汇总和绘图由
`utils/denoise_benchmark_report.py` 完成。只在服务器上手动运行。

## 比较对象

- `student_before`：masked endpoint 的微调前 best checkpoint。
- `student_ft1`：从同一个微调前 checkpoint 开始、一轮微调后的 best1。
- `student_ft3`：从同一个微调前 checkpoint 开始、三轮微调后的 best3。
- `teacher_2/4/6/8/10`：同一个原教师 checkpoint，分别直接执行 2/4/6/8/10 步。

教师保留步长 0.3、sigma 衰减 0.95。每个步数独立执行完整的分块、迭代和融合，
不使用十步轨迹的缓存读出时间，不重新调整步长。学生复用原来的
`runner_distill.infer_student`，固定节点 `[0,2,5,7,10]`，实际连续执行四步。
本次不包含基础蒸馏模型，也不写入训练 checkpoint。

输入使用 PUNet 六项 10k/50k × 1%/2%/3%，每项严格要求 20 个唯一点云。
复用原数据加载、归一化、CD 和测试用双向 P2M。关闭 SOR 与 surface projection，
与这三个学生既有测试保持一致。Score 仍为 `CD + 0.3 * P2M`，CD/P2M 为 ×10⁴。

## 测量口径

- 每个模型、每个条件在独立子进程中执行，显存中只加载一个模型。全部进程串行。
- 默认 FP32、无 autocast，patch batch=1。记录 CUDA/cuDNN/PyTorch、显卡、线程数及
  TF32 设置。matmul TF32 关闭、cuDNN TF32 开启，与正常 FP32 入口的默认口径一致。
- 每组默认用完整点云预热 2 次，再对 20 云重复计时 3 轮。预热与正式计时分开。
- 测量输入已位于 GPU 的归一化整云，到 GPU 上去噪整云输出的墙钟时间。
  包含 FPS/KNN、patch 构造、模型迭代及融合。使用 `perf_counter_ns`，前后 CUDA 同步。
  不包含加载权重、磁盘 I/O、数据预处理、H2D/D2H、指标计算、日志和随机种子设置。
- 第一轮计时的输出复制到 CPU 后留作质量评估。全部计时完成后才初始化质量指标，
  计算 CD/P2M，避免网格和指标的 GPU 分配污染推理显存及计时。
  质量结果为单次推理，不进行三轮投票或最佳结果挑选。
- 每次推理前重置显存峰值。`peak_allocated_gib` 是 PyTorch 张量分配峰值，
  包含当前模型和输入。另报 `peak_reserved_gib` 缓存预留峰值。
  CUDA context 和非 PyTorch 分配不在此统计范围内。
  48 GiB 是既有分配预算配置，不能当成实测显存。
- `latency_mean_ms` 为 20 云 × 3 轮的平均值。`latency_std_ms` 包含点云差异及重复波动。
  `round_mean_std_ms` 单独报告各轮平均延迟的标准差，原始逐次记录全部保留。
- 吞吐量是串行推理区间的总云数或输入点数除以累计实测耗时，
  不代表最大批量、并发流水线或含文件 I/O 的吞吐量。points/s 不重复计数重叠 patch。
- 加速比为同一条件下 Teacher10 平均耗时除以被测方法平均耗时。
  参数量为实际模型参数总数，额外记录每云 patch 数和批量网络调用次数。

记录各点云输入、clean/归一化参数、网格及 checkpoint 哈希，汇总时要求不同方法
使用相同数据和环境。不同条件采用固定种子的不同方法顺序，减少固定执行顺序的偏差。
运行时应让该 GPU 保持独占，硬件负载变化仍会影响测量。

训练记录中的模型、step condition、patch 推理及数据集源文件哈希必须匹配当前代码。
历史训练入口和终点损失工具因加入微调可能存在版本差异，这些差异会写入 plan，
三个学生使用同一个当前推理入口。每轮还记录当前评测源码，并在启动各 worker 前核对。

## 一次完成八组比较、绘图和 ZIP

同步本次新增的两个 Python 文件和本说明文件到服务器对应项目目录。
三个 `*_train_root` 都指向包含 `ckpt-best.pth` 的 `train` 目录。

```bash
cd /workspace
OUT_ROOT="experiments/efficiency/teacher10_student4_$(date +%Y%m%d_%H%M%S)_$$"
CUDA_VISIBLE_DEVICES=0 python -u tools/benchmark_teacher10_six.py \
  --before_train_root experiments/input_ablation/mixed_10to4_endpoint_masked_seed0_20261112_075942_40/train \
  --one_epoch_train_root experiments/input_ablation/mixed_10to4_endpoint_finetune_1epoch_seed0_20261117_032429_40/train \
  --three_epoch_train_root experiments/input_ablation/mixed_10to4_endpoint_finetune_seed0_20261116_081226_40/train \
  --teacher_ckpt experiments/L_consistency_plus/ckpt-best.pth \
  --output_root "$OUT_ROOT" \
  --device 0 --seed 0 --patch_batch 1 --warmup 2 --repeats 3
```

共 48 组，每组 20 云，默认总计 2880 次正式整云推理、96 次预热，以及 960 云质量评估。
第一次启动先在 CPU 上核对实际 checkpoint 元数据，GPU 推理由独立 worker 执行。
耗时取决于服务器，不能由训练日志估算。本次代码修改期间未执行这条命令。

最后打印 `ZIP=<绝对路径>`，返回这个 ZIP 即可。ZIP 内不含模型权重和大体积点云。

## 输出

- `summary.csv/json`：48 行，含 CD/P2M/Score、平均/中位延迟、波动、吞吐量、
  allocated/reserved 峰值、参数量和相对 Teacher10 的加速及质量差距。
- `per_cloud.csv`：960 行逐云质量与效率。
- `timing_runs.csv`：默认 2880 行逐次计时、起始已分配显存和显存峰值。
- `plots/latency_cd_six_conditions.png/pdf`：六条件耗时—CD 总图。
- `plots/latency_p2m_six_conditions.png/pdf`：六条件耗时—P2M 总图。
- 每个条件还各自输出两类 PNG/PDF 图。教师五个步数连线，三个学生用不同标记。
  横轴为实测 ms/cloud，纵轴为对应误差，左下方更好。
- 每组目录保存实际配置、环境、输入和网格指纹、原始 CSV、汇总及 `run.log`。
- `training/` 保存三个训练记录副本，`source/` 保存本次相关源码，`plan.json` 保存计划。

## 中断续跑和重新绘图

重新执行同一命令，保留原 `--output_root`，增加 `--resume`。
不要重新生成 OUT_ROOT。已有完整结果会从逐云、逐次 CSV 重新核对后跳过，
未完成的组重新执行。checkpoint、源码、计时参数改变时会拒绝拼接结果，
此时应使用新输出目录。

完整结果的图和 ZIP 可单独重新生成，不加载模型、不执行推理。

```bash
cd /workspace
python tools/benchmark_teacher10_six.py --report_only "$OUT_ROOT"
```

只有 48 组全部完整、汇总核对通过且曲线生成成功后才生成最终 ZIP。
若中断，单组日志和已完成结果仍保留。CUDA OOM 沿用现有 48 GiB 预算退出规则，
不自动降低 batch 或重试，避免不同方法采用不同测量条件。

## 验证状态

本次仅静态语法检查和代码审阅。未运行 CPU 单测、模型推理、GPU 诊断或训练。
CUDA 执行、数值一致性、实际性能和最终曲线外观需在上述服务器命令完成后核验。
