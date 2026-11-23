# Teacher4：PUNet lap/cov 与 PCNet 全场景评估

入口：`tools/benchmark_teacher4_generalization.py`。使用原始教师权重，直接执行
4 步，`step_size=0.3`、`decay=0.95`。不加载学生、不训练、不更新权重。
推理复用 `patch_based_denoise`，归一化和 CD/P2M 复用原测试实现。

## 数据与范围

- PUNet 拉普拉斯：`data/other-noise` 下目录后缀为 `_lap`。
- PUNet 非各向同性高斯：同一根目录下后缀为 `_cov`。
- PCNet：`data/PCNet/noisy` 下全部测试场景。
- 目录示例：`PUNet_10000_poisson_0.01_cov`、`PUNet_50000_poisson_0.03_lap`、
  `PCNet_10000_poisson_0.01`。支持子目录递归发现和数据目录软链接。
- PUNet clean/mesh 根目录自动选择 `data/ScoreDenoise/PUNet` 或 `data/PUNet`；
  PCNet 自动选择 `data/PCNet` 或 `data/ScoreDenoise/PCNet`。
  每个根目录必须包含 `pointclouds/test/<点数>_poisson/*.xyz` 和 `meshes/test/*.off`。
  多个候选同时存在时必须用 `--punet_root` / `--pcnet_root` 明确指定。
- 逐场景要求 noisy 与 clean 名称集合完全相同，且每个名称都有 mesh。
  不通过交集悄悄丢弃样本；点数、有限值及最终结果完整性也会检查。
  云数按完整 clean 集合确定，不将 PCNet 硬编码为 PUNet 的 20 云。
- 发现到的所有 lap/cov/PCNet 场景均计算质量；未识别为 lap/cov 的其他噪声目录
  会列在计划中并排除。每类必须具有 10k/1% 和 50k/3% 两个效率场景，否则报错。

## SOR 与效率口径

PUNet lap/cov 均报告 SOR **off/on** 两组 CD/P2M。两组复用同一次推理输出，
避免重新推理的随机差异。SOR 使用原 `sor_filter`：邻居数 20、`std_ratio=2.0`；
允许 SOR 删除点，但不允许空输出。PCNet 仅报告 SOR off。所有场景均关闭表面投影。
CD 为原测试 ChamferDistanceL2，P2M 为测试用双向点—网格指标；均乘以 1e4，越低越好。

每类只在 **10k/1% 和 50k/3%** 测效率，测试场景仍为六个。
PUNet lap/cov 各报告 SOR off/on，PCNet 仅报告 off，合计 **10 行效率结果**。
每个场景在独立进程运行，
进程串行，每次只加载一个模型。FP32、patch batch=1，整云预热 2 次，
对该场景全部点云重复计时 3 轮。非效率场景只推理一轮，不输出耗时或显存。

两组都从 GPU 上归一化整云开始，到 GPU 上的结果结束：

- **SOR off：**包含 FPS/KNN、patch 构造、四步迭代与融合。
- **SOR on：**在同一次推理的基础上，继续执行 GPU→CPU 搬运、Open3D SOR、
  CPU→GPU 搬运；从同一个起点连续计时到 SOR 结果回到 GPU，不是单独的 SOR 耗时。
- 两个计时终点都同步 CUDA；off/on 在同一次推理中成对记录。
  计时期间不做指标计算、有限值检查、日志写入或预测归档。
  为取得 off 终点，在 SOR 之前保留一次同步和峰值显存读取，其开销包含在 on 总时间中。
- 两组均排除输入的外部 H2D、最终输出的外部 D2H、归一化/反归一化、文件 I/O、指标计算；
  **SOR 内部必需的数据搬运已计入 on 组**。这里仍是整云推理（on 组附加 SOR），
  不是包含全部预处理和文件读取的端到端时间。
- 预热也包含 SOR。CD/P2M 均在所有计时结束之后计算，on 组复用首轮实际计时生成的
  SOR 输出，不重复过滤；非效率场景仍对同一预测分别计算 off/on 质量。

`latency_mean_std_seconds` 为各模式全部云 × 三轮实际总耗时的均值 ± 总体标准差，
保留三位小数；on 组标准差直接由 on 总耗时计算，不将 off 和 SOR 的标准差相加。
`core_latency_mean_seconds` 保留共同的核心推理均值，`sor_overhead_mean_seconds` 为
同次测量的 on−off 平均差值（包含 SOR 搬运与终点测量开销）。
另保留原始毫秒值、各轮均值标准差、参数量、实际分配峰值与缓存预留峰值。
显存分别取两个计时终点之前的 PyTorch allocator 峰值，包含模型和输入，不包含质量指标阶段，
也不包含 Open3D SOR 使用的 CPU 内存。
不能把配置的 48 GiB 预算当成实测峰值。没有测 Teacher10，因此本次不生成加速比。

## 服务器命令

先将本次修改的两个公共文件和三个新增运行/文档文件同步至服务器同一分支：

```text
tools/benchmark_teacher10_six.py
utils/denoise_benchmark_report.py
tools/benchmark_teacher4_generalization.py
utils/teacher4_generalization.py
docs/teacher4_generalization.md
```

在 Docker 内已经安装原 PointGPT CUDA 扩展、Open3D、PyTorch3D 的环境中，
先进入**容器内的实际项目根目录**，然后执行以下命令。权重和 data 目录必须已挂载到
容器可见位置；不要直接使用容器中不存在的宿主机 `/home/sapi/...` 路径。

```bash
OUT_ROOT="experiments/efficiency/teacher4_generalization_sor_$(date +%Y%m%d_%H%M%S)_$$"
CUDA_VISIBLE_DEVICES=0 python -u tools/benchmark_teacher4_generalization.py \
  --teacher_ckpt experiments/L_consistency_plus/ckpt-best.pth \
  --data_root data \
  --other_noise_root data/other-noise \
  --pcnet_noisy_root data/PCNet/noisy \
  --output_root "$OUT_ROOT" \
  --device 0 --seed 0 --patch_batch 1 --warmup 2 --repeats 3
```

脚本先打印所有场景、云数、SOR 和效率范围，再开始模型评估。不要求手工修改 YAML。
如需只核对目录、完整配对及文件哈希，在同一命令末尾加 `--plan_only`；
该模式不加载权重对象、不导入模型、不执行 CUDA，也不创建结果目录。

如中断，保留相同代码、权重、数据和测量设置，重新运行原命令并加 `--resume`，
`--output_root` 使用之前打印的 `RESULT_ROOT`（不要生成新时间戳）。
已完成场景会重新核对记录，未完成场景从该场景重新跑。
本次计时协议为 schema 2，需要新建结果目录，不能用 `--resume` 接续旧版仅计 SOR-off
的 schema 1 任务。旧版完整结果仍可用 `--report_only` 查看，但不能从旧数据补算未测的 on 耗时。

## 输出

- `summary.csv` / `summary.json`：全部场景及 SOR 模式的 CD/P2M、模型信息。
  `sor_enabled` 区分开关；只有非效率场景的计时/显存为空。
- `efficiency.csv`：10 行效率结果，包含各模式秒/云的均值 ± 标准差及参数量、显存。
  `timing_scope` 明确区分整云推理和整云推理加 SOR。
- `summary.md`：可直接查看的质量与效率表格。
- `per_cloud.csv`：逐云质量，保留 SOR 后点数。
- `timing_runs.csv`：六场景逐云逐轮的原始计时及显存，`sor_enabled` 区分模式。
  场景目录另存 `timing_runs_sor.csv`，on 总耗时保留独立原始记录。
- `plan.json`、各场景 `manifest.json` / `run.log`、源码与配置快照：
  数据/mesh/权重 SHA256、运行设置、GPU 与软件环境。ZIP 不含权重和数据二进制。
- 完成后自动生成与输出目录同名的 `.zip`。

若三类均为 10k/50k × 1%/2%/3%，共有 18 个推理场景、30 行质量汇总、10 行效率汇总。
脚本也会纳入发现到的其他分辨率/噪声场景，不将它们遗漏。

只从完整记录重建表格与 ZIP：

```bash
python tools/benchmark_teacher4_generalization.py --report_only /absolute/path/to/RESULT_ROOT
```

开发验证：`tests/test_teacher4_generalization.py` 是不加载模型的单元检查，覆盖目录发现、
配对缺失、重复场景、数据改变、可变云数、SOR 删点及计时配对、十行效率汇总。
实际 CUDA 结果仍需以上服务器命令验证。
