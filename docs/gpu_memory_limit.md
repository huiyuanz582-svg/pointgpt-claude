# 48 GiB 显存预算与 OOM 退出

已接入 `main.py`（预训练、微调、测试）、`tools/runner_distill.py`（蒸馏训练、测试）和 `tools/diagnose_distill_trajectory.py`（轨迹审计）。从这些 CLI 启动时默认生效；直接调用 runner 函数或运行其他独立分析/冒烟脚本不在此覆盖范围。

## 默认行为

在模型及 checkpoint 搬到 GPU 前，通过 `utils/gpu_memory.py` 设置 PyTorch CUDA 缓存分配器上限。配置缺省时也会限制；无需给启动命令新增参数。

```yaml
gpu_mem_limit_gib: 48
gpu_mem_headroom_gib: 1
gpu_mem_fraction: 0.9
```

单位为 GiB（`1024**3` 字节）。默认以 48 GiB 为目标，给 CUDA 上下文、库和其他分配器外开销预留 1 GiB，因此 PyTorch 缓存分配器实际最多使用 **47 GiB（48128 MiB）**。预算包括分配器保留的缓存，不只是活跃张量。

实际分配器预算为 `min(min(48 GiB, GPU 总显存) - 预留, GPU 总显存 × gpu_mem_fraction)`。例如 80 GiB 卡原先的 90% 设置约为 72 GiB，现在会被 47 GiB 预算收紧。较小显卡也会扣除预留；`gpu_mem_fraction` 只可进一步收紧上限，缺省或 0 不会关闭固定预算。`gpu_mem_limit_gib` 可以调低，不能超过 48；预留不能小于 1 GiB。

新的分配将突破分配器预算时，PyTorch 拒绝分配并抛出 CUDA OOM。入口捕获后以 **退出码 86** 结束当前 Python 进程，不跳过 batch、不重试、不做 OOM 应急 checkpoint；已有正常 checkpoint 保留。原有正常 checkpoint 保存逻辑和微调的周期性系统内存压力检查仍保留。日志中会显示 `[GPU memory limit]`、实际预算和退出原因；蒸馏及审计 manifest 记录 `gpu_memory_limit`。

设置失败时直接报错退出，不允许无上限继续运行。要求 PyTorch 的 `native` CUDA 分配器；`cudaMallocAsync` 或自定义分配器会被拒绝。实现使用部署环境 PyTorch 2.0.1 已有的接口，不增加依赖。

## Docker 范围

这是**每个 Python 进程、每张 CUDA 设备的 PyTorch 分配器限制**，使用 Docker 内的逻辑设备编号。单进程单卡训练适用；DataParallel 对每张可见卡分别设置，DDP 对每个进程的当前卡设置。多卡、多个进程或多个容器的预算会叠加，不能把它理解为整个容器合计 48 GiB。

PyTorch 分配器不能限制 CUDA 上下文、NCCL、某些库/扩展自行申请的显存。1 GiB 是预留而非这些分配的硬限制，**不能保证 `nvidia-smi` 显示的进程总占用绝不超过 48 GiB**，也没有针对这些外部分配的总量监控。因此代码提供的是分配器预算和 OOM 退出，不是严格的 Docker 显存隔离。如果必须硬隔离全部显存，需要服务器层面的 GPU 隔离/分区配置，不能靠这一 Python 接口保证。

Docker `--memory` 限制的是主机 RAM，不是 GPU 显存。若之前宕机由主机 RAM 耗尽、驱动错误或其他进程造成，本保护不能覆盖。此改动没有更改 Docker 资源配置、停止其他任务或修改服务器。

## 部署与验证状态

需同步这些文件后启动新的 Python 进程：`utils/gpu_memory.py`、`main.py`、`tools/runner_pretrain.py`、`tools/runner_finetune.py`、`tools/runner_distill.py`、`tools/diagnose_distill_trajectory.py` 及动态训练 YAML。已有运行进程不会自动获得限制。外层任务调度或 Docker restart 策略也可能重启失败任务；代码本身不重试。

本次只进行源码静态检查；未启动训练、测试、诊断或 GPU 程序，也未在服务器部署。`tests/test_gpu_memory.py` 提供不依赖真实 GPU 的预算、设置失败及退出行为用例，本次未执行。后续若获准验证，应先确认启动日志和 manifest 中的预算，再在隔离环境验证 OOM 退出，不能在共享服务器上刻意填满显存。

接口范围见 [PyTorch 2.0.1 CUDA memory 源码](https://github.com/pytorch/pytorch/blob/v2.0.1/torch/cuda/memory.py)；主机内存限制见 [Docker resource constraints](https://docs.docker.com/engine/containers/resource_constraints/)。
