# 基础 Teacher10 → Student4：训练、六项质量与效率

分支：`codex/paper2-teacher10-basic-distill`。
这是项目内的朴素轨迹蒸馏对照，不声称复现某篇论文的完整方法。

## 对照定义

- 原教师为 `experiments/L_consistency_plus/ckpt-best.pth`，不能传入已蒸馏学生。
- 从原教师去噪权重复制初始化 Student，并新建相同的 step conditioning。
- 固定节点 `[0,2,5,7,10]`，Teacher10 的 eta=0.3、decay=0.95；Student4 的 eta=1。
- 每段输入为原教师轨迹的起点，目标为该轨迹下一节点：
  `T0→T2`、`T2→T5`、`T5→T7`、`T7→T10`。
- 四段对应点平方 L2 损失等权取平均，独立反传后每个 optimizer batch 更新一次。
  Teacher 目标 detach；训练期间不用学生连续输出作为后段输入。
- 没有 Mixed 概率、学生输入匹配目标、终点辅助、梯度遮罩、动态节点或第二阶段微调。
- Student 按原第一阶段的 train 模式更新，包括 BN 与 bias；不套用第二阶段冻结规则。
- 优化器 fresh AdamW，LR=1e-5、weight decay=0、batch=8、clip=1，完整训练 **20 epochs**。
- 训练噪声保持 **0.5%–2%**，log-uniform；clean/CD/P2M 不进入蒸馏训练损失。
- 推理仍是四步连续 Student rollout，分块、融合、评分与已有学生相同；关闭 SOR/projection。

配置为 `cfgs/PointGPT-L/distill_10to4_baseline.yaml`。
与当前第一阶段20轮比较时预算对齐；当前最终方案还额外训练了第二阶段，比较时需单列。
如要隔离第二阶段的贡献，需要另做同等额外训练的对照，不能把额外训练量隐去。

## 验证与来源记录

每个 epoch 做完整20云验证，以 `Score=CD+0.3*P2M` 选 best；不要求最后一轮胜出。
基础训练没有新增 epoch0。沿用当前数据设置：**10k/1%、VAL_NUM=0、split=test**，
不将其描述为独立留出的验证集；不改变当前评分或为单个方法切换选模协议。

训练 manifest 增加 `baseline_distillation_protocol`、原教师 SHA256，以及训练和推理源码指纹。
评测先检查三个训练记录、完整1–20轮计划、输入/目标策略、初始化教师文件和当前源码，
再在 CPU 读取实际 `ckpt-best.pth` 元数据，检查选中的 epoch、Score、模型和 schedule。
Teacher-forced checkpoint 沿用旧格式，不另存默认 `training_input`；评测按该语义核对。
缺失记录、未完成20轮或训练后改动被审计源码均会拒绝，不绕过检查。

## 一整段训练、质量＋效率评测、ZIP 命令

先把本分支修改同步至服务器 `/workspace`，包括新增配置、文档与三个修改的 Python 文件。
使用既有 `pointgpt` 环境。不自动执行任何训练、测试、CPU 单测或 GPU 程序。

```bash
conda activate pointgpt
(
set -euo pipefail
cd /workspace
CONFIG="cfgs/PointGPT-L/distill_10to4_baseline.yaml"
TEACHER="experiments/L_consistency_plus/ckpt-best.pth"
RUN_ROOT="experiments/input_ablation/basic_10to4_seed0_$(date +%Y%m%d_%H%M%S)_$$"
OUT_ROOT="$RUN_ROOT/basic_10to4_quality_efficiency_$(date +%Y%m%d_%H%M%S)_$$"
test -f "$TEACHER"
printf 'RUN_ROOT=%s\nOUT_ROOT=%s\n' "$RUN_ROOT" "$OUT_ROOT"

# 训练：原教师初始化，完整20轮；不传 init_student_ckpt。
CUDA_VISIBLE_DEVICES=0 python -u tools/runner_distill.py --mode train \
  --config "$CONFIG" --teacher_ckpt "$TEACHER" \
  --epochs 20 --output_dir "$RUN_ROOT/train" --device 0 --seed 0

# 测试：自动选验证 best，一次完成六项质量、效率、绘图和ZIP。
CUDA_VISIBLE_DEVICES=0 python -u tools/benchmark_teacher10_six.py \
  --baseline_train_root "$RUN_ROOT/train" \
  --output_root "$OUT_ROOT" \
  --device 0 --seed 0 --patch_batch 1 --warmup 2 --repeats 3
)
```

两条 Python 命令按顺序执行，训练失败不会继续评测。`RUN_ROOT` 含时间戳与PID，拒绝覆盖旧实验。
评测仅新增 `student_baseline` 六项任务，**不重跑旧教师或已有学生的效率结果**。
`--baseline_train_root` 必须是包含 `ckpt-best.pth` 的 **train 目录**。
训练后不要修改被指纹审计的源码；返回最后打印的 `ZIP=<absolute_path>`。

在另一个 shell 单独评测时，把 `TRAIN_ROOT` 改为实际打印的训练目录：

```bash
conda activate pointgpt
cd /workspace
TRAIN_ROOT="/workspace/experiments/input_ablation/basic_10to4_seed0_<时间戳>_<PID>/train"
OUT_ROOT="$(dirname "$TRAIN_ROOT")/basic_10to4_quality_efficiency_$(date +%Y%m%d_%H%M%S)_$$"
CUDA_VISIBLE_DEVICES=0 python -u tools/benchmark_teacher10_six.py \
  --baseline_train_root "$TRAIN_ROOT" --output_root "$OUT_ROOT" \
  --device 0 --seed 0 --patch_batch 1 --warmup 2 --repeats 3
```

评测中断后，保持同一个 `OUT_ROOT` 和全部设置，在评测命令末尾加 `--resume`。
它只跳过完整且经原始CSV复核的任务；不再次训练。
全部任务已完成时，可只重建汇总、图和ZIP：

```bash
python tools/benchmark_teacher10_six.py --report_only "$OUT_ROOT"
```

## 质量、效率与输出

六条件为10k/50k × 1%/2%/3%，每项20个唯一点云。
每项完整整云预热2次，正式计时3轮：共 **6 jobs、120云质量、360条计时、12次预热**。
每项独立子进程、仅一个模型，串行执行，FP32、patch batch=1。

计时口径完全复用现有测量：GPU上的归一化整云输入到GPU上的整云输出，
包含FPS/KNN、分块、四步模型迭代和融合；CUDA前后同步。
排除加载、I/O、CPU预处理、H2D/D2H、RNG设置与CD/P2M计算。
完成全部计时与显存记录后，使用第一轮预测计算质量，不做三轮投票。

输出包括：

- `summary.csv/json`：六项 CD、P2M、Score、平均/中位延迟、延迟波动、clouds/s、points/s、
  PyTorch allocated/reserved 峰值、参数量、每云patch数及网络调用数。
- `per_cloud.csv`：120云的质量与计时汇总；`timing_runs.csv`：360条原始计时。
- `plots/`：六条件耗时—CD/P2M图及分条件图，PNG/PDF。
- `plan.json`、每项 manifest、训练三份记录、源码副本、协议与运行日志。
- 最终 `basic_10to4_quality_efficiency_<时间戳>_<PID>.zip`，不包含checkpoint权重或大型点云。

本次没有重测Teacher10，所以 `speedup_vs_teacher10` 等相对列留空/null，
不把10/4=2.5当成实测加速比，也不把历史耗时写成本次教师实测。
之后与已保存教师/学生结果作历史对比时，需要注明环境和跨run测量限制。
参数量与显存如实报告；同结构四步模型不会因为目标更简单就自动变得更小或更快。
吞吐为串行推理区间吞吐，显存限于PyTorch allocator；48GiB配置上限不是实测峰值。

## 验证状态

本地仅做Python3.8语法AST检查、配置静态对照和Git差异检查。
未执行训练、推理、GPU诊断、CPU单测或任何模型benchmark。
真实CUDA执行、质量和效率结果须由用户在服务器手动运行后确认。
