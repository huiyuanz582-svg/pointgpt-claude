# Teacher10 → Student4 高密度噪声微调

当前局部与终点联合监督的 Student 在前五个场景优于普通蒸馏，在 50k/3% 的
CD/P2M 为 1.284743/0.867593，普通蒸馏为 1.246132/0.830399（均为 ×10⁴）。纯终点
微调从一轮延长到三轮后，该场景的两项指标进一步变差。本实验尝试调整监督目标，
以当前联合监督一轮的 `ckpt-best.pth` 为起点。

新配置从同一份完整训练集采样。源云至少有 50,000 点时，局部教师损失的权重从
σ=1.25% 时的 0 线性增加到 σ=2% 时的 1。其余 patch 的局部损失权重为 0。
每个 patch 都用冻结的初始 Student 走四步，作为候选 Student 的终点目标。这项约束
限制预测漂移，能否保护前五个场景由整云结果检验。

每个 patch 的损失为 `w · mean(L_local_0, …, L_local_3) + L_reference_endpoint`，再
对整个 batch 求均值。局部 Teacher 从候选 Student 的实际阶段输入生成对应的 2/3/2/3 步
目标。终点锚定与局部目标均停止梯度。候选四步连续反传，沿用固定 BN 统计和冻结
pre-BN Conv bias。全局 Teacher10 终点继续用于既有验证日志的 `val_loss_teacher_endpoint`。

训练噪声为 **0.5%–2%**，随机种子为 0，新配置训练一轮，学习率为 `5e-7`。该轮额外
使用一个冻结 Student，增加训练时的模型显存。测试每个场景只加载同一个四步候选，
沿用原分块、融合、CD、P2M 和 `CD + 0.3 * P2M` 口径。

## 服务器手动运行

先将当前分支的改动同步到服务器 `/workspace`。下方路径指向已完成的联合监督一轮
模型，训练会校验其 SHA256。两段命令可分别在不同 Bash 会话执行。

### 训练一轮

```bash
conda activate pointgpt
(
set -euo pipefail
cd /workspace

CONFIG="cfgs/PointGPT-L/distill_10to4_dense_noise_finetune.yaml"
TEACHER="experiments/L_consistency_plus/ckpt-best.pth"
INIT="experiments/input_ablation/mixed_10to4_local_endpoint_finetune_1epoch_seed0_20261118_035447_40/train/ckpt-best.pth"
RUN_ROOT="experiments/input_ablation/mixed_10to4_dense_noise_finetune_seed0_$(date +%Y%m%d_%H%M%S)_$$"
TRAIN_ROOT="$RUN_ROOT/train"

test -f "$TEACHER"
test -f "$INIT"
printf 'TRAIN_ROOT=%s\n' "$TRAIN_ROOT"

CUDA_VISIBLE_DEVICES=0 python -u tools/runner_distill.py --mode train \
  --config "$CONFIG" --teacher_ckpt "$TEACHER" --init_student_ckpt "$INIT" \
  --epochs 1 --output_dir "$TRAIN_ROOT" --device 0 --seed 0
)
```

### 测试六场景与统计效率

用训练打印的 `TRAIN_ROOT` 替换下面的路径。命令在六个场景逐个加载第 1 轮候选。

```bash
conda activate pointgpt
(
set -euo pipefail
cd /workspace

TRAIN_ROOT="experiments/input_ablation/替换成你的新训练目录/train"
CUDA_VISIBLE_DEVICES=0 python -u tools/benchmark_teacher10_six.py \
  --dense_noise_train_root "$TRAIN_ROOT" --device 0 --seed 0 \
  --patch_batch 1 --warmup 2 --repeats 3
)
```

每场景 20 云，预热 2 次，正式计时 3 轮，共 360 条计时记录。使用与已有对照一致的
FP32、整云推理口径，包含分块、四步迭代与融合。质量用第一轮输出计算。
`summary.csv` 汇总每个场景的 CD、P2M、耗时、吞吐量、显存峰值与参数量。
加速比可用已存档的 Teacher10 均值计算跨运行参考值。

命令打印 `ZIP=...`。ZIP 包含 `summary.csv`、`per_cloud.csv`、`timing_runs.csv`、
`quality_acceptance.json`、训练日志、训练及验证 manifest、评测配置和 checkpoint 元数据。
如中断，在相同参数下用打印的 `RESULT_ROOT` 指定 `--output_root` 并加 `--resume`。

## 验收

训练照既有验证分数维护 `ckpt-best.pth`。六场景固定评估事先指定的第 1 轮 `ckpt-last.pth`，
无论原验证评分选中哪一轮。参考值来自已核验的联合监督测试包，并与初始 Student 的
SHA256 绑定。测试设置沿用 PUNet 原有六场景文件、种子和指标。

- 10k 的 1%/2%/3% 与 50k 的 1%/2%，场景均值 CD、P2M 分别小于或等于参考值。
- 50k/3% 的 CD、P2M 都严格低于当前联合监督模型。报告也单独记录两项是否都低于普通蒸馏。

使用存档全精度数值判定。两条都通过时输出 `ACCEPT_CANDIDATE=1`，否则输出
`ACCEPT_CANDIDATE=0` 并保留全部结果，论文仍使用原 checkpoint 的结果。新的质量结论
须等待整云测试。训练日志中的 `dense_noise_active_patches`、`dense_noise_local_weight_mean`
和 `loss_reference_endpoint` 记录实际加权与约束情况。
