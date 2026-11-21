# 冻结 Student 的四步幅度校准

上一版密集噪声微调在 50k/3% 得到 CD/P2M 1.271050/0.859121，前五个场景各有至少一项退化。
这版从前五个场景表现较好的原联合监督一轮模型起步。训练仅优化四个幅度系数，原 Student
权重、step condition 和 BN 缓冲区在每轮保存前逐项与初始 checkpoint 核对。

对第 i 步，Student 的原预测为 `y`，当前坐标为 `x`，校准后的结果为
`y + a * delta_i * (y - x)`。`delta_i = 0.25 * tanh(logit_i)`，初始 `delta_i=0`，
部署时每一步的位移幅度相对原输出限制在 0.75–1.25 倍。

- 训练噪声保持 0.5%–2%。用至少 50,000 点的训练源云中 σ>1.25% 的块拟合系数，拟合时应用完整修正 `a=1`。使用加权局部教师损失和 `0.1 ×` 配对干净块的终点 MSE，后者约束去噪结果接近干净坐标。使用一轮、学习率 0.003、种子 0。
- 推理从输入读取点数和已知的初始 σ。点数小于 50,000 或 σ≤2% 时直接执行原四步。达到点数门槛且 σ 大于 2% 时线性增强修正，到 3% 后保持完整幅度。每个场景使用同一 checkpoint，每个 patch 仍执行四次网络调用。该门控依赖外部提供的 σ，泛化实验需明确噪声参数是否可得。
- 50k/3% 超过训练噪声上限，这一条件下的泛化与实际质量收益待六场景评估。六场景是对已知短板的定向验收。使用原 CD/P2M 和 `CD + 0.3 * P2M`。

## 训练一轮

同步 `codex/paper2-teacher10-dense-noise-step-calibration` 到服务器 `/workspace`。运行会校验原联合监督权重的 SHA256。

```bash
conda activate pointgpt
(
set -euo pipefail
cd /workspace

CONFIG="cfgs/PointGPT-L/distill_10to4_dense_noise_step_calibration.yaml"
TEACHER="experiments/L_consistency_plus/ckpt-best.pth"
INIT="experiments/input_ablation/mixed_10to4_local_endpoint_finetune_1epoch_seed0_20261118_035447_40/train/ckpt-best.pth"
TRAIN_ROOT="experiments/input_ablation/mixed_10to4_dense_noise_step_calibration_seed0_$(date +%Y%m%d_%H%M%S)_$$/train"

test -f "$TEACHER"
test -f "$INIT"
printf 'TRAIN_ROOT=%s\n' "$TRAIN_ROOT"

CUDA_VISIBLE_DEVICES=0 python -u tools/runner_distill.py --mode train \
  --config "$CONFIG" --teacher_ckpt "$TEACHER" --init_student_ckpt "$INIT" \
  --epochs 1 --output_dir "$TRAIN_ROOT" --device 0 --seed 0
)
```

## 测试六场景并统计效率

训练完成后，用训练打印的路径替换 `TRAIN_ROOT`。固定评估第 1 轮 `ckpt-last.pth`，验证分数继续按原逻辑记录。

```bash
conda activate pointgpt
(
set -euo pipefail
cd /workspace

TRAIN_ROOT="experiments/input_ablation/替换成训练打印的目录/train"
OUT_ROOT="experiments/efficiency/dense_noise_step_calibration_$(date +%Y%m%d_%H%M%S)_$$"
CUDA_VISIBLE_DEVICES=0 python -u tools/benchmark_teacher10_six.py \
  --dense_noise_train_root "$TRAIN_ROOT" --output_root "$OUT_ROOT" \
  --device 0 --seed 0 --patch_batch 1 --warmup 2 --repeats 3
)
```

ZIP 保存六个场景的 `summary.csv`、逐云质量、360 条计时记录及 `quality_acceptance.json`。
各场景的 manifest 记录实际启用的四个幅度差值，前五个场景应为 `null`。验收依旧逐项检查
前五个场景的 CD/P2M，要求 50k/3% 的两项同时改善，同时报告是否同时低于普通蒸馏。
