#!/usr/bin/env bash
# 两项独立的只读检查，不启动训练。请从仓库根目录执行。
set -euo pipefail
TEACHER_CKPT="${1:-experiments/L_consistency_plus/ckpt-best.pth}"
STUDENT_CKPT="${2:-experiments/distill_16to4/train1/ckpt-best.pth}"
RESULT_ROOT="${3:-experiments/distill_16to4/stage1_checks}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 正常连续四步 rollout：baseline 同口径 CD/P2M。0 表示全测试集。
python tools/runner_distill.py --mode test \
  --config cfgs/PointGPT-L/distill_16to4.yaml \
  --student_ckpt "$STUDENT_CKPT" \
  --output_dir "$RESULT_ROOT/rollout" \
  --max_shapes "${TEST_MAX_SHAPES:-0}" --save_trajectory

# 独立 Teacher-forced difficulty：默认 64 个 baseline 训练 patch，0 表示一个完整采样 epoch。
python tools/analyze_curriculum_difficulty.py \
  --config cfgs/PointGPT-L/distill_16to4.yaml \
  --teacher_ckpt "$TEACHER_CKPT" --student_ckpt "$STUDENT_CKPT" \
  --output_dir "$RESULT_ROOT/difficulty" \
  --max_samples "${ANALYSIS_MAX_SAMPLES:-64}" --save_trajectories 3
