#!/usr/bin/env bash
# Manual launch only. Activate the pointgpt environment before running.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TEACHER_CKPT="${1:-experiments/L_consistency_plus/ckpt-best.pth}"
export TRAIN_SEED="${2:-0}"
export RUN_ROOT="experiments/input_ablation/mixed_matched_targets_seed${TRAIN_SEED}_$(date +%Y%m%d_%H%M%S)_$$"

python - <<'PY'
import os
from pathlib import Path
import yaml

teacher = Path(os.environ['TEACHER_CKPT'])
if not teacher.is_file():
    raise FileNotFoundError(teacher.resolve())
config = yaml.safe_load(Path('cfgs/PointGPT-L/distill_16to4_input_matched.yaml').read_text(encoding='utf-8'))
data = yaml.safe_load(Path(config['dataset']['_base_']).read_text(encoding='utf-8'))
if (data['NOISE_MIN'], data['NOISE_MAX']) != (.005, .02):
    raise ValueError('Keep baseline training noise at 0.5%-2%; 3% is evaluation only.')
if config['curriculum_mode'] != 'fixed' or config['distillation']['teacher_nodes'] != [0, 4, 8, 12, 16]:
    raise ValueError('This experiment requires fixed equal nodes.')
if config['training_input'].get('student_target') != 'teacher_from_student_input':
    raise ValueError('Matched-input Teacher targets are required.')
config['dataset']['_base_'] = data
root = Path(os.environ['RUN_ROOT'])
root.mkdir(parents=True, exist_ok=False)
(root / 'train_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
print(f'Teacher: {teacher.resolve()}', flush=True)
print(f'Output: {(root / "train").resolve()}', flush=True)
PY

python -u tools/runner_distill.py \
  --mode train \
  --config "$RUN_ROOT/train_config.yaml" \
  --teacher_ckpt "$TEACHER_CKPT" \
  --output_dir "$RUN_ROOT/train" \
  --device 0 --seed "$TRAIN_SEED" --epochs 20
