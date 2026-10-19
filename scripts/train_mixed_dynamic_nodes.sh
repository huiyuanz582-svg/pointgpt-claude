#!/usr/bin/env bash
# 手动启动入口。仅在用户执行本脚本时创建新 run 并开始训练。
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_dir"

python_bin="${PYTHON_BIN:-python}"
teacher_ckpt="${TEACHER_CKPT:-experiments/L_consistency_plus/ckpt-best.pth}"
epochs="${EPOCHS:-20}"
seed="${SEED:-0}"
device="${DEVICE:-0}"
run_root="${RUN_ROOT:-experiments/input_ablation/mixed_dynamic_nodes_seed${seed}_$(date +%Y%m%d_%H%M%S)_$$}"

"$python_bin" -u - "$teacher_ckpt" "$run_root" "$epochs" "$seed" "$device" <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml

teacher = Path(sys.argv[1]).expanduser().resolve()
run = Path(sys.argv[2]).expanduser().resolve()
epochs, seed, device = map(int, sys.argv[3:6])
if epochs < 1 or seed < 0 or device < 0:
    raise ValueError('EPOCHS must be positive; SEED and DEVICE must be nonnegative')
if not teacher.is_file():
    raise FileNotFoundError(teacher)
if run.exists():
    raise FileExistsError(f'Use a fresh RUN_ROOT: {run}')

repo = Path.cwd()
template = repo / 'cfgs/PointGPT-L/distill_16to4_input_scheduled_dynamic.yaml'
config = yaml.safe_load(template.read_text(encoding='utf-8'))
if (config['curriculum_mode'] != 'dynamic_pcd'
        or config['training_input']['type'] != 'scheduled_rollout'
        or config['curriculum_metric']['type'] != 'original_pcd'):
    raise ValueError('Expected the Mixed + original-PCD dynamic configuration')
config['epochs'] = epochs
run.mkdir(parents=True, exist_ok=False)
effective_config = run / 'config.yaml'
effective_config.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')

def git_output(*args):
    result = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True,
                            encoding='utf-8', errors='replace')
    return result.stdout.strip() if result.returncode == 0 else None

sources = [
    template.relative_to(repo).as_posix(),
    'cfgs/dataset_configs/ScoreDenoise.yaml',
    'scripts/train_mixed_dynamic_nodes.sh',
    'tools/runner_distill.py',
    'tools/shadow_global_search.py',
    'utils/curriculum_config.py',
    'utils/training_input.py',
    'models/PointGPT.py',
    'models/step_condition.py',
    'datasets/ScoreDenoiseDataset.py',
    'datasets/scoredenoise/transforms.py',
    'tools/runner_finetune.py',
    'utils/p2m_loss.py',
]
command = [sys.executable, '-u', 'tools/runner_distill.py', '--mode', 'train',
           '--config', str(effective_config), '--teacher_ckpt', str(teacher),
           '--output_dir', str(run / 'train'), '--device', str(device), '--seed', str(seed),
           '--max_shapes', '0', '--max_patch_batches', '0']
manifest = dict(
    created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    method='original_mixed_with_epoch_global_pcd_nodes',
    branch=git_output('branch', '--show-current'), head=git_output('rev-parse', 'HEAD'),
    git_status=git_output('status', '--porcelain', '--untracked-files=normal'),
    source_sha256={name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in sources},
    config_sha256=hashlib.sha256(effective_config.read_bytes()).hexdigest(),
    teacher_checkpoint=str(teacher), epochs=epochs, seed=seed, device=device,
    command=command, status='prepared',
)
manifest_path = run / 'source_manifest.json'

def save_manifest():
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

save_manifest()
print(f'RUN_ROOT={run}', flush=True)
print(f'Mixed + dynamic nodes; epochs={epochs}; fresh initialization from {teacher}', flush=True)
try:
    manifest['status'] = 'running'
    save_manifest()
    subprocess.run(command, check=True, cwd=repo)
    records = [json.loads(line) for line in (run / 'train/train.jsonl').read_text(encoding='utf-8').splitlines()]
    if ([row['epoch'] for row in records] != list(range(1, epochs + 1))
            or any(row.get('training_input') != 'scheduled_rollout'
                   or row.get('curriculum_mode') != 'dynamic_pcd'
                   or row.get('val_rollout_score') is None for row in records)):
        raise ValueError('Training did not complete the requested Mixed dynamic epochs and validation')
    for name in ('manifest.json', 'validation_manifest.json', 'calibration_manifest.json',
                 'ckpt-last.pth', 'ckpt-best.pth'):
        if not (run / 'train' / name).is_file():
            raise FileNotFoundError(run / 'train' / name)
    manifest.update(completed_epochs=len(records), best_epoch=records[-1]['best_epoch'],
                    best_val_rollout_score=records[-1]['best_val_rollout_score'])
except BaseException as error:
    manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
    save_manifest()
    raise
else:
    manifest['status'] = 'completed'
    save_manifest()
    print(f'Best checkpoint: {run / "train/ckpt-best.pth"}', flush=True)
PY
