"""Manually test a completed Teacher10 endpoint run in six conditions and zip metrics."""

import argparse
import copy
import csv
from datetime import datetime
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
DENSE_NOISE_REFERENCE = REPO_ROOT / 'cfgs/PointGPT-L/dense_noise_reference_punet.json'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_metrics(output, points, noise, checkpoint, expected_schedule):
    summary = json.loads((output / 'test_summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    with (output / 'test_metrics.csv').open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    if not (summary['complete'] and summary['full_dataset']
            and all(summary[key] == 20 for key in ('processed_shapes', 'expected_shapes', 'dataset_shapes'))
            and len(rows) == 20 and len({r['name'] for r in rows}) == 20):
        raise RuntimeError(f'{output.name}: expected full evaluation of 20 unique clouds')
    if Path(manifest['checkpoint']).resolve() != checkpoint or manifest['schedule'] != expected_schedule:
        raise RuntimeError(f'{output.name}: checkpoint/schedule mismatch')
    protocol = summary['protocol']
    if (protocol['teacher_steps'] != 10 or protocol['student_steps'] != 4
            or protocol['teacher_nodes'] != [0, 2, 5, 7, 10]):
        raise RuntimeError(f'{output.name}: expected Teacher10 / Student4')
    for row in rows:
        if (int(row['input_points']) != points or abs(float(row['sigma0']) - noise) > 1e-5
                or not all(math.isfinite(float(row[k])) for k in ('cd_x1e4', 'p2m_x1e4'))):
            raise RuntimeError(f'{output.name}: invalid noise, point count or metric')
    cd = sum(float(r['cd_x1e4']) for r in rows) / 20
    p2m = sum(float(r['p2m_x1e4']) for r in rows) / 20
    if abs(cd - summary['mean_cd_x1e4']) > 1e-8 or abs(p2m - summary['mean_p2m_x1e4']) > 1e-8:
        raise RuntimeError(f'{output.name}: CSV and summary means differ')
    return dict(condition=output.name, cd_x1e4=cd, p2m_x1e4=p2m, score=cd + 0.3 * p2m)


def dense_noise_acceptance(results, reference):
    metrics = ('cd_x1e4', 'p2m_x1e4')
    old_by_condition = reference['reference_metrics']
    new_by_condition = {row['condition']: row for row in results}
    if len(results) != 6 or new_by_condition.keys() != old_by_condition.keys():
        raise ValueError('Acceptance requires all six reference conditions')
    target = reference['target_condition']
    comparisons = []
    for condition, original in old_by_condition.items():
        candidate = {key: new_by_condition[condition][key] for key in metrics}
        delta = {key: candidate[key] - original[key] for key in metrics}
        passed = (all(value < 0 for value in delta.values()) if condition == target
                  else all(value <= 0 for value in delta.values()))
        comparisons.append(dict(condition=condition, reference=original, candidate=candidate,
                                delta=delta, passed=passed))
    vanilla_kd = reference['vanilla_kd_target_metrics']
    return dict(accepted=all(row['passed'] for row in comparisons), comparisons=comparisons,
                beats_vanilla_kd_at_50k_3pct=all(new_by_condition[target][key] < vanilla_kd[key]
                                               for key in metrics), reference_protocol=reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train_root', type=Path, required=True, help='A completed training run directory')
    parser.add_argument('--output_root', type=Path, help='Fresh result directory; default is next to train')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.device < 0 or args.seed < 0:
        raise ValueError('device and seed must be nonnegative')
    train = args.train_root.expanduser().resolve()
    for path in (train / 'manifest.json', train / 'validation_manifest.json', train / 'train.jsonl'):
        if not path.is_file():
            raise FileNotFoundError(path)
    training = json.loads((train / 'manifest.json').read_text(encoding='utf-8'))
    config = training['config']
    from utils.teacher_endpoint import resolve_teacher_endpoint, resolve_endpoint_finetune
    endpoint = resolve_teacher_endpoint(config)
    finetune = resolve_endpoint_finetune(config)
    adaptation = (finetune or {}).get('dense_noise_adaptation')
    if endpoint is None and finetune is None:
        raise ValueError('This command requires a Teacher10 endpoint-alignment run')
    if config['dataset']['_base_']['DATASET'] != 'PUNet':
        raise ValueError('This six-condition command requires the PUNet benchmark')
    logs = [json.loads(line) for line in (train / 'train.jsonl').read_text(encoding='utf-8-sig').splitlines()
            if line.strip()]
    planned_epochs = training['arguments'].get('epochs') or config['epochs']
    first_epoch = 0 if finetune is not None else 1
    if [r['epoch'] for r in logs] != list(range(first_epoch, planned_epochs + 1)):
        raise ValueError('Training has not completed the full declared epoch plan')
    best = min(logs, key=lambda row: row['val_rollout_score'])
    if logs[-1]['best_epoch'] != best['epoch']:
        raise ValueError('Inconsistent best tracking in train.jsonl')
    for name, expected in training.get('source_sha256', {}).items():
        if sha256_file(REPO_ROOT / name) != expected:
            raise ValueError(f'Current source differs from the training manifest: {name}')

    evaluated = best
    checkpoint = train / 'ckpt-best.pth'
    reference = None
    if adaptation:
        reference = json.loads(DENSE_NOISE_REFERENCE.read_text(encoding='utf-8'))
        if (planned_epochs != 1 or logs[1]['dense_noise_active_patches'] <= 0
                or args.seed != reference['test_seed']
                or adaptation['reference_sha256'] != reference['reference_checkpoint_sha256']
                or training['initialization']['checkpoint_sha256'] != reference['reference_checkpoint_sha256']):
            raise ValueError('Dense-noise acceptance requires the pinned reference, seed and full epoch-1 run')
        # Evaluate the predeclared candidate even if the legacy validation score
        # selected epoch 0. All six scenes use the same epoch-1 parameters.
        evaluated = logs[1]
        checkpoint = train / 'ckpt-last.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    # Read the actual selected checkpoint before starting GPU inference.
    import torch
    load_options = dict(map_location='cpu')
    if 'weights_only' in inspect.signature(torch.load).parameters:
        load_options['weights_only'] = False
    saved = torch.load(checkpoint, **load_options)
    if (saved['epoch'] != evaluated['epoch'] or saved['distillation'] != training['schedule']
            or saved.get('teacher_endpoint') != endpoint
            or saved.get('training_input') != config.get('training_input')
            or saved.get('teacher_endpoint_protocol') != training.get('teacher_endpoint_protocol')
            or saved.get('endpoint_finetune') != finetune
            or saved.get('endpoint_finetune_protocol') != training.get('endpoint_finetune_protocol')):
        raise ValueError(f'{checkpoint.name} does not match this run / evaluated epoch / objective')
    endpoint_protocol = saved.get('teacher_endpoint_protocol')
    if ((endpoint or {}).get('mask_pre_bn_conv_bias', False)
            and (not isinstance(endpoint_protocol, dict) or not endpoint_protocol.get('excluded_parameters'))):
        raise ValueError('Masked endpoint run is missing its actual excluded parameter names')
    if finetune is not None:
        if (saved.get('initialization') != training.get('initialization')
                or not saved.get('endpoint_finetune_protocol', {}).get('frozen_parameters')
                or logs[0].get('phase') != 'initial_validation'
                or not all(row.get('frozen_buffers_verified') and row.get('frozen_biases_verified') for row in logs)):
            raise ValueError('Endpoint fine-tuning initialization or frozen-state verification is incomplete')
    del saved

    import yaml
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    variant = 'endpoint_masked' if (endpoint or {}).get('mask_pre_bn_conv_bias', False) else 'endpoint'
    if adaptation:
        variant = 'dense_noise_finetune'
    elif finetune is not None:
        variant = 'endpoint_finetune'
        if finetune.get('local_weight', 0) > 0:
            variant = ('local_endpoint_finetune' if finetune.get('endpoint_weight', 1) > 0
                       else 'local_finetune')
    result_root = (args.output_root.expanduser().resolve() if args.output_root else
                   train.parent / f'six_condition_metrics_10to4_{variant}_{stamp}_{os.getpid()}')
    archive_path = result_root.with_suffix('.zip')
    if archive_path.exists():
        raise FileExistsError(archive_path)
    result_root.mkdir(parents=True, exist_ok=False)
    config_root = result_root / 'configs'
    config_root.mkdir()
    selection = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
                     epoch=evaluated['epoch'], validation_score=evaluated['val_rollout_score'],
                     schedule=training['schedule'], teacher_endpoint=endpoint,
                     teacher_endpoint_protocol=endpoint_protocol,
                     test_seed=args.seed, results=[])
    if finetune is not None:
        selection.update(endpoint_finetune=finetune,
                         endpoint_finetune_protocol=training['endpoint_finetune_protocol'],
                         initialization=training['initialization'],
                         epoch0_validation_score=logs[0]['val_rollout_score'])
    if adaptation:
        selection.update(selection_rule='predeclared_epoch_1', validation_best_epoch=best['epoch'])
    print(f'TRAIN_ROOT={train}\nevaluated_epoch={evaluated["epoch"]}\n'
          f'best_epoch={best["epoch"]}\nRESULT_ROOT={result_root}', flush=True)
    outputs = []
    for points in (10000, 50000):
        for noise in (0.01, 0.02, 0.03):
            label = f'{points // 1000}k_{round(noise * 100)}pct'
            test_config = copy.deepcopy(config)
            dataset = test_config['dataset']['_base_']
            dataset.update(TEST_RESOLUTION=f'{points}_poisson', TEST_NOISE=noise,
                           TEST_NOISY_DIR=f'PUNet_{points}_poisson_{noise:.2f}')
            dataset.pop('TEST_NOISY_PATH', None)
            config_path = config_root / f'{label}.yaml'
            config_path.write_text(yaml.safe_dump(test_config, sort_keys=False, allow_unicode=True),
                                   encoding='utf-8')
            output = result_root / label
            subprocess.run([
                sys.executable, '-u', str(REPO_ROOT / 'tools/runner_distill.py'),
                '--mode', 'test', '--config', str(config_path), '--student_ckpt', str(checkpoint),
                '--output_dir', str(output), '--device', str(args.device), '--seed', str(args.seed),
                '--max_shapes', '0'], cwd=REPO_ROOT, check=True)
            metrics = check_metrics(output, points, noise, checkpoint, training['schedule'])
            selection['results'].append(metrics)
            outputs.append((label, output))
            print(json.dumps(metrics), flush=True)
    selection_path = result_root / 'selected_checkpoint.json'
    selection_path.write_text(json.dumps(selection, indent=2) + '\n', encoding='utf-8')
    if adaptation:
        acceptance = dense_noise_acceptance(selection['results'], reference)
        acceptance_path = result_root / 'quality_acceptance.json'
        acceptance_path.write_text(json.dumps(acceptance, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in ('train.jsonl', 'manifest.json', 'validation_manifest.json'):
            archive.write(train / name, f'training/{name}')
        archive.write(selection_path, 'selected_checkpoint.json')
        if adaptation:
            archive.write(acceptance_path, 'quality_acceptance.json')
        for label, output in outputs:
            for name in ('manifest.json', 'test_summary.json', 'test_metrics.csv'):
                archive.write(output / name, f'{label}/{name}')
    print(f'ZIP={archive_path}', flush=True)
    if adaptation:
        print(f'ACCEPT_CANDIDATE={int(acceptance["accepted"])}', flush=True)


if __name__ == '__main__':
    main()
