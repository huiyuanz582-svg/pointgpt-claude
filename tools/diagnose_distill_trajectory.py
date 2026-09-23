"""Read-only Teacher-forced/free-rollout audit; never trains or changes a schedule.

Heavy dependencies are imported only in _run(), after CLI/path checks. See
docs/distill_trajectory_audit.md for metric definitions and server commands.
"""

import argparse
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import uuid


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PATHS = ((0, 4, 8, 12, 16), (0, 7, 10, 13, 16),
         (0, 10, 12, 14, 16), (0, 13, 14, 15, 16))
STATS = ('mean', 'median', 'std', 'p90', 'p95', 'sample_count')
STAGE_KEYS = ('noise', 'path', 'mode', 'scope', 'postprocessing', 'stage',
              'start_step', 'target_step', 'metric', 'unit', 'sample_unit')
FINAL_KEYS = ('noise', 'path', 'mode', 'postprocessing', 'metric', 'unit', 'sample_unit')


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='cfgs/PointGPT-L/distill_16to4_dynamic.yaml')
    parser.add_argument('--checkpoint',
                        default='experiments/distill_16to4_dynamic/full20_seed0/ckpt-best.pth')
    parser.add_argument('--teacher_checkpoint', '--teacher_ckpt',
                        default='experiments/L_consistency_plus/ckpt-best.pth')
    parser.add_argument('--expected_epoch', type=int, default=5)
    parser.add_argument('--expected_checkpoint_nodes', nargs=5, type=int, default=list(PATHS[2]))
    paths = parser.add_mutually_exclusive_group()
    paths.add_argument('--teacher_nodes', nargs=5, type=int, action='append', default=None,
                       help='Repeat for several diagnostic paths; default: 0 10 12 14 16')
    paths.add_argument('--all_paths', action='store_true', help='Evaluate paths A/B/C/D on one checkpoint')
    parser.add_argument('--noise_levels', nargs='+', type=float, default=[0.01, 0.02, 0.03])
    parser.add_argument('--dataset_root', default=None, help='Default: dataset ROOT in YAML')
    parser.add_argument('--clean_root', default=None, help='Parent of <resolution>/*.xyz')
    parser.add_argument('--noisy_path_template', default=None,
                        help='Directory template, e.g. /data/noisy/PUNet_{resolution}_{noise:g}')
    parser.add_argument('--mesh_root', default=None, help='Parent of test/<shape>.off')
    parser.add_argument('--resolution', default=None, help='Default: YAML TEST_RESOLUTION')
    parser.add_argument('--output_dir', default=None, help='Must NOT already exist; default: unique timestamp')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--max_shapes', type=int, default=0, help='0=all; same sorted subset at each noise')
    parser.add_argument('--patch_batch', type=int, default=None, help='Default: YAML test_patch_batch')
    parser.add_argument('--first_stage_atol', type=float, default=1e-5,
                        help='Absolute output tolerance in normalized coordinates; '
                             'inputs/patch indices still require exact equality')
    return parser


def _nodes(values):
    from tools.runner_distill import _teacher_nodes
    return _teacher_nodes(values)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _git_info():
    def read(*args):
        try:
            return subprocess.check_output(['git', *args], cwd=REPO_ROOT, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return dict(commit=read('rev-parse', 'HEAD'), branch=read('branch', '--show-current'),
                status_short=read('status', '--short'))


def _write_manifest(output, manifest):
    temporary = output / 'run_manifest.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                         encoding='utf-8')
    temporary.replace(output / 'run_manifest.json')


def _files(directory):
    if not directory.is_dir():
        raise FileNotFoundError(f'Data directory does not exist: {directory}')
    files = {path.stem: path.resolve() for path in directory.glob('*.xyz') if path.is_file()}
    if not files:
        raise ValueError(f'No *.xyz files in {directory}')
    return files


def _data_plan(args, config):
    """Validate all names across all noises before a model is constructed."""
    base = config.dataset._base_
    root = Path(args.dataset_root or base.ROOT).expanduser().resolve()
    resolution = args.resolution or base.get('TEST_RESOLUTION', '10000_poisson')
    clean_root = Path(args.clean_root or base.get('TEST_CLEAN_PATH') or
                      root / 'PUNet' / 'pointclouds' / 'test').expanduser().resolve()
    mesh_root = Path(args.mesh_root or base.get('TEST_MESH_ROOT') or
                     os.environ.get('PUNET_MESH_ROOT') or root / 'PUNet' / 'meshes').expanduser().resolve()
    if args.noisy_path_template is None and base.get('TEST_NOISY_PATH'):
        raise ValueError('YAML TEST_NOISY_PATH is one directory. Supply --noisy_path_template '
                         'explicitly for the requested noise levels.')
    template = args.noisy_path_template or str(
        root / 'examples' / 'pointclouds' / 'test' / 'PUNet_{resolution}_{noise:g}')
    clean_files = _files(clean_root / resolution)
    names = sorted(clean_files)
    selected = names[:args.max_shapes] if args.max_shapes else names
    entries = []
    for noise in args.noise_levels:
        directory = Path(template.format(resolution=resolution, noise=noise)).expanduser().resolve()
        noisy_files = _files(directory)
        if set(noisy_files) != set(clean_files):
            raise ValueError(f'Clean/noisy name sets differ at noise={noise}: '
                             f'missing={sorted(set(clean_files) - set(noisy_files))}; '
                             f'extra={sorted(set(noisy_files) - set(clean_files))}')
        entries.append(dict(noise=noise, noisy_directory=str(directory),
                            files=[dict(name=name, clean=str(clean_files[name]),
                                        noisy=str(noisy_files[name]),
                                        mesh=str(mesh_root / 'test' / (name + '.off')))
                                   for name in selected]))
    if len({row['noisy_directory'] for row in entries}) != len(entries):
        raise ValueError('Different noise levels resolve to the same directory; check the template')
    for name in selected:
        mesh = mesh_root / 'test' / (name + '.off')
        if not mesh.is_file():
            raise FileNotFoundError(f'Required whole-cloud P2M mesh missing: {mesh}')
    return dict(dataset_root=str(root), clean_root=str(clean_root), mesh_root=str(mesh_root),
                resolution=resolution, dataset_shapes=len(names), selected_shapes=len(selected),
                selected_names=selected, entries=entries)


def _csv_writer(stack, output, filename, fields):
    handle = stack.enter_context((output / filename).open('x', encoding='utf-8', newline=''))
    writer = csv.DictWriter(handle, fieldnames=list(fields))
    writer.writeheader()
    handle.flush()
    return writer, handle


class Metrics:
    """Long-format population statistics; every nonfinite value fails the run."""

    def __init__(self, np, stack, output):
        self.np = np
        self.stage = {}
        self.final = {}
        self.stage_writer, self.stage_file = _csv_writer(
            stack, output, 'stage_metrics.csv', (*STAGE_KEYS, *STATS))
        self.shape_stage_writer, self.shape_stage_file = _csv_writer(
            stack, output, 'per_shape_stage_metrics.csv', ('name', *STAGE_KEYS, *STATS))
        self.final_writer, self.final_file = _csv_writer(
            stack, output, 'final_metrics.csv', (*FINAL_KEYS, *STATS))
        self.shape_writer, self.shape_file = _csv_writer(
            stack, output, 'per_shape_metrics.csv',
            ('name', 'noise', 'path', 'mode', 'postprocessing', 'input_points', 'output_points',
             'cd_x1e4', 'p2m_x1e4', 'score'))

    def stats(self, values):
        values = self.np.asarray(values, dtype=self.np.float64).reshape(-1)
        if not values.size or not self.np.isfinite(values).all():
            raise FloatingPointError('Empty or NaN/Inf metric samples')
        return dict(mean=float(values.mean()), median=float(self.np.median(values)),
                    std=float(values.std(ddof=0)), p90=float(self.np.percentile(values, 90)),
                    p95=float(self.np.percentile(values, 95)), sample_count=int(values.size))

    def add_stage(self, context, mode, scope, stage, nodes, metric, values, unit):
        key = (context['noise'], context['path'], mode, scope, 'raw', stage,
               nodes[stage - 1], nodes[stage], metric, unit,
               'patch' if scope == 'patch' else 'shape')
        values = self.np.asarray(values, dtype=self.np.float64).reshape(-1)
        stats = self.stats(values)
        self.stage.setdefault(key, []).extend(values.tolist())
        self.shape_stage_writer.writerow(dict(name=context['name'], **dict(zip(STAGE_KEYS, key)), **stats))
        self.shape_stage_file.flush()

    def add_final(self, context, mode, postprocessing, input_points, output_points, metrics):
        metrics = dict(metrics, score=metrics['cd_x1e4'] + 0.3 * metrics['p2m_x1e4'])
        self.stats(list(metrics.values()))
        self.shape_writer.writerow(dict(**context, mode=mode, postprocessing=postprocessing,
                                       input_points=input_points, output_points=output_points, **metrics))
        self.shape_file.flush()
        for metric, value in metrics.items():
            key = (context['noise'], context['path'], mode, postprocessing, metric, 'x1e4', 'shape')
            self.final.setdefault(key, []).append(value)

    def finish(self):
        stage_rows = [dict(zip(STAGE_KEYS, key), **self.stats(values))
                      for key, values in sorted(self.stage.items())]
        final_rows = [dict(zip(FINAL_KEYS, key), **self.stats(values))
                      for key, values in sorted(self.final.items())]
        self.stage_writer.writerows(stage_rows)
        self.final_writer.writerows(final_rows)
        self.stage_file.flush()
        self.final_file.flush()
        return stage_rows, final_rows


def _seed(seed, np, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _shape_seed(seed, name, noise):
    text = f'{seed}|{name}|{noise:.17g}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(text).digest()[:4], 'little')


def _aligned(reference, trajectory, torch):
    for key in ('patch_idx', 'coverage_count'):
        if not torch.equal(reference[key], trajectory[key]):
            raise RuntimeError(f'{key} differs between runs; paired exposure gap is invalid')
    for key in ('fuse_weights', 'seeds'):
        if not torch.equal(reference[key], trajectory[key]):
            raise RuntimeError(f'{key} differs between runs')
    for key in ('patch_states', 'global_states'):
        if not torch.isfinite(trajectory[key]).all():
            raise FloatingPointError(f'Nonfinite trajectory: {key}')
    if not torch.equal(reference['patch_states'][0], trajectory['patch_states'][0]):
        raise RuntimeError('Runs did not start from identical noisy patches')


def _teacher_forced(student, teacher, noisy, sigma0, nodes, options, torch):
    """Reuse the exact outer patch loop/fusion; replace only Student input by T_t.

    No fused whole-cloud state is fed back. At eta=1 the outer update reconstructs
    the supplied Student output (up to floating-point arithmetic), as in infer_student.
    """
    from tools.runner_distill import forward_student_interval, TEACHER_DECAY
    from tools.runner_finetune import patch_based_denoise
    offset = 0
    calls = 0

    def forward(points, clean=None, type='val', name='', noise_std=None):
        nonlocal offset, calls
        stage = calls % 4
        start, target = nodes[stage:stage + 2]
        count = points.shape[0]
        teacher_input = teacher['patch_states'][start, offset:offset + count].to(points.device)
        if teacher_input.shape != points.shape:
            raise RuntimeError('Teacher-forced patch batch alignment failed')
        if stage == 0 and not torch.equal(points, teacher_input):
            raise RuntimeError('Teacher-forced patch order differs from Teacher')
        sigma = torch.full_like(noise_std, sigma0 * TEACHER_DECAY ** start)
        prediction = forward_student_interval(student, teacher_input, sigma, start, target)
        calls += 1
        if stage == 3:
            offset += count
        return prediction

    result = patch_based_denoise(forward, noisy, sigma0, **options, num_steps=4,
                                step_size=1.0, decay=TEACHER_DECAY ** 4,
                                return_trajectory=True, raise_on_memory_pressure=True)
    if offset != teacher['patch_states'].shape[1] or calls % 4:
        raise RuntimeError('Incomplete Teacher-forced rollout')
    return result


def _patch_metrics(teacher, tf, free, clean, nodes, device, ops, metrics, context, torch):
    """CD/PCD samples are patches, displacement tails are computed inside each patch."""
    from tools.runner_distill import evaluate_candidate_pcd
    clean_patches = clean.detach().cpu()[teacher['patch_idx']]
    for stage, (start, target) in enumerate(zip(nodes[:-1], nodes[1:]), 1):
        samples = {mode: {} for mode in ('teacher', 'teacher_forced', 'free_rollout')}

        def collect(mode, key, value):
            samples[mode].setdefault(key, []).append(float(value))

        for patch in range(clean_patches.shape[0]):
            t0 = teacher['patch_states'][start, patch].to(device)
            goal = teacher['patch_states'][target, patch].to(device)
            gt = clean_patches[patch].to(device)
            collect('teacher', 'cd_to_clean_x1e4', ops['cd'](goal[None], gt[None]) * 1e4)
            for mode, trajectory in (('teacher_forced', tf), ('free_rollout', free)):
                prediction = trajectory['patch_states'][stage, patch].to(device)
                source = (t0 if mode == 'teacher_forced' else
                          trajectory['patch_states'][stage - 1, patch].to(device))
                collect(mode, 'cd_to_teacher_x1e4', ops['cd'](prediction[None], goal[None]) * 1e4)
                collect(mode, 'cd_to_clean_x1e4', ops['cd'](prediction[None], gt[None]) * 1e4)
                displacement = (prediction - source).norm(dim=-1)
                collect(mode, 'displacement_norm_mean', displacement.mean())
                collect(mode, 'displacement_norm_p95', torch.quantile(displacement, 0.95))
                # For free rollout keep the SAME Teacher denominator; this is an
                # auxiliary relative error, NOT the PCD used by curriculum search.
                pcd = evaluate_candidate_pcd(t0, goal, prediction)
                rename = (dict(E_imit='E_imit', D_move='D_move', PCD='PCD') if mode == 'teacher_forced'
                          else dict(E_imit='E_to_teacher', D_move='D_move_teacher',
                                    PCD='relative_error_teacher_move'))
                for key, value in pcd.items():
                    collect(mode, rename[key], value)
        for mode, values in samples.items():
            for key, value in values.items():
                unit = ('x1e4' if key.startswith('cd_') else 'normalized_length'
                        if key.startswith('displacement_') else 'dimensionless'
                        if key in ('PCD', 'relative_error_teacher_move') else 'normalized_length_squared')
                metrics.add_stage(context, mode, 'patch', stage, nodes, key, value, unit)
        for key in ('cd_to_teacher_x1e4', 'cd_to_clean_x1e4'):
            gap = [free_value - tf_value for free_value, tf_value in
                   zip(samples['free_rollout'][key], samples['teacher_forced'][key])]
            metrics.add_stage(context, 'free_minus_teacher_forced', 'patch', stage, nodes,
                              key, gap, 'x1e4')


def _whole_metrics(teacher, tf, free, final_predictions, clean, center, scale, nodes,
                   config, raw_config, ops, device, metrics, context, torch):
    from tools.runner_distill import evaluate_baseline_metrics
    for stage, target in enumerate(nodes[1:], 1):
        goal = teacher['global_states'][target].to(device)
        values = {}
        for mode, trajectory, index in (('teacher', teacher, target),
                                        ('teacher_forced', tf, stage), ('free_rollout', free, stage)):
            # Use the original GPU fusion for the final result, exactly as formal
            # inference; intermediate readouts use the existing CPU trajectory fusion.
            prediction = (final_predictions[mode] if stage == 4 else
                          trajectory['global_states'][index].to(device))
            target_state = final_predictions['teacher'] if stage == 4 else goal
            raw_world, quality = evaluate_baseline_metrics(
                prediction, clean[None], center, scale, context['name'], raw_config, ops)
            values[mode] = dict(cd_to_clean_x1e4=quality['cd_x1e4'], p2m_x1e4=quality['p2m_x1e4'])
            if mode != 'teacher':
                values[mode]['cd_to_teacher_x1e4'] = float(
                    ops['cd'](prediction[None], target_state[None]) * 1e4)
            for key, value in values[mode].items():
                metrics.add_stage(context, mode, 'whole_cloud', stage, nodes, key, [value], 'x1e4')
            if stage == 4:
                metrics.add_final(context, mode, 'raw', clean.shape[0], raw_world.shape[0], quality)
                world, processed = evaluate_baseline_metrics(
                    prediction, clean[None], center, scale, context['name'], config, ops)
                metrics.add_final(context, mode, 'baseline_postprocessed', clean.shape[0],
                                  world.shape[0], processed)
        for key in values['teacher_forced']:
            gap = values['free_rollout'][key] - values['teacher_forced'][key]
            metrics.add_stage(context, 'free_minus_teacher_forced', 'whole_cloud', stage,
                              nodes, key, [gap], 'x1e4')


def _summary(output, manifest, stages, finals):
    lines = ['# Trajectory exposure audit', '',
             f"Status: {manifest['status']}; checkpoint epoch: {manifest['checkpoint_epoch']}.",
             f"Checkpoint-bound nodes: {manifest['checkpoint_teacher_nodes']}.",
             'All paths reuse this checkpoint; alternative paths are sensitivity probes, not fair retrained baselines.',
             'Stage metrics are raw, before SOR/projection. Positive exposure gap = free rollout has larger error.',
             'Whole-cloud intermediate states are fused readouts, never inputs to a later stage.',
             'CD and P2M below use the original metrics, scaled by 1e4. P2M is whole-cloud only.',
             'Statistics are population statistics over shapes or patches as marked in each CSV.', '',
             '|Noise|Path|Stage|TF CD to Teacher|Free CD to Teacher|Gap|TF CD to clean|Free CD to clean|Teacher CD to clean|TF P2M|Free P2M|Teacher P2M|',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    index = {(r['noise'], r['path'], r['stage'], r['mode'], r['metric']): r['mean']
             for r in stages if r['scope'] == 'whole_cloud'}
    for noise, path, stage in sorted({(r['noise'], r['path'], r['stage']) for r in stages}):
        def value(mode, metric):
            return index[(noise, path, stage, mode, metric)]
        numbers = [value('teacher_forced', 'cd_to_teacher_x1e4'),
                   value('free_rollout', 'cd_to_teacher_x1e4'),
                   value('free_minus_teacher_forced', 'cd_to_teacher_x1e4')]
        numbers += [value(mode, 'cd_to_clean_x1e4') for mode in ('teacher_forced', 'free_rollout', 'teacher')]
        numbers += [value(mode, 'p2m_x1e4') for mode in ('teacher_forced', 'free_rollout', 'teacher')]
        lines.append(f'|{noise:g}|{path}|{stage}|' + '|'.join(f'{v:.6f}' for v in numbers) + '|')
    lines += ['', '## Paired whole-cloud exposure gaps (free minus Teacher-forced)', '',
              '|Noise|Path|Stage|CD to Teacher gap|CD to clean gap|P2M gap|',
              '|---|---|---|---|---|---|']
    for noise, path, stage in sorted({(r['noise'], r['path'], r['stage']) for r in stages}):
        gaps = [index[(noise, path, stage, 'free_minus_teacher_forced', key)]
                for key in ('cd_to_teacher_x1e4', 'cd_to_clean_x1e4', 'p2m_x1e4')]
        lines.append(f'|{noise:g}|{path}|{stage}|' + '|'.join(f'{v:.6f}' for v in gaps) + '|')
    lines += ['', '## Final metrics (mean over shapes)', '',
              '|Noise|Path|Mode|Postprocessing|CD|P2M|Score|Shapes|', '|---|---|---|---|---|---|---|---|']
    final_index = {(r['noise'], r['path'], r['mode'], r['postprocessing'], r['metric']): r for r in finals}
    for key in sorted({(r['noise'], r['path'], r['mode'], r['postprocessing']) for r in finals}):
        rows = [final_index[(*key, metric)] for metric in ('cd_x1e4', 'p2m_x1e4', 'score')]
        lines.append('|' + '|'.join(map(str, key)) + '|' + '|'.join(f"{r['mean']:.6f}" for r in rows)
                     + f"|{rows[0]['sample_count']}|")
    lines += ['', '## Interpretation', '',
              '- Inspect stage-1 clean error first, then the paired exposure gaps at stages 2–4.',
              '- A positive gap is evidence of sensitivity to Student-generated inputs; it alone does not establish why 3% fails.',
              '- Teacher-forced stage 4 is Student(T14) for path C; it is not a realizable four-step Student denoiser.',
              '- Only teacher_forced/patch/PCD is the existing curriculum metric. Free-rollout relative error uses the same Teacher displacement denominator and is labeled separately.',
              '- Patch clean CD assumes matching clean/noisy point indices. Input hashes and this assumption are recorded in the manifest.',
              '- Displacement P95 is first calculated over points within each patch; CSV statistics then summarize these patch-level values.',
              '- See per_shape_stage_metrics.csv for paired per-shape summaries, and per_shape_metrics.csv for final tails.', '',
              f"Elapsed seconds: {manifest['elapsed_seconds']:.3f}.",
              f"Peak GPU allocated bytes: {manifest['peak_gpu_allocated_bytes']}."]
    with (output / 'diagnostic_summary.md').open('x', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def _run(args, output, manifest):
    import yaml
    with Path(args.config).open(encoding='utf-8') as handle:
        top_config = yaml.safe_load(handle)
    threads = int(top_config.get('cpu_threads', 8))
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(threads)
    import numpy as np
    import torch
    from easydict import EasyDict
    from utils.config import cfg_from_yaml_file
    from tools.runner_distill import (freeze_teacher, load_student_checkpoint, infer_student,
                                      baseline_metric_ops, _patch_options, TEACHER_ETA, TEACHER_DECAY)
    config = cfg_from_yaml_file(args.config)
    plan = _data_plan(args, config)
    paths = list(PATHS) if args.all_paths else [_nodes(p) for p in (args.teacher_nodes or [PATHS[2]])]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate diagnostic paths')
    expected_nodes = _nodes(args.expected_checkpoint_nodes)
    metadata = torch.load(args.checkpoint, map_location='cpu')
    if metadata.get('epoch') != args.expected_epoch:
        raise ValueError(f"Expected epoch {args.expected_epoch}, checkpoint has {metadata.get('epoch')}")
    if metadata.get('curriculum_mode') != 'dynamic_pcd':
        raise ValueError('This audit requires the trained Dynamic PCD checkpoint')
    saved_nodes = _nodes(metadata.get('distillation', {}).get('teacher_nodes', []))
    if saved_nodes != expected_nodes:
        raise ValueError(f'Checkpoint-bound nodes {saved_nodes} != expected {expected_nodes}')
    if metadata.get('model_config') != dict(config.model):
        raise ValueError('Checkpoint model_config differs from the supplied YAML')
    manifest.update(config=config, data=plan, diagnostic_paths=[list(p) for p in paths],
                    checkpoint_epoch=int(metadata['epoch']), checkpoint_teacher_nodes=list(saved_nodes),
                    checkpoint_best_epoch=metadata.get('best_epoch'),
                    checkpoint_selection=metadata.get('selection'),
                    checkpoint_next_teacher_nodes=metadata.get('next_teacher_nodes'),
                    checkpoint_recorded_teacher=metadata.get('teacher_checkpoint'),
                    checkpoint_sha256=_sha256(Path(args.checkpoint)),
                    teacher_checkpoint_sha256=_sha256(Path(args.teacher_checkpoint)),
                    expected_shape_noise_path_runs=plan['selected_shapes'] * len(plan['entries']) * len(paths),
                    completed_shape_noise_path_runs=0, inputs=[])
    del metadata
    manifest['protocol'] = dict(
        node_0='Noisy input T0, before any Teacher update',
        node_16='Teacher state after 16 updates; not ground truth clean',
        sigma='CLI noise is sigma0 in clean-cloud unit-sphere coordinates; sigma_start=sigma0*0.95**start_step',
        noise_source='Existing paired noisy XYZ files; no fresh noise is sampled',
        normalization='Existing PairedEvalDataset; center/scale derived from clean cloud',
        point_order='Patch clean CD assumes input clean/noisy rows correspond; equal counts are checked, correspondence cannot be inferred from files',
        patch_sampling='Original fixed outer FPS/KNN patches; shared indices, seeds, weights checked for every run',
        fusion='Original patch_based_denoise; per-stage fusion readouts are not fed back',
        cd='Original ChamferDistanceL2; no patch renormalization; clean unit sphere; x1e4',
        p2m='Whole-cloud only: original bidirectional compute_p2m(test), mesh unit sphere, x1e4',
        exposure_gap='Per-paired-sample free minus teacher_forced; positive means worse free rollout',
        pcd='Teacher-forced only: original evaluate_candidate_pcd; patch E_imit / (D_move + 1e-12)',
        free_relative_error='E_to_teacher / (D_move_teacher + 1e-12); auxiliary, not curriculum PCD',
        stage_postprocessing='none', final_postprocessing=['raw', 'baseline_postprocessed'],
        sor_enable=bool(config.get('sor_enable', True)),
        surface_projection=dict(config.get('surface_projection') or {}),
        statistics='mean/median/population_std/P90/P95/count; equal weight per patch or per shape as labeled',
        no_training=True, no_search=True, inference_only=True)
    manifest['stage_conditions'] = [
        dict(nodes=list(nodes), conditions=[[t / 16, u / 16, (u - t) / 16]
                                            for t, u in zip(nodes[:-1], nodes[1:])],
             sigma_by_noise=[dict(noise=noise, sigma_start=[noise * TEACHER_DECAY ** t
                                                           for t in nodes[:-1]])
                             for noise in args.noise_levels]) for nodes in paths]
    _write_manifest(output, manifest)
    if not torch.cuda.is_available():
        raise RuntimeError('Run on the existing CUDA training/test environment with its compiled extensions')
    torch.cuda.set_device(args.device)
    device = torch.device(f'cuda:{args.device}')
    torch.set_num_threads(threads)
    torch.cuda.set_per_process_memory_fraction(float(config.get('gpu_mem_fraction', 0.9)), args.device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    _seed(args.seed, np, torch)
    from tools import builder
    from tools.runner_finetune import patch_based_denoise
    options = _patch_options(config, args.patch_batch or config.test_patch_batch)
    if options['patch_batch'] < 1 or options['patch_size'] < 1:
        raise ValueError('patch size and batch must be positive')
    manifest['patch_options'] = options
    manifest['runtime'] = dict(torch=torch.__version__, cuda=torch.version.cuda,
                               gpu=torch.cuda.get_device_name(device), numpy=np.__version__)
    teacher = builder.model_builder(config.model).to(device)
    builder.load_model(teacher, args.teacher_checkpoint)
    freeze_teacher(teacher)
    if getattr(teacher, 'step_condition', None) is not None:
        raise ValueError('Teacher must not have a condition branch')
    student = builder.model_builder(config.model).to(device)
    load_student_checkpoint(student, args.checkpoint, builder, expected_nodes=expected_nodes)
    if getattr(student, 'step_condition', None) is None:
        raise ValueError('Checkpoint does not contain trained Step Condition weights')
    student.eval().requires_grad_(False)
    config.dataset._base_.TEST_MESH_ROOT = plan['mesh_root']
    ops = baseline_metric_ops(config, device)
    raw_config = copy.deepcopy(config)
    raw_config.sor_enable = False
    raw_config.surface_projection = EasyDict(enable=False)
    print(f"Checkpoint epoch={manifest['checkpoint_epoch']}, bound nodes={saved_nodes}, diagnostic paths={paths}", flush=True)
    print('No optimizer, backward, training, or PCD search. Stage metrics are raw.', flush=True)
    _write_manifest(output, manifest)
    with ExitStack() as stack, torch.no_grad():
        metrics = Metrics(np, stack, output)
        for entry in plan['entries']:
            noise = entry['noise']
            dataset_config = copy.deepcopy(config.dataset._base_)
            dataset_config.ROOT = plan['dataset_root']
            dataset_config.TEST_RESOLUTION = plan['resolution']
            dataset_config.TEST_CLEAN_PATH = plan['clean_root']
            dataset_config.TEST_NOISY_PATH = entry['noisy_directory']
            dataset_config.TEST_NOISE = noise
            dataset_config.NUM_WORKERS = 0
            _, loader = builder.dataset_builder(
                argparse.Namespace(distributed=False, local_rank=0),
                EasyDict(_base_=dataset_config, others=EasyDict(subset='test', bs=1)))
            if list(loader.dataset.names)[:plan['selected_shapes']] != plan['selected_names']:
                raise RuntimeError('Dataset loader order differs from the preflight file list')
            for index, (noisy_batch, clean_batch, _, centers, scales, names) in enumerate(loader):
                if index >= plan['selected_shapes']:
                    break
                name = str(names[0])
                noisy, clean = noisy_batch[0].to(device), clean_batch[0].to(device)
                # The original collate returns center_list/scale_list, not tensors.
                center, scale = torch.as_tensor(centers[0]), torch.as_tensor(scales[0])
                if (noisy.shape != clean.shape or noisy.ndim != 2 or noisy.shape[1] != 3 or
                        noisy.shape[0] < options['patch_size'] or not torch.isfinite(noisy).all() or
                        not torch.isfinite(clean).all() or not torch.isfinite(center).all() or
                        not torch.isfinite(scale).all() or (scale <= 0).any()):
                    raise ValueError(f'{name}: malformed, nonfinite, too small, or unpaired input clouds')
                seed = _shape_seed(args.seed, name, noise)
                files = entry['files'][index]
                manifest['inputs'].append(dict(
                    name=name, noise=noise, sigma0=noise, points=int(noisy.shape[0]), seed=seed,
                    center=center.tolist(), scale=scale.tolist(),
                    paired_residual_coordinate_rms=float((noisy - clean).square().mean().sqrt()),
                    paired_residual_coordinate_std=(noisy - clean).std(dim=0, unbiased=False).tolist(),
                    clean_sha256=_sha256(Path(files['clean'])), noisy_sha256=_sha256(Path(files['noisy'])),
                    mesh_sha256=_sha256(Path(files['mesh']))))
                manifest['active_run'] = dict(name=name, noise=noise, phase='teacher_trajectory')
                _write_manifest(output, manifest)
                print(f'[Teacher 16-step] {name} noise={noise:g}', flush=True)
                _seed(seed, np, torch)
                teacher_final, teacher_trajectory = patch_based_denoise(
                    teacher, noisy, noise, **options, num_steps=16, step_size=TEACHER_ETA,
                    decay=TEACHER_DECAY, return_trajectory=True, raise_on_memory_pressure=True)
                _aligned(teacher_trajectory, teacher_trajectory, torch)
                manifest['inputs'][-1]['patch_count'] = int(teacher_trajectory['patch_idx'].shape[0])
                manifest['inputs'][-1]['uncovered_points'] = int((teacher_trajectory['coverage_count'] == 0).sum())
                for nodes in paths:
                    context = dict(name=name, noise=noise, path='-'.join(map(str, nodes)))
                    manifest['active_run'] = dict(**context, phase='student_rollouts_and_metrics')
                    _write_manifest(output, manifest)
                    print(f'[TF/free audit] {name} noise={noise:g} nodes={list(nodes)}', flush=True)
                    _seed(seed, np, torch)
                    free_final, free = infer_student(student, noisy, noise, options,
                                                      return_trajectory=True, teacher_nodes=nodes)
                    _aligned(teacher_trajectory, free, torch)
                    _seed(seed, np, torch)
                    tf_final, tf = _teacher_forced(student, teacher_trajectory, noisy, noise,
                                                   nodes, options, torch)
                    _aligned(teacher_trajectory, tf, torch)
                    # Independent float32 GPU forwards can differ slightly (the
                    # model uses index_add_ reductions). Do not require bitwise
                    # output equality or overwrite either measured trajectory.
                    first_delta = (tf['patch_states'][1] - free['patch_states'][1]).abs().reshape(-1)
                    first_diff = float(first_delta.max())
                    first_check = dict(
                        **context, value=first_diff, mean_abs_difference=float(first_delta.mean()),
                        p95_abs_difference=float(torch.quantile(first_delta, 0.95)),
                        coordinates_above_atol=int((first_delta > args.first_stage_atol).sum()),
                        fraction_above_atol=float((first_delta > args.first_stage_atol).float().mean()),
                        coordinate_count=int(first_delta.numel()), atol=args.first_stage_atol,
                        passed=first_diff <= args.first_stage_atol)
                    # Persist failed checks too, so a rerun never loses the evidence.
                    manifest.setdefault('first_stage_max_abs_differences', []).append(first_check)
                    _write_manifest(output, manifest)
                    if not first_check['passed']:
                        raise RuntimeError(
                            f'TF/free first stage exceeds output tolerance: max={first_diff:.9g}, '
                            f'atol={args.first_stage_atol:.9g}, '
                            f'mean={first_check["mean_abs_difference"]:.9g}, '
                            f'p95={first_check["p95_abs_difference"]:.9g}; '
                            'see run_manifest.json before changing the tolerance')
                    _patch_metrics(teacher_trajectory, tf, free, clean, nodes, device, ops, metrics, context, torch)
                    _whole_metrics(teacher_trajectory, tf, free,
                                   dict(teacher=teacher_final, teacher_forced=tf_final, free_rollout=free_final),
                                   clean, center, scale, nodes, config, raw_config, ops, device,
                                   metrics, context, torch)
                    manifest['completed_shape_noise_path_runs'] += 1
                    _write_manifest(output, manifest)
                    print(f"[{manifest['completed_shape_noise_path_runs']}/{manifest['expected_shape_noise_path_runs']}] "
                          f"{name} noise={noise:g} nodes={list(nodes)} first_stage_max_diff={first_diff:.3g}", flush=True)
                    del tf, free, tf_final, free_final
                del teacher_trajectory, teacher_final, noisy, clean
        if manifest['completed_shape_noise_path_runs'] != manifest['expected_shape_noise_path_runs']:
            raise RuntimeError('Incomplete shape/noise/path enumeration')
        if (student.training or teacher.training or any(p.requires_grad or p.grad is not None
                                                       for model in (teacher, student) for p in model.parameters())):
            raise RuntimeError('Frozen/eval/no-gradient integrity check failed')
        stage_rows, final_rows = metrics.finish()
    torch.cuda.synchronize(device)
    manifest.update(status='completed', peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device),
                    integrity=dict(teacher_frozen=True, student_frozen=True, eval=True, no_grad=True,
                                   shared_patches_verified=True, first_stage_agreement_verified=True,
                                   all_metrics_finite=True, no_training=True, no_search=True),
                    full_dataset=plan['selected_shapes'] == plan['dataset_shapes'])
    manifest['active_run'] = None
    return stage_rows, final_rows


class _Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main():
    args = _build_parser().parse_args()
    if (args.max_shapes < 0 or args.expected_epoch < 1 or not 0 <= args.seed < 2 ** 32 or
            args.device < 0 or (args.patch_batch is not None and args.patch_batch < 1) or
            not math.isfinite(args.first_stage_atol) or args.first_stage_atol <= 0):
        raise ValueError('Invalid count, device, seed, patch batch, epoch, or tolerance')
    if (len(set(args.noise_levels)) != len(args.noise_levels) or
            any(not math.isfinite(n) or n <= 0 for n in args.noise_levels)):
        raise ValueError('Noise levels must be unique finite positive values')
    for key in ('config', 'checkpoint', 'teacher_checkpoint'):
        path = Path(getattr(args, key)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'{key}: {path}')
        setattr(args, key, str(path))
    if args.checkpoint == args.teacher_checkpoint:
        raise ValueError('Student and Teacher checkpoint paths must differ')
    output = (Path(args.output_dir).expanduser().resolve() if args.output_dir else
              REPO_ROOT / 'diagnostics' / 'trajectory_audit' /
              (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]))
    # Refuse even an existing empty directory: exclusive creation prevents concurrent
    # runs from sharing CSV files. A failed run is preserved and must use a new path.
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(REPO_ROOT)
    start = time.perf_counter()
    manifest = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                    checkpoint=args.checkpoint, teacher_checkpoint=args.teacher_checkpoint,
                    arguments=vars(args), argv=sys.argv, output_dir=str(output), git=_git_info(),
                    source_sha256={name: _sha256(REPO_ROOT / name) for name in (
                        'tools/diagnose_distill_trajectory.py', 'tools/runner_distill.py',
                        'tools/runner_finetune.py', 'models/PointGPT.py', 'models/step_condition.py',
                        'datasets/ScoreDenoiseDataset.py', 'utils/p2m_loss.py')})
    _write_manifest(output, manifest)
    with (output / 'run.log').open('x', encoding='utf-8') as log:
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            try:
                stage_rows, final_rows = _run(args, output, manifest)
                manifest['elapsed_seconds'] = time.perf_counter() - start
                _summary(output, manifest, stage_rows, final_rows)
                _write_manifest(output, manifest)
                print(f"COMPLETED: {output}; elapsed={manifest['elapsed_seconds']:.3f}s", flush=True)
            except BaseException as error:
                manifest.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                                elapsed_seconds=time.perf_counter() - start,
                                error=f'{type(error).__name__}: {error}')
                _write_manifest(output, manifest)
                traceback.print_exc()
                raise


if __name__ == '__main__':
    main()
