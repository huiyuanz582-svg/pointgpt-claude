"""Frozen, standalone fixed/greedy/global shadow comparison on training patches.

No optimizer, backward, checkpoint saving, or training target changes.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='cfgs/PointGPT-L/distill_16to4.yaml')
    parser.add_argument('--teacher_ckpt', required=True)
    parser.add_argument('--student_ckpt', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--max_batches', type=int, default=16)
    parser.add_argument('--interval_patch_batch', type=int, default=8)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--device', type=int, default=0)
    return parser.parse_args()


def run(args, result):
    import numpy as np
    import torch
    from tools import builder, runner_distill as runner
    from tools.shadow_global_search import (TEACHER_PATHS, TEACHER_INTERVALS,
                                           build_interval_pcd_cache, compare_cached_shadow_paths,
                                           summarize_path_comparison)
    from utils.config import cfg_from_yaml_file

    started = time.monotonic()
    output = Path(args.output_dir)
    config = cfg_from_yaml_file(args.config)
    options = runner._shadow_search_config(config)
    target = options['threshold']
    balance = float(config.pcd_dynamic.lambda_balance)
    if args.max_batches < 1 or args.interval_patch_batch < 1:
        raise ValueError('Batch limits must be positive')
    assert runner.TEACHER_NODES == (0, 4, 8, 12, 16)
    assert config.student_patch_batch == config.total_bs == 8
    assert config.teacher_patch_batch == 1
    assert torch.cuda.is_available()
    torch.cuda.set_device(args.device)
    torch.set_num_threads(int(config.cpu_threads))
    torch.cuda.set_per_process_memory_fraction(float(config.gpu_mem_fraction), args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda', args.device)
    debug_indices = set(random.Random(1709).sample(range(args.max_batches * 8), min(8, args.max_batches * 8)))

    def emit(key, value):
        result[key] = value
        print(json.dumps({key: value}, allow_nan=False), flush=True)

    emit('environment', dict(torch=torch.__version__, cuda=torch.version.cuda,
                             gpu=torch.cuda.get_device_name(device), visible_devices=os.getenv('CUDA_VISIBLE_DEVICES')))
    result['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for folder in ('models', 'tools', 'datasets', 'utils', 'cfgs')
                               for p in sorted((ROOT / folder).rglob('*'))
                               if p.is_file() and p.suffix in ('.py', '.yaml')}
    emit('protocol', dict(seed=args.seed, fixed_nodes=list(runner.TEACHER_NODES),
                          PCD_target=target, lambda_balance=balance, pcd_eps=runner.PCD_EPS,
                          train_loader_batch=8, teacher_patch_batch=1,
                          analysis_interval_patch_batch=args.interval_patch_batch,
                          paths_per_patch=len(TEACHER_PATHS), unique_intervals_per_patch=len(TEACHER_INTERVALS),
                          selection='mean_abs_pcd_distance + lambda_balance * population_std',
                          imbalance_definition='population std/range across four stage means; per-patch values also saved'))

    result['phase'] = 'checkpoint_loading'
    teacher_path, student_path = Path(args.teacher_ckpt), Path(args.student_ckpt)
    checkpoint_stats = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in (teacher_path, student_path)}
    teacher = builder.model_builder(config.model).to(device)
    builder.load_model(teacher, str(teacher_path))
    student = copy.deepcopy(teacher)
    checkpoint = torch.load(student_path, map_location='cpu')
    weights = checkpoint.get('base_model', checkpoint.get('model'))
    condition_present = any('step_condition.' in key for key in weights)
    runner.load_student_state(student, weights)
    emit('checkpoints', dict(teacher=str(teacher_path), student=str(student_path),
                             student_epoch=checkpoint.get('epoch'), student_selection=checkpoint.get('selection'),
                             strict_backbone_load=True, condition_present_in_checkpoint=condition_present,
                             condition_state='loaded' if condition_present else 'existing_compatibility_initialization',
                             condition_parameters=sum(p.numel() for p in student.step_condition.parameters())))
    del checkpoint, weights
    runner.freeze_teacher(teacher)
    runner.freeze_teacher(student)
    versions = [{name: p._version for name, p in model.named_parameters()} for model in (teacher, student)]
    buffers = [{name: b.detach().cpu().clone() for name, b in model.named_buffers()} for model in (teacher, student)]
    counts = dict(teacher=0, student=0, patch_interval_evaluations=0, paths_scored=0)
    last_condition = None

    def model_hook(model, inputs, kwargs):
        nonlocal last_condition
        assert not torch.is_grad_enabled() and not model.training
        assert inputs[0].is_cuda and inputs[0].shape[-2:] == (1024, 3)
        if model is teacher:
            assert inputs[0].shape[0] == 1 and 'target_step' not in kwargs
            counts['teacher'] += 1
        else:
            t, u = int(kwargs['start_step']), int(kwargs['target_step'])
            assert (t, u) in TEACHER_INTERVALS and kwargs['noise_std'].is_cuda
            assert 1 <= inputs[0].shape[0] <= args.interval_patch_batch
            last_condition = (t, u)
            counts['student'] += 1

    def condition_hook(module, inputs):
        t, u = last_condition
        expected = torch.tensor([t, u, u - t], device=device, dtype=inputs[0].dtype) / 16
        assert torch.equal(inputs[0], expected[None].expand_as(inputs[0]))

    handles = [model.register_forward_pre_hook(model_hook, with_kwargs=True) for model in (teacher, student)]
    handles.append(student.step_condition.mlp[0].register_forward_pre_hook(condition_hook))
    result['phase'] = 'shadow_statistics'
    loader = runner._train_loader(config)
    if args.max_batches > len(loader):
        raise ValueError('Requested batches exceed one training epoch')
    emit('sampling', dict(dataset_patches=len(loader.dataset), batch_size=loader.batch_size,
                          loader_batches=len(loader), drop_last=loader.drop_last))
    rows, examples = [], []
    iterator = iter(loader)
    with torch.no_grad(), (output / 'selected_paths.jsonl').open('w') as selected_file, \
            (output / 'interval_metrics.jsonl').open('w') as interval_file:
        for batch_index in range(args.max_batches):
            noisy, clean, sigmas, _, _, names = next(iterator)
            assert noisy.shape == clean.shape == (8, 1024, 3)
            full = runner.capture_teacher(teacher, noisy, sigmas, 1, return_full_trajectory=True)
            assert full.shape == (17, 8, 1024, 3) and not full.requires_grad
            cache = build_interval_pcd_cache(student, full, sigmas, args.interval_patch_batch)
            before_selection = counts['student']
            current = compare_cached_shadow_paths(full, cache, target, balance)
            assert counts['student'] == before_selection
            counts['patch_interval_evaluations'] += cache['patch_interval_evaluations']
            counts['paths_scored'] += len(TEACHER_PATHS) * len(current)
            for row in current:
                index = row['batch_sample']
                row.update(patch_index=batch_index * 8 + index, shape_name=names[index], sigma0=float(sigmas[index]))
                rows.append(row)
                selected_file.write(json.dumps(row, allow_nan=False) + '\n')
                intervals = [dict(start_step=t, target_step=u,
                                  **{key: float(value[index]) for key, value in metric.items()})
                             for (t, u), metric in cache['metrics'].items()]
                interval_file.write(json.dumps(dict(patch_index=row['patch_index'], intervals=intervals), allow_nan=False) + '\n')
                if row['patch_index'] in debug_indices:
                    examples.append(dict(patch_index=row['patch_index'], shape_name=row['shape_name'],
                                         selected_nodes={method: row[method]['nodes'] for method in
                                                         ('fixed', 'greedy_dynamic', 'global_dynamic')},
                                         greedy_candidates=row['greedy_stage_search'], all_intervals=intervals))
            selected_file.flush()
            interval_file.flush()
            summary = summarize_path_comparison(rows)
            print(json.dumps(dict(batch=batch_index + 1, patches=len(rows),
                                  mean_PCD={name: values['mean_PCD_per_stage'] for name, values in summary.items()},
                                  elapsed_seconds=time.monotonic() - started), allow_nan=False), flush=True)
            del full, cache, current
    del iterator
    emit('patches', len(rows))
    emit('batches', args.max_batches)
    emit('comparison', summarize_path_comparison(rows))
    (output / 'candidate_examples.json').write_text(json.dumps(examples, indent=2, allow_nan=False), encoding='utf-8')
    for model, old_versions, old_buffers in zip((teacher, student), versions, buffers):
        assert all(not m.training for m in model.modules())
        for name, p in model.named_parameters():
            assert p.grad is None and not p.requires_grad and p._version == old_versions[name]
        for name, value in model.named_buffers():
            assert torch.equal(value.cpu(), old_buffers[name]), name
    for path in (teacher_path, student_path):
        assert (path.stat().st_size, path.stat().st_mtime_ns) == checkpoint_stats[str(path)]
    for handle in handles:
        handle.remove()
    torch.cuda.synchronize()
    emit('compute', counts)
    emit('integrity', dict(all_conditions_checked=True, final_edge_in_every_path_score=True,
                           each_patch_interval_evaluated_once=True, selectors_use_identical_cache=True,
                           global_score_no_worse_than_fixed_or_greedy_every_patch=True,
                           teacher_student_eval_frozen_no_grad=True, model_weights_and_buffers_unchanged=True,
                           checkpoint_files_unchanged=True, checkpoint_written=False))
    emit('peak_gpu_allocated_gib', torch.cuda.max_memory_allocated() / 1024 ** 3)
    emit('elapsed_seconds', time.monotonic() - started)
    result.update(passed=True, phase='complete')


def main():
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use an empty output directory to preserve previous analysis results')
    result = dict(passed=False, phase='imports', optimizer_steps=0, backward_calls=0)
    try:
        run(args, result)
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        (output / 'global_shadow_result.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps(dict(passed=result['passed'], phase=result['phase'])), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
