"""Manual fixed-checkpoint TF/FR study across paths and calibration seeds; no training/search."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.curriculum_diagnostic_study import resolve_study, seed_configuration, summarize_seeds, write_seed_summary
from utils.curriculum_diagnostic_study import write_json as write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='cfgs/PointGPT-L/distill_16to4_rollout_diagnostics.yaml')
    parser.add_argument('--teacher_ckpt', required=True)
    parser.add_argument('--student_ckpt', required=True, help='Fixed trained, conditioned Student; loaded once')
    parser.add_argument('--output_dir', required=True, help='New directory only')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0, help='General runtime seed; does not change calibration seeds')
    parser.add_argument('--calibration_seeds', nargs='+', type=int, help='Override the actual calibration bank seeds')
    parser.add_argument('--patches_per_level', type=int)
    parser.add_argument('--patch_batch', type=int)
    args = parser.parse_args()
    if args.device < 0 or not 0 <= args.seed < 2 ** 32:
        raise ValueError('Invalid device or runtime seed')
    config_path = Path(args.config).resolve()
    teacher_path, student_path = Path(args.teacher_ckpt).resolve(), Path(args.student_ckpt).resolve()
    output = Path(args.output_dir).resolve()
    for path in (config_path, teacher_path, student_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    import yaml
    study = resolve_study(yaml.safe_load(config_path.read_text(encoding='utf-8')),
                          calibration_seeds=args.calibration_seeds,
                          patches_per_level=args.patches_per_level, patch_batch=args.patch_batch)
    base_path = (ROOT / study['base_config']).resolve()
    raw = yaml.safe_load(base_path.read_text(encoding='utf-8'))
    # Validate every effective configuration before CUDA imports or output creation.
    for seed in study['calibration_seeds']:
        seed_configuration(raw, study, seed)
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    manifest = dict(status='running', purpose='fixed_checkpoint_multiseed_stage_diagnostics',
                    diagnostic_only=True, training=False, search=False, changes_selection=False,
                    arguments=vars(args), study=study, runs=[], summary=None,
                    model_state=dict(id=f'{output}::student_seed_{args.seed}', kind='loaded_student_checkpoint',
                                     checkpoint=str(student_path), teacher_checkpoint=str(teacher_path), seed=args.seed))
    write_report(output / 'manifest.json', manifest)
    try:
        source_names = ('tools/diagnose_rollout_curriculum.py', 'utils/curriculum_diagnostic_study.py',
                        'tools/rollout_curriculum.py', 'tools/runner_distill.py', 'utils/curriculum_config.py',
                        'utils/curriculum_diagnostics.py', 'utils/gpu_memory.py')
        manifest['source_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                     for name in source_names}
        manifest['config_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in (config_path, base_path, ROOT / raw['dataset']['_base_'])}
        manifest['checkpoint_files'] = {
            label: dict(path=str(path), bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
            for label, path in (('teacher', teacher_path), ('student', student_path))}
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[name] = str(raw.get('cpu_threads', 8))
        import numpy as np
        import torch
        from utils.config import cfg_from_yaml_file
        from utils.gpu_memory import apply_gpu_memory_limit
        base_config = cfg_from_yaml_file(str(base_path))
        if not torch.cuda.is_available():
            raise RuntimeError('Manual diagnostics require the original CUDA environment')
        torch.cuda.set_device(args.device)
        manifest['gpu_memory_limit'] = apply_gpu_memory_limit(base_config, args.device)
        torch.set_num_threads(int(base_config.cpu_threads))
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        from tools import builder, runner_distill as runner
        from tools.rollout_curriculum import build_stratified_bank, diagnose_stage_paths
        runner.configured_teacher_nodes(base_config)  # Validate the original 16 -> 4 protocol.
        device = torch.device('cuda', args.device)
        write_report(output / 'manifest.json', manifest)
        teacher = builder.model_builder(base_config.model).to(device)
        builder.load_model(teacher, str(teacher_path))
        student = copy.deepcopy(teacher).to(device)
        runner.load_student_checkpoint(student, student_path, builder)
        if getattr(student, 'step_condition', None) is None:
            raise ValueError('Expanded diagnostics require a trained, conditioned Student checkpoint')
        manifest['model_state']['current_checkpoint_nodes'] = list(student.distillation_teacher_nodes)
        runner.freeze_teacher(teacher)
        student.eval()
        dataset = runner._train_loader(base_config).dataset  # Never iterate the loader or start workers.
        reports = []
        for seed in study['calibration_seeds']:
            seed_dir = output / f'calibration_seed_{seed}'
            seed_dir.mkdir(exist_ok=False)
            config, resolved = seed_configuration(base_config, study, seed)
            run = dict(calibration_seed=seed, directory=seed_dir.name, status='running')
            manifest['runs'].append(run)
            write_report(output / 'manifest.json', manifest)
            print(f'[stage study] calibration_seed={seed}; patches/noise={study["patches_per_level"]}; '
                  f'expected Student forwards={study["expected_cost"]["student_forwards_per_seed"]}', flush=True)
            try:
                write_report(seed_dir / 'resolved_curriculum.json', resolved)
                started = time.perf_counter()
                bank, metadata = build_stratified_bank(config, teacher, dataset, resolved, runner.capture_teacher)
                metadata.update(purpose='fixed_checkpoint_stage_diagnostics', calibration_seconds=time.perf_counter() - started)
                write_report(seed_dir / 'calibration_manifest.json', metadata)
                report = diagnose_stage_paths(
                    student, bank, resolved, runner.forward_student_interval, named_paths=study['paths'], epoch=0,
                    model_state=dict(manifest['model_state'], calibration_seed=seed), output=seed_dir,
                    decay=runner.TEACHER_DECAY)
                del bank  # Only one seed's Teacher trajectory bank is retained at a time.
                if report['student_forward_calls'] != study['expected_cost']['student_forwards_per_seed']:
                    raise AssertionError('Diagnostic forward count differs from the requested study')
                run.update(status='completed', data_sha256=metadata['data_sha256'],
                           calibration_seconds=metadata['calibration_seconds'],
                           stage_diagnostics=dict(report['files'], seconds=report['diagnostic_seconds'],
                                                  student_forward_calls=report['student_forward_calls']),
                           nonfinite_count=report['nonfinite_count'],
                           small_local_denominator_count=report['small_local_denominator_count'])
                reports.append(report)
            except BaseException as error:
                run.update(status='failed', error=f'{type(error).__name__}: {error}')
                raise
            finally:
                write_report(output / 'manifest.json', manifest)
        summary = summarize_seeds(reports, study['calibration_seeds'])
        manifest['summary'] = write_seed_summary(output, summary)
        manifest.update(status='completed', student_forward_calls=sum(r['student_forward_calls'] for r in reports),
                        nonfinite_count=sum(r['nonfinite_count'] for r in reports))
        print(json.dumps(dict(status=manifest['status'], output_dir=str(output), summary=manifest['summary'],
                              student_forward_calls=manifest['student_forward_calls']), allow_nan=False), flush=True)
    except BaseException as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_report(output / 'manifest.json', manifest)


if __name__ == '__main__':
    from utils.gpu_memory import exit_on_cuda_oom
    with exit_on_cuda_oom():
        main()
