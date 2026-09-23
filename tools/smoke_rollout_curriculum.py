"""Manual server scoring smoke: no training/optimizer; explicit paths only."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='cfgs/PointGPT-L/distill_16to4_rollout_smoke.yaml')
    parser.add_argument('--teacher_ckpt', required=True)
    parser.add_argument('--student_ckpt', help='Optional trained Student; otherwise clone Teacher and enable condition')
    parser.add_argument('--output_dir', required=True, help='New directory only')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.device < 0 or not 0 <= args.seed < 2 ** 32:
        raise ValueError('Invalid device or seed')
    config_path = Path(args.config).resolve()
    teacher_path = Path(args.teacher_ckpt).resolve()
    student_path = Path(args.student_ckpt).resolve() if args.student_ckpt else None
    for path in (config_path, teacher_path, student_path):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    output = Path(args.output_dir).resolve()
    os.chdir(ROOT)
    import yaml
    from utils.curriculum_config import resolve_curriculum, candidate_paths
    raw = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    resolved = resolve_curriculum(raw)
    if (resolved['metric']['type'] != 'rollout_aware' or
            resolved['search']['candidate_set'] != 'explicit' or
            len(candidate_paths(resolved['search'])) > 8 or
            resolved['calibration']['patches_per_level'] > 4):
        raise ValueError('Smoke requires explicit <=8 paths and <=4 patches per noise level')
    output.mkdir(parents=True, exist_ok=False)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(raw.get('cpu_threads', 8))
    import numpy as np
    import torch
    from utils.config import cfg_from_yaml_file
    from utils.gpu_memory import apply_gpu_memory_limit
    config = cfg_from_yaml_file(str(config_path))
    if not torch.cuda.is_available():
        raise RuntimeError('Manual server smoke requires the original CUDA environment')
    torch.cuda.set_device(args.device)
    budget = apply_gpu_memory_limit(config, args.device)
    torch.set_num_threads(int(config.cpu_threads))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    from tools import builder, runner_distill as runner
    from tools.rollout_curriculum import write_report
    device = torch.device('cuda', args.device)
    manifest = dict(status='running', purpose='scoring_smoke_no_training', arguments=vars(args),
                    resolved_curriculum=resolved, gpu_memory_limit=budget,
                    source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                   for name in ('tools/rollout_curriculum.py', 'tools/runner_distill.py',
                                                'utils/curriculum_config.py')})
    write_report(output / 'manifest.json', manifest)
    try:
        teacher = builder.model_builder(config.model).to(device)
        builder.load_model(teacher, str(teacher_path))
        student = copy.deepcopy(teacher).to(device)
        if student_path:
            runner.load_student_checkpoint(student, student_path, builder)
            if getattr(student, 'step_condition', None) is None:
                raise ValueError('Rollout scoring smoke requires a conditioned Student checkpoint')
        else:
            runner.enable_student_condition(student)
        runner.freeze_teacher(teacher)
        student.eval()
        dataset = runner._train_loader(config).dataset  # Never iterate the training loader.
        bank, metadata = runner.curriculum_calibration_bank(config, teacher, dataset)
        write_report(output / 'calibration_manifest.json', metadata)
        result = runner.update_dynamic_curriculum(student, bank, config, report_path=output / 'search.json')
        manifest.update(status='completed', student_forward_calls=result['student_forward_calls'],
                        nodes=result['nodes'], Jrobust=result['Jrobust'], search_seconds=result['search_seconds'])
        print(json.dumps(manifest, ensure_ascii=False, allow_nan=False), flush=True)
    except BaseException as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_report(output / 'manifest.json', manifest)


if __name__ == '__main__':
    from utils.gpu_memory import exit_on_cuda_oom
    with exit_on_cuda_oom():
        main()
