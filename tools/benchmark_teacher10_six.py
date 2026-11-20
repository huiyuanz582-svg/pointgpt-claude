"""Manual Teacher10/Student4 quality and efficiency evaluation.

Each variant/condition runs in a fresh, sequential subprocess with one model.
No training, optimizer updates, trajectory capture, or checkpoint writes.
--dense_noise_train_root measures the predeclared epoch-1 candidate alone.
"""

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.denoise_benchmark_report import (
    CONDITIONS, DENSE_NOISE_STUDENT, STUDENTS, build_report, condition_label,
    read_result, summarize, write_csv, write_json)

NODES = [0, 2, 5, 7, 10]
SCHEDULE = dict(teacher_nodes=NODES, teacher_steps=10, teacher_eta=0.3, teacher_decay=0.95,
                student_steps=4, student_eta=1.0,
                student_decay=[0.95 ** (u - t) for t, u in zip(NODES[:-1], NODES[1:])])
INFERENCE_SOURCES = ('models/PointGPT.py', 'models/step_condition.py',
                     'tools/runner_finetune.py', 'datasets/ScoreDenoiseDataset.py')
SOURCES = INFERENCE_SOURCES + ('tools/runner_distill.py', 'tools/builder.py',
                              'utils/teacher_endpoint.py', 'utils/training_input.py',
                              'utils/misc.py', 'utils/p2m_loss.py', 'utils/gpu_memory.py',
                              'tools/benchmark_teacher10_six.py', 'utils/denoise_benchmark_report.py')
SOURCES += ('tools/test_teacher_endpoint_six.py', 'datasets/scoredenoise/transforms.py',
            'cfgs/PointGPT-L/dense_noise_reference_punet.json')
TRAIN_RECORDS = ('manifest.json', 'validation_manifest.json', 'train.jsonl')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for name in SOURCES}


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def inspect_training(train_root, variant_id, actual_sources, reference=None):
    """Read logs and fingerprints without importing torch in the coordinator."""
    train_root = train_root.expanduser().resolve()
    dense_noise = variant_id == DENSE_NOISE_STUDENT
    checkpoint = train_root / ('ckpt-last.pth' if dense_noise else 'ckpt-best.pth')
    for path in [checkpoint] + [train_root / name for name in TRAIN_RECORDS]:
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = load_json(train_root / 'manifest.json')
    config = manifest['config']
    logs = [json.loads(line) for line in (train_root / 'train.jsonl').read_text(encoding='utf-8-sig').splitlines()
            if line.strip()]
    finetune = config.get('endpoint_finetune') is not None
    planned = manifest['arguments'].get('epochs') or config['epochs']
    if [r['epoch'] for r in logs] != list(range(0 if finetune else 1, planned + 1)):
        raise ValueError(f'Incomplete training plan: {train_root}')
    if any(not math.isfinite(float(r['val_rollout_score'])) for r in logs):
        raise ValueError(f'Nonfinite validation score: {train_root}')
    best = min(logs, key=lambda r: r['val_rollout_score'])
    if logs[-1]['best_epoch'] != best['epoch'] or manifest['schedule'] != SCHEDULE:
        raise ValueError(f'Best tracking or Teacher10 schedule mismatch: {train_root}')
    expected_epochs = {'student_ft1': 1, 'student_ft3': 3}
    evaluated = best
    if dense_noise:
        from utils.teacher_endpoint import resolve_endpoint_finetune
        options = resolve_endpoint_finetune(config) or {}
        adaptation = options.get('dense_noise_adaptation') or {}
        expected_sha = reference['reference_checkpoint_sha256']
        recorded_sources = manifest.get('source_sha256', {})
        if (manifest.get('mode') != 'train' or planned != 1 or not adaptation
                or logs[0].get('phase') != 'initial_validation' or logs[1]['dense_noise_active_patches'] <= 0
                or adaptation['reference_sha256'] != expected_sha
                or manifest['initialization']['checkpoint_sha256'] != expected_sha
                or not manifest.get('endpoint_finetune_protocol', {}).get('frozen_parameters')):
            raise ValueError('Expected a complete dense-noise epoch-1 run from the pinned reference')
        if not set(INFERENCE_SOURCES).issubset(recorded_sources):
            raise ValueError('Dense-noise training record is missing inference fingerprints')
        for name, expected in recorded_sources.items():
            if sha256_file(ROOT / name) != expected:
                raise ValueError(f'Current source differs from the training manifest: {name}')
        evaluated = logs[1]
    elif variant_id == 'student_before':
        if finetune or not config.get('teacher_endpoint', {}).get('mask_pre_bn_conv_bias'):
            raise ValueError('The before checkpoint must be the masked endpoint Student4')
    elif not finetune or planned != expected_epochs[variant_id]:
        raise ValueError(f'{variant_id} requires a completed {expected_epochs[variant_id]}-epoch fine-tune')
    elif best['epoch'] != expected_epochs[variant_id]:
        raise ValueError(f'{variant_id}: ckpt-best is epoch {best["epoch"]}, so it cannot be labelled '
                         f'as the {expected_epochs[variant_id]}-epoch checkpoint')
    if finetune and not all(r.get('frozen_buffers_verified') and r.get('frozen_biases_verified') for r in logs):
        raise ValueError(f'Missing frozen-state verification: {train_root}')
    if config['dataset']['_base_']['DATASET'] != 'PUNet':
        raise ValueError('This benchmark requires PUNet')
    if config.get('sor_enable', True) or config.get('surface_projection', {}).get('enable', False):
        raise ValueError('This comparison requires the existing SOR/projection-disabled protocol')
    differences = {}
    if not dense_noise:
        differences = {name: dict(training=expected, current=actual_sources.get(name))
                       for name, expected in manifest.get('source_sha256', {}).items()
                       if expected != actual_sources.get(name)}
        for name in INFERENCE_SOURCES:
            if manifest.get('source_sha256', {}).get(name) != actual_sources[name]:
                raise ValueError(f'Inference implementation differs from the training record: {name}')
    # runner_distill/teacher_endpoint may differ after adding the fine-tuning
    # training path. Record these differences; load all three with one current
    # infer_student implementation and keep the immutable inference files equal.
    variant = dict(id=variant_id, kind='student', steps=4, train_root=str(train_root),
                   checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
                   epoch=evaluated['epoch'], validation_score=evaluated['val_rollout_score'],
                   planned_epochs=planned, training_manifest=manifest,
                   training_record_sha256={name: sha256_file(train_root / name) for name in TRAIN_RECORDS},
                   source_differences_from_training=differences)
    if dense_noise:
        variant.update(selection_rule='predeclared_epoch_1', validation_best_epoch=best['epoch'])
    return variant


def comparison_variants(args, current_sources):
    roots = (args.before_train_root, args.one_epoch_train_root, args.three_epoch_train_root)
    variants = [inspect_training(root, label, current_sources) for root, label in zip(roots, STUDENTS)]
    teacher = args.teacher_ckpt.expanduser().resolve()
    if not teacher.is_file():
        raise FileNotFoundError(teacher)
    reference = variants[0]['training_manifest']['config']
    shared_keys = ('model', 'dataset', 'inference_patch_size', 'seed_ratio', 'fuse_tau_ratio',
                   'sor_enable', 'surface_projection', 'cpu_threads', 'gpu_mem_limit_gib',
                   'gpu_mem_headroom_gib', 'gpu_mem_fraction')
    for variant in variants:
        manifest = variant['training_manifest']
        if any(manifest['config'].get(k) != reference.get(k) for k in shared_keys):
            raise ValueError(f'Model/data/inference settings differ: {variant["id"]}')
        if Path(manifest['checkpoint']).resolve() != teacher:
            raise ValueError('Teacher path differs from the original training manifest')
        if variant['id'] != 'student_before':
            initialization = manifest['initialization']
            if (initialization['checkpoint_sha256'] != variants[0]['checkpoint_sha256']
                    or initialization['source_epoch'] != variants[0]['epoch']):
                raise ValueError('Fine-tunes must start from the supplied before checkpoint')
    teacher_hash = sha256_file(teacher)
    variants.extend(dict(id=f'teacher_{steps}', kind='teacher', steps=steps,
                         checkpoint=str(teacher), checkpoint_sha256=teacher_hash)
                    for steps in (2, 4, 6, 8, 10))
    return variants


def make_plan(args):
    current_sources = source_hashes()
    quality_reference = None
    if args.dense_noise_train_root is not None:
        from tools.test_teacher_endpoint_six import DENSE_NOISE_REFERENCE
        quality_reference = load_json(DENSE_NOISE_REFERENCE)
        if args.seed != quality_reference['test_seed'] or args.patch_batch != 1:
            raise ValueError('Dense-noise quality acceptance requires the reference seed and patch batch 1')
        variants = [inspect_training(args.dense_noise_train_root, DENSE_NOISE_STUDENT,
                                     current_sources, quality_reference)]
    else:
        variants = comparison_variants(args, current_sources)
    config = copy.deepcopy(variants[0]['training_manifest']['config'])
    config['test_patch_batch'] = args.patch_batch
    # Use the same six-condition inputs and normalization in both modes.
    for key in ('TEST_NOISY_PATH', 'TEST_CLEAN_PATH', 'TEST_MESH_ROOT'):
        if config['dataset']['_base_'].get(key):
            raise ValueError(f'Use the standard six-condition PUNet inputs, without {key}')
    settings = dict(seed=args.seed, device=args.device, repeats=args.repeats, warmup=args.warmup,
                    patch_batch=args.patch_batch, cpu_threads=int(config.get('cpu_threads', 8)),
                    timing_scope='GPU-resident normalized noisy cloud -> GPU denoised cloud; FPS/KNN/patches/rollout/fusion',
                    timing_excludes='loading/normalization/H2D/D2H/quality metrics/file I/O/RNG setup',
                    timing_clock='perf_counter_ns with CUDA synchronization before and after',
                    throughput_scope='serial inference-only: total clouds or input points / summed measured intervals',
                    quality_repeat=0, vote_times=1, precision='float32, no autocast',
                    memory_scope='PyTorch allocator peak including one resident model and input; excludes metric phase',
                    memory_limit_note='The configured 48 GiB cap is not a measured peak',
                    source_sha256=current_sources)
    jobs = [dict(variant=v, condition=condition_label(n, s), points=n, noise=s, config=config)
            for n, s in CONDITIONS for v in variants]
    plan = dict(schema_version=1, settings=settings, variants=variants, jobs=jobs)
    if quality_reference is not None:
        plan.update(benchmark_mode='dense_noise_finetune', quality_reference=quality_reference)
    return plan


def load_checkpoint(torch, path):
    options = dict(map_location='cpu')
    if 'weights_only' in inspect.signature(torch.load).parameters:
        options['weights_only'] = False
    return torch.load(path, **options)


def validate_student(saved, variant):
    manifest = variant['training_manifest']
    config = manifest['config']
    if (saved.get('epoch') != variant['epoch'] or saved.get('distillation') != SCHEDULE
            or saved.get('model_config') != config['model']
            or saved.get('best_epoch') != variant.get('validation_best_epoch', variant['epoch'])):
        raise ValueError('Selected checkpoint does not match training metadata')
    selection = saved.get('selection', {})
    if selection.get('metric') != 'val_rollout_score' or not math.isclose(
            selection.get('value', float('inf')), variant['validation_score'], rel_tol=1e-10):
        raise ValueError('Checkpoint and recorded validation score differ')
    for name in ('teacher_endpoint', 'endpoint_finetune', 'training_input'):
        if saved.get(name) != config.get(name):
            raise ValueError(f'Checkpoint objective mismatch: {name}')
    for name in ('teacher_endpoint_protocol', 'endpoint_finetune_protocol', 'initialization'):
        if saved.get(name) != manifest.get(name):
            raise ValueError(f'Checkpoint provenance mismatch: {name}')
    if saved.get('teacher_checkpoint') != manifest['checkpoint']:
        raise ValueError('Checkpoint and recorded Teacher differ')


def check_saved_checkpoints(plan):
    """Validate actual saved metadata on CPU before starting any GPU worker."""
    import torch
    seen = set()
    for variant in plan['variants']:
        if variant['checkpoint'] in seen:
            continue
        seen.add(variant['checkpoint'])
        saved = load_checkpoint(torch, variant['checkpoint'])
        if variant['kind'] == 'student':
            validate_student(saved, variant)
        elif saved.get('distillation') is not None:
            raise ValueError('The Teacher path points to a distilled Student checkpoint')
        del saved


def worker(plan_path, job_index):
    plan = load_json(plan_path)
    job, settings = plan['jobs'][job_index], plan['settings']
    if source_hashes() != settings['source_sha256']:
        raise ValueError('Source files changed after the comparison plan was created')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(settings['cpu_threads'])
    os.environ['MPLBACKEND'] = 'Agg'
    import numpy as np
    import torch
    from easydict import EasyDict
    from utils.gpu_memory import apply_gpu_memory_limit, exit_on_cuda_oom

    if not torch.cuda.is_available():
        raise RuntimeError('Run manually in the original CUDA environment')
    torch.cuda.set_device(settings['device'])
    torch.set_num_threads(settings['cpu_threads'])
    device = torch.device('cuda', settings['device'])
    config = EasyDict(copy.deepcopy(job['config']))
    data = config.dataset._base_
    data.update(TEST_RESOLUTION=f'{job["points"]}_poisson', TEST_NOISE=job['noise'],
                TEST_NOISY_DIR=f'PUNet_{job["points"]}_poisson_{job["noise"]:.2f}')
    output = plan_path.parent / job['variant']['id'] / job['condition']
    output.mkdir(parents=True, exist_ok=True)
    budget = apply_gpu_memory_limit(config, device)
    with exit_on_cuda_oom():
        from tools import builder
        from tools import runner_distill as runner
        from tools.runner_finetune import patch_based_denoise

        def seed():
            random.seed(settings['seed'])
            np.random.seed(settings['seed'])
            torch.manual_seed(settings['seed'])
            torch.cuda.manual_seed_all(settings['seed'])

        seed()
        # Same backend flags for every job, matching the normal FP32 tester.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        variant = job['variant']
        saved = load_checkpoint(torch, variant['checkpoint'])
        checkpoint_epoch = saved.get('epoch', -1)
        model = builder.model_builder(config.model)
        weights = saved.get('base_model') if saved.get('model') is None else saved['model']
        if weights is None:
            raise ValueError('Checkpoint is missing its model state')
        if variant['kind'] == 'student':
            validate_student(saved, variant)
            if not any(k.replace('module.', '').startswith('step_condition.') for k in weights):
                raise ValueError('Student checkpoint has no trained step condition')
            runner.load_student_state(model, weights)
        else:
            model.load_state_dict({k.replace('module.', ''): v for k, v in weights.items()}, strict=True)
        del weights, saved
        model = model.to(device).eval().requires_grad_(False)
        parameters = sum(p.numel() for p in model.parameters())
        options = runner._patch_options(config, settings['patch_batch'])

        def infer(noisy):
            with torch.no_grad():
                if variant['kind'] == 'student':
                    return runner.infer_student(model, noisy, job['noise'], options, teacher_nodes=NODES)
                return patch_based_denoise(model, noisy, job['noise'], **options,
                                          num_steps=variant['steps'], step_size=0.3, decay=0.95,
                                          return_trajectory=False, raise_on_memory_pressure=True)

        _, loader = builder.dataset_builder(argparse.Namespace(distributed=False, local_rank=0),
                                           EasyDict(_base_=data, others=EasyDict(subset='test', bs=1)))
        clouds = []
        for noisy, clean, sigmas, centers, scales, names in loader:
            name = str(names[0])
            if noisy.shape != (1, job['points'], 3) or clean.shape != noisy.shape:
                raise ValueError(f'Incorrect point counts: {name}')
            if sigmas is not None and not math.isclose(float(sigmas.reshape(-1)[0]), job['noise'], abs_tol=1e-7):
                raise ValueError(f'Incorrect noise: {name}')
            tensors = (noisy, clean, torch.as_tensor(centers[0]), torch.as_tensor(scales[0]))
            digest = hashlib.sha256()
            for tensor in tensors:
                array = tensor.detach().cpu().contiguous().numpy()
                digest.update(str((array.shape, str(array.dtype))).encode())
                digest.update(array.tobytes())
            clouds.append(dict(name=name, noisy=noisy[0].clone(), clean=clean.clone(),
                               center=tensors[2].clone(), scale=tensors[3].clone(), sha256=digest.hexdigest()))
        del loader
        if len(clouds) != 20 or len({c['name'] for c in clouds}) != 20:
            raise ValueError('Each condition requires exactly 20 unique PUNet clouds')
        if any(c['noisy'].dtype != torch.float32 for c in clouds):
            raise ValueError('Expected float32 input')
        properties = torch.cuda.get_device_properties(device)
        environment = dict(gpu_name=properties.name, gpu_total_bytes=properties.total_memory,
                           compute_capability=[properties.major, properties.minor],
                           torch=str(torch.__version__), cuda=torch.version.cuda,
                           cudnn=torch.backends.cudnn.version(), python=platform.python_version(),
                           platform=platform.platform(), cpu_threads=torch.get_num_threads(),
                           cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                           cudnn_benchmark=torch.backends.cudnn.benchmark,
                           cudnn_deterministic=torch.backends.cudnn.deterministic,
                           matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                           cudnn_tf32=torch.backends.cudnn.allow_tf32, precision='float32_no_autocast')
        patches = max(1, int(options['seed_ratio'] * job['points'] / options['patch_size']))
        manifest = dict(job=job, settings=settings, config=config, environment=environment,
                        checkpoint_epoch=checkpoint_epoch, parameters=parameters,
                        patches_per_cloud=patches,
                        model_calls_per_cloud=math.ceil(patches / settings['patch_batch']) * variant['steps'],
                        input_sha256={c['name']: c['sha256'] for c in clouds}, gpu_memory_limit=budget,
                        quality_protocol=dict(cd='ChamferDistanceL2, clean unit sphere, x1e4',
                                              p2m='compute_p2m(test), bidirectional mesh unit sphere, x1e4',
                                              score='CD + 0.3 * P2M', aggregation='equal weight per cloud'))
        write_json(output / 'manifest.json', manifest)
        # Metrics are evaluated only AFTER all timed repeats. Mesh/CD allocations
        # can therefore never pollute inference memory peaks or timing warmup.
        torch.cuda.empty_cache()
        warm_input = clouds[0]['noisy'].to(device)
        for _ in range(settings['warmup']):
            warm_output = infer(warm_input)
            torch.cuda.synchronize(device)
            del warm_output
        del warm_input
        timings, predictions = [], {}
        timing_fields = ['name', 'repeat', 'input_points', 'output_points', 'latency_ms',
                         'allocated_before_bytes', 'peak_allocated_bytes', 'peak_reserved_bytes']
        with (output / 'timing_runs.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=timing_fields)
            writer.writeheader()
            for repeat in range(settings['repeats']):
                seed()  # Warmup/repeats do not advance the quality measurement RNG.
                for cloud in clouds:
                    noisy = cloud['noisy'].to(device)
                    torch.cuda.synchronize(device)
                    allocated_before = torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    start = time.perf_counter_ns()
                    prediction = infer(noisy)
                    torch.cuda.synchronize(device)
                    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                    peak_allocated = torch.cuda.max_memory_allocated(device)
                    peak_reserved = torch.cuda.max_memory_reserved(device)
                    if prediction.shape != noisy.shape or not bool(torch.isfinite(prediction).all()):
                        raise ValueError(f'Invalid denoised cloud: {cloud["name"]}')
                    row = dict(name=cloud['name'], repeat=repeat, input_points=noisy.shape[0],
                               output_points=prediction.shape[0], latency_ms=elapsed_ms,
                               allocated_before_bytes=allocated_before,
                               peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved)
                    timings.append(row)
                    writer.writerow(row)
                    handle.flush()
                    if repeat == 0:
                        predictions[cloud['name']] = prediction.detach().cpu()
                    del prediction, noisy
                    print(f'[{variant["id"]}/{job["condition"]}] repeat={repeat + 1} '
                          f'{cloud["name"]}: {elapsed_ms:.2f} ms', flush=True)
        ops = runner.baseline_metric_ops(config, device)
        quality = []
        mesh_root = Path(ops['mesh_root'])
        manifest['mesh_sha256'] = {}
        for cloud in clouds:
            name = cloud['name']
            world, metrics = runner.evaluate_baseline_metrics(
                predictions[name].to(device), cloud['clean'], cloud['center'], cloud['scale'], name, config, ops)
            quality.append(dict(name=name, sigma0=job['noise'], input_points=job['points'],
                                output_points=world.shape[0], **metrics))
            manifest['mesh_sha256'][name] = sha256_file(mesh_root / 'test' / f'{name}.off')
            del world
        write_csv(output / 'test_metrics.csv', quality)
        manifest['quality_protocol']['mesh_root'] = str(mesh_root)
        write_json(output / 'manifest.json', manifest)
        summary = summarize(timings, quality, job['points'], settings['repeats'])
        write_json(output / 'summary.json', dict(complete=True, **summary))
        print(json.dumps(summary, allow_nan=False), flush=True)


def package(root):
    archive = root.with_name(root.name + '.zip')
    temporary = root.with_name(root.name + '.zip.tmp')
    # Rebuild only this result's archive on resume/report-only, never input ZIPs.
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix != '.tmp':
                zipped.write(path, path.relative_to(root).as_posix())
    temporary.replace(archive)
    print(f'RESULT_ROOT={root}\nZIP={archive}', flush=True)


def finalize(root, plan):
    rows = build_report(root)
    acceptance = None
    if plan.get('benchmark_mode') == 'dense_noise_finetune':
        from tools.test_teacher_endpoint_six import dense_noise_acceptance
        acceptance = dense_noise_acceptance(rows, plan['quality_reference'])
        write_json(root / 'quality_acceptance.json', acceptance)
    package(root)
    if acceptance is not None:
        print(f'ACCEPT_CANDIDATE={int(acceptance["accepted"])}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before_train_root', type=Path)
    parser.add_argument('--one_epoch_train_root', type=Path)
    parser.add_argument('--three_epoch_train_root', type=Path)
    parser.add_argument('--teacher_ckpt', type=Path)
    parser.add_argument('--dense_noise_train_root', type=Path,
                        help='Measure the dense-noise epoch-1 candidate in all six conditions')
    parser.add_argument('--output_root', type=Path)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--patch_batch', type=int, default=1)
    parser.add_argument('--resume', action='store_true', help='Continue this exact existing result plan')
    parser.add_argument('--report_only', type=Path, help='Rebuild CSVs/plots/ZIP from complete records; no inference')
    parser.add_argument('--worker_plan', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--job_index', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.worker_plan is not None:
        worker(args.worker_plan.resolve(), args.job_index)
        return
    if args.report_only is not None:
        root = args.report_only.expanduser().resolve()
        finalize(root, load_json(root / 'plan.json'))
        return
    comparison_inputs = (args.before_train_root, args.one_epoch_train_root,
                         args.three_epoch_train_root, args.teacher_ckpt)
    if args.dense_noise_train_root is not None:
        if any(value is not None for value in comparison_inputs):
            parser.error('--dense_noise_train_root is a standalone candidate evaluation')
    elif any(value is None for value in comparison_inputs):
        parser.error('Provide --dense_noise_train_root or all three historical train roots and --teacher_ckpt')
    if not 0 <= args.seed < 2 ** 32 or args.device < 0 or min(args.repeats, args.warmup, args.patch_batch) < 1:
        parser.error('Require nonnegative device/uint32 seed and positive repeats/warmup/patch_batch')
    if args.resume and args.output_root is None:
        parser.error('--resume requires the existing --output_root')
    # Check plotting dependency before the long GPU run; import has no CUDA work.
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot  # noqa: F401
    plan = make_plan(args)
    check_saved_checkpoints(plan)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = 'dense_noise_quality_efficiency' if args.dense_noise_train_root else 'teacher10_student4'
    root = (args.output_root.expanduser().resolve() if args.output_root else
            ROOT / 'experiments/efficiency' / f'{prefix}_{stamp}_{os.getpid()}')
    plan_path = root / 'plan.json'
    if args.resume:
        if load_json(plan_path) != plan:
            raise ValueError('Resume requires the same checkpoints, code, records and measurement settings')
    else:
        if root.exists() or root.with_name(root.name + '.zip').exists():
            raise FileExistsError(f'Choose a new output_root: {root}')
        root.mkdir(parents=True)
        write_json(plan_path, plan)
        write_json(root / 'created.json', dict(utc=datetime.now(timezone.utc).isoformat(), command=sys.argv))
        for variant in plan['variants']:
            if variant['kind'] == 'student':
                dest = root / 'training' / variant['id']
                dest.mkdir(parents=True)
                for name in TRAIN_RECORDS:
                    shutil.copyfile(Path(variant['train_root']) / name, dest / name)
        for name in SOURCES:
            dest = root / 'source' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)
        shutil.copyfile(ROOT / 'docs/teacher10_efficiency.md', root / 'measurement_protocol.md')
        if args.dense_noise_train_root is not None:
            shutil.copyfile(ROOT / 'docs/teacher10_dense_noise_finetune.md', root / 'dense_noise_protocol.md')
    print(f'RESULT_ROOT={root}\n{len(plan["jobs"])} jobs, each 20 clouds x {args.repeats} timed repeats', flush=True)
    # Different method order per condition reduces a fixed early/late ordering
    # effect. Processes remain strictly sequential; no GPU concurrency is used.
    order = []
    variant_count = len(plan['variants'])
    for condition_index in range(len(CONDITIONS)):
        indices = list(range(condition_index * variant_count, (condition_index + 1) * variant_count))
        random.Random(args.seed + condition_index).shuffle(indices)
        order.extend(indices)
    compared_inputs, compared_environment = {}, []

    def accept_result(output, job):
        _, manifest, _, _ = read_result(output, job, plan['settings'])
        inputs = (manifest['input_sha256'], manifest['mesh_sha256'])
        if compared_inputs.setdefault(job['condition'], inputs) != inputs:
            raise ValueError(f'Input/mesh data changed across methods: {job["condition"]}')
        if compared_environment and compared_environment[0] != manifest['environment']:
            raise ValueError('Hardware/software/precision changed between jobs')
        if not compared_environment:
            compared_environment.append(manifest['environment'])

    for index in order:
        job = plan['jobs'][index]
        output = root / job['variant']['id'] / job['condition']
        if args.resume and (output / 'summary.json').is_file():
            accept_result(output, job)
            print(f'[resume] verified {job["variant"]["id"]}/{job["condition"]}', flush=True)
            continue
        output.mkdir(parents=True, exist_ok=True)
        print(f'[job] {job["variant"]["id"]}/{job["condition"]}', flush=True)
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker_plan', str(plan_path),
                   '--job_index', str(index)]
        with (output / 'run.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace', bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        accept_result(output, job)
    # Detect a checkpoint overwritten while the long benchmark was running.
    for path, expected in {v['checkpoint']: v['checkpoint_sha256'] for v in plan['variants']}.items():
        if sha256_file(path) != expected:
            raise ValueError(f'Checkpoint changed during benchmarking: {path}')
    finalize(root, plan)


if __name__ == '__main__':
    main()
