"""Manual six-condition quality and efficiency evaluation.

Each variant/condition runs in a fresh, sequential subprocess with one model.
No training, optimizer updates, trajectory capture, or checkpoint writes.
Use --local_endpoint_train_root to evaluate just the original joint Student4.
"""

import argparse
import copy
import csv
from contextlib import ExitStack
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
    CONDITIONS, LOCAL_ENDPOINT_STUDENT, STUDENTS, build_report, condition_label,
    read_result, summarize, write_csv, write_json)

NODES = [0, 2, 5, 7, 10]
SCHEDULE = dict(teacher_nodes=NODES, teacher_steps=10, teacher_eta=0.3, teacher_decay=0.95,
                student_steps=4, student_eta=1.0,
                student_decay=[0.95 ** (u - t) for t, u in zip(NODES[:-1], NODES[1:])])
INFERENCE_SOURCES = ('models/PointGPT.py', 'models/step_condition.py',
                     'tools/runner_finetune.py', 'datasets/ScoreDenoiseDataset.py')
JOINT_TRAINING_SOURCES = INFERENCE_SOURCES + ('tools/runner_distill.py', 'utils/teacher_endpoint.py',
                                             'utils/training_input.py')
SOURCES = INFERENCE_SOURCES + ('tools/runner_distill.py', 'tools/builder.py',
                              'utils/teacher_endpoint.py', 'utils/training_input.py',
                              'utils/misc.py', 'utils/p2m_loss.py', 'utils/gpu_memory.py',
                              'tools/benchmark_teacher10_six.py', 'utils/denoise_benchmark_report.py')
TRAIN_RECORDS = ('manifest.json', 'validation_manifest.json', 'train.jsonl')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(extra=()):
    return {name: hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for name in SOURCES + tuple(extra)}


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def inspect_training(train_root, variant_id, actual_sources):
    """Read logs and fingerprints without importing torch in the coordinator."""
    train_root = train_root.expanduser().resolve()
    checkpoint = train_root / 'ckpt-best.pth'
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
    expected_epochs = {'student_ft1': 1, 'student_ft3': 3, LOCAL_ENDPOINT_STUDENT: 1}
    if variant_id == 'student_before':
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
    recorded_sources = manifest.get('source_sha256', {})
    differences = {}
    if variant_id == LOCAL_ENDPOINT_STUDENT:
        protocol = manifest.get('endpoint_finetune_protocol', {})
        if (manifest.get('mode') != 'train' or protocol.get('objective') != 'local_endpoint'
                or not protocol.get('frozen_parameters') or logs[0].get('phase') != 'initial_validation'):
            raise ValueError('Expected the completed 1-epoch local + endpoint fine-tuning run')
        for name in JOINT_TRAINING_SOURCES:
            if recorded_sources.get(name) not in (actual_sources[name], sha256_file(ROOT / name)):
                raise ValueError(f'Current source differs from the joint Student training record: {name}')
    else:
        differences = {name: dict(training=expected, current=actual_sources.get(name))
                       for name, expected in recorded_sources.items() if expected != actual_sources.get(name)}
        for name in INFERENCE_SOURCES:
            if recorded_sources.get(name) != actual_sources[name]:
                raise ValueError(f'Inference implementation differs from the training record: {name}')
    # runner_distill/teacher_endpoint may differ after adding the fine-tuning
    # training path. Record these differences; load all three with one current
    # infer_student implementation and keep the immutable inference files equal.
    return dict(id=variant_id, kind='student', steps=4, train_root=str(train_root),
                checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
                epoch=best['epoch'], validation_score=best['val_rollout_score'],
                planned_epochs=planned, training_manifest=manifest,
                training_record_sha256={name: sha256_file(train_root / name) for name in TRAIN_RECORDS},
                source_differences_from_training=differences)


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
    local_endpoint = args.local_endpoint_train_root is not None
    if local_endpoint:
        variant = inspect_training(args.local_endpoint_train_root, LOCAL_ENDPOINT_STUDENT, current_sources)
        if (args.expected_checkpoint_sha256 is not None
                and variant['checkpoint_sha256'] != args.expected_checkpoint_sha256.lower()):
            raise ValueError('ckpt-best.pth SHA256 differs from the given checkpoint identity')
        variants = [variant]
    else:
        variants = comparison_variants(args, current_sources)
    config = copy.deepcopy(variants[0]['training_manifest']['config'])
    config['test_patch_batch'] = args.patch_batch
    # Dataset/normalization and score code are shared by both modes.
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
    if local_endpoint:
        plan['benchmark_mode'] = 'local_endpoint_finetune'
    return plan


def load_checkpoint(torch, path):
    options = dict(map_location='cpu')
    if 'weights_only' in inspect.signature(torch.load).parameters:
        options['weights_only'] = False
    return torch.load(path, **options)


def validate_student(saved, variant):
    if variant.get('checkpoint_validation') == 'fixed_10to4':
        validate_fixed_student(saved, variant)
        return
    manifest = variant['training_manifest']
    config = manifest['config']
    if (saved.get('epoch') != variant['epoch'] or saved.get('distillation') != SCHEDULE
            or saved.get('model_config') != config['model']
            or saved.get('best_epoch') != variant['epoch']):
        raise ValueError('Selected checkpoint does not match training metadata')
    selection = saved.get('selection', {})
    if selection.get('metric') != 'val_rollout_score' or not math.isclose(
            selection.get('value', float('inf')), variant['validation_score'], rel_tol=1e-10):
        raise ValueError('Checkpoint and best validation score differ')
    for name in ('teacher_endpoint', 'endpoint_finetune', 'training_input'):
        if saved.get(name) != config.get(name):
            raise ValueError(f'Checkpoint objective mismatch: {name}')
    for name in ('teacher_endpoint_protocol', 'endpoint_finetune_protocol', 'initialization'):
        if saved.get(name) != manifest.get(name):
            raise ValueError(f'Checkpoint provenance mismatch: {name}')
    if saved.get('teacher_checkpoint') != manifest['checkpoint']:
        raise ValueError('Checkpoint and recorded Teacher differ')


def validate_fixed_student(saved, variant):
    """Validate the two frozen Student checkpoints without requiring moved logs."""
    if (saved.get('distillation') != SCHEDULE
            or saved.get('model_config') != variant['model_config']
            or saved.get('epoch') != variant['expected_epoch']
            or saved.get('best_epoch') != variant['expected_epoch']):
        raise ValueError(f'{variant["id"]}: checkpoint model/schedule/selected epoch mismatch')
    selection = saved.get('selection') or {}
    score = selection.get('value')
    if (selection.get('metric') != 'val_rollout_score'
            or not isinstance(score, (int, float)) or not math.isfinite(score)):
        raise ValueError(f'{variant["id"]}: missing finite best-checkpoint selection')
    weights = saved.get('model')
    if weights is None:
        weights = saved.get('base_model')
    if not weights or not any(k.replace('module.', '').startswith('step_condition.') for k in weights):
        raise ValueError(f'{variant["id"]}: Student checkpoint has no trained step condition')
    if variant['id'] == 'student_baseline':
        # The saver deliberately omits the default teacher-forced input metadata.
        training_input = saved.get('training_input') or {'type': 'teacher_forced'}
        if (saved.get('teacher_endpoint') is not None or saved.get('endpoint_finetune') is not None
                or training_input.get('type') != 'teacher_forced'):
            raise ValueError('Ordinary distillation requires teacher-forced local training only')
    elif variant['id'] == LOCAL_ENDPOINT_STUDENT:
        protocol = saved.get('endpoint_finetune_protocol') or {}
        finetune = saved.get('endpoint_finetune') or {}
        if (protocol.get('objective') != 'local_endpoint'
                or finetune.get('local_weight') != 1.0 or finetune.get('endpoint_weight') != 1.0):
            raise ValueError('The method requires the frozen joint local + endpoint Student4')
    else:
        raise ValueError(f'Unsupported frozen Student role: {variant["id"]}')


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


def timed_inference(torch, infer, noisy, device, sor=None):
    """Measure a shared inference prefix and, optionally, its full SOR extension.

    Both intervals start at the same GPU input and finish with GPU output. No
    metrics, validation, logging or prediction archival runs between endpoints.
    """
    torch.cuda.synchronize(device)
    allocated_before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter_ns()
    prediction = infer(noisy)
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    sor_prediction, sor_row = None, None
    if sor is not None:
        # Open3D SOR runs on CPU. Include its D2H and H2D transfers so both
        # configurations have the same GPU-to-GPU input/output boundary.
        sor_prediction = sor(prediction).to(device)
        torch.cuda.synchronize(device)
        sor_elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        sor_row = dict(input_points=noisy.shape[0], output_points=sor_prediction.shape[0],
                       latency_ms=sor_elapsed_ms, allocated_before_bytes=allocated_before,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    row = dict(input_points=noisy.shape[0], output_points=prediction.shape[0], latency_ms=elapsed_ms,
               allocated_before_bytes=allocated_before, peak_allocated_bytes=peak_allocated,
               peak_reserved_bytes=peak_reserved)
    return prediction, row, sor_prediction, sor_row


def worker(plan_path, job_index):
    plan = load_json(plan_path)
    job, settings = plan['jobs'][job_index], plan['settings']
    if source_hashes(settings.get('extra_source_files', ())) != settings['source_sha256']:
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
        from tools.runner_finetune import patch_based_denoise, sor_filter

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
            if saved.get('distillation') is not None or any(
                    k.replace('module.', '').startswith('step_condition.') for k in weights):
                raise ValueError('The Teacher path points to a distilled Student checkpoint')
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
        expected_clouds = job.get('expected_clouds', 20)
        if len(clouds) != expected_clouds or len({c['name'] for c in clouds}) != expected_clouds:
            raise ValueError(f'Each condition requires exactly {expected_clouds} unique clouds')
        if 'expected_names' in job and {c['name'] for c in clouds} != set(job['expected_names']):
            raise ValueError('Loaded cloud names differ from the complete planned test set')
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
        measure_efficiency = job.get('measure_efficiency', True)
        measure_sor = job.get('measure_sor_efficiency', False)
        if measure_sor and (not measure_efficiency or not job.get('evaluate_sor_both', False)):
            raise ValueError('SOR timing requires an efficiency scene with both SOR quality modes')
        torch.cuda.empty_cache()
        warm_input = clouds[0]['noisy'].to(device)
        for _ in range(settings['warmup'] if measure_efficiency else 0):
            warm_output = infer(warm_input)
            if measure_sor:
                warm_sor = sor_filter(warm_output).to(device)
                del warm_sor
            torch.cuda.synchronize(device)
            del warm_output
        del warm_input
        timings, predictions, timings_sor, predictions_sor = [], {}, [], {}
        timing_fields = ['name', 'repeat', 'input_points', 'output_points', 'latency_ms',
                         'allocated_before_bytes', 'peak_allocated_bytes', 'peak_reserved_bytes']
        with ExitStack() as stack:
            handle = stack.enter_context((output / 'timing_runs.csv').open('w', newline='', encoding='utf-8'))
            writer = csv.DictWriter(handle, fieldnames=timing_fields)
            writer.writeheader()
            if measure_sor:
                sor_handle = stack.enter_context(
                    (output / 'timing_runs_sor.csv').open('w', newline='', encoding='utf-8'))
                sor_writer = csv.DictWriter(sor_handle, fieldnames=timing_fields)
                sor_writer.writeheader()
            for repeat in range(settings['repeats'] if measure_efficiency else 1):
                seed()  # Warmup/repeats do not advance the quality measurement RNG.
                for cloud in clouds:
                    noisy = cloud['noisy'].to(device)
                    if measure_efficiency:
                        prediction, row, sor_prediction, sor_row = timed_inference(
                            torch, infer, noisy, device, sor=sor_filter if measure_sor else None)
                    else:
                        prediction = infer(noisy)
                    if prediction.shape != noisy.shape or not bool(torch.isfinite(prediction).all()):
                        raise ValueError(f'Invalid denoised cloud: {cloud["name"]}')
                    if measure_efficiency:
                        row.update(name=cloud['name'], repeat=repeat)
                        timings.append(row)
                        writer.writerow(row)
                        handle.flush()
                    if measure_sor:
                        if (sor_prediction.ndim != 2 or sor_prediction.shape[1] != 3
                                or not 0 < sor_prediction.shape[0] <= noisy.shape[0]
                                or not bool(torch.isfinite(sor_prediction).all())):
                            raise ValueError(f'Invalid SOR cloud: {cloud["name"]}')
                        sor_row.update(name=cloud['name'], repeat=repeat)
                        timings_sor.append(sor_row)
                        sor_writer.writerow(sor_row)
                        sor_handle.flush()
                        if repeat == 0:
                            predictions_sor[cloud['name']] = sor_prediction.detach().cpu()
                        del sor_prediction
                    if repeat == 0:
                        predictions[cloud['name']] = prediction.detach().cpu()
                    del prediction, noisy
                    elapsed = f'{row["latency_ms"]:.2f} ms' if measure_efficiency else 'quality inference'
                    if measure_sor:
                        elapsed += f', including SOR {sor_row["latency_ms"]:.2f} ms'
                    print(f'[{variant["id"]}/{job["condition"]}] repeat={repeat + 1} '
                          f'{cloud["name"]}: {elapsed}', flush=True)
        ops = runner.baseline_metric_ops(config, device)
        quality, quality_sor = [], []
        sor_config = copy.deepcopy(config)
        sor_config.sor_enable = True
        mesh_root = Path(ops['mesh_root'])
        manifest['mesh_sha256'] = {}
        for cloud in clouds:
            name = cloud['name']
            world, metrics = runner.evaluate_baseline_metrics(
                predictions[name].to(device), cloud['clean'], cloud['center'], cloud['scale'], name, config, ops)
            quality.append(dict(name=name, sigma0=job['noise'], input_points=job['points'],
                                output_points=world.shape[0], **metrics))
            if job.get('evaluate_sor_both', False):
                # Reuse the actual first timed SOR output; do not filter it a second time.
                sor_input = predictions_sor[name] if measure_sor else predictions[name]
                sor_world, sor_metrics = runner.evaluate_baseline_metrics(
                    sor_input.to(device), cloud['clean'], cloud['center'], cloud['scale'],
                    name, config if measure_sor else sor_config, ops)
                quality_sor.append(dict(name=name, sigma0=job['noise'], input_points=job['points'],
                                        output_points=sor_world.shape[0], **sor_metrics))
                del sor_world
            manifest['mesh_sha256'][name] = sha256_file(mesh_root / 'test' / f'{name}.off')
            del world
        write_csv(output / 'test_metrics.csv', quality)
        if quality_sor:
            write_csv(output / 'test_metrics_sor.csv', quality_sor)
            sor_summary = summarize(timings_sor, quality_sor, job['points'], settings['repeats'],
                                    expected_clouds=expected_clouds, measure_efficiency=measure_sor,
                                    allow_filtered_output=True)
            write_json(output / 'summary_sor.json', dict(complete=True, **sor_summary))
            manifest['quality_protocol']['sor_comparison'] = dict(
                modes=['off', 'on'], same_prediction=True, nb_neighbors=20, std_ratio=2.0,
                timing_includes_sor=measure_sor)
        manifest['quality_protocol']['mesh_root'] = str(mesh_root)
        write_json(output / 'manifest.json', manifest)
        summary = summarize(timings, quality, job['points'], settings['repeats'],
                            expected_clouds=expected_clouds, measure_efficiency=measure_efficiency)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before_train_root', type=Path)
    parser.add_argument('--one_epoch_train_root', type=Path)
    parser.add_argument('--three_epoch_train_root', type=Path)
    parser.add_argument('--teacher_ckpt', type=Path)
    parser.add_argument('--local_endpoint_train_root', type=Path,
                        help='Evaluate only the joint Student4 ckpt-best in all six conditions')
    parser.add_argument('--expected_checkpoint_sha256', help='Verify the joint Student4 checkpoint identity')
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
        build_report(root)
        package(root)
        return
    comparison_inputs = (args.before_train_root, args.one_epoch_train_root,
                         args.three_epoch_train_root, args.teacher_ckpt)
    local_endpoint = args.local_endpoint_train_root is not None
    if local_endpoint:
        if any(value is not None for value in comparison_inputs):
            parser.error('--local_endpoint_train_root runs a standalone Student evaluation')
    else:
        if any(value is None for value in comparison_inputs):
            parser.error('Provide --local_endpoint_train_root or all three train roots and --teacher_ckpt')
        if args.expected_checkpoint_sha256 is not None:
            parser.error('--expected_checkpoint_sha256 requires --local_endpoint_train_root')
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
    prefix = 'local_endpoint_quality_efficiency' if local_endpoint else 'teacher10_student4'
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
    build_report(root)
    package(root)


if __name__ == '__main__':
    main()
