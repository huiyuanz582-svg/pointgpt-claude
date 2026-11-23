"""Manual Teacher4 CD/P2M suite: PUNet lap/cov and PCNet, with six timing cases.

The coordinator discovers existing test data. GPU jobs are sequential, with a
fresh process per scene. Only --plan_only avoids model/checkpoint loading.
"""

import argparse
import copy
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / 'tools'):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

# Import the standalone module, not tools/__init__.py (which eagerly imports CUDA code).
import benchmark_teacher10_six as benchmark
from utils.denoise_benchmark_report import write_json
from utils.teacher4_generalization import (
    FAMILIES, build_report, discover_scenes, scene_job, validate_result, verify_inputs)

EXTRA_SOURCES = ('tools/benchmark_teacher4_generalization.py', 'utils/teacher4_generalization.py')
DEFAULT_CONFIG = 'cfgs/PointGPT-L/distill_10to4_input_scheduled.yaml'


def load_config(path, patch_batch):
    # This small resolver deliberately avoids importing the training/model packages.
    import yaml
    with path.open(encoding='utf-8-sig') as handle:
        config = yaml.safe_load(handle)
    dataset = config['dataset']['_base_']
    if isinstance(dataset, str):
        dataset_path = (ROOT / dataset).resolve()
        with dataset_path.open(encoding='utf-8-sig') as handle:
            dataset = yaml.safe_load(handle)
    else:
        dataset_path = None
    config['dataset']['_base_'] = copy.deepcopy(dataset)
    config['dataset']['_base_']['NUM_WORKERS'] = 0
    config['test_patch_batch'] = patch_batch
    config['sor_enable'] = False  # SOR-on quality is a paired, post-inference evaluation.
    config['surface_projection'] = dict(enable=False)
    if config['model']['NAME'] != 'PointTransformer' or config.get('ablation'):
        raise ValueError('Use the unmodified PointGPT-L model configuration for this Teacher baseline')
    return config, dataset_path


def make_plan(args):
    config_path = args.config.expanduser().resolve()
    config, dataset_path = load_config(config_path, args.patch_batch)
    checkpoint = args.teacher_ckpt.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Missing original Teacher checkpoint: {checkpoint}')
    checkpoint_hash = benchmark.sha256_file(checkpoint)
    if args.expected_checkpoint_sha256 and checkpoint_hash != args.expected_checkpoint_sha256.lower():
        raise ValueError('Teacher checkpoint SHA256 differs from the requested identity')
    scenes, ignored = discover_scenes(
        args.data_root, other_noise_root=args.other_noise_root, pcnet_noisy_root=args.pcnet_noisy_root,
        punet_root=args.punet_root, pcnet_root=args.pcnet_root)
    variant = dict(id='teacher_4', kind='teacher', steps=4, checkpoint=str(checkpoint),
                   checkpoint_sha256=checkpoint_hash)
    settings = dict(seed=args.seed, device=args.device, repeats=args.repeats, warmup=args.warmup,
                    patch_batch=args.patch_batch, cpu_threads=int(config.get('cpu_threads', 8)),
                    teacher_steps=4, teacher_step_size=0.3, teacher_decay=0.95,
                    timing_scope='GPU-resident normalized noisy cloud -> GPU denoised cloud; FPS/KNN/patches/rollout/fusion',
                    timing_excludes='SOR/surface projection/loading/normalization/H2D/D2H/quality metrics/file I/O/RNG setup',
                    timing_clock='perf_counter_ns with CUDA synchronization before and after',
                    throughput_scope='serial inference-only: total clouds or input points / summed measured intervals',
                    quality_repeat=0, vote_times=1, precision='float32, no autocast',
                    memory_scope='PyTorch allocator peak including one resident model and input; excludes metric phase',
                    memory_limit_note='The configured memory budget is not a measured peak',
                    latency_std_scope='population SD over all clouds x repeats, not SD of the round means',
                    extra_source_files=list(EXTRA_SOURCES), source_sha256=benchmark.source_hashes(EXTRA_SOURCES))
    if settings['cpu_threads'] < 1:
        raise ValueError('cpu_threads must be positive')
    config_files = [config_path] + ([dataset_path] if dataset_path else [])
    return dict(schema_version=1, benchmark_mode='teacher4_generalization', settings=settings,
                config_files_sha256={str(p): benchmark.sha256_file(p) for p in config_files},
                variants=[variant], jobs=[scene_job(scene, variant, config) for scene in scenes],
                ignored_other_noise_directories=ignored)


def print_plan(plan):
    print('Teacher4: step_size=0.3, decay=0.95, no training, no surface projection', flush=True)
    for family in FAMILIES:
        jobs = [j for j in plan['jobs'] if j['dataset'] == family]
        print(f'{family}: {len(jobs)} scenes; SOR={"off" if family == "pcnet" else "off + on"}', flush=True)
        for job in jobs:
            mode = 'quality + efficiency' if job['measure_efficiency'] else 'quality only'
            print(f'  {job["points"]} / {job["noise"]:.1%}: {job["expected_clouds"]} clouds, '
                  f'{mode}\n    {job["noisy_dir"]}', flush=True)
    if plan['ignored_other_noise_directories']:
        print('Other noise directories excluded (neither lap nor cov):', flush=True)
        for path in plan['ignored_other_noise_directories']:
            print(f'  {path}', flush=True)
    print('Six efficiency cases; SOR and metrics are excluded from measured latency.', flush=True)


def run_worker(plan_path, index):
    plan = benchmark.load_json(plan_path)
    job = plan['jobs'][index]
    variant = job['variant']
    verify_inputs(job)
    if benchmark.sha256_file(variant['checkpoint']) != variant['checkpoint_sha256']:
        raise ValueError('Teacher checkpoint changed after planning')
    benchmark.worker(plan_path, index)
    verify_inputs(job)
    if benchmark.sha256_file(variant['checkpoint']) != variant['checkpoint_sha256']:
        raise ValueError('Teacher checkpoint changed during evaluation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher_ckpt', type=Path)
    parser.add_argument('--data_root', type=Path, default=ROOT / 'data')
    parser.add_argument('--config', type=Path, default=ROOT / DEFAULT_CONFIG)
    parser.add_argument('--other_noise_root', type=Path)
    parser.add_argument('--pcnet_noisy_root', type=Path)
    parser.add_argument('--punet_root', type=Path, help='Contains pointclouds/test and meshes/test')
    parser.add_argument('--pcnet_root', type=Path, help='Contains pointclouds/test and meshes/test')
    parser.add_argument('--expected_checkpoint_sha256')
    parser.add_argument('--output_root', type=Path)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--patch_batch', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plan_only', action='store_true', help='Inspect paths and hashes; no model or CUDA work')
    parser.add_argument('--report_only', type=Path, help='Rebuild tables/ZIP from completed records; no model')
    parser.add_argument('--worker_plan', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--job_index', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.worker_plan:
        if args.job_index is None:
            parser.error('--worker_plan requires --job_index')
        run_worker(args.worker_plan.resolve(), args.job_index)
        return
    if args.report_only:
        root = args.report_only.expanduser().resolve()
        build_report(root, benchmark.load_json(root / 'plan.json'))
        benchmark.package(root)
        return
    if args.teacher_ckpt is None:
        parser.error('--teacher_ckpt must point to the original Teacher weight file')
    if not 0 <= args.seed < 2 ** 32 or args.device < 0 or min(args.patch_batch, args.warmup, args.repeats) < 1:
        parser.error('Require uint32 seed, nonnegative device and positive patch_batch/warmup/repeats')
    if args.resume and args.output_root is None:
        parser.error('--resume requires the existing --output_root')
    plan = make_plan(args)
    print_plan(plan)
    if args.plan_only:
        return
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(plan['settings']['cpu_threads'])
    benchmark.check_saved_checkpoints(plan)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    root = (args.output_root.expanduser().resolve() if args.output_root else
            ROOT / 'experiments/efficiency' / f'teacher4_generalization_{stamp}_{os.getpid()}')
    plan_path = root / 'plan.json'
    if args.resume:
        if benchmark.load_json(plan_path) != plan:
            raise ValueError('Resume requires identical data/checkpoint/code/config/measurement settings')
    else:
        if root.exists() or root.with_name(root.name + '.zip').exists():
            raise FileExistsError(f'Choose a new output_root: {root}')
        root.mkdir(parents=True)
        write_json(plan_path, plan)
        write_json(root / 'created.json', dict(utc=datetime.now(timezone.utc).isoformat(), command=sys.argv))
        for name in plan['settings']['source_sha256']:
            target = root / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        for index, path in enumerate(plan['config_files_sha256']):
            target = root / 'configs' / f'{index}_{Path(path).name}'
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        shutil.copyfile(ROOT / 'docs/teacher4_generalization.md', root / 'measurement_protocol.md')
    print(f'RESULT_ROOT={root}', flush=True)
    environment = None
    for index, job in enumerate(plan['jobs']):
        output = root / job['variant']['id'] / job['condition']
        complete = (output / 'summary.json').is_file()
        if args.resume and complete:
            print(f'[resume] {job["condition"]}', flush=True)
        else:
            output.mkdir(parents=True, exist_ok=True)
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
        _, manifest, _, _ = validate_result(root, job, plan['settings'])
        if environment is not None and environment != manifest['environment']:
            raise ValueError('Hardware/software/precision settings changed between jobs')
        environment = manifest['environment']
    variant = plan['variants'][0]
    if benchmark.sha256_file(variant['checkpoint']) != variant['checkpoint_sha256']:
        raise ValueError('Teacher checkpoint changed during the suite')
    build_report(root, plan)
    benchmark.package(root)


if __name__ == '__main__':
    main()
