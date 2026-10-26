"""PUNet Gaussian inference cost/quality sweep for a fine-tuned PointGPT.

Run from the repository root on the same GPU used for the paper's evaluation.
The timed region is the existing full-cloud patch_based_denoise pipeline only:
FPS/KNN tiling, T network steps per patch, and cross-patch fusion. Loading,
host-to-device transfer, CD/P2M, and file output are outside the timer.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from datasets.ScoreDenoiseDataset import ScoreDenoise
from extensions.chamfer_dist import ChamferDistanceL2
from tools import builder
from tools.runner_finetune import normalize_unit_sphere, patch_based_denoise
from utils import p2m_loss
from utils.config import cfg_from_yaml_file


RAW_FIELDS = [
    'resolution', 'noise', 'steps', 'step_size', 'decay', 'seed_ratio', 'cloud', 'n_points',
    'patches', 'model_forwards', 'latency_s', 'peak_allocated_gib',
    'cd_x1e4', 'point_to_face_x1e4', 'face_to_point_x1e4', 'p2m_x1e4',
]
SUMMARY_FIELDS = [
    'resolution', 'noise', 'steps', 'step_size', 'decay', 'seed_ratio', 'n_clouds',
    'n_points', 'patches_per_cloud', 'model_forwards_per_cloud',
    'mean_latency_s', 'median_latency_s', 'std_latency_s', 'p95_latency_s',
    'clouds_per_s', 'points_per_s', 'max_peak_allocated_gib',
    'mean_cd_x1e4', 'mean_point_to_face_x1e4',
    'mean_face_to_point_x1e4', 'mean_p2m_x1e4',
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='cfgs/PointGPT-L/finetune_scoredenoise.yaml')
    parser.add_argument('--checkpoint', required=True, help='Fine-tuned ckpt-best.pth, not the pre-training checkpoint')
    parser.add_argument('--data-root', default='data/ScoreDenoise')
    parser.add_argument('--resolution', type=int, choices=(10000, 50000), required=True)
    parser.add_argument('--noise', type=float, choices=(0.01, 0.02, 0.03), required=True)
    parser.add_argument('--steps', type=int, nargs='+', default=[1, 5, 10, 20, 30])
    parser.add_argument('--step-size', type=float, default=None, help='Default: config.langevin.step_size (0.3)')
    parser.add_argument('--decay', type=float, default=None, help='Default: config.langevin.decay (0.95)')
    parser.add_argument('--seed-ratio', type=float, default=3.0,
                        help='Overlapping patches per cloud point, r (default: 3)')
    parser.add_argument('--patch-batch', type=int, default=None, help='Default: config.test_patch_batch')
    parser.add_argument('--warmup', type=int, default=1, help='Untimed runs before each T; default 1')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--expected-clouds', type=int, default=0, help='Optional completeness check, e.g. 20')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--resume', action='store_true', help='Continue an interrupted run with identical settings')
    args = parser.parse_args()
    if len(set(args.steps)) != len(args.steps) or any(t < 1 for t in args.steps):
        parser.error('--steps must contain distinct positive integers')
    if args.warmup < 0 or args.expected_clouds < 0:
        parser.error('--warmup and --expected-clouds must be nonnegative')
    if args.patch_batch is not None and args.patch_batch < 1:
        parser.error('--patch-batch must be positive')
    if not math.isfinite(args.seed_ratio) or args.seed_ratio <= 0:
        parser.error('--seed-ratio must be a finite positive number')
    return args


def checked_names(data_root, resolution, noise, expected_clouds):
    resolution_dir = f'{resolution}_poisson'
    noisy_label = f'PUNet_{resolution_dir}_{noise:.2f}'
    clean_dir = data_root / 'PUNet' / 'pointclouds' / 'test' / resolution_dir
    noisy_dir = data_root / 'examples' / 'pointclouds' / 'test' / noisy_label
    mesh_dir = data_root / 'PUNet' / 'meshes' / 'test'
    for path in (clean_dir, noisy_dir, mesh_dir):
        if not path.is_dir():
            raise FileNotFoundError(f'Missing PUNet directory: {path}')
    clean = {p.stem for p in clean_dir.glob('*.xyz')}
    noisy = {p.stem for p in noisy_dir.glob('*.xyz')}
    meshes = {p.stem for p in mesh_dir.glob('*.off')}
    if not clean or clean != noisy or not clean.issubset(meshes):
        raise ValueError(
            f'Incomplete paired test set: clean={len(clean)}, noisy={len(noisy)}, '
            f'meshes={len(meshes)}; clean-only={sorted(clean - noisy)}, '
            f'noisy-only={sorted(noisy - clean)}, no-mesh={sorted(clean - meshes)}'
        )
    if expected_clouds and len(clean) != expected_clouds:
        raise ValueError(f'Expected {expected_clouds} clouds, found {len(clean)}')
    return noisy_label, len(clean)


def run_denoise(model, noisy, sigma, patch_size, patch_batch,
                steps, step_size, decay, seed_ratio, fuse_tau):
    return patch_based_denoise(
        model, noisy, sigma, patch_size=patch_size, seed_ratio=seed_ratio,
        patch_batch=patch_batch, num_steps=steps, step_size=step_size,
        decay=decay, fuse_tau_ratio=fuse_tau, memory_guard=False,
    )


def write_summary(raw_path, summary_path, steps, n_clouds):
    with raw_path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(steps) * n_clouds:
        raise RuntimeError(f'Incomplete run: {len(rows)} rows; expected {len(steps) * n_clouds}')
    with summary_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for t in steps:
            group = [row for row in rows if int(row['steps']) == t]
            if len(group) != n_clouds or len({row['cloud'] for row in group}) != n_clouds:
                raise RuntimeError(f'Incomplete or duplicate results for T={t}')
            latency = [float(row['latency_s']) for row in group]
            n_points = [int(row['n_points']) for row in group]
            total_s = sum(latency)
            writer.writerow({
                'resolution': group[0]['resolution'],
                'noise': group[0]['noise'],
                'steps': t,
                'step_size': group[0]['step_size'],
                'decay': group[0]['decay'],
                'seed_ratio': group[0]['seed_ratio'],
                'n_clouds': n_clouds,
                'n_points': group[0]['n_points'],
                'patches_per_cloud': group[0]['patches'],
                'model_forwards_per_cloud': group[0]['model_forwards'],
                'mean_latency_s': statistics.mean(latency),
                'median_latency_s': statistics.median(latency),
                'std_latency_s': statistics.stdev(latency) if n_clouds > 1 else 0.0,
                'p95_latency_s': float(np.percentile(latency, 95)),
                'clouds_per_s': n_clouds / total_s,
                'points_per_s': sum(n_points) / total_s,
                'max_peak_allocated_gib': max(float(row['peak_allocated_gib']) for row in group),
                'mean_cd_x1e4': statistics.mean(float(row['cd_x1e4']) for row in group),
                'mean_point_to_face_x1e4': statistics.mean(float(row['point_to_face_x1e4']) for row in group),
                'mean_face_to_point_x1e4': statistics.mean(float(row['face_to_point_x1e4']) for row in group),
                'mean_p2m_x1e4': statistics.mean(float(row['p2m_x1e4']) for row in group),
            })


def main():
    args = parse_args()
    launch_dir = Path.cwd()
    config_path = (launch_dir / args.config).resolve()
    checkpoint_path = (launch_dir / args.checkpoint).resolve()
    data_root = (launch_dir / args.data_root).resolve()
    output_dir = (launch_dir / args.output_dir).resolve()
    for path in (config_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the model and GPU memory measurement')

    # Config _base_ paths and the project CUDA extensions resolve from the repository root.
    os.chdir(REPO_ROOT)
    config = cfg_from_yaml_file(str(config_path))
    if bool(getattr(config.surface_projection, 'enable', False)):
        raise ValueError('Disable surface_projection for the Gaussian no-post-processing comparison')
    langevin = config.langevin
    step_size = float(args.step_size if args.step_size is not None else langevin.step_size)
    decay = float(args.decay if args.decay is not None else langevin.decay)
    seed_ratio = float(args.seed_ratio)
    patch_batch = int(args.patch_batch if args.patch_batch is not None else config.test_patch_batch)
    patch_size = int(config.npoints)
    fuse_tau = float(config.fuse_tau_ratio)
    if not math.isfinite(step_size) or step_size <= 0 or not math.isfinite(decay) or not (0 < decay <= 1) or patch_batch < 1:
        raise ValueError('Invalid step size, decay, or patch batch')

    noisy_label, n_clouds = checked_names(data_root, args.resolution, args.noise, args.expected_clouds)
    test_config = config.dataset.test._base_
    test_config.ROOT = str(data_root)
    test_config.TEST_RESOLUTION = f'{args.resolution}_poisson'
    test_config.TEST_NOISY_DIR = noisy_label
    test_config.TEST_NOISE = args.noise
    test_config.TEST_NOISY_PATH = None
    test_config.TEST_CLEAN_PATH = None
    test_config.TEST_MESH_ROOT = None
    test_config.NUM_WORKERS = 0
    p2m_loss._MESH_ROOT = str(data_root / 'PUNet' / 'meshes')

    cpu_threads = int(getattr(config, 'cpu_threads', 0))
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')
    gpu_fraction = float(getattr(config, 'gpu_mem_fraction', 0))
    if 0 < gpu_fraction < 1:
        torch.cuda.set_per_process_memory_fraction(gpu_fraction, device=device)
    loader = ScoreDenoise(SimpleNamespace(distributed=False), test_config).test_dataloader()[1]
    if len(loader) != n_clouds:
        raise RuntimeError(f'DataLoader has {len(loader)} samples, expected {n_clouds}')

    builder.inject_ablation(config)
    model = builder.model_builder(config.model)
    builder.load_model(model, str(checkpoint_path))
    model = model.to(device).eval()
    cd_metric = ChamferDistanceL2().to(device)
    checkpoint_stat = checkpoint_path.stat()
    metadata = {
        'config': str(config_path), 'checkpoint': str(checkpoint_path),
        'checkpoint_bytes': checkpoint_stat.st_size,
        'checkpoint_mtime_ns': checkpoint_stat.st_mtime_ns,
        'data_root': str(data_root), 'resolution': args.resolution,
        'noise': args.noise, 'steps': args.steps, 'step_size': step_size,
        'decay': decay, 'patch_batch': patch_batch, 'patch_size': patch_size,
        'seed_ratio': seed_ratio, 'fuse_tau_ratio': fuse_tau, 'warmup': args.warmup,
        'metric_fields': ['cd_x1e4', 'point_to_face_x1e4', 'face_to_point_x1e4', 'p2m_x1e4'],
        'seed': args.seed, 'postprocessing': 'none',
        'timed_region': 'FPS + KNN tiling + all iterative model steps + patch fusion',
        'memory_metric': 'torch.cuda.max_memory_allocated, includes model weights and inference tensors',
        'gpu': torch.cuda.get_device_name(device), 'torch': torch.__version__,
        'cuda': torch.version.cuda,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / 'run_info.json'
    raw_path = output_dir / 'per_cloud.csv'
    summary_path = output_dir / 'summary.csv'
    if args.resume:
        if not metadata_path.is_file() or not raw_path.is_file():
            raise FileNotFoundError('--resume needs existing run_info.json and per_cloud.csv')
        old_metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if old_metadata != metadata:
            raise ValueError('Existing run_info.json does not match these settings')
        with raw_path.open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != RAW_FIELDS:
                raise ValueError('Existing per_cloud.csv has a different metric schema; choose a new output directory')
            old_rows = list(reader)
        done = {(int(row['steps']), row['cloud']) for row in old_rows}
        if len(done) != len(old_rows):
            raise ValueError('Duplicate rows in per_cloud.csv')
    else:
        if raw_path.exists() or metadata_path.exists():
            raise FileExistsError(f'{output_dir} already contains a run; choose a new directory or use --resume')
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
        done = set()

    print(f'GPU: {metadata["gpu"]}; clouds: {n_clouds}; steps: {args.steps}')
    print('T=1 uses the same step_size as other rows; this is a one-step partial update.')
    with raw_path.open('a' if args.resume else 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        if not args.resume:
            writer.writeheader()
            handle.flush()
        with torch.no_grad():
            for steps in args.steps:
                if len([key for key in done if key[0] == steps]) == n_clouds:
                    continue
                first_noisy, _, _, _, _, _ = next(iter(loader))
                warm_input = first_noisy[0].to(device)
                warm_sigma = torch.tensor([args.noise], dtype=torch.float32, device=device)
                for _ in range(args.warmup):
                    warm_output = run_denoise(model, warm_input, warm_sigma, patch_size,
                                              patch_batch, steps, step_size, decay, seed_ratio, fuse_tau)
                    del warm_output
                torch.cuda.synchronize(device)
                del warm_input, warm_sigma, first_noisy

                for noisy, clean, _, centers, scales, names in loader:
                    cloud = names[0]
                    if (steps, cloud) in done:
                        continue
                    if noisy.shape[1] != args.resolution or clean.shape[1] != args.resolution:
                        raise ValueError(f'{cloud}: expected {args.resolution} points in both clouds')
                    noisy_gpu = noisy[0].to(device)
                    sigma_gpu = torch.tensor([args.noise], dtype=torch.float32, device=device)
                    torch.manual_seed(args.seed)
                    torch.cuda.manual_seed_all(args.seed)
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    start = time.perf_counter()
                    denoised = run_denoise(model, noisy_gpu, sigma_gpu, patch_size,
                                           patch_batch, steps, step_size, decay, seed_ratio, fuse_tau)
                    torch.cuda.synchronize(device)
                    latency_s = time.perf_counter() - start
                    peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

                    # Match test() metrics, but keep mesh loading and CUDA metric kernels untimed.
                    center_gpu = centers[0].to(device)
                    scale_gpu = scales[0].to(device)
                    clean_world = clean.to(device) * scale_gpu + center_gpu
                    denoised_world = denoised.unsqueeze(0) * scale_gpu + center_gpu
                    p2m_components = p2m_loss.compute_p2m_components(denoised_world[0], cloud, 'test')
                    _, clean_center, clean_scale = normalize_unit_sphere(clean_world)
                    cd = cd_metric((denoised_world - clean_center) / clean_scale,
                                   (clean_world - clean_center) / clean_scale) * 1e4

                    n_points = int(noisy_gpu.shape[0])
                    patches = max(1, int(seed_ratio * n_points / patch_size))
                    row = {
                        'resolution': args.resolution, 'noise': args.noise,
                        'steps': steps, 'step_size': step_size, 'decay': decay,
                        'seed_ratio': seed_ratio,
                        'cloud': cloud, 'n_points': n_points, 'patches': patches,
                        'model_forwards': math.ceil(patches / patch_batch) * steps,
                        'latency_s': latency_s, 'peak_allocated_gib': peak_gib,
                        'cd_x1e4': float(cd.item()),
                        'point_to_face_x1e4': float(p2m_components['point_to_face'].item() * 1e4),
                        'face_to_point_x1e4': float(p2m_components['face_to_point'].item() * 1e4),
                        'p2m_x1e4': float(p2m_components['p2m'].item() * 1e4),
                    }
                    writer.writerow(row)
                    handle.flush()
                    done.add((steps, cloud))
                    print(f'{args.resolution} {args.noise:.2f} T={steps:2d} {cloud}: '
                          f'{latency_s:.3f}s CD={row["cd_x1e4"]:.3f} '
                          f'point→face={row["point_to_face_x1e4"]:.3f} '
                          f'face→point={row["face_to_point_x1e4"]:.3f} '
                          f'P2M={row["p2m_x1e4"]:.3f} peak={peak_gib:.2f} GiB', flush=True)
                    del (noisy_gpu, sigma_gpu, denoised, center_gpu, scale_gpu,
                         clean_world, denoised_world, clean_center, clean_scale, cd, p2m_components)

    write_summary(raw_path, summary_path, args.steps, n_clouds)
    print(f'Complete: {summary_path}')


if __name__ == '__main__':
    main()
