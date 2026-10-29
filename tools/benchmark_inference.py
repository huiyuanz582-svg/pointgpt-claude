"""PUNet/PCNet inference cost and quality sweep for a fine-tuned PointGPT.

Run from the repository root on the same GPU used for the paper's evaluation.
The timed region is the existing full-cloud patch_based_denoise pipeline only:
FPS/KNN tiling, T network steps per patch, and cross-patch fusion. Loading,
host-to-device transfer, CD/P2M, SOR, and file output are outside that timer.
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
import matplotlib

matplotlib.use('Agg')

from datasets.ScoreDenoiseDataset import ScoreDenoise
from extensions.chamfer_dist import ChamferDistanceL2
from tools import builder
from tools.runner_finetune import normalize_unit_sphere, patch_based_denoise, sor_filter
from utils import p2m_loss
from utils.config import cfg_from_yaml_file


RAW_FIELDS = [
    'dataset', 'resolution', 'noise_family', 'noise', 'inference_sigma', 'sor',
    'steps', 'step_size', 'decay', 'seed_ratio', 'cloud', 'n_points', 'output_points',
    'patches', 'model_forwards', 'profiled_matmul_conv_gflops',
    'latency_s', 'sor_latency_s', 'inference_plus_sor_latency_s', 'peak_allocated_gib',
    'cd_x1e4', 'point_to_face_x1e4', 'face_to_point_x1e4', 'p2m_x1e4',
]
SUMMARY_FIELDS = [
    'dataset', 'resolution', 'noise_family', 'noise', 'inference_sigma', 'sor',
    'steps', 'step_size', 'decay', 'seed_ratio', 'n_clouds',
    'n_points', 'mean_output_points', 'patches_per_cloud', 'model_forwards_per_cloud',
    'profiled_matmul_conv_gflops_per_cloud',
    'mean_latency_s', 'median_latency_s', 'std_latency_s', 'p95_latency_s',
    'mean_sor_latency_s', 'mean_inference_plus_sor_latency_s',
    'clouds_per_s', 'points_per_s', 'clouds_per_s_including_sor',
    'points_per_s_including_sor',
    'max_peak_allocated_gib',
    'mean_cd_x1e4', 'mean_point_to_face_x1e4',
    'mean_face_to_point_x1e4', 'mean_p2m_x1e4',
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='cfgs/PointGPT-L/finetune_scoredenoise.yaml')
    parser.add_argument('--checkpoint', required=True, help='Fine-tuned ckpt-best.pth, not the pre-training checkpoint')
    parser.add_argument('--data-root', default='data/ScoreDenoise')
    parser.add_argument('--dataset', choices=('PUNet', 'PCNet'), default='PUNet')
    parser.add_argument('--resolution', type=int, choices=(10000, 50000), required=True)
    parser.add_argument('--noise', type=float, required=True,
                        help='Nominal noise level (e.g. 0.01, 0.025); selects the test condition and default model sigma')
    parser.add_argument('--noise-family', choices=('gaussian', 'cov', 'lap'), default='gaussian')
    parser.add_argument('--noisy-dir', default=None,
                        help='Directory of paired noisy .xyz files; required for cov/lap and PCNet')
    parser.add_argument('--clean-root', default=None,
                        help='Clean point-cloud root containing <resolution>_poisson/; required for PCNet')
    parser.add_argument('--mesh-root', default=None,
                        help='Mesh root containing test/<cloud>.off; required for PCNet')
    sigma_group = parser.add_mutually_exclusive_group()
    sigma_group.add_argument('--inference-sigma', type=float, default=None,
                             help='Model noise input; default is the nominal --noise level')
    sigma_group.add_argument('--inference-sigma-ratio', type=float, default=None,
                             help='Model noise input as a multiple of nominal --noise, for sigma0 sensitivity sweeps')
    parser.add_argument('--sor', choices=('off', 'on'), default='off',
                        help='Apply the same SOR as main.py --test after core inference')
    parser.add_argument('--steps', type=int, nargs='+', default=[1, 5, 10, 20, 30])
    parser.add_argument('--step-size', type=float, default=None, help='Default: config.langevin.step_size (0.3)')
    parser.add_argument('--decay', type=float, default=None, help='Default: config.langevin.decay (0.95)')
    parser.add_argument('--seed-ratio', type=float, default=3.0,
                        help='Overlapping patches per cloud point, r (default: 3)')
    parser.add_argument('--patch-batch', type=int, default=None, help='Default: config.test_patch_batch')
    parser.add_argument('--warmup', type=int, default=1, help='Untimed runs before each T; default 1')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--expected-clouds', type=int, default=0, help='Optional completeness check, e.g. 20')
    parser.add_argument('--profile-flops', action='store_true',
                        help='Estimate counted model matmul/conv FLOPs outside the timed region')
    parser.add_argument('--visualize-clouds', nargs='+', default=None, metavar='NAME',
                        help='Save full .xyz and a comparison PNG for named clouds, or use all')
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
    if not math.isfinite(args.noise) or args.noise <= 0:
        parser.error('--noise must be a finite positive number')
    if args.noise_family != 'gaussian' and not args.noisy_dir:
        parser.error('--noisy-dir is required for cov/lap noise')
    if args.dataset == 'PCNet' and not (args.noisy_dir and args.clean_root and args.mesh_root):
        parser.error('PCNet requires --noisy-dir, --clean-root, and --mesh-root')
    if args.inference_sigma is not None and (not math.isfinite(args.inference_sigma) or args.inference_sigma <= 0):
        parser.error('--inference-sigma must be a finite positive number')
    if args.inference_sigma_ratio is not None and (
            not math.isfinite(args.inference_sigma_ratio) or args.inference_sigma_ratio <= 0):
        parser.error('--inference-sigma-ratio must be a finite positive number')
    if args.visualize_clouds and 'all' in args.visualize_clouds and len(args.visualize_clouds) != 1:
        parser.error('Use --visualize-clouds all by itself, or list cloud names')
    return args


def checked_names(data_root, dataset, resolution, noise, expected_clouds,
                  clean_root_override=None, noisy_dir_override=None, mesh_root_override=None):
    resolution_dir = f'{resolution}_poisson'
    noisy_label = f'{dataset}_{resolution_dir}_{noise:g}'
    clean_root = clean_root_override or data_root / dataset / 'pointclouds' / 'test'
    noisy_dir = noisy_dir_override or data_root / 'examples' / 'pointclouds' / 'test' / noisy_label
    mesh_root = mesh_root_override or data_root / dataset / 'meshes'
    clean_dir = clean_root / resolution_dir
    mesh_dir = mesh_root / 'test'
    for path in (clean_dir, noisy_dir, mesh_dir):
        if not path.is_dir():
            raise FileNotFoundError(f'Missing {dataset} directory: {path}')
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
    return noisy_label, clean_root, noisy_dir, mesh_root, sorted(clean)


def profile_model_flops(model, sample, sigma, patch_size, patch_batch, patches):
    """Count profiler-supported model ops for each patch-batch size, then scale by T.

    PyTorch counts matmul and 2D convolution FLOPs, not all model/custom CUDA
    operations or FPS/KNN/fusion. Run this once, outside latency measurements.
    """
    full_batches, remainder = divmod(patches, patch_batch)
    batch_sizes = ([patch_batch] if full_batches else []) + ([remainder] if remainder else [])
    per_batch = {}
    patch = sample[:patch_size].unsqueeze(0)
    for batch_size in batch_sizes:
        points = patch.repeat(batch_size, 1, 1)
        noise_std = torch.full((batch_size,), sigma, dtype=points.dtype, device=points.device)
        with torch.no_grad(), torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True, with_flops=True) as profiler:
            output = model(points, None, 'val', '', noise_std=noise_std)
            torch.cuda.synchronize(points.device)
        per_batch[batch_size] = sum(int(event.flops or 0) for event in profiler.key_averages())
        del output, points, noise_std
    counted = full_batches * per_batch.get(patch_batch, 0) + per_batch.get(remainder, 0)
    if counted <= 0:
        raise RuntimeError('PyTorch profiler counted no FLOPs; rerun without --profile-flops and report this limitation')
    return counted, per_batch


def save_visualization_clouds(output_dir, cloud, steps, noisy_world, clean_world, denoised_world):
    """Save full-resolution world-coordinate clouds outside the timed region."""
    cloud_dir = output_dir / 'visualization' / cloud
    cloud_dir.mkdir(parents=True, exist_ok=True)
    for name, points in (
            ('noisy.xyz', noisy_world), ('clean.xyz', clean_world),
            (f'T{steps:02d}.xyz', denoised_world)):
        path = cloud_dir / name
        if name.startswith('T') or not path.exists():
            np.savetxt(path, points.detach().cpu().numpy(), fmt='%.8f')


def write_visualization_overviews(output_dir, clouds, steps, raw_path):
    """Render the same camera and point sample for every T; retain full .xyz for zooming."""
    import matplotlib.pyplot as plt

    with raw_path.open(newline='', encoding='utf-8') as handle:
        rows = {(row['cloud'], int(row['steps'])): row for row in csv.DictReader(handle)}
    for cloud in clouds:
        cloud_dir = output_dir / 'visualization' / cloud
        panels = [('Noisy', cloud_dir / 'noisy.xyz'), ('Clean', cloud_dir / 'clean.xyz')]
        panels += [(f'T={t}', cloud_dir / f'T{t:02d}.xyz') for t in steps]
        missing = [str(path) for _, path in panels if not path.is_file()]
        if missing:
            raise RuntimeError(f'Missing visualization files for {cloud}: {missing}')
        points = [np.loadtxt(path, dtype=np.float32) for _, path in panels]
        if any(pcl.ndim != 2 or pcl.shape[1] != 3 or len(pcl) == 0 or not np.isfinite(pcl).all()
               for pcl in points):
            raise ValueError(f'Invalid visualization point cloud for {cloud}')
        bounds = np.concatenate(points, axis=0)
        mid = (bounds.max(axis=0) + bounds.min(axis=0)) / 2
        half_span = max(float((bounds.max(axis=0) - bounds.min(axis=0)).max()) * 0.55, 1e-6)
        ncols = 3
        nrows = math.ceil(len(panels) / ncols)
        fig = plt.figure(figsize=(4.5 * ncols, 4.3 * nrows))
        for i, ((label, _), pcl) in enumerate(zip(panels, points), start=1):
            ax = fig.add_subplot(nrows, ncols, i, projection='3d')
            sample = pcl[np.linspace(0, len(pcl) - 1, min(len(pcl), 4000), dtype=int)]
            ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=0.25, c='#243b53', rasterized=True)
            ax.set(xlim=(mid[0] - half_span, mid[0] + half_span),
                   ylim=(mid[1] - half_span, mid[1] + half_span),
                   zlim=(mid[2] - half_span, mid[2] + half_span))
            if hasattr(ax, 'set_box_aspect'):
                ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=20, azim=-60)
            ax.set_axis_off()
            if label.startswith('T='):
                row = rows[(cloud, int(label[2:]))]
                label += (f'\nCD={float(row["cd_x1e4"]):.3f}  P2M={float(row["p2m_x1e4"]):.3f}'
                          f'\np2f={float(row["point_to_face_x1e4"]):.3f}  '
                          f'f2p={float(row["face_to_point_x1e4"]):.3f}')
            ax.set_title(label, fontsize=9)
        fig.suptitle(f'{cloud} | same camera and axes | CD/P2M x1e4', fontsize=12)
        fig.tight_layout()
        fig.savefig(cloud_dir / 'overview.png', dpi=180)
        plt.close(fig)


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
            sor_latency = [float(row['sor_latency_s']) for row in group]
            inference_plus_sor = [float(row['inference_plus_sor_latency_s']) for row in group]
            n_points = [int(row['n_points']) for row in group]
            total_s = sum(latency)
            total_with_sor_s = sum(inference_plus_sor)
            writer.writerow({
                'dataset': group[0]['dataset'],
                'resolution': group[0]['resolution'],
                'noise_family': group[0]['noise_family'],
                'noise': group[0]['noise'],
                'inference_sigma': group[0]['inference_sigma'],
                'sor': group[0]['sor'],
                'steps': t,
                'step_size': group[0]['step_size'],
                'decay': group[0]['decay'],
                'seed_ratio': group[0]['seed_ratio'],
                'n_clouds': n_clouds,
                'n_points': group[0]['n_points'],
                'mean_output_points': statistics.mean(int(row['output_points']) for row in group),
                'patches_per_cloud': group[0]['patches'],
                'model_forwards_per_cloud': group[0]['model_forwards'],
                'profiled_matmul_conv_gflops_per_cloud': group[0]['profiled_matmul_conv_gflops'],
                'mean_latency_s': statistics.mean(latency),
                'median_latency_s': statistics.median(latency),
                'std_latency_s': statistics.stdev(latency) if n_clouds > 1 else 0.0,
                'p95_latency_s': float(np.percentile(latency, 95)),
                'mean_sor_latency_s': statistics.mean(sor_latency),
                'mean_inference_plus_sor_latency_s': statistics.mean(inference_plus_sor),
                'clouds_per_s': n_clouds / total_s,
                'points_per_s': sum(n_points) / total_s,
                'clouds_per_s_including_sor': n_clouds / total_with_sor_s,
                'points_per_s_including_sor': sum(n_points) / total_with_sor_s,
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
    clean_root_override = (launch_dir / args.clean_root).resolve() if args.clean_root else None
    noisy_dir_override = (launch_dir / args.noisy_dir).resolve() if args.noisy_dir else None
    mesh_root_override = (launch_dir / args.mesh_root).resolve() if args.mesh_root else None
    output_dir = (launch_dir / args.output_dir).resolve()
    inference_sigma = float(
        args.inference_sigma if args.inference_sigma is not None else
        args.noise * args.inference_sigma_ratio if args.inference_sigma_ratio is not None else
        args.noise)
    if not args.resume and ((output_dir / 'run_info.json').exists() or (output_dir / 'per_cloud.csv').exists()):
        raise FileExistsError(f'{output_dir} already contains a run; choose a new directory or use --resume')
    for path in (config_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the model and GPU memory measurement')

    # Config _base_ paths and the project CUDA extensions resolve from the repository root.
    os.chdir(REPO_ROOT)
    config = cfg_from_yaml_file(str(config_path))
    if bool(getattr(config.surface_projection, 'enable', False)):
        raise ValueError('Disable surface_projection for a comparable iteration sweep')
    langevin = config.langevin
    step_size = float(args.step_size if args.step_size is not None else langevin.step_size)
    decay = float(args.decay if args.decay is not None else langevin.decay)
    seed_ratio = float(args.seed_ratio)
    patch_batch = int(args.patch_batch if args.patch_batch is not None else config.test_patch_batch)
    patch_size = int(config.npoints)
    fuse_tau = float(config.fuse_tau_ratio)
    if not math.isfinite(step_size) or step_size <= 0 or not math.isfinite(decay) or not (0 < decay <= 1) or patch_batch < 1:
        raise ValueError('Invalid step size, decay, or patch batch')

    noisy_label, clean_root, noisy_dir, mesh_root, cloud_names = checked_names(
        data_root, args.dataset, args.resolution, args.noise, args.expected_clouds,
        clean_root_override, noisy_dir_override, mesh_root_override)
    n_clouds = len(cloud_names)
    if args.visualize_clouds == ['all']:
        visualize_clouds = cloud_names
    else:
        visualize_clouds = args.visualize_clouds or []
        unknown = sorted(set(visualize_clouds) - set(cloud_names))
        if unknown:
            raise ValueError(f'Unknown --visualize-clouds names: {unknown}')
    test_config = config.dataset.test._base_
    test_config.ROOT = str(data_root)
    test_config.TEST_RESOLUTION = f'{args.resolution}_poisson'
    test_config.TEST_NOISY_DIR = noisy_label
    test_config.TEST_NOISE = inference_sigma
    test_config.TEST_NOISY_PATH = str(noisy_dir)
    test_config.TEST_CLEAN_PATH = str(clean_root) if clean_root_override or args.dataset == 'PCNet' else None
    test_config.TEST_MESH_ROOT = str(mesh_root)
    test_config.NUM_WORKERS = 0
    p2m_loss._MESH_ROOT = str(mesh_root)

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
    patches = max(1, int(seed_ratio * args.resolution / patch_size))
    profiled_flops_per_step = None
    profiled_flops_by_batch = None
    if args.profile_flops:
        first_noisy = next(iter(loader))[0][0].to(device)
        profiled_flops_per_step, profiled_flops_by_batch = profile_model_flops(
            model, first_noisy, inference_sigma, patch_size, patch_batch, patches)
        del first_noisy
        torch.cuda.synchronize(device)
    checkpoint_stat = checkpoint_path.stat()
    metadata = {
        'config': str(config_path), 'checkpoint': str(checkpoint_path),
        'checkpoint_bytes': checkpoint_stat.st_size,
        'checkpoint_mtime_ns': checkpoint_stat.st_mtime_ns,
        'data_root': str(data_root), 'dataset': args.dataset,
        'clean_root': str(clean_root), 'noisy_dir': str(noisy_dir), 'mesh_root': str(mesh_root),
        'resolution': args.resolution, 'noise_family': args.noise_family,
        'noise': args.noise, 'inference_sigma': inference_sigma,
        'sor': args.sor, 'visualize_clouds': visualize_clouds,
        'steps': args.steps, 'step_size': step_size,
        'decay': decay, 'patch_batch': patch_batch, 'patch_size': patch_size,
        'seed_ratio': seed_ratio, 'fuse_tau_ratio': fuse_tau, 'warmup': args.warmup,
        'metric_fields': ['cd_x1e4', 'point_to_face_x1e4', 'face_to_point_x1e4', 'p2m_x1e4'],
        'profile_flops': args.profile_flops,
        'profiled_matmul_conv_flops_per_step': profiled_flops_per_step,
        'profiled_matmul_conv_flops_by_patch_batch': (
            {str(k): v for k, v in profiled_flops_by_batch.items()}
            if profiled_flops_by_batch is not None else None),
        'flops_scope': 'PyTorch profiler-counted model matmul/2D-conv only; excludes unsupported ops and FPS/KNN/fusion',
        'seed': args.seed, 'postprocessing': 'SOR nb=20/std=2.0' if args.sor == 'on' else 'none',
        'timed_region': 'FPS + KNN tiling + all iterative model steps + patch fusion',
        'sor_timing': 'separate wall time after core inference; excludes CD/P2M and file output',
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

    print(f'GPU: {metadata["gpu"]}; dataset: {args.dataset}; noise: {args.noise_family}/{args.noise:g}; '
          f'SOR: {args.sor}; clouds: {n_clouds}; steps: {args.steps}')
    if args.profile_flops:
        print('FLOPs are profiler-counted model matmul/2D-conv estimates, not full-pipeline FLOPs.')
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
                warm_sigma = torch.tensor([inference_sigma], dtype=torch.float32, device=device)
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
                    sigma_gpu = torch.tensor([inference_sigma], dtype=torch.float32, device=device)
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

                    sor_latency_s = 0.0
                    if args.sor == 'on':
                        sor_start = time.perf_counter()
                        denoised_eval = sor_filter(denoised)
                        sor_latency_s = time.perf_counter() - sor_start
                    else:
                        denoised_eval = denoised

                    # Match test() metrics, but keep mesh loading and CUDA metric kernels untimed.
                    center_gpu = centers[0].to(device)
                    scale_gpu = scales[0].to(device)
                    clean_world = clean.to(device) * scale_gpu + center_gpu
                    denoised_world = denoised_eval.to(device).unsqueeze(0) * scale_gpu + center_gpu
                    p2m_components = p2m_loss.compute_p2m_components(denoised_world[0], cloud, 'test')
                    _, clean_center, clean_scale = normalize_unit_sphere(clean_world)
                    cd = cd_metric((denoised_world - clean_center) / clean_scale,
                                   (clean_world - clean_center) / clean_scale) * 1e4

                    n_points = int(noisy_gpu.shape[0])
                    patches = max(1, int(seed_ratio * n_points / patch_size))
                    row = {
                        'dataset': args.dataset, 'resolution': args.resolution,
                        'noise_family': args.noise_family,
                        'noise': args.noise, 'inference_sigma': inference_sigma,
                        'sor': args.sor,
                        'steps': steps, 'step_size': step_size, 'decay': decay,
                        'seed_ratio': seed_ratio,
                        'cloud': cloud, 'n_points': n_points,
                        'output_points': int(denoised_world.shape[1]), 'patches': patches,
                        'model_forwards': math.ceil(patches / patch_batch) * steps,
                        'profiled_matmul_conv_gflops': (
                            profiled_flops_per_step * steps / 1e9
                            if profiled_flops_per_step is not None else ''),
                        'latency_s': latency_s, 'sor_latency_s': sor_latency_s,
                        'inference_plus_sor_latency_s': latency_s + sor_latency_s,
                        'peak_allocated_gib': peak_gib,
                        'cd_x1e4': float(cd.item()),
                        'point_to_face_x1e4': float(p2m_components['point_to_face'].item() * 1e4),
                        'face_to_point_x1e4': float(p2m_components['face_to_point'].item() * 1e4),
                        'p2m_x1e4': float(p2m_components['p2m'].item() * 1e4),
                    }
                    if cloud in visualize_clouds:
                        noisy_world = noisy_gpu.unsqueeze(0) * scale_gpu + center_gpu
                        save_visualization_clouds(
                            output_dir, cloud, steps, noisy_world[0], clean_world[0], denoised_world[0])
                        del noisy_world
                    writer.writerow(row)
                    handle.flush()
                    done.add((steps, cloud))
                    print(f'{args.resolution} {args.noise:g} T={steps:2d} {cloud}: '
                          f'{latency_s:.3f}s CD={row["cd_x1e4"]:.3f} '
                          f'point→face={row["point_to_face_x1e4"]:.3f} '
                          f'face→point={row["face_to_point_x1e4"]:.3f} '
                          f'P2M={row["p2m_x1e4"]:.3f} peak={peak_gib:.2f} GiB '
                          f'SOR={sor_latency_s:.3f}s', flush=True)
                    del (noisy_gpu, sigma_gpu, denoised, denoised_eval, center_gpu, scale_gpu,
                         clean_world, denoised_world, clean_center, clean_scale, cd, p2m_components)

    write_summary(raw_path, summary_path, args.steps, n_clouds)
    if visualize_clouds:
        write_visualization_overviews(output_dir, visualize_clouds, args.steps, raw_path)
    print(f'Complete: {summary_path}')


if __name__ == '__main__':
    main()
