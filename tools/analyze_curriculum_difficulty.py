"""只读的四区间难度分析：Teacher forcing，绝不执行 Student 连续 rollout 或训练。"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'tools'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from runner_distill import TEACHER_NODES, TEACHER_DECAY, TEACHER_ETA, _sigma_batch, freeze_teacher

METRICS = ('D_move', 'D_remain', 'E_imit', 'R_relative')
DATA_FIELDS = ('dataset', 'resolution', 'noise_type', 'noise_level', 'noisy_path', 'sigma0')


def test_data_metadata(dataset, config):
    """Record the same resolved noisy directory used by test_dataloader()."""
    sigma0 = float(config.TEST_NOISE)
    if not math.isfinite(sigma0) or sigma0 <= 0:
        raise ValueError('TEST_NOISE must be finite and positive')
    noisy_path = dataset.test_noisy_path or os.path.join(
        dataset.root, 'examples', 'pointclouds', 'test', dataset.test_noisy_dir)
    return dict(dataset=str(config.DATASET), resolution=str(dataset.test_resolution),
                noise_type='Gaussian', noise_level=sigma0,
                noisy_path=str(Path(noisy_path).resolve()), sigma0=sigma0)


def capture_teacher_cloud(teacher, noisy, sigma0, patch_options, denoise_fn=None):
    """Run the original whole-cloud Teacher, retaining its actual patch states."""
    import torch
    if denoise_fn is None:
        from tools.runner_finetune import patch_based_denoise
        denoise_fn = patch_based_denoise
    freeze_teacher(teacher)
    device = next(teacher.parameters()).device
    with torch.no_grad():
        _, trajectory = denoise_fn(
            teacher, noisy.detach().to(device), sigma0, **patch_options,
            num_steps=16, step_size=TEACHER_ETA, decay=TEACHER_DECAY,
            return_trajectory=True, raise_on_memory_pressure=True)
    for key, value in trajectory.items():
        if torch.is_tensor(value):
            trajectory[key] = value.detach().cpu()
            if not torch.isfinite(trajectory[key]).all():
                raise FloatingPointError(f'Nonfinite Teacher trajectory: {key}')
    return trajectory


def fuse_cloud_patches(patches, trajectory):
    """Same weighted fusion/uncovered-point fallback as baseline trajectory readout."""
    import torch
    noisy = trajectory['global_states'][0]
    indices = trajectory['patch_idx'].reshape(-1)
    weights = trajectory['fuse_weights'].reshape(-1, 1)
    accum = torch.zeros_like(noisy)
    wsum = torch.zeros_like(noisy[:, :1])
    accum.index_add_(0, indices, patches.reshape(-1, 3) * weights)
    wsum.index_add_(0, indices, weights)
    fused = accum / wsum.clamp_min(1e-8)
    uncovered = (wsum < 1e-8).squeeze(-1)
    fused[uncovered] = noisy[uncovered]
    return fused


def analyze_cloud_stages(student, trajectory, clean, sigma0, patch_batch=1,
                         denominator_eps=1e-12):
    """Teacher-forced patch forwards, then metrics over all N fused cloud points.

    Baseline evolves fixed outer patches independently. Fused Teacher states are
    readouts, so do not re-extract patches from them or feed them into the Teacher.
    """
    import torch
    states = trajectory['global_states']
    patch_states = trajectory['patch_states']
    indices = trajectory['patch_idx']
    clean = clean.detach().cpu()
    if (states.shape != (17, *clean.shape) or clean.ndim != 2 or
            clean.shape[-1] != 3 or patch_batch < 1 or indices.shape[0] < 1):
        raise ValueError('Expected paired whole-cloud [N,3] and a 17-state Teacher trajectory')
    stage_batches = []
    for offset in range(0, indices.shape[0], patch_batch):
        end = offset + patch_batch
        # Clean uses exactly the noisy patch's original point indices.
        _, outputs = analyze_stages(student, patch_states[:, offset:end],
                                   clean[indices[offset:end]], sigma0, denominator_eps)
        stage_batches.append(outputs)
    patch_predictions = torch.cat(stage_batches, dim=1)
    predictions = torch.stack([fuse_cloud_patches(p, trajectory) for p in patch_predictions])
    rows = []
    for stage, (start, end) in enumerate(zip(TEACHER_NODES[:-1], TEACHER_NODES[1:])):
        move = (states[start] - states[end]).square().sum(-1).mean()
        remain = (states[start] - clean).square().sum(-1).mean()
        imitation = (predictions[stage] - states[end]).square().sum(-1).mean()
        values = torch.stack([move, remain, imitation, move / (remain + denominator_eps)])
        if not torch.isfinite(values).all():
            raise FloatingPointError('Nonfinite whole-cloud difficulty metrics')
        rows.append(dict(stage=stage, teacher_start=start, teacher_target=end,
                         sigma_start=float(sigma0) * TEACHER_DECAY ** start,
                         **dict(zip(METRICS, values.tolist()))))
    return rows, predictions, patch_predictions


def capture_full_teacher(teacher, noisy, sigma0, patch_batch=1):
    """同一个已采好的测试 patch，保存 T0..T16；公式与原 Teacher patch 内迭代一致。"""
    import torch
    if noisy.ndim != 3 or noisy.shape[-1] != 3 or noisy.shape[0] < 1 or patch_batch < 1:
        raise ValueError('需要 noisy [B,N,3] 和正数 patch_batch')
    device = next(teacher.parameters()).device
    sigmas = _sigma_batch(sigma0, noisy.shape[0], device, noisy.dtype)
    freeze_teacher(teacher)
    all_batches = []
    with torch.no_grad():
        for offset in range(0, noisy.shape[0], patch_batch):
            x = noisy[offset:offset + patch_batch].detach().to(device)
            sigma = sigmas[offset:offset + patch_batch].clone()
            states = [x.detach().cpu().clone()]
            for _ in range(16):
                out = teacher(x, None, 'val', '', noise_std=sigma)
                eps = (out - x) / sigma[:, None, None]
                x = x + TEACHER_ETA * sigma[:, None, None] * eps
                sigma = sigma * TEACHER_DECAY
                states.append(x.detach().cpu().clone())
            all_batches.append(torch.stack(states))
    trajectory = torch.cat(all_batches, dim=1).detach()
    if not torch.isfinite(trajectory).all():
        raise FloatingPointError('Teacher trajectory 非有限')
    return trajectory


def analyze_stages(student, teacher_states, clean, sigma0, denominator_eps=1e-12):
    """每个 stage 从独立 Teacher 起点出发，返回逐样本原始指标和四个预测状态。"""
    import torch
    if (teacher_states.ndim != 4 or teacher_states.shape[0] != 17 or
            tuple(clean.shape) != tuple(teacher_states.shape[1:]) or clean.shape[-1] != 3):
        raise ValueError('需要 Teacher [17,B,N,3] 和逐点对应的 clean [B,N,3]')
    if not math.isfinite(denominator_eps) or denominator_eps <= 0:
        raise ValueError('denominator_eps 必须为有限正数')
    device = next(student.parameters()).device
    freeze_teacher(student)  # 分析阶段同样完全冻结 Student，包括 BatchNorm buffers。
    sigma_batch = _sigma_batch(sigma0, clean.shape[0], device, clean.dtype)
    rows, predictions = [], []
    with torch.no_grad():
        clean = clean.detach().to(device)
        for stage, (start, end) in enumerate(zip(TEACHER_NODES[:-1], TEACHER_NODES[1:])):
            # 禁止使用上一 stage 的 prediction；每次都直接索引 Teacher 的真实状态。
            x = teacher_states[start].detach().to(device)
            target = teacher_states[end].detach().to(device)
            sigma = sigma_batch * TEACHER_DECAY ** start
            prediction = student(x, None, 'val', '', noise_std=sigma)
            if prediction.shape != target.shape:
                raise ValueError('Student 输出 shape 与 Teacher target 不一致')
            move = (x - target).square().sum(-1).mean(-1)
            remain = (x - clean).square().sum(-1).mean(-1)
            imitation = (prediction - target).square().sum(-1).mean(-1)
            relative = move / (remain + denominator_eps)
            values = torch.stack([move, remain, imitation, relative], dim=-1)
            if not torch.isfinite(values).all():
                raise FloatingPointError('difficulty 原始指标非有限')
            for index, vector in enumerate(values.cpu().tolist()):
                rows.append(dict(batch_sample=index, stage=stage, teacher_start=start,
                                 teacher_target=end, sigma_start=float(sigma[index]),
                                 **dict(zip(METRICS, vector))))
            predictions.append(prediction.detach().cpu().clone())
    return rows, torch.stack(predictions)


def write_stage_summary(output, rows):
    with (output / 'stage_summary.csv').open('w', newline='', encoding='utf-8') as handle:
        fields = ['stage', 'teacher_start', 'teacher_target', 'samples'] + [f'mean_{m}' for m in METRICS]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for stage in range(4):
            selected = [row for row in rows if row['stage'] == stage]
            if selected:
                writer.writerow(dict(stage=stage, teacher_start=TEACHER_NODES[stage],
                                     teacher_target=TEACHER_NODES[stage + 1], samples=len(selected),
                                     **{f'mean_{m}': sum(row[m] for row in selected) / len(selected)
                                        for m in METRICS}))


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='cfgs/PointGPT-L/distill_16to4.yaml')
    parser.add_argument('--teacher_ckpt', required=True)
    parser.add_argument('--student_ckpt', required=True, help='第一阶段已训练的 best Student；不会重新训练')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_samples', type=int, default=0, help='完整测试点云数上限；0=全部测试点云')
    parser.add_argument('--save_trajectories', type=int, default=1, help='前 N 个样本保存完整 T0..T16；0=不落盘')
    parser.add_argument('--denominator_eps', type=float, default=1e-12)
    return parser


def main():
    args = _build_parser().parse_args()
    if args.max_samples < 0 or args.save_trajectories < 0:
        raise ValueError('样本数限制必须非负')
    if not math.isfinite(args.denominator_eps) or args.denominator_eps <= 0:
        raise ValueError('denominator_eps 必须为有限正数')
    config_path = Path(args.config).expanduser().resolve()
    teacher_path = Path(args.teacher_ckpt).expanduser().resolve()
    student_path = Path(args.student_ckpt).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    for path in (config_path, teacher_path, student_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'分析目录非空: {output}')
    os.chdir(ROOT)
    import yaml
    raw = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(raw.get('cpu_threads', 8))
    import numpy as np
    import torch
    from utils.config import cfg_from_yaml_file
    from runner_distill import schedule, _patch_options
    from datasets.ScoreDenoiseDataset import ScoreDenoise
    config = cfg_from_yaml_file(str(config_path))
    if dict(config.distillation) != schedule():
        raise ValueError('分析只支持第一阶段固定 16→4 日程')
    if not torch.cuda.is_available():
        raise RuntimeError('真实分析需要原 PointGPT CUDA 环境、数据和两个 checkpoint')
    torch.cuda.set_device(args.device)
    torch.set_num_threads(int(config.cpu_threads))
    torch.cuda.set_per_process_memory_fraction(float(config.gpu_mem_fraction), args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    from tools import builder
    device = torch.device(f'cuda:{args.device}')
    teacher = builder.model_builder(config.model).to(device)
    builder.load_model(teacher, str(teacher_path))
    student = builder.model_builder(config.model).to(device)
    builder.load_model(student, str(student_path))
    freeze_teacher(teacher)
    freeze_teacher(student)
    dataset_config = config.dataset._base_
    dataset = ScoreDenoise(argparse.Namespace(distributed=False, local_rank=0), dataset_config)
    metadata = test_data_metadata(dataset, dataset_config)
    _, loader = dataset.test_dataloader()
    patch_options = _patch_options(config, config.test_patch_batch)
    available = len(loader.dataset)
    expected = min(args.max_samples, available) if args.max_samples else available
    if expected == 0:
        raise ValueError('Test dataset is empty')
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(analysis='independent_teacher_forced_stages', dataset_split='test', **metadata,
                    noise_level_units='fraction (0.01 = 1%); precomputed Gaussian noisy files',
                    patch_sampling='baseline patch_based_denoise FPS/KNN and weighted full-cloud fusion',
                    patch_options=patch_options, metric_scope='all N points after fusion; equal weight per cloud',
                    teacher_state='fixed outer patch trajectories; global states are fused readouts only',
                    student_state='independent Teacher patch starts; same indices and weights for fusion',
                    postprocessing='none for trajectory distances; normal rollout CD/P2M remains separate',
                    teacher_checkpoint=str(teacher_path), student_checkpoint=str(student_path),
                    distance='mean over corresponding points of squared L2; normalized coordinates; no x1e4',
                    ratio='D_move / (D_remain + denominator_eps); per sample, then aggregate',
                    denominator_eps=args.denominator_eps, schedule=schedule(), expected_samples=expected,
                    processed_samples=0, complete=False, config=config, arguments=vars(args))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    fields = ['sample_id', 'name', *DATA_FIELDS, 'num_points', 'num_patches', 'uncovered_points',
              'stage', 'teacher_start', 'teacher_target', 'sigma_start', *METRICS]
    all_rows, count = [], 0
    with (output / 'per_sample_stage.csv').open('w', newline='', encoding='utf-8') as handle, \
            (output / 'analysis.log').open('w', encoding='utf-8') as log:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        message = '[test-data] ' + json.dumps(metadata, ensure_ascii=False)
        print(message, flush=True)
        log.write(message + '\n')
        log.flush()
        for noisy, clean, _, centers, scales, names in loader:
            if count >= expected:
                break
            if (noisy.ndim != 3 or noisy.shape[0] != 1 or noisy.shape[-1] != 3 or
                    noisy.shape != clean.shape):
                raise ValueError('Test data must be paired whole clouds [1,N,3]')
            if not torch.isfinite(noisy).all() or not torch.isfinite(clean).all():
                raise ValueError('Test noisy/clean points must be finite')
            trajectory = capture_teacher_cloud(teacher, noisy[0], metadata['sigma0'], patch_options)
            rows, predictions, patch_predictions = analyze_cloud_stages(
                student, trajectory, clean[0], metadata['sigma0'],
                int(config.test_patch_batch), args.denominator_eps)
            cloud_info = dict(num_points=noisy.shape[1], num_patches=trajectory['patch_idx'].shape[0],
                              uncovered_points=int((trajectory['coverage_count'] == 0).sum()))
            for row in rows:
                row.update(sample_id=count, name=str(names[0]), **metadata, **cloud_info)
                writer.writerow(row)
                all_rows.append(row)
            if count < args.save_trajectories:
                folder = output / 'trajectories'
                folder.mkdir(exist_ok=True)
                np.savez_compressed(folder / f'sample_{count:06d}.npz',
                                    teacher_states=trajectory['global_states'].numpy(),
                                    teacher_patch_states=trajectory['patch_states'].numpy(),
                                    student_stage_outputs=predictions.numpy(),
                                    student_stage_patch_outputs=patch_predictions.numpy(),
                                    clean=clean[0].numpy(), name=str(names[0]), **metadata,
                                    patch_indices=trajectory['patch_idx'].numpy(),
                                    fuse_weights=trajectory['fuse_weights'].numpy(),
                                    coverage_count=trajectory['coverage_count'].numpy(),
                                    teacher_nodes=np.asarray(TEACHER_NODES),
                                    center=torch.as_tensor(centers[0]).numpy(),
                                    scale=torch.as_tensor(scales[0]).numpy())
            count += 1
            handle.flush()
            write_stage_summary(output, all_rows)
            message = (f'[difficulty] {count}/{expected} clouds; {names[0]}; '
                       f'{cloud_info}; 4 independent Teacher-forced stages; whole-cloud metrics')
            print(message, flush=True)
            log.write(message + '\n')
            log.flush()
            manifest.update(processed_samples=count, complete=count == expected)
            (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
