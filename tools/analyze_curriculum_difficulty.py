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


def capture_full_teacher(teacher, noisy, sigma0, patch_batch=1):
    """同一个已采好的训练 patch，保存 T0..T16；公式与原 Teacher patch 内迭代一致。"""
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
    parser.add_argument('--max_samples', type=int, default=0, help='分析训练 patch 数；0=原 DataLoader 一个 epoch')
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
    from runner_distill import _train_loader, schedule
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
    loader = _train_loader(config)
    available = len(loader) * loader.batch_size
    expected = min(args.max_samples, available) if args.max_samples else available
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(analysis='independent_teacher_forced_stages', dataset_split='baseline_train_patches',
                    teacher_checkpoint=str(teacher_path), student_checkpoint=str(student_path),
                    distance='mean over corresponding points of squared L2; normalized coordinates; no x1e4',
                    ratio='D_move / (D_remain + denominator_eps); per sample, then aggregate',
                    denominator_eps=args.denominator_eps, schedule=schedule(), expected_samples=expected,
                    processed_samples=0, complete=False, config=config, arguments=vars(args))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    fields = ['sample_id', 'name', 'sigma0', 'stage', 'teacher_start', 'teacher_target', 'sigma_start', *METRICS]
    all_rows, count = [], 0
    with (output / 'per_sample_stage.csv').open('w', newline='', encoding='utf-8') as handle, \
            (output / 'analysis.log').open('w', encoding='utf-8') as log:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for noisy, clean, sigmas, centers, scales, names in loader:
            if count >= expected:
                break
            size = min(len(noisy), expected - count)
            full = capture_full_teacher(teacher, noisy[:size], sigmas[:size], int(config.teacher_patch_batch))
            for offset in range(0, size, int(config.student_patch_batch)):
                end = min(size, offset + int(config.student_patch_batch))
                rows, predictions = analyze_stages(student, full[:, offset:end], clean[offset:end],
                                                   sigmas[offset:end], args.denominator_eps)
                for row in rows:
                    index = offset + row.pop('batch_sample')
                    row.update(sample_id=count + index, name=str(names[index]), sigma0=float(sigmas[index]))
                    writer.writerow(row)
                    all_rows.append(row)
                for index in range(offset, end):
                    if count + index < args.save_trajectories:
                        folder = output / 'trajectories'
                        folder.mkdir(exist_ok=True)
                        np.savez_compressed(folder / f'sample_{count + index:06d}.npz',
                                            teacher_states=full[:, index].numpy(),
                                            student_stage_outputs=predictions[:, index - offset].numpy(),
                                            clean=clean[index].numpy(), sigma0=float(sigmas[index]),
                                            teacher_nodes=np.asarray(TEACHER_NODES),
                                            center=torch.as_tensor(centers[index]).numpy(),
                                            scale=torch.as_tensor(scales[index]).numpy())
            count += size
            handle.flush()
            write_stage_summary(output, all_rows)
            message = f'[difficulty] {count}/{expected} patches; 4 independent Teacher-forced stages per patch'
            print(message, flush=True)
            log.write(message + '\n')
            log.flush()
            manifest.update(processed_samples=count, complete=count == expected)
            (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
