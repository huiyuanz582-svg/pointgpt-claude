"""固定 16 -> 4 trajectory distillation；从仓库根目录运行，--help 不加载 CUDA。"""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TEACHER_NODES = (0, 4, 8, 12, 16)
TEACHER_ETA = 0.3
TEACHER_DECAY = 0.95


def schedule():
    return dict(teacher_nodes=list(TEACHER_NODES), teacher_steps=16,
                teacher_eta=TEACHER_ETA, teacher_decay=TEACHER_DECAY,
                student_steps=4, student_eta=1.0,
                student_decay=TEACHER_DECAY ** 4)


def freeze_teacher(teacher):
    teacher.eval()
    teacher.requires_grad_(False)
    for parameter in teacher.parameters():
        parameter.grad = None
    return teacher


def _sigma_batch(sigma0, batch_size, device, dtype):
    import torch
    sigma = torch.as_tensor(sigma0, device=device, dtype=dtype).detach().reshape(-1)
    if sigma.numel() == 1:
        sigma = sigma.expand(batch_size)
    if sigma.numel() != batch_size or not torch.isfinite(sigma).all() or (sigma <= 0).any():
        raise ValueError('sigma0 必须为有限正数或长度为 batch size 的正数向量')
    return sigma


def capture_teacher(teacher, noisy, sigma0, patch_batch=1):
    """在 baseline 已采出的同一 patch 上迭代，不再次外层 FPS/KNN、重排或融合。"""
    import torch
    if noisy.ndim != 3 or noisy.shape[-1] != 3 or noisy.shape[0] < 1 or patch_batch < 1:
        raise ValueError('noisy 必须为非空 [B,N,3]，patch_batch 必须为正数')
    device = next(teacher.parameters()).device
    sigmas = _sigma_batch(sigma0, noisy.shape[0], device, noisy.dtype)
    freeze_teacher(teacher)
    batches = []
    with torch.no_grad():
        for offset in range(0, noisy.shape[0], patch_batch):
            x = noisy[offset:offset + patch_batch].detach().to(device)
            sigma = sigmas[offset:offset + patch_batch].clone()
            states = [x.detach().cpu().clone()]
            for step in range(16):
                # 与 runner_finetune.patch_based_denoise 的 patch 内更新逐式一致。
                out = teacher(x, None, 'val', '', noise_std=sigma)
                eps = (out - x) / sigma[:, None, None]
                x = x + TEACHER_ETA * sigma[:, None, None] * eps
                sigma = sigma * TEACHER_DECAY
                if step + 1 in TEACHER_NODES:
                    states.append(x.detach().cpu().clone())
            batches.append(torch.stack(states))
    nodes = torch.cat(batches, dim=1).detach()
    if not torch.isfinite(nodes).all():
        raise FloatingPointError('Teacher trajectory 包含非有限坐标')
    return nodes


def backward_stages(student, nodes, sigma0, loss_scale=1.0):
    """四个 teacher-forced stage，逐 stage backward，仅累计 Student 梯度。"""
    import torch
    if nodes.ndim != 4 or nodes.shape[0] != 5 or nodes.shape[-1] != 3:
        raise ValueError('需要五个 Teacher 节点，形状为 [5, B, N, 3]')
    device = next(student.parameters()).device
    sigmas = _sigma_batch(sigma0, nodes.shape[1], device, nodes.dtype)
    losses = []
    for stage, start in enumerate(TEACHER_NODES[:-1]):
        x = nodes[stage].detach().to(device)
        target = nodes[stage + 1].detach().to(device)
        sigma = sigmas * TEACHER_DECAY ** start
        # type='val' 只选择坐标输出分支，不会关闭 autograd，也不计算 clean loss。
        # backbone 内部仍预测 epsilon；返回值就是 eta_student=1 的一步状态。
        next_state = student(x, None, 'val', '', noise_std=sigma)
        loss = (next_state - target).square().sum(dim=-1).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'stage {stage} trajectory loss 非有限')
        if not loss.requires_grad:
            raise RuntimeError('Student trajectory loss 没有梯度')
        # micro-batch 按样本数加权，保持完整 DataLoader batch 的原 L_traj 均值。
        (loss * loss_scale / 4).backward()
        losses.append(float(loss.detach()))
    return losses


def check_gradients(teacher, student):
    """运行时检查冻结与梯度；不要求未参与 forward 的分类参数有梯度。"""
    import torch
    if teacher.training or any(p.requires_grad or p.grad is not None
                               for p in teacher.parameters()):
        raise RuntimeError('Teacher 未完全冻结')
    grads = [p.grad for p in student.parameters() if p.grad is not None]
    if not grads or any(not torch.isfinite(g).all() for g in grads):
        raise RuntimeError('Student 梯度缺失或非有限')
    return any(bool(torch.count_nonzero(g)) for g in grads)


def infer_student(student, noisy, sigma0, patch_options, denoise_fn=None,
                  return_trajectory=False):
    """真实连续四步：sigma 指数 0,4,8,12；每步使用上一步 Student 的 x。"""
    import torch
    if denoise_fn is None:
        from tools.runner_finetune import patch_based_denoise
        denoise_fn = patch_based_denoise
    student.eval()
    with torch.no_grad():
        return denoise_fn(
            student, noisy, sigma0, **patch_options,
            num_steps=4, step_size=1.0, decay=TEACHER_DECAY ** 4,
            return_trajectory=return_trajectory, raise_on_memory_pressure=True)


def _patch_options(config, batch):
    return dict(patch_size=int(config.inference_patch_size),
                seed_ratio=int(config.seed_ratio), patch_batch=int(batch),
                fuse_tau_ratio=float(config.fuse_tau_ratio))


def _train_loader(config):
    """直接复用原 train_dataloader，包括 oversample、shuffle、drop_last 和 collate。"""
    from datasets.ScoreDenoiseDataset import ScoreDenoise
    cfg = copy.deepcopy(config.dataset._base_)
    # 与 main.py / builder.dataset_builder 的 baseline batch size 下发方式一致。
    cfg.TRAIN_BATCH_SIZE = int(config.total_bs)
    data_module = ScoreDenoise(argparse.Namespace(distributed=False), cfg)
    _, loader = data_module.train_dataloader()
    if not len(loader):
        raise ValueError('原训练 DataLoader 没有完整 batch')
    return loader


def _save_checkpoint(path, student, optimizer, epoch, teacher_path, config):
    import torch
    temporary = path.with_suffix('.tmp')
    torch.save(dict(
        base_model=student.state_dict(), optimizer=optimizer.state_dict(),
        epoch=epoch, distillation=schedule(), teacher_checkpoint=str(teacher_path),
        model_config=dict(config.model)), temporary)
    temporary.replace(path)


def train(args, config, builder, device, checkpoint_path, output):
    import numpy as np
    import torch
    teacher = builder.model_builder(config.model).to(device)
    # 严格加载已微调的去噪权重，绝不调用会重置输出头的 fine-tune 初始化流程。
    builder.load_model(teacher, str(checkpoint_path))
    student = copy.deepcopy(teacher).to(device)
    student.requires_grad_(True)
    freeze_teacher(teacher)
    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=float(config.learning_rate),
                                 weight_decay=float(config.weight_decay))
    loader = _train_loader(config)
    batch_size = int(config.student_patch_batch)
    epochs = args.epochs or int(config.epochs)
    gradient_checked = False
    print(f'[sampling] dataset_patches={len(loader.dataset)} '
          f'effective_patches={len(loader) * loader.batch_size} '
          f'batch_size={loader.batch_size} drop_last={loader.drop_last}', flush=True)
    with (output / 'train.jsonl').open('w', encoding='utf-8') as log:
        for epoch in range(1, epochs + 1):
            totals = np.zeros(4, dtype=np.float64)
            seen = 0
            for batch_index, (noisy, clean, sigmas, _centers, _scales, _names) in enumerate(loader):
                if args.max_patch_batches and batch_index >= args.max_patch_batches:
                    break
                if args.max_shapes and seen >= args.max_shapes:
                    break
                if noisy.shape != clean.shape or sigmas is None:
                    raise ValueError('baseline patch 必须 noisy/clean 对齐且提供每个样本的 sigma')
                if args.max_shapes:
                    noisy = noisy[:args.max_shapes - seen]
                    sigmas = sigmas[:args.max_shapes - seen]
                nodes = capture_teacher(teacher, noisy, sigmas, int(config.teacher_patch_batch))
                count = noisy.shape[0]
                optimizer.zero_grad(set_to_none=True)
                for offset in range(0, count, batch_size):
                    batch_nodes = nodes[:, offset:offset + batch_size]
                    size = batch_nodes.shape[1]
                    losses = backward_stages(student, batch_nodes, sigmas[offset:offset + batch_size],
                                             loss_scale=size / count)
                    totals += np.asarray(losses) * size
                nonzero = check_gradients(teacher, student)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    student.parameters(), float(config.grad_norm_clip), error_if_nonfinite=True)
                if not gradient_checked:
                    print(f'[grad-check] Teacher eval/frozen/grad=None; '
                          f'Student finite gradients, nonzero={nonzero}, '
                          f'norm={float(grad_norm):.6g}', flush=True)
                    gradient_checked = True
                optimizer.step()
                seen += count
                del nodes
                print(f'[train] epoch={epoch} batch={batch_index + 1}/{len(loader)} '
                      f'patches_seen={seen}', flush=True)
            if not seen:
                raise RuntimeError('没有执行任何 Student 更新')
            means = (totals / seen).tolist()
            record = dict(epoch=epoch, patches=seen, stage_losses=means,
                          loss_traj=sum(means) / 4)
            log.write(json.dumps(record) + '\n')
            log.flush()
            _save_checkpoint(output / 'ckpt-last.pth', student, optimizer, epoch,
                             checkpoint_path, config)
            print(json.dumps(record), flush=True)


def test(args, config, builder, device, checkpoint_path, output):
    import numpy as np
    import torch
    from easydict import EasyDict
    student = builder.model_builder(config.model).to(device)
    builder.load_model(student, str(checkpoint_path))
    dataset_config = EasyDict(_base_=config.dataset._base_,
                              others=EasyDict(subset='test', bs=1))
    _, loader = builder.dataset_builder(
        argparse.Namespace(distributed=False, local_rank=0), dataset_config)
    options = _patch_options(config, config.test_patch_batch)
    for index, (noisy, _clean, noise_std, centers, scales, names) in enumerate(loader):
        if args.max_shapes and index >= args.max_shapes:
            break
        sigma0 = (float(noise_std.reshape(-1)[0]) if noise_std is not None
                  else float(config.dataset._base_.get('TEST_NOISE', 0.01)))
        result = infer_student(student, noisy[0].to(device), sigma0, options,
                               return_trajectory=args.save_trajectory)
        if args.save_trajectory:
            prediction, trajectory = result
        else:
            prediction = result
        center = torch.as_tensor(centers[0]).cpu().numpy()
        scale = torch.as_tensor(scales[0]).cpu().numpy()
        name = Path(str(names[0])).name
        world = prediction.detach().cpu().numpy() * scale + center
        np.savetxt(output / f'{name}.xyz', world, fmt='%.8f')
        if args.save_trajectory:
            np.savez_compressed(
                output / f'{name}_trajectory.npz',
                global_states=trajectory['global_states'].numpy(),
                patch_states=trajectory['patch_states'].numpy(),
                patch_idx=trajectory['patch_idx'].numpy(),
                fuse_weights=trajectory['fuse_weights'].numpy(),
                sigma_before=trajectory['sigma_before'].numpy(),
                center=center, scale=scale)
        print(f'[test] {name}: Student 连续 4 步，sigma0={sigma0}; raw XYZ 已保存', flush=True)


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='cfgs/PointGPT-L/distill_16to4.yaml')
    parser.add_argument('--mode', choices=['train', 'test'], required=True)
    parser.add_argument('--teacher_ckpt', help='训练必填：最佳 PointGPT-L 去噪 checkpoint')
    parser.add_argument('--student_ckpt', help='测试必填：蒸馏后的 Student checkpoint')
    parser.add_argument('--output_dir', required=True, help='新的输出目录，拒绝覆盖已有文件')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--max_shapes', type=int, default=0, help='冒烟：训练 patch 样本数/测试整云数；0=全部')
    parser.add_argument('--max_patch_batches', type=int, default=0, help='训练 DataLoader batch 数上限；0=全部')
    parser.add_argument('--save_trajectory', action='store_true', help='测试保存 S0..S4 的 NPZ')
    return parser


def main():
    args = _build_parser().parse_args()
    checkpoint_arg = args.teacher_ckpt if args.mode == 'train' else args.student_ckpt
    if not checkpoint_arg:
        raise ValueError('train 必须指定 --teacher_ckpt；test 必须指定 --student_ckpt')
    if args.max_shapes < 0 or args.max_patch_batches < 0 or (args.epochs is not None and args.epochs < 1):
        raise ValueError('epochs 必须为正数，冒烟限制必须非负')
    checkpoint_path = Path(checkpoint_arg).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'checkpoint 不存在: {checkpoint_path}')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'输出目录非空，请换一个目录: {output}')
    os.chdir(REPO_ROOT)
    import yaml
    with config_path.open(encoding='utf-8') as handle:
        raw = yaml.safe_load(handle)
    cpu_threads = int(raw.get('cpu_threads', 8))
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(cpu_threads)
    import numpy as np
    import torch
    from utils.config import cfg_from_yaml_file
    config = cfg_from_yaml_file(str(config_path))
    if dict(config.distillation) != schedule():
        raise ValueError('第一版仅支持固定节点 [0,4,8,12,16] 和指定 Teacher/Student 日程')
    if any(int(config[key]) < 1 for key in (
            'teacher_patch_batch', 'student_patch_batch', 'test_patch_batch',
            'epochs', 'inference_patch_size', 'seed_ratio', 'total_bs')):
        raise ValueError('步数、patch 大小及 batch 必须为正整数')
    if not torch.cuda.is_available():
        raise RuntimeError('真实 PointGPT 训练/测试需要原 CUDA 环境及其扩展')
    torch.cuda.set_device(args.device)
    torch.set_num_threads(cpu_threads)
    torch.cuda.set_per_process_memory_fraction(float(config.gpu_mem_fraction), args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    from tools import builder
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'manifest.json').open('w', encoding='utf-8') as handle:
        json.dump(dict(mode=args.mode, checkpoint=str(checkpoint_path),
                       config=config, arguments=vars(args), schedule=schedule()),
                  handle, indent=2, ensure_ascii=False)
    device = torch.device(f'cuda:{args.device}')
    if args.mode == 'train':
        train(args, config, builder, device, checkpoint_path, output)
    else:
        test(args, config, builder, device, checkpoint_path, output)


if __name__ == '__main__':
    main()
