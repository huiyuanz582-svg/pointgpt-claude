"""固定 16 -> 4 trajectory distillation；从仓库根目录运行，--help 不加载 CUDA。"""

import argparse
import copy
import csv
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


def enable_student_condition(student):
    if hasattr(student, 'enable_step_condition'):
        student.enable_step_condition()
    return student


def forward_student_interval(student, points, sigma, start_step, target_step):
    """One explicit interval interface for fixed training, diagnostics and future targets."""
    condition = {}
    if getattr(student, 'step_condition', None) is not None:
        condition = dict(start_step=start_step, target_step=target_step)
    return student(points, None, 'val', '', noise_std=sigma, **condition)


def load_student_state(student, state):
    """Strict backbone loading; only a wholly absent new condition branch is initialized."""
    enable_student_condition(student)
    weights = {key.replace('module.', ''): value for key, value in state.items()}
    if (getattr(student, 'step_condition', None) is not None and
            not any(key.startswith('step_condition.') for key in weights)):
        weights.update({'step_condition.' + key: value
                        for key, value in student.step_condition.state_dict().items()})
    student.load_state_dict(weights, strict=True)


def load_student_checkpoint(student, checkpoint_path, builder):
    """Match the condition branch in the checkpoint; preserve legacy inference."""
    if not hasattr(student, 'enable_step_condition'):
        return builder.load_model(student, str(checkpoint_path))
    import torch
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if checkpoint.get('distillation', schedule()) != schedule():
        raise ValueError('Student checkpoint must use the fixed 16-to-4 schedule')
    weights = checkpoint.get('model')
    if weights is None:
        weights = checkpoint.get('base_model')
    if weights is None:
        raise ValueError('Student checkpoint must contain model or base_model')
    weights = {key.replace('module.', ''): value for key, value in weights.items()}
    if any(key.startswith('step_condition.') for key in weights):
        load_student_state(student, weights)
    else:
        student.load_state_dict(weights, strict=True)


def restore_student_optimizer(optimizer, state, student):
    """Keep legacy AdamW state/hyperparameters; append empty slots only for new parameters."""
    old_groups = state['param_groups']
    new_groups = optimizer.param_groups
    if len(old_groups) == len(new_groups) and any(
            len(old['params']) != len(new['params']) for old, new in zip(old_groups, new_groups)):
        condition = getattr(student, 'step_condition', None)
        condition_params = list(condition.parameters()) if condition is not None else []
        state = copy.deepcopy(state)
        next_id = max((pid for group in old_groups for pid in group['params']), default=-1) + 1
        for old, new in zip(state['param_groups'], new_groups):
            added = new['params'][len(old['params']):]
            if len(new['params']) < len(old['params']) or any(
                    not any(p is q for q in condition_params) for p in added):
                raise ValueError('Optimizer mismatch outside the new Student condition branch')
            old['params'].extend(range(next_id, next_id + len(added)))
            next_id += len(added)
    optimizer.load_state_dict(state)


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
        next_state = forward_student_interval(student, x, sigma, start, TEACHER_NODES[stage + 1])
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
    model_for_patches = student
    if getattr(student, 'step_condition', None) is not None:
        calls = 0
        def fixed_interval_forward(points, clean=None, type='val', name='', noise_std=None):
            nonlocal calls
            # The original denoiser runs all four stages inside each patch batch.
            stage = calls % 4
            calls += 1
            return forward_student_interval(student, points, noise_std,
                                            TEACHER_NODES[stage], TEACHER_NODES[stage + 1])
        model_for_patches = fixed_interval_forward
    with torch.no_grad():
        return denoise_fn(
            model_for_patches, noisy, sigma0, **patch_options,
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


def _validation_bank(config, teacher):
    """固定验证 noisy/patch/Teacher target，跨 epoch 使用同一份 CPU 缓存。"""
    import torch
    from datasets.ScoreDenoiseDataset import ScoreDenoise
    cfg = copy.deepcopy(config.dataset._base_)
    data_module = ScoreDenoise(argparse.Namespace(distributed=False), cfg)
    seed = int(getattr(config, 'validation_seed', 2024))
    patches_per_cloud = int(getattr(config, 'validation_patches_per_cloud', 4))
    if patches_per_cloud < 1:
        raise ValueError('validation_patches_per_cloud 必须为正整数')
    device = next(teacher.parameters()).device
    cuda_devices = [device.index] if device.type == 'cuda' else []
    bank, samples = [], []
    # 验证准备不能推进训练采样所使用的 RNG。
    with torch.random.fork_rng(devices=cuda_devices):
        _, loader = data_module.val_dataloader()
        for index in range(len(loader.dataset)):
            sample = loader.dataset[index]
            noisy = sample['pcl_noisy']
            sigma0 = float(sample['noise_std'])
            generator = torch.Generator().manual_seed(seed + index)
            seeds = torch.randint(len(noisy), (patches_per_cloud,), generator=generator)
            patch_size = min(int(cfg.PATCH_SIZE), len(noisy))
            indices = ((noisy[None] - noisy[seeds, None]) ** 2).sum(-1).topk(
                patch_size, dim=-1, largest=False).indices
            nodes = capture_teacher(teacher, noisy[indices], sigma0,
                                    int(config.teacher_patch_batch))
            bank.append((nodes, torch.full((patches_per_cloud,), sigma0)))
            samples.append(dict(name=sample['name'], sigma0=sigma0,
                                seed_indices=seeds.tolist(), patch_size=patch_size))
    if not bank:
        raise ValueError('验证集为空，无法选择 best checkpoint')
    metadata = dict(metric='val_loss_traj', teacher_forced=True, seed=seed,
                    patches_per_cloud=patches_per_cloud, samples=samples,
                    split='held_out_train' if data_module.val_num > 0 else 'test',
                    schedule=schedule())
    return bank, metadata


def validate_trajectory(student, bank, patch_batch):
    """只评固定验证状态对的 L_traj；不更新梯度或 BatchNorm running statistics。"""
    import torch
    if patch_batch < 1:
        raise ValueError('validation patch_batch 必须为正数')
    device = next(student.parameters()).device
    was_training = student.training
    student.eval()
    totals, count = [0.0] * 4, 0
    try:
        with torch.no_grad():
            for nodes, sigma0 in bank:
                sigmas = _sigma_batch(sigma0, nodes.shape[1], device, nodes.dtype)
                for offset in range(0, nodes.shape[1], patch_batch):
                    batch_nodes = nodes[:, offset:offset + patch_batch].detach().to(device)
                    size = batch_nodes.shape[1]
                    for stage, start in enumerate(TEACHER_NODES[:-1]):
                        sigma = sigmas[offset:offset + size] * TEACHER_DECAY ** start
                        prediction = forward_student_interval(
                            student, batch_nodes[stage], sigma, start, TEACHER_NODES[stage + 1])
                        loss = (prediction - batch_nodes[stage + 1]).square().sum(-1).mean()
                        if not torch.isfinite(loss):
                            raise FloatingPointError('验证 trajectory loss 非有限，不选择 best')
                        totals[stage] += float(loss) * size
                    count += size
    finally:
        student.train(was_training)
    if not count:
        raise ValueError('验证 patch 数为零')
    means = [value / count for value in totals]
    return dict(val_loss_traj=sum(means) / 4, val_stage_losses=means, val_patches=count)


def _rollout_validation_loader(config):
    """Reuse baseline's complete validation clouds, fixed noise and normalization."""
    from datasets.ScoreDenoiseDataset import ScoreDenoise
    cfg = copy.deepcopy(config.dataset._base_)
    data_module = ScoreDenoise(argparse.Namespace(distributed=False), cfg)
    _, loader = data_module.val_dataloader()
    mesh_split = 'train' if data_module.val_num > 0 else 'test'
    return loader, mesh_split


def validate_rollout(student, config):
    """Diagnostic whole-cloud Student rollout using the existing test pipeline.

    infer_student runs four consecutive updates inside each overlapping patch,
    then baseline patch_based_denoise fuses them. No Teacher states are involved.
    """
    import numpy as np
    import torch
    device = next(student.parameters()).device
    cuda_devices = [device.index] if device.type == 'cuda' else []
    was_training = student.training
    python_state, numpy_state = random.getstate(), np.random.get_state()
    seed = int(getattr(config, 'validation_seed', 2024))
    total_cd, total_p2m, count = 0.0, 0.0, 0
    student.eval()
    try:
        # Fixed evaluation randomness without advancing subsequent training RNGs.
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if device.type == 'cuda':
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            loader, mesh_split = _rollout_validation_loader(config)
            options = _patch_options(config, config.test_patch_batch)
            ops = baseline_metric_ops(config, device)
            for noisy, clean, sigmas, centers, scales, names in loader:
                if (noisy.ndim != 3 or noisy.shape[0] != 1 or noisy.shape[-1] != 3 or
                        noisy.shape != clean.shape or sigmas is None):
                    raise ValueError('Rollout validation requires complete paired [1,N,3] clouds and sigma')
                sigma0 = float(sigmas.reshape(-1)[0])
                prediction = infer_student(student, noisy[0].to(device), sigma0, options)
                _, metrics = evaluate_baseline_metrics(
                    prediction, clean, centers[0], scales[0], str(names[0]), config, ops,
                    mesh_split=mesh_split)
                total_cd += metrics['cd_x1e4']
                total_p2m += metrics['p2m_x1e4']
                count += 1
                print(f'[rollout-validation] {count}/{len(loader)} {names[0]}: '
                      f'CD={metrics["cd_x1e4"]:.6f} P2M={metrics["p2m_x1e4"]:.6f} (x1e4)',
                      flush=True)
    finally:
        student.train(was_training)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    if not count:
        raise ValueError('Rollout validation dataset is empty')
    cd, p2m = total_cd / count, total_p2m / count
    score = cd + 0.3 * p2m
    if not all(math.isfinite(value) for value in (cd, p2m, score)):
        raise FloatingPointError('Nonfinite rollout validation metrics')
    return dict(val_rollout_cd=cd, val_rollout_p2m=p2m,
                val_rollout_score=score, val_rollout_clouds=count)


def _save_checkpoint(path, student, optimizer, epoch, teacher_path, config, selection=None,
                     best_val_rollout_score=float('inf'), best_epoch=None):
    import torch
    temporary = path.with_suffix('.tmp')
    payload = dict(
        base_model=student.state_dict(),
        epoch=epoch, distillation=schedule(), teacher_checkpoint=str(teacher_path),
        model_config=dict(config.model), selection=selection,
        best_val_rollout_score=best_val_rollout_score, best_epoch=best_epoch)
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    torch.save(payload, temporary)
    temporary.replace(path)


def save_epoch_checkpoints(output, student, optimizer, epoch, teacher_path, config,
                           validation, best_score, best_epoch):
    # Missing/None means no rollout was evaluated this epoch. Never substitute L_traj.
    value = validation.get('val_rollout_score')
    if value is not None:
        value = float(value)
        if not math.isfinite(value):
            raise FloatingPointError('验证 rollout score 非有限，不能保存为 best')
    improved = value is not None and value < best_score
    if improved:
        best_score, best_epoch = value, epoch
    selection = dict(metric='val_rollout_score', value=value,
                     best_value=best_score if math.isfinite(best_score) else None,
                     best_epoch=best_epoch)
    # 只保留 last 和 best；last 含 optimizer，best 仅在验证指标改善时更新。
    _save_checkpoint(output / 'ckpt-last.pth', student, optimizer, epoch,
                     teacher_path, config, selection,
                     best_val_rollout_score=best_score, best_epoch=best_epoch)
    if improved:
        _save_checkpoint(output / 'ckpt-best.pth', student, None, epoch,
                         teacher_path, config, selection,
                         best_val_rollout_score=best_score, best_epoch=best_epoch)
    return best_score, best_epoch, improved


def train(args, config, builder, device, checkpoint_path, output):
    import numpy as np
    import torch
    teacher = builder.model_builder(config.model).to(device)
    # 严格加载已微调的去噪权重，绝不调用会重置输出头的 fine-tune 初始化流程。
    builder.load_model(teacher, str(checkpoint_path))
    student = copy.deepcopy(teacher).to(device)
    if getattr(config, 'student_step_condition', False):
        enable_student_condition(student)
    student.requires_grad_(True)
    freeze_teacher(teacher)
    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=float(config.learning_rate),
                                 weight_decay=float(config.weight_decay))
    start_epoch = 1
    best_score, best_epoch = float('inf'), None
    if getattr(args, 'resume', None):
        resumed = torch.load(args.resume, map_location='cpu')
        if resumed.get('distillation') != schedule() or 'optimizer' not in resumed:
            raise ValueError('--resume 需要相同日程且包含 optimizer 的 ckpt-last.pth')
        if getattr(config, 'student_step_condition', False):
            load_student_state(student, resumed['base_model'])
            restore_student_optimizer(optimizer, resumed['optimizer'], student)
        else:
            student.load_state_dict(resumed['base_model'], strict=True)
            optimizer.load_state_dict(resumed['optimizer'])
        start_epoch = int(resumed['epoch']) + 1
        best_score = float(resumed.get('best_val_rollout_score', float('inf')))
        best_epoch = resumed.get('best_epoch', None)
        del resumed
    loader = _train_loader(config)
    batch_size = int(config.student_patch_batch)
    rollout_interval = int(getattr(config, 'rollout_val_interval', 5))
    if rollout_interval < 1:
        raise ValueError('rollout_val_interval 必须为正整数')
    epochs = args.epochs or int(config.epochs)
    if start_epoch > epochs:
        raise ValueError('epochs 是目标总 epoch 数，必须大于已完成的 resume epoch')
    validation_bank, validation_metadata = _validation_bank(config, teacher)
    with (output / 'validation_manifest.json').open('w', encoding='utf-8') as handle:
        json.dump(validation_metadata, handle, indent=2, ensure_ascii=False)
    print(f'[validation] fixed_patches={sum(nodes.shape[1] for nodes, _ in validation_bank)} '
          f'split={validation_metadata["split"]} diagnostic=val_loss_traj '
          f'best_metric=val_rollout_score', flush=True)
    gradient_checked = False
    print(f'[sampling] dataset_patches={len(loader.dataset)} '
          f'effective_patches={len(loader) * loader.batch_size} '
          f'batch_size={loader.batch_size} drop_last={loader.drop_last}', flush=True)
    with (output / 'train.jsonl').open('w', encoding='utf-8') as log:
        for epoch in range(start_epoch, epochs + 1):
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
            validation = validate_trajectory(student, validation_bank, int(config.test_patch_batch))
            if epoch % rollout_interval == 0:
                validation.update(validate_rollout(student, config))
            for key in ('val_rollout_cd', 'val_rollout_p2m', 'val_rollout_score'):
                validation.setdefault(key, None)
            best_score, best_epoch, improved = save_epoch_checkpoints(
                output, student, optimizer, epoch, checkpoint_path, config,
                validation, best_score, best_epoch)
            record = dict(epoch=epoch, patches=seen, stage_losses=means,
                          loss_traj=sum(means) / 4, **validation,
                          best_val_rollout_score=best_score if math.isfinite(best_score) else None,
                          best_epoch=best_epoch, is_best=improved)
            log.write(json.dumps(record) + '\n')
            log.flush()
            print(json.dumps(record), flush=True)


def baseline_metric_ops(config, device):
    """直接复用第一篇 baseline 测试所调用的度量与后处理实现。"""
    from extensions.chamfer_dist import ChamferDistanceL2
    from tools.runner_finetune import normalize_unit_sphere, sor_filter, local_surface_projection
    import utils.p2m_loss as p2m
    mesh_root = config.dataset._base_.get('TEST_MESH_ROOT', None)
    if mesh_root:
        p2m._MESH_ROOT = mesh_root
    return dict(cd=ChamferDistanceL2().to(device), p2m=p2m.compute_p2m,
                normalize=normalize_unit_sphere, sor=sor_filter,
                project=local_surface_projection, mesh_root=str(p2m._MESH_ROOT))


def evaluate_baseline_metrics(prediction, clean, center, scale, name, config, ops, mesh_split='test'):
    """与 runner_finetune.test 相同：后处理 -> 世界坐标 -> 各自度量的归一化。"""
    import torch
    device = prediction.device
    sp = getattr(config, 'surface_projection', None) or {}
    with torch.no_grad():
        filtered = ops['sor'](prediction) if getattr(config, 'sor_enable', True) else prediction.cpu()
        if sp.get('enable', False):
            filtered = ops['project'](filtered, k=int(sp.get('k', 16)),
                                      num_iters=int(sp.get('num_iters', 1)), blend=float(sp.get('blend', 1.0)))
        if filtered.ndim != 2 or filtered.shape[0] == 0 or filtered.shape[-1] != 3:
            raise ValueError('后处理产生空点云或非法 shape，无法计算 CD/P2M')
        center = torch.as_tensor(center, device=device)
        scale = torch.as_tensor(scale, device=device)
        world = filtered.unsqueeze(0).to(device) * scale + center
        clean_world = clean.to(device) * scale + center
        # P2M 必须调用测试用双向实现；不能换成训练用单向 compute_p2m_train。
        p2m_value = ops['p2m'](world[0], name, mesh_split) * 1e4
        _, metric_center, metric_scale = ops['normalize'](clean_world)
        cd_value = ops['cd']((world - metric_center) / metric_scale,
                             (clean_world - metric_center) / metric_scale) * 1e4
        metrics = dict(cd_x1e4=float(cd_value), p2m_x1e4=float(p2m_value))
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f'{name}: CD/P2M 非有限')
        return world[0].detach().cpu(), metrics


def write_test_summary(output, rows, protocol, expected_shapes, dataset_shapes):
    """按整云等权平均，单样本冒烟不会标记成全量评估。"""
    summary = dict(protocol=protocol, processed_shapes=len(rows), expected_shapes=expected_shapes,
                   dataset_shapes=dataset_shapes, complete=len(rows) == expected_shapes,
                   full_dataset=expected_shapes == dataset_shapes,
                   mean_cd_x1e4=sum(r['cd_x1e4'] for r in rows) / len(rows),
                   mean_p2m_x1e4=sum(r['p2m_x1e4'] for r in rows) / len(rows))
    temporary = output / 'test_summary.tmp'
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(output / 'test_summary.json')
    return summary


def test(args, config, builder, device, checkpoint_path, output):
    import numpy as np
    import torch
    from easydict import EasyDict
    student = builder.model_builder(config.model).to(device)
    load_student_checkpoint(student, checkpoint_path, builder)
    dataset_config = EasyDict(_base_=config.dataset._base_,
                              others=EasyDict(subset='test', bs=1))
    _, loader = builder.dataset_builder(
        argparse.Namespace(distributed=False, local_rank=0), dataset_config)
    options = _patch_options(config, config.test_patch_batch)
    ops = baseline_metric_ops(config, device)
    protocol = dict(mode='student_continuous_4_step_rollout', reference='tools/runner_finetune.py::test',
                    cd='ChamferDistanceL2; clean-cloud unit sphere; x1e4',
                    p2m='compute_p2m(test); bidirectional mesh unit sphere; x1e4',
                    aggregation='equal weight per cloud', vote_times=1, mesh_root=ops['mesh_root'],
                    sor_enable=bool(getattr(config, 'sor_enable', True)),
                    surface_projection=dict(getattr(config, 'surface_projection', None) or {}))
    expected = min(args.max_shapes, len(loader)) if args.max_shapes else len(loader)
    if expected == 0:
        raise ValueError('测试数据集为空')
    rows = []
    with (output / 'test_metrics.csv').open('w', newline='', encoding='utf-8') as csv_file, \
            (output / 'test.log').open('w', encoding='utf-8') as log:
        writer = csv.DictWriter(csv_file, fieldnames=['name', 'sigma0', 'input_points', 'output_points',
                                                     'cd_x1e4', 'p2m_x1e4'])
        writer.writeheader()
        log.write(json.dumps(protocol, ensure_ascii=False) + '\n')
        log.flush()
        for index, (noisy, clean, noise_std, centers, scales, names) in enumerate(loader):
            if index >= expected:
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
            world, metrics = evaluate_baseline_metrics(prediction, clean, centers[0], scales[0],
                                                       str(names[0]), config, ops)
            np.savetxt(output / f'{name}.xyz', world.numpy(), fmt='%.8f')
            raw_world = prediction.detach().cpu().numpy() * scale + center
            np.savetxt(output / f'{name}_raw.xyz', raw_world, fmt='%.8f')
            if args.save_trajectory:
                np.savez_compressed(
                    output / f'{name}_trajectory.npz',
                    global_states=trajectory['global_states'].numpy(),
                    patch_states=trajectory['patch_states'].numpy(),
                    patch_idx=trajectory['patch_idx'].numpy(),
                    fuse_weights=trajectory['fuse_weights'].numpy(),
                    sigma_before=trajectory['sigma_before'].numpy(),
                    center=center, scale=scale, postprocessed=False)
            row = dict(name=name, sigma0=sigma0, input_points=noisy.shape[1],
                       output_points=world.shape[0], **metrics)
            rows.append(row)
            writer.writerow(row)
            csv_file.flush()
            summary = write_test_summary(output, rows, protocol, expected, len(loader))
            message = (f'[test] {index + 1}/{expected} {name}: CD={metrics["cd_x1e4"]:.6f} '
                       f'P2M={metrics["p2m_x1e4"]:.6f} (x1e4); '
                       f'mean CD={summary["mean_cd_x1e4"]:.6f} P2M={summary["mean_p2m_x1e4"]:.6f}')
            print(message, flush=True)
            log.write(message + '\n')
            log.flush()


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='cfgs/PointGPT-L/distill_16to4.yaml')
    parser.add_argument('--mode', choices=['train', 'test'], required=True)
    parser.add_argument('--teacher_ckpt', help='训练必填：最佳 PointGPT-L 去噪 checkpoint')
    parser.add_argument('--student_ckpt', help='测试必填：蒸馏后的 Student checkpoint')
    parser.add_argument('--resume', help='训练续跑已有 ckpt-last.pth；仍需 teacher_ckpt，使用新输出目录')
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
    if args.resume:
        if args.mode != 'train':
            raise ValueError('--resume 仅用于训练')
        args.resume = str(Path(args.resume).expanduser().resolve())
        if not Path(args.resume).is_file():
            raise FileNotFoundError(args.resume)
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
