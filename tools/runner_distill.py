"""16 -> 4 trajectory distillation with fixed or epoch-level PCD curriculum."""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TEACHER_NODES = (0, 4, 8, 12, 16)
NONUNIFORM_NODES = (0, 7, 10, 13, 16)
TEACHER_ETA = 0.3
TEACHER_DECAY = 0.95
PCD_EPS = 1e-12


def _teacher_nodes(nodes):
    nodes = tuple(nodes)
    if (len(nodes) != 5 or any(type(t) is not int for t in nodes) or
            nodes[0] != 0 or nodes[-1] != 16 or any(t >= u for t, u in zip(nodes[:-1], nodes[1:]))):
        raise ValueError('Teacher nodes require integer 0 < t1 < t2 < t3 < 16, with endpoints 0 and 16')
    return nodes


def dynamic_pcd_options(config):
    """Only dynamic mode consumes these options; all curriculum settings come from YAML."""
    if getattr(config, 'curriculum_mode', 'fixed') != 'dynamic_pcd':
        return None
    options = dict(getattr(config, 'dynamic_pcd', {}))
    required = ('initial_nodes', 'target', 'lambda_balance', 'update_every_epochs',
                'calibration_patches', 'calibration_seed', 'interval_patch_batch')
    if any(key not in options for key in required):
        raise ValueError('dynamic_pcd requires ' + ', '.join(required))
    options['initial_nodes'] = list(_teacher_nodes(options['initial_nodes']))
    for key in ('target', 'lambda_balance'):
        options[key] = float(options[key])
        if not math.isfinite(options[key]) or options[key] < 0:
            raise ValueError(f'dynamic_pcd.{key} must be finite and nonnegative')
    for key in ('update_every_epochs', 'calibration_patches', 'interval_patch_batch', 'calibration_seed'):
        value = options[key]
        if type(value) is not int or value < (0 if key == 'calibration_seed' else 1):
            raise ValueError(f'Invalid dynamic_pcd.{key}')
    if options['calibration_seed'] >= 2 ** 32:
        raise ValueError('calibration_seed must fit the NumPy seed range')
    if getattr(config, 'pcd_dynamic', {}).get('shadow_enabled', False):
        raise ValueError('dynamic_pcd uses epoch-level search; disable per-batch shadow search')
    return options


def configured_teacher_nodes(config):
    specification = getattr(config, 'distillation', None)
    mode = getattr(config, 'curriculum_mode', 'fixed')
    dynamic = dynamic_pcd_options(config)
    nodes = _teacher_nodes(dynamic['initial_nodes'] if dynamic is not None else
                           specification['teacher_nodes'] if specification is not None else TEACHER_NODES)
    expected = {'fixed': TEACHER_NODES, 'fixed_nonuniform': NONUNIFORM_NODES}
    if dynamic is None and (mode not in expected or nodes != expected[mode]):
        raise ValueError('curriculum_mode and distillation.teacher_nodes must match')
    if specification is not None and dict(specification) != schedule(nodes):
        raise ValueError('Teacher/Student schedule differs from the supported 16-to-4 protocol')
    return nodes


def schedule(teacher_nodes=TEACHER_NODES):
    nodes = _teacher_nodes(teacher_nodes)
    # Preserve legacy equal-interval checkpoint metadata exactly.
    decay = TEACHER_DECAY ** 4 if nodes == TEACHER_NODES else [
        TEACHER_DECAY ** (u - t) for t, u in zip(nodes[:-1], nodes[1:])]
    return dict(teacher_nodes=list(nodes), teacher_steps=16,
                teacher_eta=TEACHER_ETA, teacher_decay=TEACHER_DECAY,
                student_steps=4, student_eta=1.0,
                student_decay=decay)


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


def load_student_checkpoint(student, checkpoint_path, builder, expected_nodes=None):
    """Inference keeps old unconditioned checkpoints on their original forward path."""
    if not hasattr(student, 'enable_step_condition'):
        return builder.load_model(student, str(checkpoint_path))
    import torch
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    saved_schedule = checkpoint.get('distillation', schedule())
    saved_nodes = _teacher_nodes(saved_schedule['teacher_nodes'])
    if saved_schedule != schedule(saved_nodes):
        raise ValueError('Unsupported checkpoint distillation schedule')
    if checkpoint.get('curriculum_mode') == 'dynamic_pcd':
        if (list(saved_nodes) != checkpoint.get('nodes_used_this_epoch') or
                list(saved_nodes) != checkpoint.get('current_teacher_nodes')):
            raise ValueError('Dynamic checkpoint schedule must match nodes_used_this_epoch')
    if expected_nodes is not None and saved_nodes != _teacher_nodes(expected_nodes):
        raise ValueError('Student checkpoint schedule does not match the test YAML')
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
    student.distillation_teacher_nodes = saved_nodes


def restore_dynamic_curriculum(checkpoint, config):
    """A saved model is evaluated with used nodes; resumed training starts with next nodes."""
    options = dynamic_pcd_options(config)
    if checkpoint.get('curriculum_mode') != 'dynamic_pcd' or checkpoint.get('dynamic_pcd_config') != options:
        raise ValueError('Dynamic resume requires the same dynamic_pcd configuration')
    used = _teacher_nodes(checkpoint['nodes_used_this_epoch'])
    next_nodes = _teacher_nodes(checkpoint['next_teacher_nodes'])
    if (checkpoint.get('distillation') != schedule(used) or
            checkpoint.get('current_teacher_nodes') != list(used)):
        raise ValueError('Inconsistent dynamic checkpoint schedule')
    for key, value in (('pcd_target', options['target']), ('lambda_balance', options['lambda_balance']),
                       ('dynamic_update_every_epochs', options['update_every_epochs'])):
        if checkpoint.get(key) != value:
            raise ValueError(f'Inconsistent dynamic checkpoint {key}')
    history = copy.deepcopy(checkpoint['curriculum_history'])
    expected_epochs = list(range(options['update_every_epochs'], int(checkpoint['epoch']) + 1,
                                 options['update_every_epochs']))
    if [row['epoch'] for row in history] != expected_epochs:
        raise ValueError('Incomplete curriculum history')
    previous = _teacher_nodes(options['initial_nodes'])
    expected_used = previous
    for row in history:
        if _teacher_nodes(row['old_nodes']) != previous:
            raise ValueError('Discontinuous curriculum history')
        previous = _teacher_nodes(row['new_nodes'])
        if row['epoch'] < checkpoint['epoch']:
            expected_used = previous
    if used != expected_used:
        raise ValueError('nodes_used_this_epoch disagrees with curriculum history')
    if previous != next_nodes:
        raise ValueError('next_teacher_nodes disagrees with curriculum history')
    return next_nodes, history, checkpoint['calibration_metadata']


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


def capture_teacher(teacher, noisy, sigma0, patch_batch=1, return_full_trajectory=False,
                    teacher_nodes=TEACHER_NODES):
    """同一 baseline patch 的原 16 步；可额外缓存所有状态，更新公式和模型调用不变。"""
    import torch
    teacher_nodes = _teacher_nodes(teacher_nodes)
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
                if return_full_trajectory or step + 1 in teacher_nodes:
                    states.append(x.detach().cpu().clone())
            batches.append(torch.stack(states))
    nodes = torch.cat(batches, dim=1).detach()
    if not torch.isfinite(nodes).all():
        raise FloatingPointError('Teacher trajectory 包含非有限坐标')
    return nodes


def evaluate_candidate_pcd(T_start, T_target, student_pred):
    """Evaluate supplied states for any 0 <= t < u <= 16, independently of fixed nodes.

    Inputs share point order and shape [N,3] or [B,N,3]. Return detached scalar
    metrics for one cloud/patch, or [B] metrics for a batch (mean over points).
    The caller supplies T_t, T_u and the Student's one-step state from T_t;
    this function does not run a model, select targets, or change a trajectory.
    """
    import torch
    with torch.no_grad():
        points = (T_start, T_target, student_pred)
        if not all(torch.is_tensor(p) and p.is_floating_point() for p in points):
            raise ValueError('Candidate states must be floating-point tensors')
        if (T_start.ndim not in (2, 3) or T_start.shape[-1] != 3 or T_start.numel() == 0 or
                any(p.shape != T_start.shape or p.device != T_start.device for p in points)):
            raise ValueError('Candidate states must have matching [N,3] or [B,N,3] shapes and devices')
        if not all(torch.isfinite(p).all() for p in points):
            raise ValueError('Candidate states must be finite')
        # Keep 1e-12 representable even when a supplied prediction uses float16.
        dtype = torch.float64 if any(p.dtype == torch.float64 for p in points) else torch.float32
        start, target, prediction = (p.detach().to(dtype=dtype) for p in points)
        move = (target - start).square().sum(-1).mean(-1)
        imitation = (prediction - target).square().sum(-1).mean(-1)
        pcd = imitation / (move + PCD_EPS)
        return dict(D_move=move.detach(), E_imit=imitation.detach(), PCD=pcd.detach())


def search_dynamic_teacher_target(teacher_states, student_pred, start_step, remaining_stages,
                                  pcd_threshold, teacher_end=16, min_gap=1):
    """Search one patch's cached [17,N,3] Teacher states, without backward.

    Evaluate every legal u, then minimize abs(PCD - pcd_threshold). Equal
    distances select the smaller u. The final stage always selects teacher_end.
    student_pred may be a tensor or a no-grad callback(u) for a conditioned Student.
    All candidate metrics are returned as detached Python values for diagnostics.
    """
    import torch
    if (not isinstance(start_step, int) or not isinstance(remaining_stages, int) or
            teacher_end != 16 or min_gap != 1 or not 0 <= remaining_stages < 4):
        raise ValueError('Shadow search requires integer steps, min_gap=1, teacher_end=16 and 0..3 remaining stages')
    pcd_target = float(pcd_threshold)
    if not math.isfinite(pcd_target) or pcd_target < 0:
        raise ValueError('pcd_threshold must be finite and nonnegative')
    u_max = teacher_end - remaining_stages
    if not 0 <= start_step < u_max:
        raise ValueError('No valid target that reserves one Teacher step per remaining stage')
    if teacher_states.ndim != 3 or teacher_states.shape[0] != teacher_end + 1:
        raise ValueError('Search requires full Teacher states [17,N,3]')
    with torch.no_grad():
        start = teacher_states[start_step].detach()
        prediction = None if callable(student_pred) else student_pred.detach()
        candidates = []
        for u in range(start_step + min_gap, u_max + 1):
            if callable(student_pred):
                prediction = student_pred(u).detach()
            metrics = evaluate_candidate_pcd(start, teacher_states[u].detach(), prediction)
            values = {key: float(value) for key, value in metrics.items()}
            if not all(math.isfinite(value) for value in values.values()):
                raise FloatingPointError('Nonfinite candidate PCD metrics')
            candidates.append(dict(target_step=u, gap=u - start_step, **values,
                                   pcd_distance_to_target=abs(values['PCD'] - pcd_target)))
        forced_final = remaining_stages == 0
        selected = (candidates[-1] if forced_final else min(
            candidates, key=lambda row: (row['pcd_distance_to_target'], row['target_step'])))
        return dict(selected_target_step=selected['target_step'], selected_gap=selected['gap'],
                    selected_PCD=selected['PCD'], selected_D_move=selected['D_move'],
                    selected_E_imit=selected['E_imit'],
                    pcd_distance_to_target=selected['pcd_distance_to_target'], pcd_target=pcd_target,
                    stop_reason='forced_final_target' if forced_final else 'closest_pcd_target',
                    forced_final_target=forced_final, candidate_metrics=candidates)


def _shadow_search_config(config):
    options = dict(getattr(config, 'pcd_dynamic', None) or {})
    enabled = bool(options.get('shadow_enabled', False))
    if enabled:
        if any(options.get(key, default) != default for key, default in (
                ('min_gap', 1), ('teacher_end', 16), ('student_steps', 4))):
            raise ValueError('This shadow protocol requires min_gap=1, teacher_end=16, student_steps=4')
        threshold = float(options['threshold'])
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError('pcd_dynamic.threshold must be finite and nonnegative')
        options['threshold'] = threshold
    options['shadow_enabled'] = enabled
    return options


def shadow_search_teacher_targets(student, teacher_states, sigma0, threshold, patch_batch):
    """Four chained Teacher-start searches, isolated from the fixed training graph.

    Runs before the training update. Only selected Teacher states advance the
    shadow start, never Student outputs. Restores all module modes and RNGs.
    """
    import numpy as np
    import torch
    if (teacher_states.ndim != 4 or teacher_states.shape[0] != 17 or
            teacher_states.shape[-1] != 3 or teacher_states.shape[1] < 1 or patch_batch < 1):
        raise ValueError('Shadow search requires Teacher [17,B,N,3] and positive patch_batch')
    device = next(student.parameters()).device
    cuda_devices = [device.index] if device.type == 'cuda' else []
    modes = [(module, module.training) for module in student.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    rows = []
    try:
        student.eval()
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            states = teacher_states.detach().cpu()
            sigmas = _sigma_batch(sigma0, states.shape[1], device, states.dtype)
            for offset in range(0, states.shape[1], patch_batch):
                end = min(offset + patch_batch, states.shape[1])
                batch = states[:, offset:end]
                starts = [0] * (end - offset)
                for stage in range(4):
                    x = torch.stack([batch[t, i] for i, t in enumerate(starts)]).to(device)
                    decay = torch.tensor([TEACHER_DECAY ** t for t in starts], device=device,
                                         dtype=sigmas.dtype)
                    stage_sigmas = sigmas[offset:end] * decay
                    conditioned = getattr(student, 'step_condition', None) is not None
                    if not conditioned:
                        prediction = student(x, None, 'val', '', noise_std=stage_sigmas).detach().cpu()
                    for index, t in enumerate(starts):
                        if conditioned:
                            # Each candidate receives its own u condition; never trains on it.
                            def candidate_prediction(u):
                                return forward_student_interval(
                                    student, x[index:index + 1], stage_sigmas[index:index + 1],
                                    t, u)[0].detach().cpu()
                            candidate = candidate_prediction
                        else:
                            candidate = prediction[index]
                        result = search_dynamic_teacher_target(
                            batch[:, index], candidate, t, 3 - stage, threshold)
                        rows.append(dict(stage=stage, batch_sample=offset + index,
                                         start_step=t, **result))
                        starts[index] = result['selected_target_step']
    finally:
        for module, mode in modes:
            module.training = mode
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    return rows


def summarize_shadow_search(rows):
    """Per-stage epoch means/histograms, with one equally weighted row per patch."""
    fields = {'mean_shadow_target_step': 'selected_target_step',
              'mean_shadow_gap': 'selected_gap', 'mean_shadow_PCD': 'selected_PCD',
              'mean_shadow_D_move': 'selected_D_move', 'mean_shadow_E_imit': 'selected_E_imit',
              'mean_PCD_distance_to_target': 'pcd_distance_to_target'}
    summary = {key: [] for key in fields}
    summary.update(shadow_target_hist=[], closest_pcd_target_count=[],
                   forced_final_target_count=[], shadow_debug_samples=[])
    for stage in range(4):
        selected = [row for row in rows if row['stage'] == stage]
        if not selected:
            raise ValueError('Cannot summarize an empty shadow stage')
        for field, key in fields.items():
            summary[field].append(sum(row[key] for row in selected) / len(selected))
        hist = {str(step): 0 for step in range(1, 17)}
        for row in selected:
            hist[str(row['selected_target_step'])] += 1
        summary['shadow_target_hist'].append(hist)
        summary['closest_pcd_target_count'].append(sum(
            row['stop_reason'] == 'closest_pcd_target' for row in selected))
        summary['forced_final_target_count'].append(sum(row['forced_final_target'] for row in selected))
        # Bounded epoch log: two patch examples per stage, each with every candidate.
        summary['shadow_debug_samples'].extend(selected[:2])
    return summary


def backward_stages(student, nodes, sigma0, loss_scale=1.0, stage_diagnostics=None,
                    teacher_nodes=TEACHER_NODES):
    """四个 teacher-forced stage，逐 stage backward，仅累计 Student 梯度。"""
    import torch
    teacher_nodes = _teacher_nodes(teacher_nodes)
    if nodes.ndim != 4 or nodes.shape[0] != 5 or nodes.shape[-1] != 3:
        raise ValueError('需要五个 Teacher 节点，形状为 [5, B, N, 3]')
    device = next(student.parameters()).device
    sigmas = _sigma_batch(sigma0, nodes.shape[1], device, nodes.dtype)
    losses = []
    for stage, start in enumerate(teacher_nodes[:-1]):
        x = nodes[stage].detach().to(device)
        target = nodes[stage + 1].detach().to(device)
        sigma = sigmas * TEACHER_DECAY ** start
        # type='val' 只选择坐标输出分支，不会关闭 autograd，也不计算 clean loss。
        # backbone 内部仍预测 epsilon；返回值就是 eta_student=1 的一步状态。
        next_state = forward_student_interval(student, x, sigma, start, teacher_nodes[stage + 1])
        loss = (next_state - target).square().sum(dim=-1).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'stage {stage} trajectory loss 非有限')
        if not loss.requires_grad:
            raise RuntimeError('Student trajectory loss 没有梯度')
        if stage_diagnostics is not None:
            # Diagnostics only: corresponding-point squared L2, reduced per patch.
            # Average individual PCD ratios, not a ratio of batch/epoch means.
            with torch.no_grad():
                move = (target - x).square().sum(-1).mean(-1)
                imitation = (next_state.detach() - target).square().sum(-1).mean(-1)
                pcd = imitation / (move + PCD_EPS)
                stage_diagnostics.append(torch.stack(
                    (move.mean(), imitation.mean(), pcd.mean())).cpu().tolist())
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
                  return_trajectory=False, teacher_nodes=None):
    """真实连续四步；使用配置区间的 condition/sigma，不读取 Teacher state。"""
    import torch
    teacher_nodes = _teacher_nodes(teacher_nodes if teacher_nodes is not None else
                                   getattr(student, 'distillation_teacher_nodes', TEACHER_NODES))
    if denoise_fn is None:
        from tools.runner_finetune import patch_based_denoise
        denoise_fn = patch_based_denoise
    student.eval()
    model_for_patches = student
    if getattr(student, 'step_condition', None) is not None or teacher_nodes != TEACHER_NODES:
        calls = 0
        def fixed_interval_forward(points, clean=None, type='val', name='', noise_std=None):
            nonlocal calls
            # Baseline loops four steps inside each patch batch; reset per batch.
            stage = calls % 4
            calls += 1
            if teacher_nodes != TEACHER_NODES:
                # The original patch loop reconstructs out via x + sigma*(out-x)/sigma
                # at eta=1. Its scalar decay cancels; only this actual Student sigma
                # sets displacement magnitude. Keep the patch/fusion implementation.
                initial_sigma = float(torch.as_tensor(sigma0).reshape(-1)[0])
                noise_std = torch.full_like(noise_std, initial_sigma * TEACHER_DECAY ** teacher_nodes[stage])
            return forward_student_interval(student, points, noise_std,
                                            teacher_nodes[stage], teacher_nodes[stage + 1])
        model_for_patches = fixed_interval_forward
    with torch.no_grad():
        result = denoise_fn(
            model_for_patches, noisy, sigma0, **patch_options,
            num_steps=4, step_size=1.0, decay=TEACHER_DECAY ** 4,
            return_trajectory=return_trajectory, raise_on_memory_pressure=True)
    if return_trajectory and teacher_nodes != TEACHER_NODES:
        initial_sigma = float(torch.as_tensor(sigma0).reshape(-1)[0])
        trajectory = result[1]
        trajectory_device = trajectory['patch_states'].device
        trajectory['sigma_before'] = torch.tensor(
            [initial_sigma * TEACHER_DECAY ** t for t in teacher_nodes[:-1]],
            dtype=torch.float64, device=trajectory_device)
        trajectory['sigma_after'] = torch.tensor(
            [initial_sigma * TEACHER_DECAY ** t for t in teacher_nodes],
            dtype=torch.float64, device=trajectory_device)
    return result


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


def curriculum_calibration_bank(config, teacher, dataset):
    """独立固定采样原训练 Dataset；不迭代训练 loader，也不推进训练 RNG。

    PairedPatchDataset 原样完成 clean 归一化、加噪、noisy KNN 同索引切片。
    固定 noisy realization 的 T0..T16 缓存在 CPU，跨 epoch 只重评 Student。
    """
    import numpy as np
    import torch
    options = dynamic_pcd_options(config)
    count, seed = options['calibration_patches'], options['calibration_seed']
    if count > len(dataset):
        raise ValueError('calibration_patches exceeds the training dataset size')
    device = next(teacher.parameters()).device
    python_state, numpy_state = random.getstate(), np.random.get_state()
    bank, samples = [], []
    digest = hashlib.sha256()
    try:
        with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []), torch.no_grad():
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if device.type == 'cuda':
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:count].tolist()
            batch_size = options['interval_patch_batch']
            for offset in range(0, count, batch_size):
                noisy_patches, sigmas = [], []
                for index in indices[offset:offset + batch_size]:
                    sample = dataset[index]
                    noisy, clean = sample['pcl_noisy'], sample['pcl_clean']
                    if (noisy.ndim != 2 or noisy.shape[-1] != 3 or noisy.shape != clean.shape or
                            not torch.isfinite(noisy).all() or not torch.isfinite(clean).all()):
                        raise ValueError('Calibration requires finite paired [N,3] training patches')
                    sigma = float(sample['noise_std'])
                    for value in (noisy, clean, torch.tensor(sigma, dtype=torch.float64)):
                        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
                    noisy_patches.append(noisy.detach().cpu().clone())
                    sigmas.append(sigma)
                    samples.append(dict(dataset_index=index, name=str(sample['name']), sigma0=sigma,
                                        patch_size=len(noisy)))
                sigmas = torch.tensor(sigmas, dtype=noisy_patches[0].dtype)
                full = capture_teacher(teacher, torch.stack(noisy_patches), sigmas,
                                       int(config.teacher_patch_batch), return_full_trajectory=True)
                bank.append((full, sigmas))
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    metadata = dict(split='train', purpose='curriculum_only', seed=seed, patches=count,
                    samples=samples, data_sha256=digest.hexdigest(), teacher_steps=16,
                    pcd_aggregation='mean_of_per_patch_ratios')
    return bank, metadata


def update_dynamic_curriculum(student, bank, config):
    """One shared global path from mean per-patch interval PCD, not a path per patch."""
    import torch
    from tools.shadow_global_search import build_interval_pcd_cache, search_global_teacher_nodes
    options = dynamic_pcd_options(config)
    totals, count, forward_calls = {}, 0, 0
    with torch.no_grad():
        for states, sigmas in bank:
            cache = build_interval_pcd_cache(student, states, sigmas, options['interval_patch_batch'])
            size = states.shape[1]
            for edge, row in cache['metrics'].items():
                if edge not in totals:
                    totals[edge] = dict.fromkeys(('D_move', 'E_imit', 'PCD'), 0.0)
                for key, values in row.items():
                    # 先按 patch 求比值，再平均；不同大小的最后一批按样本数加权。
                    totals[edge][key] += float(values.double().sum())
            count += size
            forward_calls += cache['forward_calls']
            del cache
        if count != options['calibration_patches']:
            raise ValueError('Calibration bank size differs from dynamic_pcd.calibration_patches')
        means = {edge: {key: value / count for key, value in row.items()} for edge, row in totals.items()}
        result = search_global_teacher_nodes(means, options['target'], options['lambda_balance'])
    _teacher_nodes(result['nodes'])
    return dict(result, calibration_patches=count, student_forward_calls=forward_calls,
                patch_interval_evaluations=count * len(means), pcd_aggregation='mean_of_per_patch_ratios')


def _validation_bank(config, teacher):
    """固定验证 noisy/patch/Teacher target，跨 epoch 使用同一份 CPU 缓存。"""
    import torch
    teacher_nodes = configured_teacher_nodes(config)
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
                                    int(config.teacher_patch_batch), teacher_nodes=teacher_nodes,
                                    return_full_trajectory=getattr(config, 'curriculum_mode', 'fixed') == 'dynamic_pcd')
            bank.append((nodes, torch.full((patches_per_cloud,), sigma0)))
            samples.append(dict(name=sample['name'], sigma0=sigma0,
                                seed_indices=seeds.tolist(), patch_size=patch_size))
    if not bank:
        raise ValueError('验证集为空，无法选择 best checkpoint')
    metadata = dict(metric='val_loss_traj', teacher_forced=True, seed=seed,
                    patches_per_cloud=patches_per_cloud, samples=samples,
                    split='held_out_train' if data_module.val_num > 0 else 'test',
                    schedule=schedule(teacher_nodes))
    if getattr(config, 'curriculum_mode', 'fixed') == 'dynamic_pcd':
        metadata.update(cached_teacher_steps=list(range(17)), schedule_selection='nodes_used_this_epoch')
    return bank, metadata


def validate_trajectory(student, bank, patch_batch, teacher_nodes=TEACHER_NODES):
    """只评固定验证状态对的 L_traj；不更新梯度或 BatchNorm running statistics。"""
    import torch
    teacher_nodes = _teacher_nodes(teacher_nodes)
    if patch_batch < 1:
        raise ValueError('validation patch_batch 必须为正数')
    device = next(student.parameters()).device
    was_training = student.training
    student.eval()
    totals, count = [0.0] * 4, 0
    try:
        with torch.no_grad():
            for nodes, sigma0 in bank:
                if nodes.shape[0] == 17:
                    nodes = nodes[list(teacher_nodes)]
                if nodes.shape[0] != 5:
                    raise ValueError('Validation bank requires five selected or seventeen full Teacher states')
                sigmas = _sigma_batch(sigma0, nodes.shape[1], device, nodes.dtype)
                for offset in range(0, nodes.shape[1], patch_batch):
                    batch_nodes = nodes[:, offset:offset + patch_batch].detach().to(device)
                    size = batch_nodes.shape[1]
                    for stage, start in enumerate(teacher_nodes[:-1]):
                        sigma = sigmas[offset:offset + size] * TEACHER_DECAY ** start
                        prediction = forward_student_interval(
                            student, batch_nodes[stage], sigma, start, teacher_nodes[stage + 1])
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


def validate_rollout(student, config, teacher_nodes=None):
    """Diagnostic whole-cloud Student rollout using the existing test pipeline.

    infer_student runs four consecutive updates inside each overlapping patch,
    then baseline patch_based_denoise fuses them. No Teacher states are involved.
    """
    import numpy as np
    import torch
    teacher_nodes = _teacher_nodes(teacher_nodes if teacher_nodes is not None else configured_teacher_nodes(config))
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
                prediction = infer_student(student, noisy[0].to(device), sigma0, options,
                                           teacher_nodes=teacher_nodes)
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
                     best_val_rollout_score=float('inf'), best_epoch=None, curriculum_state=None):
    import torch
    temporary = path.with_suffix('.tmp')
    if getattr(config, 'curriculum_mode', 'fixed') == 'dynamic_pcd' and curriculum_state is None:
        raise ValueError('Dynamic checkpoint requires the used/next curriculum state')
    used_nodes = (curriculum_state['nodes_used_this_epoch'] if curriculum_state is not None
                  else configured_teacher_nodes(config))
    payload = dict(
        base_model=student.state_dict(),
        epoch=epoch, distillation=schedule(used_nodes), teacher_checkpoint=str(teacher_path),
        curriculum_mode=getattr(config, 'curriculum_mode', 'fixed'),
        model_config=dict(config.model), selection=selection,
        best_val_rollout_score=best_val_rollout_score, best_epoch=best_epoch)
    if curriculum_state is not None:
        payload.update(copy.deepcopy(curriculum_state))
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    torch.save(payload, temporary)
    temporary.replace(path)


def save_epoch_checkpoints(output, student, optimizer, epoch, teacher_path, config,
                           validation, best_score, best_epoch, curriculum_state=None):
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
                     best_val_rollout_score=best_score, best_epoch=best_epoch, curriculum_state=curriculum_state)
    if improved:
        _save_checkpoint(output / 'ckpt-best.pth', student, None, epoch,
                         teacher_path, config, selection,
                         best_val_rollout_score=best_score, best_epoch=best_epoch, curriculum_state=curriculum_state)
    return best_score, best_epoch, improved


def train(args, config, builder, device, checkpoint_path, output):
    import numpy as np
    import torch
    def timestamp():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        return time.perf_counter()

    run_started = timestamp()
    epochs = args.epochs or int(config.epochs)
    stop_after_epoch = getattr(args, 'stop_after_epoch', None)
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= epochs:
        raise ValueError('stop_after_epoch must be within the unchanged total epoch plan')
    teacher_nodes = configured_teacher_nodes(config)
    dynamic = dynamic_pcd_options(config)
    curriculum_history, resumed_calibration = [], None
    teacher = builder.model_builder(config.model).to(device)
    # 严格加载已微调的去噪权重，绝不调用会重置输出头的 fine-tune 初始化流程。
    builder.load_model(teacher, str(checkpoint_path))
    student = copy.deepcopy(teacher).to(device)
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
        if 'optimizer' not in resumed:
            raise ValueError('--resume requires ckpt-last.pth with optimizer state')
        if dynamic is not None:
            teacher_nodes, curriculum_history, resumed_calibration = restore_dynamic_curriculum(resumed, config)
        elif (resumed.get('curriculum_mode', 'fixed') != getattr(config, 'curriculum_mode', 'fixed') or
              resumed.get('distillation') != schedule(teacher_nodes)):
            raise ValueError('--resume 需要相同日程且包含 optimizer 的 ckpt-last.pth')
        load_student_state(student, resumed['base_model'])
        restore_student_optimizer(optimizer, resumed['optimizer'], student)
        start_epoch = int(resumed['epoch']) + 1
        best_score = float(resumed.get('best_val_rollout_score', float('inf')))
        best_epoch = resumed.get('best_epoch', None)
        del resumed
    loader = _train_loader(config)
    batch_size = int(config.student_patch_batch)
    shadow_options = _shadow_search_config(config)
    rollout_interval = int(getattr(config, 'rollout_val_interval', 5))
    if rollout_interval < 1:
        raise ValueError('rollout_val_interval 必须为正整数')
    if start_epoch > epochs:
        raise ValueError('epochs 是目标总 epoch 数，必须大于已完成的 resume epoch')
    if stop_after_epoch is not None and start_epoch > stop_after_epoch:
        raise ValueError('stop_after_epoch must be at or after the first resumed epoch')
    validation_bank, validation_metadata = _validation_bank(config, teacher)
    with (output / 'validation_manifest.json').open('w', encoding='utf-8') as handle:
        json.dump(validation_metadata, handle, indent=2, ensure_ascii=False)
    if dynamic is not None:
        calibration_bank, calibration_metadata = curriculum_calibration_bank(config, teacher, loader.dataset)
        if resumed_calibration is not None and calibration_metadata != resumed_calibration:
            raise ValueError('Resume calibration bank differs from the checkpoint; check dataset/order/noise')
        with (output / 'calibration_manifest.json').open('w', encoding='utf-8') as handle:
            json.dump(calibration_metadata, handle, indent=2, ensure_ascii=False)
    print(f'[validation] fixed_patches={sum(nodes.shape[1] for nodes, _ in validation_bank)} '
          f'split={validation_metadata["split"]} diagnostic=val_loss_traj '
          f'best_metric=val_rollout_score', flush=True)
    gradient_checked = False
    setup_seconds = timestamp() - run_started
    print(f'[sampling] dataset_patches={len(loader.dataset)} '
          f'effective_patches={len(loader) * loader.batch_size} '
          f'batch_size={loader.batch_size} drop_last={loader.drop_last}', flush=True)
    with (output / 'train.jsonl').open('w', encoding='utf-8') as log:
        for epoch in range(start_epoch, epochs + 1):
            epoch_started = timestamp()
            # 此变量在整个 epoch 的训练和验证中保持不变；next_nodes 只在末尾切换。
            stage_gaps = [u - t for t, u in zip(teacher_nodes[:-1], teacher_nodes[1:])]
            print(f'[curriculum] epoch={epoch} teacher_nodes={list(teacher_nodes)} stage_gaps={stage_gaps}', flush=True)
            totals = np.zeros(4, dtype=np.float64)
            pcd_totals = np.zeros((4, 3), dtype=np.float64)
            shadow_rows = []
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
                if shadow_options['shadow_enabled']:
                    full_teacher = capture_teacher(teacher, noisy, sigmas, int(config.teacher_patch_batch),
                                                   return_full_trajectory=True)
                    # Only these fixed nodes enter the training loop below.
                    nodes = full_teacher[list(teacher_nodes)]
                    shadow_rows.extend(shadow_search_teacher_targets(
                        student, full_teacher, sigmas, shadow_options['threshold'], batch_size))
                    del full_teacher
                else:
                    nodes = capture_teacher(teacher, noisy, sigmas, int(config.teacher_patch_batch),
                                            teacher_nodes=teacher_nodes)
                count = noisy.shape[0]
                optimizer.zero_grad(set_to_none=True)
                for offset in range(0, count, batch_size):
                    batch_nodes = nodes[:, offset:offset + batch_size]
                    size = batch_nodes.shape[1]
                    stage_diagnostics = []
                    losses = backward_stages(student, batch_nodes, sigmas[offset:offset + batch_size],
                                             loss_scale=size / count, stage_diagnostics=stage_diagnostics,
                                             teacher_nodes=teacher_nodes)
                    totals += np.asarray(losses) * size
                    pcd_totals += np.asarray(stage_diagnostics) * size
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
            pcd_means = pcd_totals / seen
            validation_started = timestamp()
            train_seconds = validation_started - epoch_started
            validation = validate_trajectory(student, validation_bank, int(config.test_patch_batch),
                                             teacher_nodes=teacher_nodes)
            rollout_started = timestamp()
            trajectory_validation_seconds = rollout_started - validation_started
            if epoch % rollout_interval == 0:
                validation.update(validate_rollout(student, config, teacher_nodes=teacher_nodes) if dynamic is not None
                                  else validate_rollout(student, config))
            for key in ('val_rollout_cd', 'val_rollout_p2m', 'val_rollout_score'):
                validation.setdefault(key, None)
            search_started = timestamp()
            rollout_validation_seconds = search_started - rollout_started
            next_nodes, search_result, curriculum_state = teacher_nodes, None, None
            if dynamic is not None:
                if epoch % dynamic['update_every_epochs'] == 0:
                    search_result = update_dynamic_curriculum(student, calibration_bank, config)
                    next_nodes = _teacher_nodes(search_result['nodes'])
                    curriculum_history.append(dict(
                        epoch=epoch, old_nodes=list(teacher_nodes), new_nodes=list(next_nodes),
                        stage_pcd=search_result['stage_PCD'], pcd_std=search_result['pcd_std'],
                        pcd_range=search_result['pcd_max_minus_min'],
                        mean_target_error=search_result['mean_target_error'], path_score=search_result['path_score'],
                        student_forward_calls=search_result['student_forward_calls']))
                    print(f'[dynamic-pcd] epoch={epoch} next_nodes={list(next_nodes)} '
                          f'path_score={search_result["path_score"]:.6g}', flush=True)
                curriculum_state = dict(
                    current_teacher_nodes=list(teacher_nodes), nodes_used_this_epoch=list(teacher_nodes),
                    next_teacher_nodes=list(next_nodes), pcd_target=dynamic['target'],
                    lambda_balance=dynamic['lambda_balance'],
                    dynamic_update_every_epochs=dynamic['update_every_epochs'],
                    dynamic_pcd_config=dynamic, curriculum_history=curriculum_history,
                    calibration_metadata=calibration_metadata)
            checkpoint_started = timestamp()
            curriculum_search_seconds = checkpoint_started - search_started if search_result is not None else 0.0
            best_score, best_epoch, improved = save_epoch_checkpoints(
                output, student, optimizer, epoch, checkpoint_path, config,
                validation, best_score, best_epoch, curriculum_state=curriculum_state)
            saved_at = timestamp()
            record = dict(epoch=epoch, patches=seen, stage_losses=means,
                          curriculum_mode=getattr(config, 'curriculum_mode', 'fixed'),
                          teacher_nodes=list(teacher_nodes), stage_gaps=stage_gaps,
                          mean_D_move=pcd_means[:, 0].tolist(),
                          mean_E_imit=pcd_means[:, 1].tolist(),
                          mean_PCD=pcd_means[:, 2].tolist(), pcd_eps=PCD_EPS,
                          pcd_aggregation='mean_of_per_patch_ratios',
                          loss_traj=sum(means) / 4, **validation,
                          best_val_rollout_score=best_score if math.isfinite(best_score) else None,
                          best_epoch=best_epoch, is_best=improved)
            if dynamic is not None:
                record.update(nodes_used_this_epoch=list(teacher_nodes),
                              checkpoint_teacher_nodes=list(teacher_nodes),
                              train_PCD_std=float(np.std(pcd_means[:, 2], ddof=0)),
                              train_PCD_range=float(np.ptp(pcd_means[:, 2])),
                              total_planned_epochs=epochs, stop_after_epoch=stop_after_epoch,
                              train_seconds=train_seconds,
                              trajectory_validation_seconds=trajectory_validation_seconds,
                              rollout_validation_seconds=rollout_validation_seconds,
                              curriculum_search_seconds=curriculum_search_seconds,
                              checkpoint_seconds=saved_at - checkpoint_started,
                              epoch_seconds=saved_at - epoch_started,
                              setup_seconds=setup_seconds, run_elapsed_seconds=saved_at - run_started)
                record.update(teacher_nodes_used=list(teacher_nodes),
                              dynamic_update_performed=search_result is not None,
                              next_teacher_nodes=list(next_nodes),
                              search_stage_PCD=search_result['stage_PCD'] if search_result else None,
                              search_PCD_std=search_result['pcd_std'] if search_result else None,
                              search_PCD_range=search_result['pcd_max_minus_min'] if search_result else None,
                              search_mean_target_error=search_result['mean_target_error'] if search_result else None,
                              search_path_score=search_result['path_score'] if search_result else None,
                              search_student_forward_calls=search_result['student_forward_calls'] if search_result else 0)
            if shadow_options['shadow_enabled']:
                record.update(shadow_enabled=True, shadow_threshold=shadow_options['threshold'],
                              **summarize_shadow_search(shadow_rows))
            log.write(json.dumps(record) + '\n')
            log.flush()
            print(json.dumps(record), flush=True)
            teacher_nodes = next_nodes
            if stop_after_epoch is not None and epoch >= stop_after_epoch:
                print(f'[stop-after-epoch] saved epoch={epoch}; total_plan={epochs}; '
                      f'next_epoch={epoch + 1} not started', flush=True)
                break


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


def evaluate_baseline_metrics(prediction, clean, center, scale, name, config, ops, mesh_split='test',
                              keep_on_device=False):
    """原后处理/归一化/CD/P2M；keep_on_device 仅省去搬运并保留返回点云的设备。

    默认仍返回 CPU 点云。开启后，只有完全关闭后处理时才跳过输入的 CPU 搬运；
    SOR/surface projection 启用时仍走原设备与执行路径。
    """
    import torch
    device = prediction.device
    sp = getattr(config, 'surface_projection', None) or {}
    with torch.no_grad():
        if getattr(config, 'sor_enable', True):
            filtered = ops['sor'](prediction)
        elif keep_on_device and not sp.get('enable', False):
            # No CPU postprocessing is needed. Avoid a GPU -> CPU -> GPU roundtrip.
            filtered = prediction
        else:
            filtered = prediction.cpu()
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
        return (world[0].detach() if keep_on_device else world[0].detach().cpu()), metrics


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
    teacher_nodes = configured_teacher_nodes(config)
    dynamic = getattr(config, 'curriculum_mode', 'fixed') == 'dynamic_pcd'
    load_student_checkpoint(student, checkpoint_path, builder, expected_nodes=None if dynamic else teacher_nodes)
    if dynamic:
        teacher_nodes = _teacher_nodes(student.distillation_teacher_nodes)
        manifest_path = output / 'manifest.json'
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            manifest.update(schedule=schedule(teacher_nodes), schedule_source='student_checkpoint')
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    dataset_config = EasyDict(_base_=config.dataset._base_,
                              others=EasyDict(subset='test', bs=1))
    _, loader = builder.dataset_builder(
        argparse.Namespace(distributed=False, local_rank=0), dataset_config)
    options = _patch_options(config, config.test_patch_batch)
    ops = baseline_metric_ops(config, device)
    protocol = dict(mode='student_continuous_4_step_rollout', reference='tools/runner_finetune.py::test',
                    teacher_nodes=list(teacher_nodes),
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
                                   return_trajectory=args.save_trajectory, teacher_nodes=teacher_nodes)
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
    parser.add_argument('--stop_after_epoch', type=int, default=None,
                        help='Stop after saving this epoch; leave the total epoch plan and resume format unchanged')
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
    if args.stop_after_epoch is not None and (args.mode != 'train' or args.stop_after_epoch < 1):
        raise ValueError('stop_after_epoch is a positive training-only epoch boundary')
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
    teacher_nodes = configured_teacher_nodes(config)
    if any(int(config[key]) < 1 for key in (
            'teacher_patch_batch', 'student_patch_batch', 'test_patch_batch',
            'epochs', 'inference_patch_size', 'seed_ratio', 'total_bs')):
        raise ValueError('步数、patch 大小及 batch 必须为正整数')
    if not torch.cuda.is_available():
        raise RuntimeError('真实 PointGPT 训练/测试需要原 CUDA 环境及其扩展')
    torch.cuda.set_device(args.device)
    torch.set_num_threads(cpu_threads)
    from utils.gpu_memory import apply_gpu_memory_limit
    gpu_memory_limit = apply_gpu_memory_limit(config, args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    from tools import builder
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'manifest.json').open('w', encoding='utf-8') as handle:
        json.dump(dict(mode=args.mode, checkpoint=str(checkpoint_path),
                       config=config, arguments=vars(args), schedule=schedule(teacher_nodes),
                       gpu_memory_limit=gpu_memory_limit),
                  handle, indent=2, ensure_ascii=False)
    device = torch.device(f'cuda:{args.device}')
    if args.mode == 'train':
        train(args, config, builder, device, checkpoint_path, output)
    else:
        test(args, config, builder, device, checkpoint_path, output)


if __name__ == '__main__':
    from utils.gpu_memory import exit_on_cuda_oom
    with exit_on_cuda_oom():
        main()
