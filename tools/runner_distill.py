"""第二篇第一阶段：固定 16 步教师到 4 步学生的轨迹蒸馏训练。"""

import copy
import random
import time

import torch
import torch.nn as nn

from tools import builder
from tools.runner_finetune import DenoiseMetrics, validate
from utils.AverageMeter import AverageMeter
from utils.logger import get_logger, print_log
from utils.trajectory_distill import (
    load_finetuned_checkpoint,
    rollout_distill_loss,
    rollout_student_patch,
    rollout_teacher_patch,
    teacher_forced_jump_loss,
    uniform_teacher_indices,
    validate_teacher_indices,
)


def _cfg_value(cfg, name, default):
    return getattr(cfg, name, default) if cfg is not None else default


def _build_models(config, checkpoint_path, device, logger,
                  student_checkpoint_path=None):
    """构造冻结教师和学生；学生可从已有蒸馏 checkpoint 继续初始化。"""
    builder.inject_ablation(config)

    teacher_model_cfg = copy.deepcopy(config.model)
    if getattr(teacher_model_cfg, 'distill_conditioning', None) is not None:
        teacher_model_cfg.distill_conditioning.enable = False
    teacher = builder.model_builder(teacher_model_cfg)
    teacher_ckpt = load_finetuned_checkpoint(
        teacher, checkpoint_path, allow_conditioning_missing=False)
    teacher.eval()
    teacher.requires_grad_(False)

    student = builder.model_builder(config.model)
    student_source = student_checkpoint_path or checkpoint_path
    load_finetuned_checkpoint(
        student, student_source, allow_conditioning_missing=True)

    device_ids = [device.index] if device.index is not None else None
    teacher = nn.DataParallel(teacher.to(device), device_ids=device_ids).eval()
    student = nn.DataParallel(student.to(device), device_ids=device_ids)
    epoch = teacher_ckpt.get('epoch', 'unknown')
    metrics = teacher_ckpt.get('metrics', {})
    print_log(
        f'[Distill] 教师加载完成: {checkpoint_path}, epoch={epoch}, metrics={metrics}',
        logger=logger)
    if student_checkpoint_path:
        print_log(f'[Distill] 学生初始化权重: {student_checkpoint_path}', logger=logger)
    else:
        print_log('[Distill] 学生复制教师全部已有权重；条件 MLP 以零输出初始化', logger=logger)
    return teacher, student


def _rollout_probability(epoch, teacher_forcing_epochs, rollout_ramp_epochs):
    """纯 teacher-forced 后线性增加 rollout batch 比例。"""
    if epoch < teacher_forcing_epochs:
        return 0.0
    if rollout_ramp_epochs <= 0:
        return 1.0
    progress = (epoch - teacher_forcing_epochs + 1) / float(rollout_ramp_epochs)
    return min(max(progress, 0.0), 1.0)


def _meter_avg(meter):
    return meter.avg() if meter.count() > 0 else 0.0


def _evaluate_initial_baselines(student, teacher, val_dataloader, args, config,
                                distill_cfg, logger):
    """用同一验证集和后处理协议测学生初值及教师截断轨迹。"""
    print_log('[Distill/Baseline] 开始同协议初始基线评估', logger=logger)
    student_metrics = validate(
        student, val_dataloader, -1, None, args, config, logger=logger)
    print_log(
        '[Distill/Baseline] student-initial: CD=%.6f P2M=%.6f score=%.6f' %
        (student_metrics.cd, student_metrics.p2m,
         student_metrics.cd + 0.3 * student_metrics.p2m),
        logger=logger)

    teacher_step_size = float(_cfg_value(distill_cfg, 'teacher_step_size', 0.3))
    teacher_decay = float(_cfg_value(distill_cfg, 'teacher_decay', 0.95))
    baseline_steps = list(_cfg_value(distill_cfg, 'baseline_teacher_steps', [4, 16]))
    for steps in baseline_steps:
        eval_config = copy.deepcopy(config)
        eval_config.langevin.num_steps = int(steps)
        eval_config.langevin.step_size = teacher_step_size
        eval_config.langevin.decay = teacher_decay
        for key in ('teacher_indices', 'teacher_max_steps'):
            if key in eval_config.langevin:
                del eval_config.langevin[key]
        teacher_metrics = validate(
            teacher, val_dataloader, -int(steps), None, args, eval_config,
            logger=logger)
        print_log(
            '[Distill/Baseline] teacher-%d-step: CD=%.6f P2M=%.6f score=%.6f' %
            (int(steps), teacher_metrics.cd, teacher_metrics.p2m,
             teacher_metrics.cd + 0.3 * teacher_metrics.p2m),
            logger=logger)
        torch.cuda.empty_cache()
    return student_metrics


def _distill_settings(config):
    cfg = getattr(config, 'distillation', None)
    if cfg is None or not bool(_cfg_value(cfg, 'enable', False)):
        raise ValueError('蒸馏 runner 要求 config.distillation.enable=True')

    teacher_endpoint = int(_cfg_value(cfg, 'teacher_endpoint', 16))
    student_steps = int(_cfg_value(cfg, 'student_steps', 4))
    configured = _cfg_value(cfg, 'teacher_indices', None)
    indices = (validate_teacher_indices(configured, student_steps)
               if configured is not None
               else uniform_teacher_indices(teacher_endpoint, student_steps))
    if indices[-1] != teacher_endpoint:
        raise ValueError(
            f'teacher_indices 终点 {indices[-1]} 与 teacher_endpoint '
            f'{teacher_endpoint} 不一致')
    teacher_max_steps = int(_cfg_value(cfg, 'teacher_max_steps', 30))
    if indices[-1] > teacher_max_steps:
        raise ValueError(
            f'teacher_indices 终点 {indices[-1]} 超过 teacher_max_steps={teacher_max_steps}')
    return cfg, indices


def run_net(args, config, train_writer=None, val_writer=None):
    """训练固定终点少步学生。

    前 ``teacher_forcing_epochs`` 个 epoch 随机学习一个教师区间；随后逐渐增加
    完整 4 步 rollout 的 batch 比例，并用教师 landmarks 与 clean 共同监督。
    """
    if not args.use_gpu:
        raise RuntimeError('PointGPT-L 轨迹蒸馏需要 CUDA 环境')
    if args.distributed:
        raise NotImplementedError('第一阶段蒸馏 runner 暂只支持单进程单卡/DataParallel')
    if args.ckpts is None:
        raise ValueError('--distill_model 必须通过 --ckpts 指定最终教师 checkpoint')

    logger = get_logger(args.log_name)
    distill_cfg, teacher_indices = _distill_settings(config)
    teacher_step_size = float(_cfg_value(distill_cfg, 'teacher_step_size', 0.3))
    teacher_decay = float(_cfg_value(distill_cfg, 'teacher_decay', 0.95))
    teacher_max_steps = int(_cfg_value(distill_cfg, 'teacher_max_steps', 30))
    teacher_forcing_epochs = int(_cfg_value(distill_cfg, 'teacher_forcing_epochs', 10))
    rollout_ramp_epochs = int(_cfg_value(distill_cfg, 'rollout_ramp_epochs', 0))
    jump_loss_type = str(_cfg_value(distill_cfg, 'jump_loss', 'smooth_l1'))
    jump_weight = float(_cfg_value(distill_cfg, 'jump_weight', 1.0))
    trajectory_weight = float(_cfg_value(distill_cfg, 'trajectory_weight', 0.25))
    endpoint_weight = float(_cfg_value(distill_cfg, 'endpoint_weight', 1.0))
    clean_weight = float(_cfg_value(distill_cfg, 'clean_weight', 0.5))
    normalize_state_losses = bool(
        _cfg_value(distill_cfg, 'normalize_state_losses', False))
    state_scale_floor = float(_cfg_value(distill_cfg, 'state_scale_floor', 1e-4))
    evaluate_initial_baselines = bool(
        _cfg_value(distill_cfg, 'evaluate_initial_baselines', False))
    if not 0 <= teacher_forcing_epochs <= int(config.max_epoch):
        raise ValueError('teacher_forcing_epochs 必须位于 [0, max_epoch]')
    if rollout_ramp_epochs < 0:
        raise ValueError('rollout_ramp_epochs 必须 >= 0')
    if teacher_step_size <= 0 or not 0 < teacher_decay <= 1:
        raise ValueError('teacher_step_size 必须 > 0，teacher_decay 必须位于 (0, 1]')
    if min(jump_weight, trajectory_weight, endpoint_weight, clean_weight) < 0:
        raise ValueError('所有蒸馏损失权重都必须 >= 0')
    if jump_weight + trajectory_weight + endpoint_weight + clean_weight <= 0:
        raise ValueError('至少需要启用一个蒸馏损失')
    if teacher_forcing_epochs > 0 and jump_weight <= 0:
        raise ValueError('teacher-forced 阶段要求 jump_weight > 0')
    if state_scale_floor <= 0:
        raise ValueError('state_scale_floor 必须 > 0')

    train_sampler, train_dataloader = builder.dataset_builder(args, config.dataset.train)
    _, val_dataloader = builder.dataset_builder(args, config.dataset.val)
    device = torch.device(f'cuda:{args.local_rank}')
    teacher, student = _build_models(
        config, args.ckpts, device, logger,
        student_checkpoint_path=args.start_ckpts)
    optimizer, scheduler = builder.build_opti_sche(student, config)

    start_epoch = 0
    best_metrics = DenoiseMetrics()
    metrics = DenoiseMetrics()
    if args.resume:
        # builder.resume_model 会去掉 checkpoint 的 ``module.`` 前缀，因此应加载到
        # DataParallel 内部模型，而不是外层包装器。
        start_epoch, best_state = builder.resume_model(
            student.module, args, logger=logger)
        builder.resume_optimizer(optimizer, args, logger=logger)
        if isinstance(best_state, dict):
            best_metrics = DenoiseMetrics(
                float(best_state.get('cd', 0.0)),
                float(best_state.get('p2m', 0.0)))

    accumulation = max(1, int(getattr(config, 'step_per_update', 1)))
    print_log(
        '[Distill] 固定轨迹: teacher_indices=%s, teacher step_size=%.4f, decay=%.4f, '
        'student_steps=%d, teacher-forcing epochs=%d, rollout-ramp epochs=%d, '
        'normalize-state=%s, accumulation=%d' %
        (teacher_indices, teacher_step_size, teacher_decay,
         len(teacher_indices) - 1, teacher_forcing_epochs, rollout_ramp_epochs,
         normalize_state_losses, accumulation),
        logger=logger)

    if evaluate_initial_baselines and not args.resume:
        metrics = _evaluate_initial_baselines(
            student, teacher, val_dataloader, args, config, distill_cfg, logger)
        best_metrics = metrics
        # 新实验即使后续课程暂时退化，也不会丢掉传入的学生初始化模型。
        builder.save_checkpoint(
            student, optimizer, -1, metrics, best_metrics,
            'ckpt-init', args, logger=logger)
        builder.save_checkpoint(
            student, optimizer, -1, metrics, best_metrics,
            'ckpt-best', args, logger=logger)

    student.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, config.max_epoch + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        student.train()
        teacher.eval()
        epoch_start = time.time()
        meters = {
            key: AverageMeter()
            for key in ('total', 'jump', 'trajectory', 'endpoint', 'clean')
        }
        stage_jump_meters = [AverageMeter() for _ in range(len(teacher_indices) - 1)]
        rollout_probability = _rollout_probability(
            epoch, teacher_forcing_epochs, rollout_ramp_epochs)
        teacher_forced_batches = 0
        rollout_batches = 0
        accumulated = 0

        for idx, (pcl_noisy, pcl_clean, noise_std, center, scale, name) in enumerate(train_dataloader):
            del center, scale, name
            pcl_noisy = pcl_noisy.to(device, non_blocking=True)
            pcl_clean = pcl_clean.to(device, non_blocking=True)
            if noise_std is None:
                raise RuntimeError('蒸馏训练要求每个 patch 提供 noise_std')
            sigma0 = noise_std.to(device, non_blocking=True).reshape(-1)

            try:
                use_rollout = random.random() < rollout_probability
                # teacher-forced batch 先抽阶段，只把教师运行到该区间右端，
                # 平均可少算 37.5% 教师前向。
                stage_idx = (None if use_rollout else
                             random.randrange(len(teacher_indices) - 1))
                rollout_indices = (teacher_indices[:stage_idx + 2]
                                   if stage_idx is not None else teacher_indices)
                teacher_states = rollout_teacher_patch(
                    teacher, pcl_noisy, sigma0, rollout_indices,
                    step_size=teacher_step_size, decay=teacher_decay)

                if not use_rollout:
                    jump_loss, _, _ = teacher_forced_jump_loss(
                        student, teacher_states, sigma0, stage_idx, teacher_indices,
                        teacher_decay=teacher_decay,
                        teacher_max_steps=teacher_max_steps,
                        loss_type=jump_loss_type)
                    loss = jump_weight * jump_loss
                    jump_value = jump_loss.item()
                    component_values = {}
                    stage_values = [None] * (len(teacher_indices) - 1)
                    stage_values[stage_idx] = jump_value
                else:
                    student_states, student_fields = rollout_student_patch(
                        student, pcl_noisy, sigma0, teacher_indices,
                        teacher_decay=teacher_decay,
                        teacher_max_steps=teacher_max_steps)
                    loss, terms = rollout_distill_loss(
                        student_states, student_fields, teacher_states,
                        pcl_clean, sigma0, teacher_indices,
                        teacher_decay=teacher_decay,
                        jump_weight=jump_weight,
                        trajectory_weight=trajectory_weight,
                        endpoint_weight=endpoint_weight,
                        clean_weight=clean_weight,
                        jump_loss_type=jump_loss_type,
                        normalize_state_losses=normalize_state_losses,
                        state_scale_floor=state_scale_floor)
                    jump_value = terms['jump'].item()
                    component_values = {
                        'trajectory': terms['trajectory'].item(),
                        'endpoint': terms['endpoint'].item(),
                        'clean': terms['clean'].item(),
                    }
                    stage_values = [item.item() for item in terms['jump_per_stage']]

                (loss / accumulation).backward()
                accumulated += 1
                if accumulated == accumulation:
                    if config.get('grad_norm_clip') is not None:
                        torch.nn.utils.clip_grad_norm_(
                            student.parameters(), float(config.grad_norm_clip))
                    optimizer.step()
                    student.zero_grad(set_to_none=True)
                    accumulated = 0
                meters['total'].update(loss.item())
                meters['jump'].update(jump_value)
                for key, value in component_values.items():
                    meters[key].update(value)
                for stage_meter, stage_value in zip(stage_jump_meters, stage_values):
                    if stage_value is not None:
                        stage_meter.update(stage_value)
                if use_rollout:
                    rollout_batches += 1
                else:
                    teacher_forced_batches += 1
            except RuntimeError as exc:
                if 'out of memory' not in str(exc).lower():
                    raise
                print_log(f'[Distill] OOM，跳过 batch {idx} 并清理梯度', logger=logger)
                student.zero_grad(set_to_none=True)
                accumulated = 0
                torch.cuda.empty_cache()
                continue

            if train_writer is not None:
                n_itr = epoch * len(train_dataloader) + idx
                train_writer.add_scalar('Distill/Batch/Total', loss.item(), n_itr)
                train_writer.add_scalar('Distill/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

        # 不丢弃 epoch 末尾不足 accumulation 的有效梯度。
        if accumulated > 0:
            scale = float(accumulation) / accumulated
            for parameter in student.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
            if config.get('grad_norm_clip') is not None:
                torch.nn.utils.clip_grad_norm_(student.parameters(), float(config.grad_norm_clip))
            optimizer.step()
            student.zero_grad(set_to_none=True)

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        elif scheduler is not None:
            scheduler.step(epoch)

        averages = {key: _meter_avg(meter) for key, meter in meters.items()}
        stage_averages = [_meter_avg(meter) for meter in stage_jump_meters]
        phase = ('teacher-forced' if rollout_probability == 0 else
                 'student-rollout' if rollout_probability == 1 else
                 'mixed')
        print_log(
            '[Distill] EPOCH %d/%d phase=%s rollout_p=%.3f batches(tf=%d,rollout=%d) '
            'time=%.1fs total=%.6f jump=%.6f trajectory=%.6f endpoint=%.6f '
            'clean=%.6f stage_jump=%s' %
            (epoch, config.max_epoch, phase, rollout_probability,
             teacher_forced_batches, rollout_batches, time.time() - epoch_start,
             averages['total'], averages['jump'], averages['trajectory'],
             averages['endpoint'], averages['clean'],
             '[' + ', '.join('%.6f' % value for value in stage_averages) + ']'),
            logger=logger)
        if train_writer is not None:
            for key, value in averages.items():
                train_writer.add_scalar(f'Distill/Epoch/{key.title()}', value, epoch)
            train_writer.add_scalar(
                'Distill/Epoch/RolloutProbability', rollout_probability, epoch)
            for stage_idx, value in enumerate(stage_averages):
                train_writer.add_scalar(
                    f'Distill/Epoch/JumpStage{stage_idx}', value, epoch)

        if epoch % args.val_freq == 0 and epoch != 0:
            torch.cuda.empty_cache()
            metrics = validate(
                student, val_dataloader, epoch, val_writer, args, config, logger=logger)
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(
                    student, optimizer, epoch, metrics, best_metrics,
                    'ckpt-best', args, logger=logger)

        builder.save_checkpoint(
            student, optimizer, epoch, metrics, best_metrics,
            'ckpt-last', args, logger=logger)

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()
