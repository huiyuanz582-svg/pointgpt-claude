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


def _build_models(config, checkpoint_path, device, logger):
    """构造冻结教师和学生；两者均从最终微调权重精确初始化。"""
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
    load_finetuned_checkpoint(
        student, checkpoint_path, allow_conditioning_missing=True)

    device_ids = [device.index] if device.index is not None else None
    teacher = nn.DataParallel(teacher.to(device), device_ids=device_ids).eval()
    student = nn.DataParallel(student.to(device), device_ids=device_ids)
    epoch = teacher_ckpt.get('epoch', 'unknown')
    metrics = teacher_ckpt.get('metrics', {})
    print_log(
        f'[Distill] 教师加载完成: {checkpoint_path}, epoch={epoch}, metrics={metrics}',
        logger=logger)
    print_log('[Distill] 学生复制教师全部已有权重；条件 MLP 以零输出初始化', logger=logger)
    return teacher, student


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

    前 ``teacher_forcing_epochs`` 个 epoch 随机学习一个教师区间；之后完整展开
    4 步学生并用教师 landmarks 与 clean 共同监督。
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
    jump_loss_type = str(_cfg_value(distill_cfg, 'jump_loss', 'smooth_l1'))
    jump_weight = float(_cfg_value(distill_cfg, 'jump_weight', 1.0))
    trajectory_weight = float(_cfg_value(distill_cfg, 'trajectory_weight', 0.25))
    endpoint_weight = float(_cfg_value(distill_cfg, 'endpoint_weight', 1.0))
    clean_weight = float(_cfg_value(distill_cfg, 'clean_weight', 0.5))
    if not 0 <= teacher_forcing_epochs <= int(config.max_epoch):
        raise ValueError('teacher_forcing_epochs 必须位于 [0, max_epoch]')
    if teacher_step_size <= 0 or not 0 < teacher_decay <= 1:
        raise ValueError('teacher_step_size 必须 > 0，teacher_decay 必须位于 (0, 1]')
    if min(jump_weight, trajectory_weight, endpoint_weight, clean_weight) < 0:
        raise ValueError('所有蒸馏损失权重都必须 >= 0')
    if jump_weight + trajectory_weight + endpoint_weight + clean_weight <= 0:
        raise ValueError('至少需要启用一个蒸馏损失')

    train_sampler, train_dataloader = builder.dataset_builder(args, config.dataset.train)
    _, val_dataloader = builder.dataset_builder(args, config.dataset.val)
    device = torch.device(f'cuda:{args.local_rank}')
    teacher, student = _build_models(config, args.ckpts, device, logger)
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
        'student_steps=%d, teacher-forcing epochs=%d, accumulation=%d' %
        (teacher_indices, teacher_step_size, teacher_decay,
         len(teacher_indices) - 1, teacher_forcing_epochs, accumulation),
        logger=logger)

    student.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, config.max_epoch + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        student.train()
        teacher.eval()
        epoch_start = time.time()
        losses = AverageMeter(['total', 'jump', 'trajectory', 'endpoint', 'clean'])
        accumulated = 0

        for idx, (pcl_noisy, pcl_clean, noise_std, center, scale, name) in enumerate(train_dataloader):
            del center, scale, name
            pcl_noisy = pcl_noisy.to(device, non_blocking=True)
            pcl_clean = pcl_clean.to(device, non_blocking=True)
            if noise_std is None:
                raise RuntimeError('蒸馏训练要求每个 patch 提供 noise_std')
            sigma0 = noise_std.to(device, non_blocking=True).reshape(-1)

            try:
                # 预热期先抽阶段，只把教师运行到该区间右端，平均可少算 37.5% 教师前向。
                stage_idx = (random.randrange(len(teacher_indices) - 1)
                             if epoch < teacher_forcing_epochs else None)
                rollout_indices = (teacher_indices[:stage_idx + 2]
                                   if stage_idx is not None else teacher_indices)
                teacher_states = rollout_teacher_patch(
                    teacher, pcl_noisy, sigma0, rollout_indices,
                    step_size=teacher_step_size, decay=teacher_decay)

                if epoch < teacher_forcing_epochs:
                    loss, _, _ = teacher_forced_jump_loss(
                        student, teacher_states, sigma0, stage_idx, teacher_indices,
                        teacher_decay=teacher_decay,
                        teacher_max_steps=teacher_max_steps,
                        loss_type=jump_loss_type)
                    values = [loss.item(), loss.item(), 0.0, 0.0, 0.0]
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
                        jump_loss_type=jump_loss_type)
                    values = [loss.item(), terms['jump'].item(),
                              terms['trajectory'].item(), terms['endpoint'].item(),
                              terms['clean'].item()]

                (loss / accumulation).backward()
                accumulated += 1
                if accumulated == accumulation:
                    if config.get('grad_norm_clip') is not None:
                        torch.nn.utils.clip_grad_norm_(
                            student.parameters(), float(config.grad_norm_clip))
                    optimizer.step()
                    student.zero_grad(set_to_none=True)
                    accumulated = 0
                losses.update(values)
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
                train_writer.add_scalar('Distill/Batch/Total', values[0], n_itr)
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

        print_log(
            '[Distill] EPOCH %d/%d phase=%s time=%.1fs '
            'total=%.6f jump=%.6f trajectory=%.6f endpoint=%.6f clean=%.6f' %
            (epoch, config.max_epoch,
             'teacher-forced' if epoch < teacher_forcing_epochs else 'student-rollout',
             time.time() - epoch_start, *losses.avg()),
            logger=logger)
        if train_writer is not None:
            train_writer.add_scalar('Distill/Epoch/Total', losses.avg(0), epoch)

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
