"""Teacher10 endpoint alignment, optional alongside the existing local losses."""

from collections.abc import Mapping
import math
import random


def resolve_teacher_endpoint(config):
    get = config.get if isinstance(config, Mapping) else lambda key, default=None: getattr(config, key, default)
    supplied = get('teacher_endpoint')
    if supplied is None:
        return None
    keys = {'target_grad_ratio', 'max_weight', 'warmup_epochs', 'ramp_epochs',
            'patch_batch', 'gradient_checkpointing'}
    optional_keys = {'mask_pre_bn_conv_bias'}
    if (not isinstance(supplied, Mapping) or not keys.issubset(supplied)
            or set(supplied) - keys - optional_keys):
        raise ValueError('teacher_endpoint requires: ' + ', '.join(sorted(keys))
                         + '; optional: mask_pre_bn_conv_bias')
    options = dict(supplied)
    for key in ('target_grad_ratio', 'max_weight'):
        value = options[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f'teacher_endpoint.{key} must be finite and positive')
        options[key] = float(value)
    if options['target_grad_ratio'] > 1:
        raise ValueError('teacher_endpoint.target_grad_ratio must be <= 1')
    for key, minimum in (('warmup_epochs', 0), ('ramp_epochs', 1), ('patch_batch', 1)):
        if type(options[key]) is not int or options[key] < minimum:
            raise ValueError(f'teacher_endpoint.{key} must be an integer >= {minimum}')
    if type(options['gradient_checkpointing']) is not bool:
        raise ValueError('teacher_endpoint.gradient_checkpointing must be bool')
    if 'mask_pre_bn_conv_bias' in options and type(options['mask_pre_bn_conv_bias']) is not bool:
        raise ValueError('teacher_endpoint.mask_pre_bn_conv_bias must be bool')
    training = get('training_input', {})
    if (get('curriculum_mode') != 'fixed'
            or get('distillation', {}).get('teacher_nodes') != [0, 2, 5, 7, 10]
            or training.get('type') != 'scheduled_rollout'
            or training.get('student_target') != 'teacher_from_student_input'):
        raise ValueError('teacher_endpoint requires fixed 10-to-4 with input-matched Mixed training')
    if get('pcd_dynamic', {}).get('shadow_enabled', False):
        raise ValueError('teacher_endpoint does not support shadow search')
    return options


def endpoint_grad_ratio(options, epoch):
    if type(epoch) is not int or epoch < 1:
        raise ValueError('epoch must be a positive integer')
    progress = min(1.0, max(0.0, (epoch - options['warmup_epochs']) / options['ramp_epochs']))
    return options['target_grad_ratio'] * progress


def resolve_endpoint_finetune(config):
    get = config.get if isinstance(config, Mapping) else lambda key, default=None: getattr(config, key, default)
    supplied = get('endpoint_finetune')
    if supplied is None:
        return None
    keys = {'patch_batch', 'gradient_checkpointing', 'freeze_pre_bn_conv_bias'}
    weight_keys = {'local_weight', 'endpoint_weight'}
    if (not isinstance(supplied, Mapping) or not keys.issubset(supplied)
            or set(supplied) - keys - weight_keys - {'dense_noise_adaptation'}
            or (set(supplied) & weight_keys and not weight_keys.issubset(supplied))):
        raise ValueError('endpoint_finetune requires: ' + ', '.join(sorted(keys))
                         + '; optional pair: local_weight, endpoint_weight; optional dense_noise_adaptation')
    options = dict(supplied)
    if type(options['patch_batch']) is not int or options['patch_batch'] < 1:
        raise ValueError('endpoint_finetune.patch_batch must be a positive integer')
    if type(options['gradient_checkpointing']) is not bool or options['freeze_pre_bn_conv_bias'] is not True:
        raise ValueError('endpoint_finetune requires boolean checkpointing and freeze_pre_bn_conv_bias=true')
    if weight_keys.issubset(options):
        for key in weight_keys:
            value = options[key]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f'endpoint_finetune.{key} must be finite and nonnegative')
            options[key] = float(value)
        if options['local_weight'] == options['endpoint_weight'] == 0:
            raise ValueError('endpoint_finetune requires at least one positive loss weight')
    if 'dense_noise_adaptation' in options:
        adaptation = options['dense_noise_adaptation']
        adaptation_keys = {'min_source_points', 'noise_ramp_start', 'noise_ramp_end', 'reference_sha256'}
        if (not isinstance(adaptation, Mapping) or set(adaptation) != adaptation_keys
                or not weight_keys.issubset(options)
                or min(options['local_weight'], options['endpoint_weight']) <= 0):
            raise ValueError('dense_noise_adaptation requires both positive weights and: '
                             + ', '.join(sorted(adaptation_keys)))
        start, end = adaptation['noise_ramp_start'], adaptation['noise_ramp_end']
        points, digest = adaptation['min_source_points'], adaptation['reference_sha256']
        if (type(points) is not int or points < 1
                or any(type(value) not in (int, float) for value in (start, end))
                or not 0.005 <= start < end <= 0.02
                or not isinstance(digest, str) or len(digest) != 64
                or any(char not in '0123456789abcdef' for char in digest)):
            raise ValueError('Invalid dense_noise_adaptation point count, noise ramp or reference SHA256')
        options['dense_noise_adaptation'] = dict(adaptation)
    if (get('curriculum_mode') != 'fixed'
            or get('distillation', {}).get('teacher_nodes') != [0, 2, 5, 7, 10]
            or get('teacher_endpoint') is not None or get('training_input') is not None
            or get('pcd_dynamic', {}).get('shadow_enabled', False)):
        raise ValueError('endpoint_finetune requires fixed Teacher10/Student4 with its own fine-tuning objective')
    return options


def finetune_metadata(options, student=None):
    import torch
    metadata = dict(config=dict(options), target='cached_teacher_T10_from_same_noisy_T0',
                    loss='mean_per_point_squared_l2', loss_weight=1.0,
                    student_rollout='continuous_without_teacher_inputs_or_interstage_detach',
                    student_mode='eval_with_autograd', batchnorm_running_statistics='fixed_from_initial_student',
                    frozen_parameters=pre_bn_conv_biases(student, torch) if student is not None else None,
                    other_parameters='trainable_including_bn_affine', optimizer='fresh_AdamW',
                    best_selection='val_rollout_score_including_epoch0', clean_target_used=False)
    # Preserve historical endpoint-only checkpoint metadata when weights are absent.
    if 'local_weight' in options:
        local, endpoint = options['local_weight'], options['endpoint_weight']
        metadata.pop('loss_weight')
        metadata.update(
            objective=('local_endpoint' if local and endpoint else 'local' if local else 'endpoint'),
            target=('input_matched_teacher_intervals_and_cached_teacher_T10' if local and endpoint
                    else 'input_matched_teacher_intervals' if local else metadata['target']),
            loss_weights=dict(local=local, endpoint=endpoint),
            local_target='frozen_teacher_from_each_actual_student_input_with_nominal_interval_sigma',
            local_teacher_intervals=[[0, 2], [2, 5], [5, 7], [7, 10]],
            local_reduction='arithmetic_mean_of_four_mean_per_point_squared_l2_losses',
            teacher_targets_detached=True,
            student_gradient_scope='all_preceding_student_steps_for_each_active_loss',
            teacher_training_steps_per_patch=(10 if local else 0) + (10 if endpoint else 0))
    if 'dense_noise_adaptation' in options:
        metadata.pop('teacher_training_steps_per_patch')
        metadata.update(
            objective='dense_noise_local_with_student_endpoint_anchor',
            target='input_matched_teacher_intervals_and_frozen_initial_student_endpoint',
            loss_weights=dict(local=local, reference_endpoint=endpoint),
            local_reduction='per_patch_noise_ramp_times_four_stage_mean_then_batch_mean',
            local_weight_support='dense_source_clouds_with_sigma_above_ramp_start',
            reference_target='frozen_initial_student_four_steps_from_same_noisy_patch',
            reference_target_detached=True,
            reference_steps_per_patch=4,
            teacher_steps_per_active_patch=10,
            experimental_candidate='predeclared_epoch_1')
    return metadata


def student_rollout_endpoint(student, state, sigma0, forward_fn,
                             teacher_nodes=(0, 2, 5, 7, 10), teacher_decay=0.95,
                             gradient_checkpointing=False):
    """Apply the four conditioned updates to a patch batch."""
    from torch.utils.checkpoint import checkpoint
    for start, stop in zip(teacher_nodes[:-1], teacher_nodes[1:]):
        sigma = sigma0 * teacher_decay ** start

        # Bind interval values for delayed checkpoint recomputation.
        def step(points, start_step=start, target_step=stop, interval_sigma=sigma):
            return forward_fn(student, points, interval_sigma, start_step, target_step)

        state = checkpoint(step, state, use_reentrant=False) if gradient_checkpointing else step(state)
    return state


def continuous_endpoint_loss(student, state, target, sigma0, forward_fn,
                             teacher_nodes=(0, 2, 5, 7, 10), teacher_decay=0.95,
                             gradient_checkpointing=True):
    """One micro-batch through the connected four-step Student rollout."""
    import torch
    state, target, sigma0 = state.detach(), target.detach(), sigma0.detach()
    if not torch.isfinite(state).all() or not torch.isfinite(target).all():
        raise FloatingPointError('Nonfinite cached Teacher endpoint/input')
    state = student_rollout_endpoint(student, state, sigma0, forward_fn, teacher_nodes,
                                     teacher_decay, gradient_checkpointing)
    loss = (state - target).square().sum(-1).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite endpoint loss')
    return loss


def continuous_finetune_losses(student, state, target, sigma0, forward_fn,
                               teacher_target_fn=None, local_weight=0.0, endpoint_weight=1.0,
                               teacher_nodes=(0, 2, 5, 7, 10), teacher_decay=0.95,
                               gradient_checkpointing=True, local_patch_weights=None):
    """Connected Student rollout with detached, input-matched local Teacher targets.

    teacher_target_fn(points, sigma0, start, stop) receives the original sigma0;
    the callback applies the nominal start-step decay. Teacher targets are made
    outside Student checkpoint closures, so backward never reruns the Teacher.
    Optional local_patch_weights reduce over the full batch, including zero-weight
    patches. The endpoint target may come from a frozen reference Student.
    Disabled loss terms are None, not a misleading measured zero.
    """
    import torch
    from torch.utils.checkpoint import checkpoint
    for weight in (local_weight, endpoint_weight):
        if (isinstance(weight, bool) or not isinstance(weight, (int, float))
                or not math.isfinite(weight) or weight < 0):
            raise ValueError('Fine-tuning loss weights must be finite and nonnegative')
    if local_weight == endpoint_weight == 0:
        raise ValueError('At least one fine-tuning loss must be active')
    if endpoint_weight and (target is None or target.shape != state.shape):
        raise ValueError('Endpoint supervision requires a matching Teacher10 target')
    if not local_weight:
        endpoint = continuous_endpoint_loss(
            student, state, target, sigma0, forward_fn, teacher_nodes, teacher_decay,
            gradient_checkpointing)
        return dict(loss=endpoint if endpoint_weight == 1 else endpoint * endpoint_weight,
                    local=None, endpoint=endpoint, stages=None)
    if teacher_target_fn is None:
        raise ValueError('Local supervision requires an input-matched Teacher callback')
    state, sigma0 = state.detach(), sigma0.detach()
    target = target.detach() if endpoint_weight else None
    if not torch.isfinite(state).all() or (target is not None and not torch.isfinite(target).all()):
        raise FloatingPointError('Nonfinite fine-tuning input or endpoint target')
    if local_patch_weights is not None:
        weights = local_patch_weights.detach().to(state)
        if weights.shape != (len(state),) or not ((weights >= 0) & (weights <= 1)).all():
            raise ValueError('Expected one local weight in [0, 1] per patch')
        active = (weights > 0).nonzero(as_tuple=True)[0]
    stages = []
    for start, stop in zip(teacher_nodes[:-1], teacher_nodes[1:]):
        if local_patch_weights is None or active.numel():
            with torch.no_grad():
                inputs = state.detach() if local_patch_weights is None else state.detach()[active]
                sigmas = sigma0 if local_patch_weights is None else sigma0[active]
                local_target = teacher_target_fn(inputs, sigmas, start, stop).detach()
            if local_target.shape != inputs.shape or not torch.isfinite(local_target).all():
                raise FloatingPointError('Invalid input-matched local Teacher target')
        sigma = sigma0 * teacher_decay ** start

        def step(points, start_step=start, target_step=stop, interval_sigma=sigma):
            return forward_fn(student, points, interval_sigma, start_step, target_step)

        # No detach between Student stages: later losses also update earlier stages.
        state = checkpoint(step, state, use_reentrant=False) if gradient_checkpointing else step(state)
        if local_patch_weights is None:
            stages.append((state - local_target).square().sum(-1).mean())
        elif active.numel():
            per_patch = (state[active] - local_target).square().sum(-1).mean(-1)
            stages.append((per_patch * weights[active]).sum() / len(state))
        else:
            stages.append(state.new_zeros(()))
    local = torch.stack(stages).mean()
    endpoint = (state - target).square().sum(-1).mean() if endpoint_weight else None
    loss = local * local_weight
    if endpoint is not None:
        loss = loss + endpoint * endpoint_weight
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite local/endpoint fine-tuning loss')
    return dict(loss=loss, local=local, endpoint=endpoint, stages=stages)


def pre_bn_conv_biases(student, torch):
    """Find biases of consecutive Conv -> BatchNorm children in Sequential."""
    names = set()
    conv_types = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
    bn_types = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
    for prefix, module in student.named_modules():
        if not isinstance(module, torch.nn.Sequential):
            continue
        children = list(module.named_children())
        for (key, layer), (_, following) in zip(children[:-1], children[1:]):
            if (isinstance(layer, conv_types) and isinstance(following, bn_types)
                    and layer.bias is not None and layer.out_channels == following.num_features):
                names.add('.'.join(value for value in (prefix, key, 'bias') if value))
    return sorted(names)


def endpoint_excluded_parameters(student, options):
    if not options.get('mask_pre_bn_conv_bias', False):
        return []
    import torch
    trainable = {name for name, parameter in student.named_parameters() if parameter.requires_grad}
    excluded = sorted(trainable.intersection(pre_bn_conv_biases(student, torch)))
    if not excluded:
        raise ValueError('Masked endpoint alignment requires trainable pre-BatchNorm Conv biases')
    return excluded


def endpoint_metadata(options, student=None):
    metadata = dict(config=dict(options), target='cached_teacher_T10_from_same_noisy_T0',
                    loss='mean_per_point_squared_l2', student_steps=4,
                    student_rollout='continuous_without_teacher_inputs_or_interstage_detach',
                    student_mode='eval_with_autograd', teacher_target_detached=True,
                    weighting='min(max_weight, target_ratio * local_grad_norm / endpoint_grad_norm)',
                    gradient_norm_scope='all_trainable_student_parameters_before_combined_clipping',
                    rng_isolated=True, clean_target_used=False)
    if options.get('mask_pre_bn_conv_bias', False):
        metadata.update(
            gradient_norm_scope='same_trainable_parameters_excluding_pre_bn_conv_biases_before_combined_clipping',
            endpoint_gradient_mask='bias_of_conv_directly_followed_by_batchnorm_in_sequential',
            excluded_parameters=endpoint_excluded_parameters(student, options) if student is not None else None,
            excluded_parameter_local_gradients='preserved',
            gradient_diagnostics='cosine_and_top_level_parameter_groups_before_combined_clipping')
    return metadata


def gradient_norm(gradients):
    import torch
    norms = [g.detach().norm(2).double() for g in gradients if g is not None]
    value = float(torch.stack(norms).norm(2)) if norms else 0.0
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite local/endpoint gradient norm')
    return value


def gradient_alignment(named_parameters, endpoint_gradients, weight, local_norm, endpoint_norm):
    """Reduce on the parameter device, then transfer only scalar diagnostics."""
    import torch
    groups = {}
    for (name, parameter), gradient in zip(named_parameters, endpoint_gradients):
        groups.setdefault(name.split('.')[0], []).append((parameter.grad, gradient))
    diagnostics = {}
    total_dot = 0.0
    for name, pairs in groups.items():
        left_norm = gradient_norm(left for left, _ in pairs)
        right_norm = gradient_norm(right for _, right in pairs)
        dots = [torch.dot(left.detach().reshape(-1), right.detach().reshape(-1)).double()
                for left, right in pairs if left is not None and right is not None]
        dot = float(torch.stack(dots).sum()) if dots else 0.0
        if not math.isfinite(dot):
            raise FloatingPointError('Nonfinite local/endpoint gradient dot product')
        total_dot += dot
        diagnostics[name] = dict(
            local_norm=left_norm, endpoint_norm=right_norm,
            weighted_endpoint_to_local_ratio=weight * right_norm / left_norm if left_norm > 0 else None,
            cosine_to_local=(max(-1.0, min(1.0, dot / (left_norm * right_norm)))
                             if left_norm > 0 and right_norm > 0 else None),
            endpoint_squared_norm_fraction=(right_norm / endpoint_norm) ** 2 if endpoint_norm > 0 else 0.0)
    cosine = (max(-1.0, min(1.0, total_dot / (local_norm * endpoint_norm)))
              if local_norm > 0 and endpoint_norm > 0 else None)
    return cosine, diagnostics


def accumulate_teacher_endpoint(student, nodes, sigma0, options, epoch, forward_fn,
                                teacher_nodes=(0, 2, 5, 7, 10), teacher_decay=0.95):
    """Add a gradient-balanced endpoint gradient to accumulated local gradients.

    Call once per optimizer batch, after all local micro-batches and before
    clipping/step. autograd.grad keeps the two gradient vectors separate until
    their norms are measured. Endpoint micro-batches are sample-weighted before
    summation, so balancing is independent of the endpoint patch batch size.
    """
    excluded = endpoint_excluded_parameters(student, options)
    ratio = endpoint_grad_ratio(options, epoch)
    result = dict(loss=None, weighted_loss=0.0, weight=0.0, target_grad_ratio=ratio,
                  actual_grad_ratio=0.0, local_grad_norm=None, endpoint_grad_norm=None,
                  excluded_parameters=excluded, grad_cosine=None, group_diagnostics={})
    if ratio == 0:
        return result

    import numpy as np
    import torch

    if tuple(teacher_nodes) != (0, 2, 5, 7, 10):
        raise ValueError('Endpoint alignment requires the fixed 10-to-4 nodes')
    if (nodes.ndim != 4 or nodes.shape[0] != 5 or nodes.shape[1] < 1
            or nodes.shape[2] < 1 or nodes.shape[-1] != 3):
        raise ValueError('Endpoint alignment requires cached Teacher states [5,B,N,3]')
    excluded_set = set(excluded)
    named_parameters = [(name, parameter) for name, parameter in student.named_parameters()
                        if parameter.requires_grad and name not in excluded_set]
    parameters = [parameter for _, parameter in named_parameters]
    if not parameters:
        raise ValueError('Endpoint alignment requires trainable Student parameters')
    device = parameters[0].device
    sigmas = torch.as_tensor(sigma0, device=device, dtype=nodes.dtype).detach().reshape(-1)
    if sigmas.numel() == 1:
        sigmas = sigmas.expand(nodes.shape[1])
    if (sigmas.numel() != nodes.shape[1] or not torch.isfinite(sigmas).all()
            or (sigmas <= 0).any()):
        raise ValueError('Endpoint alignment requires finite positive per-patch sigma')
    local_norm = gradient_norm(p.grad for p in parameters)
    modes = [(module, module.training) for module in student.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cuda_devices = [device.index] if device.type == 'cuda' else []
    endpoint_gradients = None
    loss_sum = 0.0
    try:
        # Match deployment and avoid additional BatchNorm updates/checkpoint
        # recomputation side effects. eval() does not disable autograd.
        student.eval()
        with torch.random.fork_rng(devices=cuda_devices), torch.enable_grad():
            for offset in range(0, nodes.shape[1], options['patch_batch']):
                end = offset + options['patch_batch']
                state = nodes[0, offset:end].detach().to(device)
                target = nodes[-1, offset:end].detach().to(device)
                loss = continuous_endpoint_loss(
                    student, state, target, sigmas[offset:end], forward_fn,
                    teacher_nodes, teacher_decay, options['gradient_checkpointing'])
                if not torch.isfinite(loss) or not loss.requires_grad:
                    raise FloatingPointError('Endpoint loss is nonfinite or has no gradient')
                size = target.shape[0]
                gradients = torch.autograd.grad(
                    loss * (size / nodes.shape[1]), parameters, allow_unused=True)
                if endpoint_gradients is None:
                    endpoint_gradients = list(gradients)
                else:
                    for index, gradient in enumerate(gradients):
                        if gradient is not None:
                            if endpoint_gradients[index] is None:
                                endpoint_gradients[index] = gradient
                            else:
                                endpoint_gradients[index].add_(gradient)
                loss_sum += float(loss.detach()) * size
                del gradients, state, target, loss
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(python_state)
        np.random.set_state(numpy_state)

    endpoint_norm = gradient_norm(endpoint_gradients)
    weight = (min(options['max_weight'], ratio * local_norm / endpoint_norm)
              if endpoint_norm > 0 and local_norm > 0 else 0.0)
    if options.get('mask_pre_bn_conv_bias', False):
        cosine, groups = gradient_alignment(named_parameters, endpoint_gradients, weight,
                                            local_norm, endpoint_norm)
        result.update(grad_cosine=cosine, group_diagnostics=groups)
    with torch.no_grad():
        for parameter, gradient in zip(parameters, endpoint_gradients):
            if gradient is not None and weight > 0:
                if parameter.grad is None:
                    parameter.grad = gradient.mul_(weight)
                else:
                    parameter.grad.add_(gradient, alpha=weight)
    mean_loss = loss_sum / nodes.shape[1]
    result.update(loss=mean_loss, weighted_loss=weight * mean_loss, weight=weight,
                  local_grad_norm=local_norm, endpoint_grad_norm=endpoint_norm,
                  actual_grad_ratio=weight * endpoint_norm / local_norm if local_norm > 0 else 0.0)
    return result
