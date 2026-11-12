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
    from torch.utils.checkpoint import checkpoint

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
                if not torch.isfinite(state).all() or not torch.isfinite(target).all():
                    raise FloatingPointError('Nonfinite cached Teacher endpoint/input')
                for start, stop in zip(teacher_nodes[:-1], teacher_nodes[1:]):
                    sigma = sigmas[offset:end] * teacher_decay ** start

                    # Bind interval values for delayed checkpoint recomputation.
                    def step(points, start_step=start, target_step=stop, interval_sigma=sigma):
                        return forward_fn(student, points, interval_sigma, start_step, target_step)

                    state = (checkpoint(step, state, use_reentrant=False)
                             if options['gradient_checkpointing'] else step(state))
                loss = (state - target).square().sum(-1).mean()
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
