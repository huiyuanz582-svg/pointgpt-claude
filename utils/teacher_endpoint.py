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
    if not isinstance(supplied, Mapping) or set(supplied) != keys:
        raise ValueError('teacher_endpoint requires exactly: ' + ', '.join(sorted(keys)))
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


def endpoint_metadata(options):
    return dict(config=dict(options), target='cached_teacher_T10_from_same_noisy_T0',
                loss='mean_per_point_squared_l2', student_steps=4,
                student_rollout='continuous_without_teacher_inputs_or_interstage_detach',
                student_mode='eval_with_autograd', teacher_target_detached=True,
                weighting='min(max_weight, target_ratio * local_grad_norm / endpoint_grad_norm)',
                gradient_norm_scope='all_trainable_student_parameters_before_combined_clipping',
                rng_isolated=True, clean_target_used=False)


def gradient_norm(gradients):
    import torch
    norms = [g.detach().norm(2).double() for g in gradients if g is not None]
    value = float(torch.stack(norms).norm(2)) if norms else 0.0
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite local/endpoint gradient norm')
    return value


def accumulate_teacher_endpoint(student, nodes, sigma0, options, epoch, forward_fn,
                                teacher_nodes=(0, 2, 5, 7, 10), teacher_decay=0.95):
    """Add a gradient-balanced endpoint gradient to accumulated local gradients.

    Call once per optimizer batch, after all local micro-batches and before
    clipping/step. autograd.grad keeps the two gradient vectors separate until
    their norms are measured. Endpoint micro-batches are sample-weighted before
    summation, so balancing is independent of the endpoint patch batch size.
    """
    ratio = endpoint_grad_ratio(options, epoch)
    result = dict(loss=None, weighted_loss=0.0, weight=0.0, target_grad_ratio=ratio,
                  actual_grad_ratio=0.0, local_grad_norm=None, endpoint_grad_norm=None)
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
    parameters = [p for p in student.parameters() if p.requires_grad]
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
