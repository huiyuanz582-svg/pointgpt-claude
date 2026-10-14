"""Student input policy, independent of curriculum/path selection.

Configuration and schedule helpers are torch-free. Mask sampling imports torch
only when called, uses a dedicated CPU generator, and never samples per point.
"""

from collections.abc import Mapping
import math


def resolve_training_input(config):
    supplied = (config.get('training_input', {}) if isinstance(config, Mapping)
                 else getattr(config, 'training_input', {}))
    if not isinstance(supplied, Mapping):
        raise ValueError('training_input must be a mapping')
    mode = supplied.get('type', 'teacher_forced')
    if mode not in ('teacher_forced', 'scheduled_rollout', 'student_rollout'):
        raise ValueError('Unknown training_input.type')
    defaults = {'type': mode}
    if mode == 'scheduled_rollout':
        defaults.update(warmup_epochs=2, decay_epochs=6, teacher_prob_start=1.0,
                        teacher_prob_end=0.2, sampling_seed=2025)
        if 'noise_teacher_anchor' in supplied:
            defaults['noise_teacher_anchor'] = None
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(f'Unknown training_input keys for {mode}: {sorted(unknown)}')
    resolved = dict(defaults, **supplied)
    if mode == 'scheduled_rollout':
        for key, minimum in (('warmup_epochs', 0), ('decay_epochs', 1), ('sampling_seed', 0)):
            if type(resolved[key]) is not int or resolved[key] < minimum:
                raise ValueError(f'{key} must be an integer >= {minimum}')
        if resolved['sampling_seed'] >= 2 ** 32:
            raise ValueError('sampling_seed must be < 2**32')
        for key in ('teacher_prob_start', 'teacher_prob_end'):
            value = resolved[key]
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f'{key} must be a finite probability in [0, 1]')
            resolved[key] = float(value)
        if resolved['teacher_prob_end'] > resolved['teacher_prob_start']:
            raise ValueError('teacher_prob_end must be <= teacher_prob_start')
        if 'noise_teacher_anchor' in resolved:
            anchor = resolved['noise_teacher_anchor']
            required = {'sigma_start', 'sigma_end', 'teacher_prob_high'}
            if not isinstance(anchor, Mapping) or set(anchor) != required:
                raise ValueError('noise_teacher_anchor requires exactly sigma_start, '
                                 'sigma_end and teacher_prob_high')
            for key, value in anchor.items():
                if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                        not math.isfinite(value)):
                    raise ValueError(f'noise_teacher_anchor.{key} must be finite numeric')
            anchor = {key: float(value) for key, value in anchor.items()}
            if not 0 < anchor['sigma_start'] < anchor['sigma_end']:
                raise ValueError('noise_teacher_anchor requires 0 < sigma_start < sigma_end')
            if not resolved['teacher_prob_end'] <= anchor['teacher_prob_high'] <= resolved['teacher_prob_start']:
                raise ValueError('teacher_prob_high must lie between teacher_prob_end and teacher_prob_start')
            resolved['noise_teacher_anchor'] = anchor
    return resolved


def teacher_input_probability(training, epoch):
    """Probability of Teacher input at stages 2..4; stage 1 always starts at T0.

    The decay window includes both endpoints. With defaults: epochs 1,2 = 1;
    epochs 3..8 = 1,.84,.68,.52,.36,.2; epoch 9 onward = .2.
    """
    if type(epoch) is not int or epoch < 1:
        raise ValueError('epoch must be a positive integer')
    if training['type'] == 'teacher_forced':
        return 1.0
    if training['type'] == 'student_rollout':
        return 0.0
    start, end = training['teacher_prob_start'], training['teacher_prob_end']
    after_warmup = epoch - training['warmup_epochs']
    if after_warmup <= 0:
        return start
    if training['decay_epochs'] == 1 or after_warmup >= training['decay_epochs']:
        return end
    fraction = (after_warmup - 1) / (training['decay_epochs'] - 1)
    return start + (end - start) * fraction


def training_input_metadata(training, epoch=None):
    result = dict(config=dict(training), first_stage='teacher_T0',
                  student_state_detached=True, sampling='per_patch_per_stage',
                  target='teacher_target_node', sigma='nominal_teacher_start_step',
                  student_input_resume_enabled=False)
    if training['type'] == 'scheduled_rollout':
        result['rng'] = 'dedicated_cpu_generator_seed_(sampling_seed+epoch-1)_mod_2**32'
    if epoch is not None:
        result['teacher_probability'] = teacher_input_probability(training, epoch)
    if 'noise_teacher_anchor' in training:
        result.update(teacher_probability_scope='base_before_noise_anchor',
                      noise_probability_basis='initial_sigma0',
                      noise_probability_interpolation='linear_in_log_sigma',
                      noise_probability_formula='q_base + max(q_high-q_base,0) * '
                                                'clip(log(sigma0/sigma_start)/log(sigma_end/sigma_start),0,1)')
    return result


def noise_anchored_teacher_probabilities(training, epoch, sigma0_values):
    """Per-patch probabilities; original sigma0 is used at all stages, not decayed sigma."""
    base = teacher_input_probability(training, epoch)
    anchor = training.get('noise_teacher_anchor')
    probabilities = []
    for value in sigma0_values:
        if isinstance(value, bool):
            raise ValueError('sigma0 must be finite and positive')
        sigma = float(value)
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError('sigma0 must be finite and positive')
        weight = (0.0 if anchor is None else min(1.0, max(0.0,
                  math.log(sigma / anchor['sigma_start']) /
                  math.log(anchor['sigma_end'] / anchor['sigma_start']))))
        probabilities.append(base if anchor is None else
                             base + max(anchor['teacher_prob_high'] - base, 0.0) * weight)
    return probabilities


def new_noise_anchor_statistics():
    return {band: dict(patches=0, probability_sum=0.0, probability_min=None,
                       probability_max=None, teacher_counts=[0, 0, 0])
            for band in ('low_noise', 'high_noise')}


def accumulate_noise_anchor_statistics(totals, training, sigma0_values, probabilities, mask):
    """CPU/Python counts from the actual mask, before any micro-batch slicing."""
    count = len(probabilities)
    if len(sigma0_values) != count or len(mask) != 4 or any(len(row) != count for row in mask):
        raise ValueError('Noise-anchor statistics require matching samples and [4,B] mask')
    threshold = training['noise_teacher_anchor']['sigma_start']
    for index, (sigma, probability) in enumerate(zip(sigma0_values, probabilities)):
        band = totals['low_noise' if sigma <= threshold else 'high_noise']
        band['patches'] += 1
        band['probability_sum'] += probability
        band['probability_min'] = probability if band['probability_min'] is None else min(band['probability_min'], probability)
        band['probability_max'] = probability if band['probability_max'] is None else max(band['probability_max'], probability)
        for stage in range(3):
            band['teacher_counts'][stage] += int(mask[stage + 1][index])


def summarize_noise_anchor_statistics(totals, training):
    threshold = training['noise_teacher_anchor']['sigma_start']
    rows = []
    for name, band in totals.items():
        count = band['patches']
        rows.append(dict(band=name, sigma_rule=f'sigma0 {"<=" if name == "low_noise" else ">"} {threshold:g}',
                         patches=count,
                         planned_teacher_probability_mean=band['probability_sum'] / count if count else None,
                         planned_teacher_probability_min=band['probability_min'],
                         planned_teacher_probability_max=band['probability_max'],
                         stages=[dict(stage=stage + 2, teacher_count=value, student_count=count - value,
                                      teacher_fraction=value / count if count else None)
                                 for stage, value in enumerate(band['teacher_counts'])]))
    return rows


def sample_teacher_input_mask(count, probability, generator=None):
    """Return CPU bool [4, B]; sample once before dividing an optimizer batch.

    Extreme probabilities do not consume RNG. Fractional probabilities require a
    dedicated generator so data augmentation/dropout/calibration RNG stays separate.
    """
    import torch
    if type(count) is not int or count < 1:
        raise ValueError('count must be a positive integer')
    vector = isinstance(probability, (list, tuple))
    values = list(probability) if vector else [probability]
    if vector and len(values) != count:
        raise ValueError('Per-patch probability must have length count')
    if vector:
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
               not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError('probability must contain finite numbers in [0, 1]')
    elif not math.isfinite(probability) or not 0 <= probability <= 1:
        # Keep the original scalar API and threshold arithmetic for old configs.
        raise ValueError('probability must be in [0, 1]')
    mask = torch.ones((4, count), dtype=torch.bool, device='cpu')
    if all(value == 0 for value in values):
        mask[1:] = False
    elif not all(value == 1 for value in values):
        if generator is None or generator.device.type != 'cpu':
            raise ValueError('Mixed inputs require a dedicated CPU generator')
        threshold = (torch.tensor(values, dtype=torch.get_default_dtype())[:, None]
                     if vector else probability)
        mask[1:] = (torch.rand((count, 3), generator=generator, device='cpu') < threshold).T
    return mask


def summarize_input_stages(totals):
    """Summarize [teacher_count, student_count, teacher_loss_sum, student_loss_sum]."""
    rows = []
    for stage, (teacher_count, student_count, teacher_sum, student_sum) in enumerate(totals):
        count = teacher_count + student_count
        rows.append(dict(stage=stage + 1, teacher_count=int(teacher_count),
                         student_count=int(student_count),
                         teacher_fraction=teacher_count / count if count else None,
                         teacher_input_loss=teacher_sum / teacher_count if teacher_count else None,
                         student_input_loss=student_sum / student_count if student_count else None))
    return rows
