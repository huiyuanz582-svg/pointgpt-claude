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
    if mode != 'teacher_forced' and 'student_target' in supplied:
        defaults['student_target'] = 'teacher_target_node'
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
    if resolved.get('student_target', 'teacher_target_node') not in (
            'teacher_target_node', 'teacher_from_student_input'):
        raise ValueError('student_target must be teacher_target_node or teacher_from_student_input')
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
    if training.get('student_target') == 'teacher_from_student_input':
        result.update(target='input_matched_teacher_interval',
                      teacher_input_target='teacher_target_node',
                      student_input_target='teacher_from_detached_student_input',
                      teacher_interval='start_step_to_target_step',
                      teacher_interval_initial_sigma='sigma0_times_teacher_decay_to_start_step',
                      teacher_target_detached=True, extra_teacher_rng_isolated=True)
    return result


def sample_teacher_input_mask(count, probability, generator=None):
    """Return CPU bool [4, B]; sample once before dividing an optimizer batch.

    Extreme probabilities do not consume RNG. Fractional probabilities require a
    dedicated generator so data augmentation/dropout/calibration RNG stays separate.
    """
    import torch
    if type(count) is not int or count < 1:
        raise ValueError('count must be a positive integer')
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('probability must be in [0, 1]')
    mask = torch.ones((4, count), dtype=torch.bool, device='cpu')
    if probability == 0:
        mask[1:] = False
    elif probability != 1:
        if generator is None or generator.device.type != 'cpu':
            raise ValueError('Mixed inputs require a dedicated CPU generator')
        mask[1:] = (torch.rand((count, 3), generator=generator, device='cpu') < probability).T
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


def summarize_training_targets(totals):
    """Counts are samples and additional Teacher sample-steps, not batched calls."""
    return [dict(stage=index + 1, teacher_node_target_count=int(node_count),
                 student_input_teacher_target_count=int(matched_count),
                 extra_teacher_sample_steps=int(steps))
            for index, (node_count, matched_count, steps) in enumerate(totals)]
