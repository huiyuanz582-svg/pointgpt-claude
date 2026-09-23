"""Lightweight configuration boundary; importing this module does not import torch."""

from collections.abc import Mapping
from itertools import combinations
import math


ALL_PATHS = tuple((0,) + nodes + (16,) for nodes in combinations(range(1, 16), 3))


def _get(config, key, default=None):
    return config.get(key, default) if isinstance(config, Mapping) else getattr(config, key, default)


def _block(config, name, defaults):
    supplied = _get(config, name, {})
    if not isinstance(supplied, Mapping):
        raise ValueError(f'{name} must be a mapping')
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(f'Unknown {name} keys: {sorted(unknown)}')
    return dict(defaults, **supplied)


def _number(block, key, minimum=0.0, strict=False):
    value = block[key]
    if isinstance(value, bool):
        raise ValueError(f'{key} must be numeric, not boolean')
    value = float(value)
    if not math.isfinite(value) or (value <= minimum if strict else value < minimum):
        raise ValueError(f'{key} must be finite and {">" if strict else ">="} {minimum}')
    block[key] = value


def _integer(block, key, minimum=1):
    if type(block[key]) is not int or block[key] < minimum:
        raise ValueError(f'{key} must be an integer >= {minimum}')


def candidate_paths(search):
    if search['candidate_set'] == 'all_455':
        if search.get('candidate_paths') is not None:
            raise ValueError('candidate_paths requires candidate_set: explicit')
        return ALL_PATHS
    if search['candidate_set'] != 'explicit':
        raise ValueError('candidate_set must be all_455 or explicit')
    values = search.get('candidate_paths')
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError('explicit search requires nonempty candidate_paths')
    paths = []
    for values in values:
        if (not isinstance(values, (list, tuple)) or len(values) != 5 or
                any(type(t) is not int for t in values) or values[0] != 0 or values[-1] != 16 or
                any(t >= u for t, u in zip(values[:-1], values[1:]))):
            raise ValueError('Candidate must satisfy 0 < n1 < n2 < n3 < 16')
        paths.append(tuple(values))
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate candidate paths')
    return tuple(sorted(paths))


def resolve_curriculum(config):
    """Defaults dispatch to the actual legacy implementation, never an approximation."""
    training = _block(config, 'training_input', {'type': 'teacher_forced'})
    if training['type'] == 'scheduled_rollout':
        raise NotImplementedError('scheduled_rollout training is not implemented; use teacher_forced')
    if training['type'] != 'teacher_forced':
        raise ValueError('Unknown training_input.type')
    metric = _block(config, 'curriculum_metric', dict(
        type='original_pcd', distance='pointwise_squared_l2', normalization='teacher_endpoint',
        eps=1e-12, alpha=1.0, beta=0.25, gamma=0.25, lambda_worst=0.25,
        denominator_warn_threshold=1e-10))
    if metric['type'] == 'original_pcd':
        # No new search/calibration settings can silently alter the old branch.
        if any(_get(config, key, {}) for key in ('curriculum_search', 'curriculum_calibration')):
            raise ValueError('New search/calibration blocks require rollout_aware')
        if set(_get(config, 'curriculum_metric', {})) - {'type'}:
            raise ValueError('original_pcd takes its settings from the existing dynamic_pcd block')
        return dict(metric={'type': 'original_pcd'}, training_input=training)
    if metric['type'] != 'rollout_aware':
        raise ValueError('Unknown curriculum_metric.type')
    if _get(config, 'curriculum_mode') != 'dynamic_pcd':
        raise ValueError('rollout_aware requires the existing dynamic_pcd epoch scheduler')
    if metric['distance'] != 'pointwise_squared_l2' or metric['normalization'] != 'teacher_endpoint':
        raise ValueError('Only pointwise_squared_l2 / teacher_endpoint is implemented')
    for key in ('alpha', 'beta', 'gamma', 'lambda_worst', 'denominator_warn_threshold'):
        _number(metric, key)
    _number(metric, 'eps', strict=True)
    if metric['alpha'] + metric['beta'] + metric['gamma'] == 0:
        raise ValueError('alpha, beta and gamma cannot all be zero')
    search = _block(config, 'curriculum_search', dict(
        candidate_set='all_455', candidate_paths=None, patch_batch=8, update_every_epochs=2,
        ema_decay=0.8, switch_relative_threshold=0.02, switch_absolute_threshold=0.0,
        min_hold_epochs=2))
    for key in ('patch_batch', 'update_every_epochs'):
        _integer(search, key)
    _integer(search, 'min_hold_epochs', minimum=0)
    for key in ('ema_decay', 'switch_relative_threshold', 'switch_absolute_threshold'):
        _number(search, key)
    if search['ema_decay'] >= 1:
        raise ValueError('ema_decay must be in [0, 1)')
    candidate_paths(search)
    calibration = _block(config, 'curriculum_calibration', dict(
        mode='stratified_sigma', noise_levels=[0.005, 0.01, 0.02], patches_per_level=16, seed=2025))
    if calibration['mode'] != 'stratified_sigma':
        raise ValueError('rollout_aware currently requires stratified_sigma calibration')
    _integer(calibration, 'patches_per_level')
    _integer(calibration, 'seed', minimum=0)
    if calibration['seed'] >= 2 ** 32:
        raise ValueError('calibration seed must fit NumPy seed range')
    levels = calibration['noise_levels']
    if not isinstance(levels, (list, tuple)) or not levels:
        raise ValueError('noise_levels must be a nonempty list')
    if any(isinstance(x, bool) for x in levels):
        raise ValueError('noise_levels must be numeric')
    levels = [float(x) for x in levels]
    if (any(not math.isfinite(x) or not 0.005 <= x <= 0.02 for x in levels) or
            len(set(levels)) != len(levels)):
        raise ValueError('noise_levels must be unique values in [0.005, 0.02]; 3% is held out')
    calibration['noise_levels'] = sorted(levels)
    return dict(metric=metric, training_input=training, search=search, calibration=calibration,
                implementation=dict(ema_enabled=False, switch_suppression_enabled=False,
                                    rollout_resume_enabled=False, scheduled_training_enabled=False,
                                    selection='raw_argmin_Jrobust'))


def require_supported_resume(resolved, checkpoint=None):
    saved = (checkpoint or {}).get('rollout_curriculum')
    if resolved['metric']['type'] == 'rollout_aware' or saved is not None:
        raise NotImplementedError('Rollout-aware resume/method migration is not implemented; '
                                  'start a new run. Legacy checkpoints still resume in legacy mode.')
