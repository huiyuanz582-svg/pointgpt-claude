"""Rollout-aware scoring only. No scheduled training, EMA, or resume implementation.

No torch/model imports at module scope. The search never calls backward/step and
keeps only a DFS branch of point states for the current calibration micro-batch.
"""

from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import random
import time
import warnings

from utils.curriculum_config import candidate_paths


def write_report(path, report):
    if path is None:
        return
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


@contextmanager
def isolated_evaluation(model):
    """Restore mixed module modes and all RNGs, including on failure."""
    import numpy as np
    import torch
    device = next(model.parameters()).device
    modes = [(module, module.training) for module in model.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        model.eval()
        with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []), torch.no_grad():
            yield device
    finally:
        for module, mode in modes:
            module.training = mode
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def squared_distance(x, y):
    """Per-patch corresponding-point squared L2; no coordinate-axis averaging."""
    import torch
    dtype = torch.float64 if x.dtype == torch.float64 or y.dtype == torch.float64 else torch.float32
    return (x.detach().to(dtype) - y.detach().to(dtype)).square().sum(-1).mean(-1)


def path_components(errors):
    """[patch, stage] -> [patch, final/mean/max], BEFORE patch aggregation."""
    import torch
    if errors.ndim != 2 or errors.shape[1] != 4:
        raise ValueError('Expected [patch, 4] normalized errors')
    return torch.stack((errors[:, 3], errors.mean(-1), errors.max(-1).values), dim=-1)


def weighted_score(components, metric):
    return (metric['alpha'] * components[..., 0] + metric['beta'] * components[..., 1]
            + metric['gamma'] * components[..., 2])


def prefix_tree(paths):
    tree = {}
    for path in paths:
        branch = tree
        for node in path[1:]:
            branch = branch.setdefault(node, {})
    return tree


def prefix_count(paths):
    return len({path[:length] for path in paths for length in range(2, 6)})


def _statistics(values):
    import torch
    values = values.detach().double().cpu()
    if not values.numel():
        return dict(count=0, min=None, median=None, mean=None, p95=None)
    return dict(count=values.numel(), min=float(values.min()),
                median=float(torch.quantile(values, .5)), mean=float(values.mean()),
                p95=float(torch.quantile(values, .95)))


def search_rollout_paths(student, bank, resolved, forward_fn, *, report_path=None, decay=0.95):
    """Bank entries: (T[17,B,N,3], sigma[B], group_id[B]); exact prefix DFS.

    forward_fn(student, x, sigma0*decay**start, start, target) is the existing
    runner interface. No outer patch resampling/fusion is performed during search.
    Invalid values abort selection and preserve a failed report, never omit data.
    """
    import torch
    metric, search, calibration = (resolved[key] for key in ('metric', 'search', 'calibration'))
    paths = candidate_paths(search)
    levels = calibration['noise_levels']
    tree, path_indices = prefix_tree(paths), {path: index for index, path in enumerate(paths)}
    groups = len(levels)
    sums = torch.zeros(groups, len(paths), 3, dtype=torch.float64)
    counts = torch.zeros(groups, dtype=torch.long)
    denominators = [[] for _ in levels]
    report = dict(status='running', metric=dict(metric), search_config=dict(search),
                  calibration_config=dict(calibration), implementation=resolved['implementation'],
                  candidates_evaluated=len(paths), unique_prefixes=prefix_count(paths),
                  student_forward_calls=0, patch_forward_evaluations=0, micro_batches=0,
                  nonfinite_counts={}, warnings=[], aggregation='mean_patch_J_then_equal_weight_noise_robust',
                  normalization='per_patch_distance_T0_T16_plus_eps_no_clamp',
                  noise_groups=[dict(sigma=sigma, valid_patches=0) for sigma in levels])
    started = time.perf_counter()

    def finite(value, label):
        bad = int((~torch.isfinite(value)).sum())
        report['nonfinite_counts'][label] = report['nonfinite_counts'].get(label, 0) + bad
        if bad:
            raise FloatingPointError(f'{label}: {bad} nonfinite values; refusing partial selection')

    try:
        with isolated_evaluation(student) as device:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            for states, sigmas, group_ids in bank:
                if (states.ndim != 4 or states.shape[0] != 17 or states.shape[-1] != 3 or
                        states.shape[1] < 1 or states.shape[2] < 1 or not states.is_floating_point()):
                    raise ValueError('Teacher bank must contain floating [17,B,N,3] states')
                sigmas = torch.as_tensor(sigmas).detach().cpu().reshape(-1)
                group_ids = torch.as_tensor(group_ids).detach().cpu().reshape(-1)
                if (len(sigmas) != states.shape[1] or len(group_ids) != states.shape[1] or
                        group_ids.dtype not in (torch.int32, torch.int64)):
                    raise ValueError('sigma/group IDs must match patch count; group IDs must be integers')
                finite(sigmas, 'sigma')
                if (sigmas <= 0).any() or (group_ids < 0).any() or (group_ids >= groups).any():
                    raise ValueError('Invalid sigma or group ID')
                expected = torch.tensor(levels, dtype=sigmas.dtype)[group_ids.long()]
                if not torch.allclose(sigmas, expected, rtol=1e-6, atol=1e-10):
                    raise ValueError('Actual sigma does not match its configured noise group')
                for offset in range(0, states.shape[1], search['patch_batch']):
                    teacher = states[:, offset:offset + search['patch_batch']].detach().to(device)
                    size = teacher.shape[1]
                    sigma0 = sigmas[offset:offset + size].to(device=device, dtype=teacher.dtype)
                    ids = group_ids[offset:offset + size].long()
                    finite(teacher, 'teacher_states')
                    denominator = squared_distance(teacher[0], teacher[16]) + metric['eps']
                    finite(denominator, 'denominator')
                    if (denominator <= 0).any():
                        raise FloatingPointError('Nonpositive denominator; eps may underflow in computation dtype')
                    d_cpu = denominator.double().cpu()
                    for group in range(groups):
                        selected = ids == group
                        denominators[group].append(d_cpu[selected])
                    # Scalar metrics only, not point-state caches: ~43 KiB at 455 paths, B=8.
                    components = torch.empty(len(paths), size, 3, dtype=denominator.dtype, device=device)

                    def visit(prefix, x, errors, branch):
                        for target, children in sorted(branch.items()):
                            start = prefix[-1]
                            report['active_prefix'] = list(prefix + (target,))
                            report['student_forward_calls'] += 1
                            report['patch_forward_evaluations'] += size
                            prediction = forward_fn(student, x, sigma0 * decay ** start, start, target).detach()
                            if prediction.shape != x.shape or prediction.device != x.device:
                                raise ValueError('Student output shape/device differs from its input')
                            finite(prediction, 'student_states')
                            error = squared_distance(prediction, teacher[target]) / denominator
                            finite(error, 'normalized_errors')
                            next_errors = errors + (error,)
                            next_prefix = prefix + (target,)
                            if children:
                                visit(next_prefix, prediction, next_errors, children)
                            else:
                                components[path_indices[next_prefix]] = path_components(torch.stack(next_errors, -1))
                            # Only the active depth-four branch retains point states.
                            del prediction, error, next_errors

                    visit((0,), teacher[0], (), tree)
                    finite(components, 'components')
                    cpu_components = components.double().cpu()
                    for group in range(groups):
                        selected = ids == group
                        sums[group] += cpu_components[:, selected].sum(1)
                        counts[group] += int(selected.sum())
                    report['micro_batches'] += 1
                    # Drop closure cells as well as arrays before the next micro-batch.
                    del visit, components, cpu_components, teacher, denominator, sigma0
            if (counts == 0).any():
                raise ValueError('Every configured noise group must contain valid patches')
            group_components = sums / counts[:, None, None]
            group_scores = weighted_score(group_components, metric)
            robust = group_scores.mean(0) + metric['lambda_worst'] * group_scores.max(0).values
            finite(group_scores, 'group_scores')
            finite(robust, 'robust_scores')
            rows = []
            for index, path in enumerate(paths):
                per_group = [dict(sigma=sigma, valid_patches=int(counts[group]),
                                  final=float(group_components[group, index, 0]),
                                  mean=float(group_components[group, index, 1]),
                                  max=float(group_components[group, index, 2]),
                                  Jb=float(group_scores[group, index]))
                             for group, sigma in enumerate(levels)]
                average = group_components[:, index].mean(0)
                rows.append(dict(nodes=list(path), final=float(average[0]), mean=float(average[1]),
                                 max=float(average[2]), noise_groups=per_group,
                                 Jrobust=float(robust[index])))
            rows.sort(key=lambda row: (row['Jrobust'], row['nodes']))
            for rank, row in enumerate(rows, 1):
                row['rank'] = rank
            report.update(status='completed', candidates=rows, nodes=rows[0]['nodes'],
                          path_score=rows[0]['Jrobust'], Jrobust=rows[0]['Jrobust'],
                          runner_up_gap=rows[1]['Jrobust'] - rows[0]['Jrobust'] if len(rows) > 1 else None,
                          theoretical_forward_calls=prefix_count(paths) * report['micro_batches'])
            if report['student_forward_calls'] != report['theoretical_forward_calls']:
                raise AssertionError('Unexpected forward count')
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        all_denominators = []
        for group, values in enumerate(denominators):
            combined = torch.cat(values) if values else torch.empty(0)
            all_denominators.append(combined)
            report['noise_groups'][group].update(valid_patches=int(counts[group]), D=_statistics(combined))
        combined = torch.cat(all_denominators)
        report['D'] = _statistics(combined)
        small_count = int((combined < metric['denominator_warn_threshold']).sum())
        report['small_denominator_count'] = small_count
        if small_count:
            message = f'{small_count} denominators below {metric["denominator_warn_threshold"]:g}; no clamp applied'
            report['warnings'].append(message)
            warnings.warn(message, RuntimeWarning)
        report['nonfinite_count'] = sum(report['nonfinite_counts'].values())
        report['search_seconds'] = time.perf_counter() - started
        if report['status'] == 'completed':
            report.pop('active_prefix', None)
        write_report(report_path, report)
    return report


def build_stratified_bank(config, teacher, dataset, resolved, capture_fn, *, transform_factory=None):
    """Clone only the dataset wrapper; retain production normalization/noise/KNN.

    Same deterministic dataset indices at every sigma, independent noise draws;
    no mutation of the training dataset/loader or its transforms. All T stored on CPU.
    """
    import numpy as np
    import torch
    if transform_factory is None:
        from torchvision.transforms import Compose
        from datasets.scoredenoise.transforms import NormalizeUnitSphere, AddNoise
        transform_factory = lambda sigma: Compose([NormalizeUnitSphere(), AddNoise(sigma, sigma)])
    options = resolved['calibration']
    count, seed = options['patches_per_level'], options['seed']
    if count > len(dataset) or not dataset.on_the_fly or dataset.flag != 'train':
        raise ValueError('Calibration requires enough on-the-fly paired training patches')
    if any(sigma < dataset.noise_min or sigma > dataset.noise_max for sigma in options['noise_levels']):
        raise ValueError('Calibration sigma lies outside the actual training noise range')
    bank, samples = [], []
    digest = hashlib.sha256()
    with isolated_evaluation(teacher) as device:
        random.seed(seed)
        np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        if device.type == 'cuda':
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:count].tolist()
        for group, sigma in enumerate(options['noise_levels']):
            local_dataset = copy.copy(dataset)
            local_dataset.transform = transform_factory(sigma)
            for offset in range(0, count, resolved['search']['patch_batch']):
                noisy, sigmas = [], []
                for index in indices[offset:offset + resolved['search']['patch_batch']]:
                    sample = local_dataset[index]
                    x, clean = sample['pcl_noisy'], sample['pcl_clean']
                    if (x.shape != clean.shape or x.ndim != 2 or x.shape[-1] != 3 or
                            not torch.isfinite(x).all() or not torch.isfinite(clean).all() or
                            abs(float(sample['noise_std']) - sigma) > 1e-10):
                        raise ValueError('Invalid stratified paired sample or actual sigma')
                    for value in (x, clean, torch.tensor(sigma, dtype=torch.float64)):
                        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
                    noisy.append(x.detach().cpu())
                    sigmas.append(sigma)
                    samples.append(dict(dataset_index=index, name=str(sample['name']), sigma0=sigma,
                                        group=group, patch_size=len(x)))
                sigmas = torch.tensor(sigmas, dtype=noisy[0].dtype)
                states = capture_fn(teacher, torch.stack(noisy), sigmas,
                                    int(config.teacher_patch_batch), return_full_trajectory=True)
                bank.append((states.detach().cpu(), sigmas, torch.full((len(noisy),), group, dtype=torch.long)))
    return bank, dict(split='train', purpose='curriculum_only', mode='stratified_sigma',
                      noise_levels=options['noise_levels'], patches_per_level=count,
                      patches=count * len(options['noise_levels']), seed=seed, samples=samples,
                      data_sha256=digest.hexdigest(), teacher_steps=16)


def update_rollout_epoch(student, bank, config, resolved, epoch, used_nodes, history,
                         metadata, output, update_fn):
    """Raw argmin only. No EMA/threshold/hold behavior is claimed or applied."""
    next_nodes, result = tuple(used_nodes), None
    if epoch % resolved['search']['update_every_epochs'] == 0:
        destination = Path(output) / f'curriculum_search_epoch{epoch:04d}.json'
        if destination.exists():
            raise FileExistsError(f'Refusing to overwrite search report: {destination}')
        result = update_fn(student, bank, config, report_path=destination)
        next_nodes = tuple(result['nodes'])
        history.append(dict(epoch=epoch, old_nodes=list(used_nodes), new_nodes=list(next_nodes),
                            metric='rollout_aware', path_score=result['Jrobust'],
                            report=destination.name, student_forward_calls=result['student_forward_calls']))
    state = dict(current_teacher_nodes=list(used_nodes), nodes_used_this_epoch=list(used_nodes),
                 next_teacher_nodes=list(next_nodes), curriculum_history=copy.deepcopy(history),
                 calibration_metadata=metadata, rollout_curriculum=resolved,
                 dynamic_update_every_epochs=resolved['search']['update_every_epochs'])
    return next_nodes, result, state


def rollout_log_fields(result, resolved):
    return dict(curriculum_metric='rollout_aware', training_input='teacher_forced',
                rollout_implementation=resolved['implementation'],
                search_Jrobust=result['Jrobust'] if result else None,
                search_noise_groups=result['candidates'][0]['noise_groups'] if result else None,
                search_runner_up_gap=result['runner_up_gap'] if result else None,
                search_nonfinite_count=result['nonfinite_count'] if result else None)
