"""Exact global four-stage selection shared by shadow analysis and epoch curriculum."""
from importlib import import_module
from itertools import combinations
import math
import random


TEACHER_PATHS = tuple((0,) + nodes + (16,) for nodes in combinations(range(1, 16), 3))
TEACHER_INTERVALS = tuple(sorted({edge for nodes in TEACHER_PATHS for edge in zip(nodes[:-1], nodes[1:])}))


def score_teacher_path(nodes, interval_metrics, pcd_target, lambda_balance):
    """Population std over all four edges, including the last edge to T16."""
    import torch
    with torch.no_grad():
        nodes = tuple(nodes)
        target, balance = float(pcd_target), float(lambda_balance)
        if (len(nodes) != 5 or nodes[0] != 0 or nodes[-1] != 16 or
                any(not isinstance(t, int) for t in nodes) or
                any(t >= u for t, u in zip(nodes[:-1], nodes[1:]))):
            raise ValueError('A path must have integer nodes 0 < t1 < t2 < t3 < 16')
        if not all(math.isfinite(v) and v >= 0 for v in (target, balance)):
            raise ValueError('PCD target and lambda_balance must be finite and nonnegative')
        metrics = []
        for edge in zip(nodes[:-1], nodes[1:]):
            values = {key: float(value.detach() if torch.is_tensor(value) else value)
                      for key, value in interval_metrics[edge].items()}
            if not all(math.isfinite(v) and v >= 0 for v in values.values()):
                raise ValueError('Interval metrics must be finite and nonnegative')
            metrics.append(values)
        pcd = [row['PCD'] for row in metrics]
        mean = sum(pcd) / 4
        std = math.sqrt(sum((value - mean) ** 2 for value in pcd) / 4)
        error = sum(abs(value - target) for value in pcd) / 4
        return dict(nodes=list(nodes), stage_metrics=metrics, stage_PCD=pcd,
                    mean_target_error=error, pcd_std=std, pcd_max_minus_min=max(pcd) - min(pcd),
                    path_score=error + balance * std)


def search_global_teacher_nodes(interval_metrics, pcd_target, lambda_balance, return_path_scores=False):
    """Exact enumeration of all 455 complete paths, with lexicographic tie breaking."""
    import torch
    with torch.no_grad():
        missing = set(TEACHER_INTERVALS) - set(interval_metrics)
        if missing:
            raise ValueError('Missing cached Teacher intervals: ' + str(sorted(missing)[:5]))
        paths = [score_teacher_path(nodes, interval_metrics, pcd_target, lambda_balance)
                 for nodes in TEACHER_PATHS]
        selected = min(paths, key=lambda row: (row['path_score'], row['nodes']))
        result = dict(selected, paths_evaluated=len(paths), unique_intervals=len(TEACHER_INTERVALS))
        if return_path_scores:
            result['candidate_paths'] = [dict(nodes=row['nodes'], path_score=row['path_score'],
                                              mean_target_error=row['mean_target_error'], pcd_std=row['pcd_std'])
                                         for row in paths]
        return result


def build_interval_pcd_cache(student, teacher_states, sigma0, patch_batch=8):
    """Evaluate each (patch,t,u) once; share detached metrics/predictions across all selectors.

    Patches may be batched in eval mode. No training batch setting is changed.
    Prediction readouts allow the unchanged greedy function to reuse the cache
    without another Student forward. Modes and all RNG states are restored.
    """
    import numpy as np
    import torch
    runner = import_module('tools.runner_distill')
    if (teacher_states.ndim != 4 or teacher_states.shape[0] != 17 or
            teacher_states.shape[-1] != 3 or teacher_states.shape[1] < 1 or patch_batch < 1):
        raise ValueError('Cache requires [17,B,N,3] Teacher states and positive patch_batch')
    device = next(student.parameters()).device
    states = teacher_states.detach().cpu()
    sigmas = runner._sigma_batch(sigma0, states.shape[1], device, states.dtype)
    modes = [(m, m.training) for m in student.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    metrics, predictions = {}, {}
    forward_calls = 0
    try:
        student.eval()
        with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []), torch.no_grad():
            for t, u in TEACHER_INTERVALS:
                parts, values = [], []
                for offset in range(0, states.shape[1], patch_batch):
                    x = states[t, offset:offset + patch_batch].to(device)
                    sigma = sigmas[offset:offset + patch_batch] * runner.TEACHER_DECAY ** t
                    prediction = runner.forward_student_interval(student, x, sigma, t, u)
                    forward_calls += 1
                    prediction = prediction.detach().cpu().clone()
                    # Match the original shadow function's CPU metric reduction.
                    values.append(runner.evaluate_candidate_pcd(
                        states[t, offset:offset + patch_batch], states[u, offset:offset + patch_batch], prediction))
                    parts.append(prediction)
                metrics[t, u] = {key: torch.cat([value[key].cpu() for value in values]).detach()
                                 for key in ('D_move', 'E_imit', 'PCD')}
                predictions[t, u] = torch.cat(parts).detach()
    finally:
        for module, mode in modes:
            module.training = mode
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    return dict(metrics=metrics, predictions=predictions, forward_calls=forward_calls,
                patch_interval_evaluations=len(TEACHER_INTERVALS) * states.shape[1])


def compare_cached_shadow_paths(teacher_states, cache, pcd_target, lambda_balance):
    """Fixed, original greedy and exact global selection on the same cached intervals."""
    import torch
    runner = import_module('tools.runner_distill')
    results = []
    with torch.no_grad():
        states = teacher_states.detach().cpu()
        for index in range(states.shape[1]):
            metrics = {edge: {key: float(value[index]) for key, value in row.items()}
                       for edge, row in cache['metrics'].items()}
            nodes, greedy_stages = [0], []
            for stage in range(4):
                t = nodes[-1]
                # Only cached point states are read; no Student call here.
                selected = runner.search_dynamic_teacher_target(
                    states[:, index], lambda u: cache['predictions'][t, u][index],
                    t, 3 - stage, pcd_target)
                nodes.append(selected['selected_target_step'])
                greedy_stages.append(selected)
            fixed = score_teacher_path(runner.TEACHER_NODES, metrics, pcd_target, lambda_balance)
            greedy = score_teacher_path(nodes, metrics, pcd_target, lambda_balance)
            global_path = search_global_teacher_nodes(metrics, pcd_target, lambda_balance)
            if global_path['path_score'] > min(fixed['path_score'], greedy['path_score']) + 1e-12:
                raise AssertionError('Global result must dominate every feasible fixed/greedy path score')
            results.append(dict(batch_sample=index, fixed=fixed, greedy_dynamic=greedy,
                                global_dynamic=global_path, greedy_stage_search=greedy_stages))
    return results


def summarize_path_comparison(rows):
    """Report both imbalance of stage means and average per-patch path difficulty."""
    if not rows:
        raise ValueError('Cannot summarize empty shadow results')
    summary = {}
    for method in ('fixed', 'greedy_dynamic', 'global_dynamic'):
        paths = [row[method] for row in rows]
        stage_pcd = [sum(path['stage_PCD'][k] for path in paths) / len(paths) for k in range(4)]
        mean = sum(stage_pcd) / 4
        mean_nodes = [sum(path['nodes'][k] for path in paths) / len(paths) for k in range(5)]
        histograms, boundaries = [], []
        for k in range(4):
            histograms.append({str(u): sum(path['nodes'][k + 1] == u for path in paths) for u in range(1, 17)})
            percent = lambda predicate: 100 * sum(predicate(path['nodes']) for path in paths) / len(paths)
            boundaries.append(dict(stage=k, percentage_at_t_plus_1=percent(lambda n: n[k + 1] == n[k] + 1),
                                   percentage_at_u_max=percent(lambda n: n[k + 1] == 13 + k),
                                   percentage_at_fixed_target=percent(lambda n: n[k + 1] == 4 * (k + 1))))
        summary[method] = dict(mean_PCD_per_stage=stage_pcd,
                              PCD_std=math.sqrt(sum((v - mean) ** 2 for v in stage_pcd) / 4),
                              PCD_max_minus_min=max(stage_pcd) - min(stage_pcd),
                              mean_target_error=sum(p['mean_target_error'] for p in paths) / len(paths),
                              mean_per_patch_PCD_std=sum(p['pcd_std'] for p in paths) / len(paths),
                              mean_per_patch_PCD_range=sum(p['pcd_max_minus_min'] for p in paths) / len(paths),
                              mean_path_score=sum(p['path_score'] for p in paths) / len(paths),
                              mean_nodes=mean_nodes, mean_gap=[mean_nodes[k + 1] - mean_nodes[k] for k in range(4)],
                              target_histogram=histograms, boundary_selection=boundaries)
    return summary
