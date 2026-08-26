"""第 0 阶段教师轨迹的 CPU 几何统计工具。

这里只依赖 NumPy/SciPy，不依赖 CUDA 扩展。所有状态必须是 SOR 前、点数和点身份保持不变的
归一化点云。对 PUNet 默认数据可使用索引对应；外部数据若不能保证 noisy/clean 点顺序一致，
请把 ``assume_correspondence`` 设为 False，此时索引相关指标会显式写为 NaN。
"""

from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree


_EPS = 1e-12


def validate_budgets(budgets, max_steps=None):
    """校验并保留用户给定顺序；重复、非正数和越界都直接报错。"""
    parsed = []
    for value in budgets:
        # argparse 和 YAML 正常都会给 int。这里仍做严格校验，避免 1.5 被
        # int() 静默截断成 1，导致实验标签与实际 rollout 不一致。
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f'budgets 必须全部为整数，收到 {value!r}')
        parsed.append(int(value))
    if not parsed:
        raise ValueError('budgets 不能为空')
    if any(v <= 0 for v in parsed):
        raise ValueError(f'budgets 必须全部为正整数，当前为 {parsed}')
    if len(set(parsed)) != len(parsed):
        raise ValueError(f'budgets 不能重复，当前为 {parsed}')
    if max_steps is not None and any(v > int(max_steps) for v in parsed):
        raise ValueError(f'budgets 不能超过 max_steps={max_steps}，当前为 {parsed}')
    return parsed


def deterministic_sample_indices(num_points, max_points):
    """在不引入额外随机性的前提下，均匀选择最多 max_points 个索引。"""
    num_points = int(num_points)
    max_points = int(max_points)
    if num_points <= 0:
        raise ValueError('点数必须 > 0')
    if max_points <= 0 or max_points >= num_points:
        return np.arange(num_points, dtype=np.int64)
    return np.unique(np.linspace(0, num_points - 1, max_points, dtype=np.int64))


def bidirectional_distance_metrics(pred, clean):
    """返回双向最近邻欧氏 HD/HD95，以及与仓库 CD 同量纲的平方距离统计。"""
    pred = np.asarray(pred, dtype=np.float64)
    clean = np.asarray(clean, dtype=np.float64)
    if pred.ndim != 2 or clean.ndim != 2 or pred.shape[1] != 3 or clean.shape[1] != 3:
        raise ValueError('pred/clean 必须为 [N,3]')
    if len(pred) == 0 or len(clean) == 0:
        raise ValueError('pred/clean 不能为空')

    pred_to_clean = cKDTree(clean).query(pred, k=1)[0]
    clean_to_pred = cKDTree(pred).query(clean, k=1)[0]
    pooled = np.concatenate([pred_to_clean, clean_to_pred])
    hd = float(max(pred_to_clean.max(), clean_to_pred.max()))
    hd95 = float(np.percentile(pooled, 95))
    cd_sq = float(np.mean(pred_to_clean ** 2) + np.mean(clean_to_pred ** 2))
    return {
        'hd_euclidean': hd,
        'hd95_euclidean': hd95,
        'hd_sq_x1e4': hd * hd * 1e4,
        'hd95_sq_x1e4': hd95 * hd95 * 1e4,
        'cpu_cd_sq_x1e4': cd_sq * 1e4,
    }


def _query_knn_indices(tree, points, anchor_indices, k):
    """查询 anchor 的 k 个邻居并排除自身索引；对重复点也保持固定输出形状。"""
    num_points = len(points)
    if num_points < 2:
        return np.empty((len(anchor_indices), 0), dtype=np.int64)
    k = min(int(k), num_points - 1)
    query_k = min(num_points, k + 2)
    raw = tree.query(points[anchor_indices], k=query_k)[1]
    if raw.ndim == 1:
        raw = raw[:, None]
    result = np.empty((len(anchor_indices), k), dtype=np.int64)
    for row_idx, anchor_idx in enumerate(anchor_indices):
        candidates = [int(v) for v in np.atleast_1d(raw[row_idx]) if int(v) != int(anchor_idx)]
        if len(candidates) < k:
            # 极端重复点下 query 结果可能没有包含自身；扩大查询到全体以保证输出完整。
            full = np.atleast_1d(tree.query(points[anchor_idx], k=num_points)[1])
            candidates = [int(v) for v in full if int(v) != int(anchor_idx)]
        if not candidates:
            candidates = [int(anchor_idx)]
        candidates = (candidates + [candidates[-1]] * k)[:k]
        result[row_idx] = np.asarray(candidates, dtype=np.int64)
    return result


def _estimate_normals(clean, anchor_indices, k):
    """在 clean 全云上以 kNN-PCA 估计指定点的法向和置信度。"""
    clean = np.asarray(clean, dtype=np.float64)
    tree = cKDTree(clean)
    neighbors = _query_knn_indices(tree, clean, anchor_indices, k)
    if neighbors.shape[1] == 0:
        normals = np.zeros((len(anchor_indices), 3), dtype=np.float64)
        confidence = np.zeros(len(anchor_indices), dtype=np.float64)
        return normals, confidence, neighbors

    neighborhoods = np.concatenate(
        [clean[anchor_indices, None, :], clean[neighbors]], axis=1)
    centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
    covariance = np.einsum('nki,nkj->nij', centered, centered) / max(neighborhoods.shape[1], 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]
    normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(normal_norm, _EPS)
    # 平面：lambda0 << lambda1≈lambda2，置信度高；线状/重复邻域：lambda0≈lambda1，置信度低。
    confidence = (eigenvalues[:, 1] - eigenvalues[:, 0]) / np.maximum(eigenvalues[:, 2], _EPS)
    return normals, confidence, neighbors


def _neighbor_retention(current, reference):
    if current.shape[1] == 0:
        return float('nan')
    matched = (current[:, :, None] == reference[:, None, :]).any(axis=2)
    return float(matched.sum(axis=1).mean() / current.shape[1])


def _safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float('nan')


def analyze_trajectory(states, clean, sigma0, step_size, decay, max_steps=None,
                       sample_points=2048, knn_k=16, normal_confidence_threshold=0.05,
                       assume_correspondence=True):
    """分析完整 SOR 前轨迹。

    Args:
        states: [T+1,N,3]，state[0] 是 noisy，state[k] 是第 k 次更新后的诊断整云。
        clean: [N,3] clean 参考点云；默认 PUNet 点索引与 noisy 对应。
        sigma0/step_size/decay: 教师日程。
        max_steps: 用于 teacher_t=max_steps-update_step 的标签；缺省 T。

    Returns:
        rows, diagnostics；rows 每个状态一行，diagnostics 含采样索引、参考法向和置信度。
    """
    states = np.asarray(states, dtype=np.float64)
    clean = np.asarray(clean, dtype=np.float64)
    if states.ndim != 3 or states.shape[-1] != 3 or states.shape[0] < 1 or states.shape[1] < 1:
        raise ValueError('states 必须为 [T+1,N,3]')
    if clean.shape != states.shape[1:]:
        raise ValueError(f'clean 形状 {clean.shape} 与 states 单帧 {states.shape[1:]} 不一致')
    if not np.isfinite(states).all() or not np.isfinite(clean).all():
        raise ValueError('states/clean 含 NaN 或 Inf')
    if (not np.isfinite(sigma0) or float(sigma0) <= 0 or
            not np.isfinite(step_size) or float(step_size) <= 0 or
            not np.isfinite(decay) or float(decay) <= 0):
        raise ValueError('sigma0/step_size/decay 必须为有限正数')
    if int(sample_points) <= 0 or int(knn_k) <= 0:
        raise ValueError('sample_points/knn_k 必须 > 0')
    if (not np.isfinite(normal_confidence_threshold) or
            float(normal_confidence_threshold) < 0):
        raise ValueError('normal_confidence_threshold 必须为有限非负数')
    num_updates = states.shape[0] - 1
    max_steps = num_updates if max_steps is None else int(max_steps)
    if max_steps < num_updates:
        raise ValueError('max_steps 不能小于轨迹实际更新数')

    sample_idx = deterministic_sample_indices(states.shape[1], sample_points)
    clean_tree = cKDTree(clean)
    reference_normals, normal_confidence, clean_neighbors = _estimate_normals(
        clean, sample_idx, knn_k)
    normal_valid = normal_confidence >= float(normal_confidence_threshold)

    clean_edge_lengths = None
    if assume_correspondence and clean_neighbors.shape[1] > 0:
        clean_edge_lengths = np.linalg.norm(
            clean[sample_idx, None, :] - clean[clean_neighbors], axis=2)

    rows = []
    previous_neighbors = None
    previous_edge_lengths = None
    previous_nn_rmse = None

    for update_step, current in enumerate(states):
        current_sample = current[sample_idx]
        current_tree = cKDTree(current)
        current_neighbors = _query_knn_indices(current_tree, current, sample_idx, knn_k)

        nn_distances, nearest_clean_idx = clean_tree.query(current_sample, k=1)
        nearest_clean_rmse = float(np.sqrt(np.mean(nn_distances ** 2)))
        nearest_clean_mae = float(np.mean(nn_distances))
        error_drop = (float('nan') if previous_nn_rmse is None
                      else float(previous_nn_rmse - nearest_clean_rmse))

        if update_step == 0:
            delta = np.zeros_like(current)
        else:
            delta = current - states[update_step - 1]
        delta_norm = np.linalg.norm(delta, axis=1)
        delta_sample = delta[sample_idx]

        if assume_correspondence:
            normals = reference_normals
            valid = normal_valid
            paired_rmse = float(np.sqrt(np.mean((current - clean) ** 2)))
        else:
            # 外部数据无索引对应时，按当前点最近的 clean 点重新估计参考法向。
            normals, step_confidence, _ = _estimate_normals(clean, nearest_clean_idx, knn_k)
            valid = step_confidence >= float(normal_confidence_threshold)
            paired_rmse = float('nan')

        normal_scalar = np.sum(delta_sample * normals, axis=1)
        normal_abs = np.abs(normal_scalar)
        tangent_vec = delta_sample - normal_scalar[:, None] * normals
        tangent_norm = np.linalg.norm(tangent_vec, axis=1)
        valid_normal_abs = normal_abs[valid]
        valid_tangent = tangent_norm[valid]
        valid_delta = np.linalg.norm(delta_sample[valid], axis=1) if np.any(valid) else np.empty(0)
        normal_energy_ratio = (
            float(np.sum(valid_normal_abs ** 2) / max(np.sum(valid_delta ** 2), _EPS))
            if valid_normal_abs.size else float('nan'))

        retention_prev = (1.0 if previous_neighbors is None
                          else _neighbor_retention(current_neighbors, previous_neighbors))
        retention_clean = (_neighbor_retention(current_neighbors, clean_neighbors)
                           if assume_correspondence else float('nan'))

        edge_error_clean = float('nan')
        edge_change_prev = float('nan')
        current_edge_lengths = None
        if assume_correspondence and clean_edge_lengths is not None:
            current_edge_lengths = np.linalg.norm(
                current[sample_idx, None, :] - current[clean_neighbors], axis=2)
            edge_error_clean = float(np.mean(
                np.abs(current_edge_lengths - clean_edge_lengths) /
                np.maximum(clean_edge_lengths, _EPS)))
            if previous_edge_lengths is not None:
                edge_change_prev = float(np.mean(
                    np.abs(current_edge_lengths - previous_edge_lengths) /
                    np.maximum(previous_edge_lengths, _EPS)))

        sigma_before = float(sigma0 * (decay ** max(update_step - 1, 0)))
        sigma_after = float(sigma0 * (decay ** update_step))
        row = {
            'update_step': int(update_step),
            'teacher_t': int(max_steps - update_step),
            'sigma_before': sigma_before,
            'sigma_after': sigma_after,
            'step_size': float(step_size),
            'num_points': int(states.shape[1]),
            'stats_sample_points': int(len(sample_idx)),
            'normal_valid_fraction': float(np.mean(valid)) if len(valid) else 0.0,
            'step_disp_mean': float(delta_norm.mean()),
            'step_disp_rms': float(np.sqrt(np.mean(delta_norm ** 2))),
            'step_disp_p50': float(np.percentile(delta_norm, 50)),
            'step_disp_p95': float(np.percentile(delta_norm, 95)),
            'step_disp_max': float(delta_norm.max()),
            'cumulative_disp_mean': float(np.linalg.norm(current - states[0], axis=1).mean()),
            'normal_disp_abs_mean': _safe_mean(valid_normal_abs),
            'tangent_disp_mean': _safe_mean(valid_tangent),
            'normal_energy_ratio': normal_energy_ratio,
            'nearest_clean_mae': nearest_clean_mae,
            'nearest_clean_rmse': nearest_clean_rmse,
            'nearest_clean_error_drop': error_drop,
            'paired_coordinate_rmse': paired_rmse,
            'knn_retention_prev': retention_prev,
            'knn_churn_prev': float(1.0 - retention_prev),
            'knn_retention_clean': retention_clean,
            'local_edge_rel_error_clean': edge_error_clean,
            'local_edge_rel_change_prev': edge_change_prev,
        }
        rows.append(row)
        previous_neighbors = current_neighbors
        previous_edge_lengths = current_edge_lengths
        previous_nn_rmse = nearest_clean_rmse

    diagnostics = {
        'sample_indices': sample_idx,
        'reference_normals': reference_normals,
        'normal_confidence': normal_confidence,
        'clean_neighbors': clean_neighbors,
    }
    return rows, diagnostics


def aggregate_numeric_rows(rows, group_keys):
    """按 group_keys 聚合数值列，同时为每列输出有效值数量，避免静默吞掉 NaN。"""
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    output = []
    for group_values, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        result = dict(zip(group_keys, group_values))
        result['num_records'] = len(group_rows)
        numeric_keys = sorted({
            key for row in group_rows for key, value in row.items()
            if key not in group_keys and isinstance(value, (int, float, np.integer, np.floating))
        })
        for key in numeric_keys:
            values = np.asarray([row.get(key, np.nan) for row in group_rows], dtype=np.float64)
            finite = values[np.isfinite(values)]
            result[key] = float(finite.mean()) if finite.size else float('nan')
            result[key + '__valid_count'] = int(finite.size)
        output.append(result)
    return output
