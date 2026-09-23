"""Stage-diagnostic statistics/serialization; never scores or selects a path."""

import csv
import json
from pathlib import Path


FIXED_REFERENCE = (0, 4, 8, 12, 16)
NORMALIZATIONS = {
    'A_TF': 'none; corresponding-point squared L2 in normalized coordinates',
    'M': 'none; Teacher interval corresponding-point squared L2',
    'C_TF': 'per-patch A_TF / (M + eps)',
    'A_FR': 'none; corresponding-point squared L2 in normalized coordinates',
    'E_FR': 'per-patch A_FR / (d(T0,T16) + eps)',
    'A_FR_minus_A_TF': 'none; paired absolute-error difference, FR minus TF',
}
CSV_FIELDS = ('epoch', 'model_state_id', 'checkpoint', 'path_id', 'path', 'path_role',
              'overlap_roles', 'noise', 'metric', 'stage', 'start_step', 'target_step',
              'normalization', 'aggregation', 'sample_count', 'total_patch_count',
              'undefined_count', 'mean', 'median', 'p90', 'p95', 'value')


def diagnostic_paths(current, selected):
    """Deduplicate actual paths while preserving all three role aliases."""
    paths = {}
    for role, values in (('fixed_reference', FIXED_REFERENCE), ('current', current), ('selected', selected)):
        nodes = tuple(values)
        if (len(nodes) != 5 or any(type(t) is not int for t in nodes) or nodes[0] != 0 or
                nodes[-1] != 16 or any(t >= u for t, u in zip(nodes[:-1], nodes[1:]))):
            raise ValueError('Diagnostic paths require 0 < n1 < n2 < n3 < 16')
        paths.setdefault(nodes, []).append(role)
    return [dict(path_id=f'P{i}', path=list(path), path_roles=roles)
            for i, (path, roles) in enumerate(paths.items())]


def distribution(values, total_count=None):
    import torch
    values = values.detach().double().cpu().reshape(-1)
    if not torch.isfinite(values).all():
        raise FloatingPointError('Nonfinite diagnostic statistic; samples cannot be silently dropped')
    count = values.numel()
    result = dict(sample_count=count, total_patch_count=count if total_count is None else total_count,
                  undefined_count=0 if total_count is None else total_count - count,
                  mean=None, median=None, p90=None, p95=None, value=None)
    if count:
        quantiles = torch.quantile(values, torch.tensor([.5, .9, .95], dtype=torch.float64)).tolist()
        result.update(mean=float(values.mean()), median=quantiles[0], p90=quantiles[1], p95=quantiles[2])
    return result


def summarize_path(path_record, values, group_ids, levels, *, epoch, model_state):
    """Values: metric -> CPU [patch,4]. Stage maxima/std use population statistics.

    all_noise pools patches, not group percentiles. CSV/JSON rows repeat role aliases
    without repeating model computation; do not add sample counts across aliases.
    """
    import torch
    if set(values) != set(NORMALIZATIONS):
        raise ValueError('Missing or unexpected stage metrics')
    size = len(group_ids)
    for name, value in values.items():
        if value.shape != (size, 4) or not torch.isfinite(value).all():
            raise ValueError(f'Invalid [patch,4] values for {name}')
    rows = []
    groups = [(float(sigma), group_ids == group) for group, sigma in enumerate(levels)]
    groups.append(('all_noise', torch.ones(size, dtype=torch.bool)))
    for noise, mask in groups:
        count = int(mask.sum())
        if not count:
            raise ValueError(f'Empty diagnostic noise group: {noise}')
        base = dict(epoch=epoch, model_state_id=model_state['id'], checkpoint=model_state.get('checkpoint'),
                    path_id=path_record['path_id'], path=path_record['path'],
                    overlap_roles=path_record['path_roles'], noise=noise)
        group_values = {key: value[mask].double() for key, value in values.items()}
        for name, value in group_values.items():
            for stage in range(4):
                row = dict(base, metric=name, stage=stage + 1,
                           start_step=path_record['path'][stage], target_step=path_record['path'][stage + 1],
                           normalization=NORMALIZATIONS[name], aggregation='pooled_patch_distribution',
                           **distribution(value[:, stage]))
                rows.extend(dict(row, path_role=role) for role in path_record['path_roles'])

        # Both aggregation orders are useful, so label them explicitly.
        tf = group_values['A_TF']
        total_tf = tf.sum(-1)
        defined = total_tf > 0
        share = tf[defined, 0] / total_tf[defined]  # 0/0 is undefined, not silently stabilized.
        patch_summaries = {'first_stage_A_TF_share': (share, 'A_TF_stage1 / sum_four_stages(A_TF); zero sum undefined')}
        stage_means = {key: value.mean(0) for key, value in group_values.items()}
        total_mean_tf = float(stage_means['A_TF'].sum())
        summary_values = {'first_stage_A_TF_share': (float(stage_means['A_TF'][0]) / total_mean_tf
                                                   if total_mean_tf > 0 else None)}
        for metric in ('C_TF', 'E_FR'):
            stage_values = group_values[metric]
            patch_summaries[metric + '_stage_std'] = (stage_values.std(-1, unbiased=False), NORMALIZATIONS[metric])
            patch_summaries[metric + '_stage_range'] = (stage_values.max(-1).values - stage_values.min(-1).values,
                                                       NORMALIZATIONS[metric])
            summary_values[metric + '_stage_std'] = float(stage_means[metric].std(unbiased=False))
            summary_values[metric + '_stage_range'] = float(stage_means[metric].max() - stage_means[metric].min())
        for name, (samples, normalization) in patch_summaries.items():
            common = dict(base, metric=name, stage=None, start_step=None, target_step=None,
                          normalization=normalization)
            row = dict(common, aggregation='per_patch_four_stage_summary', **distribution(samples, count))
            rows.extend(dict(row, path_role=role) for role in path_record['path_roles'])
            row = dict(common, aggregation='summary_of_four_stage_patch_means', sample_count=count,
                       total_patch_count=count, undefined_count=count if summary_values[name] is None else 0,
                       mean=None, median=None, p90=None, p95=None, value=summary_values[name])
            rows.extend(dict(row, path_role=role) for role in path_record['path_roles'])
    return rows


def compact_table(report):
    lines = ['[curriculum stage diagnostics] values=patch means; all_noise=pooled patches',
             f'epoch={report["epoch"]}; model_state_id={report["model_state"]["id"]}']
    for path in report['paths']:
        lines.append(f'{path["path_id"]}={path["path"]}; roles={"/".join(path["path_roles"])}; computed_once=true')
    lines.append('Path | Noise | Metric | Stage1 | Stage2 | Stage3 | Stage4')
    for path in report['paths']:
        role = path['path_roles'][0]  # Role aliases have identical rows; print once.
        for noise in list(report['noise_levels']) + ['all_noise']:
            for metric in ('C_TF', 'A_TF', 'E_FR', 'A_FR'):
                rows = sorted((row for row in report['rows'] if row['path_id'] == path['path_id'] and
                               row['path_role'] == role and row['noise'] == noise and row['metric'] == metric),
                              key=lambda row: row['stage'])
                lines.append(f'{path["path_id"]} | {noise} | {metric} | ' + ' | '.join(f'{row["mean"]:.6g}' for row in rows))
    lines.append(f'diagnostic_student_forward_calls={report["student_forward_calls"]}; '
                 f'separate_from_search=true; elapsed={report["diagnostic_seconds"]:.3f}s')
    return '\n'.join(lines)


def write_stage_report(output, report):
    output = Path(output)
    stem = f'curriculum_stage_diagnostics_epoch_{report["epoch"]:03d}'
    json_path, csv_path = output / (stem + '.json'), output / (stem + '.csv')
    if json_path.exists() or csv_path.exists():
        raise FileExistsError(f'Refusing to overwrite stage diagnostics: {stem}')
    temporary = json_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(json_path)
    if report['status'] == 'completed':
        temporary = csv_path.with_suffix('.csv.tmp')
        with temporary.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in report['rows']:
                row = dict(row, path=json.dumps(row['path']), overlap_roles='|'.join(row['overlap_roles']))
                writer.writerow(row)
        temporary.replace(csv_path)
    return dict(json=json_path.name, csv=csv_path.name if report['status'] == 'completed' else None)
