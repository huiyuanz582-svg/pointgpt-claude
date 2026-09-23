"""Bounded diagnostic-study configuration and seed summaries; stdlib only."""

from collections.abc import Mapping
import copy
import csv
import json
import math
from pathlib import Path
import statistics

from utils.curriculum_config import resolve_curriculum
from utils.curriculum_diagnostics import diagnostic_paths


STUDY_KEYS = {'base_config', 'paths', 'noise_levels', 'patches_per_level', 'calibration_seeds', 'patch_batch'}
ROW_ID = ('noise', 'metric', 'stage', 'start_step', 'target_step', 'normalization', 'aggregation')
ROW_VALUES = ('sample_count', 'total_patch_count', 'undefined_count', 'mean', 'median', 'p90', 'p95', 'value')


def resolve_study(raw, *, calibration_seeds=None, patches_per_level=None, patch_batch=None):
    if not isinstance(raw, Mapping) or set(raw) != STUDY_KEYS:
        raise ValueError(f'Diagnostic config requires exactly these keys: {sorted(STUDY_KEYS)}')
    study = copy.deepcopy(dict(raw))
    for key, override in (('calibration_seeds', calibration_seeds), ('patches_per_level', patches_per_level),
                          ('patch_batch', patch_batch)):
        if override is not None:
            study[key] = override
    if not isinstance(study['base_config'], str) or not study['base_config'].strip():
        raise ValueError('base_config must name a model/dataset YAML')
    paths = diagnostic_paths(named_paths=study['paths'])
    if len(study['paths']) > 8:
        raise ValueError('Diagnostic study supports at most 8 named paths')
    for key, maximum in (('patches_per_level', 32), ('patch_batch', 8)):
        if type(study[key]) is not int or not 1 <= study[key] <= maximum:
            raise ValueError(f'{key} must be an integer in [1, {maximum}]')
    seeds = study['calibration_seeds']
    if (not isinstance(seeds, (list, tuple)) or not 1 <= len(seeds) <= 5 or
            any(type(seed) is not int or not 0 <= seed < 2 ** 32 for seed in seeds) or
            len(set(seeds)) != len(seeds)):
        raise ValueError('Use 1..5 distinct calibration seeds in [0, 2**32)')
    levels = study['noise_levels']
    if not isinstance(levels, (list, tuple)) or list(levels) != [.005, .01, .02]:
        raise ValueError('Diagnostic study uses noise_levels [0.005, 0.01, 0.02]; 3% remains held out')
    study['calibration_seeds'] = list(seeds)
    study['paths'] = {name: list(nodes) for name, nodes in study['paths'].items()}
    micro_batches = len(levels) * math.ceil(study['patches_per_level'] / study['patch_batch'])
    study['expected_cost'] = dict(unique_paths=len(paths), micro_batches_per_seed=micro_batches,
                                  student_forwards_per_seed=7 * len(paths) * micro_batches,
                                  student_forwards_total=7 * len(paths) * micro_batches * len(seeds),
                                  includes_teacher_calibration=False, search_forwards=0)
    return study


def seed_configuration(base_config, study, seed):
    """Override the actual bank seed, never the general model/runtime seed."""
    if seed not in study['calibration_seeds']:
        raise ValueError('Calibration seed was not requested in this study')
    config = copy.deepcopy(base_config)
    config.setdefault('curriculum_search', {}).update(
        candidate_set='explicit', candidate_paths=[p['path'] for p in diagnostic_paths(named_paths=study['paths'])],
        patch_batch=study['patch_batch'])
    config.setdefault('curriculum_calibration', {}).update(
        mode='stratified_sigma', noise_levels=list(study['noise_levels']),
        patches_per_level=study['patches_per_level'], seed=seed)
    resolved = resolve_curriculum(config)
    if resolved['metric']['type'] != 'rollout_aware':
        raise ValueError('Diagnostic study requires a rollout_aware base configuration')
    return config, resolved


def summarize_seeds(reports, expected_seeds):
    """Equal-weight seed means of per-seed statistics, never pooled quantiles.

    Rows are deduplicated by actual node path before aggregation. Missing runs or
    mismatched rows fail; null first-stage shares retain their undefined counts.
    """
    expected_seeds = list(expected_seeds)
    if not expected_seeds or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError('Expected seeds must be nonempty and unique')
    indexed, identity, schema, reference = {}, None, None, None
    for report in reports:
        if report['status'] != 'completed' or report['nonfinite_count'] != 0:
            raise ValueError('Only completed, finite diagnostic runs may be summarized')
        state = report['model_state']
        seed = state['calibration_seed']
        if seed in indexed or seed not in expected_seeds:
            raise ValueError('Duplicate or unexpected calibration seed')
        run_identity = {key: state.get(key) for key in
                        ('id', 'kind', 'checkpoint', 'teacher_checkpoint', 'seed', 'current_checkpoint_nodes')}
        run_schema = dict(paths=report['paths'], noise_levels=report['noise_levels'], eps=report['eps'])
        if identity is None:
            identity, schema = run_identity, run_schema
        if run_identity != identity or run_schema != schema:
            raise ValueError('Cannot combine different model states, paths, noise levels or eps')
        rows = {}
        for row in report['rows']:
            key = (tuple(row['path']),) + tuple(row[field] for field in ROW_ID)
            if key in rows and any(rows[key][field] != row[field] for field in ROW_VALUES):
                raise ValueError('Path role aliases disagree')
            rows.setdefault(key, row)
        if not rows or (reference is not None and rows.keys() != reference.keys()):
            raise ValueError('Missing or mismatched per-seed diagnostic rows')
        if reference is None:
            reference = rows
        indexed[seed] = rows
    if set(indexed) != set(expected_seeds):
        raise ValueError('All requested calibration seeds must complete before aggregation')
    summary_rows = []
    for key, first in reference.items():
        field = 'value' if first['aggregation'] == 'summary_of_four_stage_patch_means' else 'mean'
        seed_values = []
        for seed in expected_seeds:
            row = indexed[seed][key]
            value = row[field]
            if value is not None and (isinstance(value, bool) or not math.isfinite(value)):
                raise FloatingPointError('Nonfinite cross-seed statistic')
            seed_values.append(dict(calibration_seed=seed, value=value,
                                    **{name: row[name] for name in ROW_VALUES[:3]}))
        defined = [item['value'] for item in seed_values if item['value'] is not None]
        summary_rows.append(dict(
            path_id=first['path_id'], path=first['path'], path_roles=first['overlap_roles'],
            **{name: first[name] for name in ROW_ID if name != 'aggregation'},
            within_seed_aggregation=first['aggregation'], source_statistic=field,
            seed_count=len(expected_seeds), defined_seed_count=len(defined),
            undefined_seed_count=len(expected_seeds) - len(defined),
            mean_across_seeds=statistics.mean(defined) if defined else None,
            std_across_seeds=statistics.pstdev(defined) if defined else None,
            min_across_seeds=min(defined) if defined else None,
            max_across_seeds=max(defined) if defined else None, seed_values=seed_values))
    return dict(status='completed', diagnostic_only=True, changes_selection=False,
                model_state=identity, calibration_seeds=expected_seeds, **schema,
                aggregation='equal-weight mean of per-seed means or summary values; population std (ddof=0)',
                quantiles='P90/P95 stay in per-seed artifacts; no pooled or averaged quantiles reported',
                sampling='Calibration seeds can share shapes/indices; these are not independent training runs',
                role_aliases='Deduplicated by actual path; counts are never added across aliases',
                rows=summary_rows)


def write_json(path, report):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_seed_summary(output, report):
    """Only called after every seed completes; refuse existing aggregate artifacts."""
    output = Path(output)
    json_path, csv_path = output / 'seed_summary.json', output / 'seed_summary.csv'
    if json_path.exists() or csv_path.exists():
        raise FileExistsError('Refusing to overwrite seed summary')
    temporary = csv_path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report['rows'][0]))
        writer.writeheader()
        for row in report['rows']:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (list, dict)) else value for key, value in row.items()})
    temporary.replace(csv_path)
    write_json(json_path, report)
    return dict(json=json_path.name, csv=csv_path.name)
