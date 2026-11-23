"""Read benchmark records, check completeness, and draw quality/latency figures.

This module does not import torch or execute a model. Times are measured whole-
cloud denoising intervals, not the duration of CD/P2M evaluation or file I/O.
"""

import csv
import json
import math
from pathlib import Path
import statistics


CONDITIONS = tuple((n, s) for n in (10000, 50000) for s in (0.01, 0.02, 0.03))
STUDENTS = ('student_before', 'student_ft1', 'student_ft3')
LOCAL_ENDPOINT_STUDENT = 'student_local_endpoint'
TEACHERS = tuple(f'teacher_{t}' for t in (2, 4, 6, 8, 10))
LABELS = dict(student_before='Student4 before FT', student_ft1='Student4 FT 1 epoch',
              student_ft3='Student4 FT 3 epochs', student_local_endpoint='Student4 local + endpoint',
              **{f'teacher_{t}': f'Teacher {t} steps' for t in (2, 4, 6, 8, 10)})


def condition_label(points, noise):
    return f'{points // 1000}k_{round(noise * 100)}pct'


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        raise ValueError('Cannot write an empty result table')
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def summarize(timings, quality, points, repeats, *, expected_clouds=20, measure_efficiency=True,
              allow_filtered_output=False):
    """Equal cloud weights; every cloud must have exactly the same repeat count."""
    names = [r['name'] for r in quality]
    if expected_clouds < 1 or len(names) != expected_clouds or len(set(names)) != expected_clouds:
        raise ValueError(f'Expected quality results for exactly {expected_clouds} unique clouds')
    for row in quality:
        output_points = int(row['output_points'])
        valid_output = 0 < output_points <= points if allow_filtered_output else output_points == points
        if int(row['input_points']) != points or not valid_output:
            raise ValueError('Quality point counts differ from the condition')
        for field in ('cd_x1e4', 'p2m_x1e4'):
            if not math.isfinite(float(row[field])) or float(row[field]) < 0:
                raise ValueError(f'Invalid quality metric: {field}')
    cd = statistics.mean(float(r['cd_x1e4']) for r in quality)
    p2m = statistics.mean(float(r['p2m_x1e4']) for r in quality)
    if not measure_efficiency:
        if timings:
            raise ValueError('Quality-only conditions must not contain timing records')
        return dict(clouds=expected_clouds, timing_observations=0, repeats=0,
                    cd_x1e4=cd, p2m_x1e4=p2m, score=cd + 0.3 * p2m,
                    **{key: None for key in ('latency_mean_ms', 'latency_median_ms', 'latency_std_ms',
                       'round_mean_std_ms', 'clouds_per_second', 'points_per_second',
                       'peak_allocated_gib', 'peak_reserved_gib')}, round_mean_ms=[])
    if repeats < 1 or len(timings) != expected_clouds * repeats:
        raise ValueError('Incomplete timing records')
    seen = set()
    for row in timings:
        key = (row['name'], int(row['repeat']))
        if key in seen or key[0] not in names or not 0 <= key[1] < repeats:
            raise ValueError(f'Duplicate or unexpected timing observation: {key}')
        seen.add(key)
        for field in ('latency_ms', 'peak_allocated_bytes', 'peak_reserved_bytes'):
            if not math.isfinite(float(row[field])) or float(row[field]) <= 0:
                raise ValueError(f'Invalid {field}')
        if not (0 < int(row['allocated_before_bytes']) <= int(row['peak_allocated_bytes'])
                <= int(row['peak_reserved_bytes'])):
            raise ValueError('Inconsistent allocated/reserved memory peaks')
        if int(row['input_points']) != points or int(row['output_points']) != points:
            raise ValueError('Timing point counts differ from the condition')
    latencies = [float(r['latency_ms']) for r in timings]
    mean_ms = statistics.mean(latencies)
    round_means = [statistics.mean(float(r['latency_ms']) for r in timings if int(r['repeat']) == i)
                   for i in range(repeats)]
    return dict(clouds=expected_clouds, timing_observations=len(timings), repeats=repeats,
                cd_x1e4=cd, p2m_x1e4=p2m, score=cd + 0.3 * p2m,
                latency_mean_ms=mean_ms, latency_median_ms=statistics.median(latencies),
                latency_std_ms=statistics.pstdev(latencies),
                round_mean_std_ms=statistics.pstdev(round_means),
                round_mean_ms=round_means,
                clouds_per_second=1000.0 / mean_ms, points_per_second=points * 1000.0 / mean_ms,
                peak_allocated_gib=max(int(r['peak_allocated_bytes']) for r in timings) / 1024 ** 3,
                peak_reserved_gib=max(int(r['peak_reserved_bytes']) for r in timings) / 1024 ** 3)


def read_result(output, job, settings):
    """Recompute all reported aggregates from raw rows, including on --resume."""
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['job'] != job or manifest['settings'] != settings or not summary.get('complete'):
        raise ValueError(f'Incomplete or incompatible result: {output}')
    timing = read_csv(output / 'timing_runs.csv')
    quality = read_csv(output / 'test_metrics.csv')
    check = summarize(timing, quality, job['points'], settings['repeats'],
                      expected_clouds=job.get('expected_clouds', 20),
                      measure_efficiency=job.get('measure_efficiency', True))
    for key, value in check.items():
        observed = summary[key]
        if value is None:
            equal = observed is None
        elif isinstance(value, list):
            equal = len(value) == len(observed) and all(
                math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10) for a, b in zip(value, observed))
        else:
            equal = math.isclose(value, observed, rel_tol=1e-10, abs_tol=1e-10)
        if not equal:
            raise ValueError(f'Raw data and {key} differ: {output}')
    for row in quality:
        if not math.isclose(float(row['sigma0']), job['noise'], abs_tol=1e-7):
            raise ValueError(f'Incorrect test noise: {output}')
    if set(manifest['input_sha256']) != {r['name'] for r in quality}:
        raise ValueError(f'Input fingerprint and cloud names differ: {output}')
    if set(manifest['mesh_sha256']) != {r['name'] for r in quality}:
        raise ValueError(f'Mesh fingerprint and cloud names differ: {output}')
    return summary, manifest, quality, timing


def collect_results(root):
    plan = json.loads((root / 'plan.json').read_text(encoding='utf-8'))
    rows, per_cloud, timings = [], [], []
    shared_inputs, shared_environment = {}, None
    for job in plan['jobs']:
        output = root / job['variant']['id'] / job['condition']
        summary, manifest, quality, raw_timing = read_result(output, job, plan['settings'])
        fingerprints = (manifest['input_sha256'], manifest['mesh_sha256'])
        reference = shared_inputs.setdefault(job['condition'], fingerprints)
        if reference != fingerprints:
            raise ValueError(f'Input/mesh data changed across methods: {job["condition"]}')
        environment = manifest['environment']
        if shared_environment is None:
            shared_environment = environment
        elif shared_environment != environment:
            raise ValueError('Hardware/software/precision settings differ across runs')
        row = dict(variant=job['variant']['id'], condition=job['condition'],
                   points=job['points'], noise=job['noise'], steps=job['variant']['steps'],
                   checkpoint_epoch=manifest['checkpoint_epoch'],
                   parameters=manifest['parameters'], parameters_m=manifest['parameters'] / 1e6,
                   patches_per_cloud=manifest['patches_per_cloud'],
                   model_calls_per_cloud=manifest['model_calls_per_cloud'])
        row.update({k: v for k, v in summary.items() if k not in ('round_mean_ms', 'complete')})
        rows.append(row)
        for q in quality:
            cloud_timing = [r for r in raw_timing if r['name'] == q['name']]
            ms = statistics.mean(float(r['latency_ms']) for r in cloud_timing)
            per_cloud.append(dict(variant=row['variant'], condition=row['condition'], **q,
                                  latency_mean_ms=ms,
                                  latency_repeat_std_ms=statistics.pstdev(float(r['latency_ms']) for r in cloud_timing),
                                  peak_allocated_gib=max(int(r['peak_allocated_bytes']) for r in cloud_timing) / 1024 ** 3,
                                  peak_reserved_gib=max(int(r['peak_reserved_bytes']) for r in cloud_timing) / 1024 ** 3))
        timings.extend(dict(variant=row['variant'], condition=row['condition'], **r) for r in raw_timing)
    mode = plan.get('benchmark_mode', 'teacher10_comparison')
    variants = {'teacher10_comparison': STUDENTS + TEACHERS,
                'local_endpoint_finetune': (LOCAL_ENDPOINT_STUDENT,)}[mode]
    expected = {(v, condition_label(n, s)) for v in variants for n, s in CONDITIONS}
    if (len(rows) != len(expected) or {(r['variant'], r['condition']) for r in rows} != expected
            or {v['id'] for v in plan['variants']} != set(variants)):
        raise ValueError(f'Expected all {len(variants)} variants x 6 conditions')
    teachers = {r['condition']: r for r in rows if r['variant'] == 'teacher_10'}
    for row in rows:
        teacher = teachers.get(row['condition'])
        row['speedup_vs_teacher10'] = teacher['latency_mean_ms'] / row['latency_mean_ms'] if teacher else None
        row['cd_change_vs_teacher10_pct'] = (row['cd_x1e4'] / teacher['cd_x1e4'] - 1) * 100 if teacher else None
        row['p2m_change_vs_teacher10_pct'] = (row['p2m_x1e4'] / teacher['p2m_x1e4'] - 1) * 100 if teacher else None
    write_csv(root / 'summary.csv', rows)
    write_csv(root / 'per_cloud.csv', per_cloud)
    write_csv(root / 'timing_runs.csv', timings)
    write_json(root / 'summary.json', dict(complete=True, settings=plan['settings'],
                                          environment=shared_environment, results=rows))
    return rows


def draw_curves(root, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plots = root / 'plots'
    plots.mkdir(exist_ok=True)
    styles = dict(student_before=('#0072B2', 's'), student_ft1=('#009E73', '^'),
                  student_ft3=('#D55E00', 'D'), student_local_endpoint=('#009E73', 'o'))
    present = {row['variant'] for row in rows}

    def panel(ax, selected, metric, title):
        teachers = sorted((r for r in selected if r['variant'] in TEACHERS), key=lambda r: r['steps'])
        if teachers:
            ax.plot([r['latency_mean_ms'] for r in teachers], [r[metric] for r in teachers],
                    color='#555555', marker='o', lw=1.5)
        for row in selected:
            x, y = row['latency_mean_ms'], row[metric]
            if row['variant'] in TEACHERS:
                ax.annotate(f'T{row["steps"]}', (x, y), xytext=(4, 6), textcoords='offset points', fontsize=8)
            else:
                color, marker = styles[row['variant']]
                ax.scatter(x, y, color=color, marker=marker, s=55, zorder=4,
                           edgecolors='white', linewidths=0.6)
        ax.set(title=title, xlabel='Whole-cloud denoising latency (ms)',
               ylabel=('CD' if metric == 'cd_x1e4' else 'P2M') + r' ($\times 10^4$)')
        ax.grid(alpha=0.22)
        ax.margins(x=0.12, y=0.18)

    legend = ([Line2D([], [], color='#555555', marker='o', label='Teacher prefixes (2/4/6/8/10)')]
              if present.intersection(TEACHERS) else [])
    legend += [Line2D([], [], color=c, marker=m, linestyle='None', label=LABELS[v])
               for v, (c, m) in styles.items() if v in present]
    with plt.rc_context({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42}):
        for metric, slug in (('cd_x1e4', 'latency_cd'), ('p2m_x1e4', 'latency_p2m')):
            fig, axes = plt.subplots(2, 3, figsize=(16, 9))
            for ax, (points, noise) in zip(axes.flat, CONDITIONS):
                label = condition_label(points, noise)
                selected = [r for r in rows if r['condition'] == label]
                panel(ax, selected, metric, f'{points // 1000}k points, {noise:.0%} noise')
                single, single_ax = plt.subplots(figsize=(7.3, 5.4))
                panel(single_ax, selected, metric, f'{points // 1000}k points, {noise:.0%} noise')
                single.legend(handles=legend, loc='lower center', ncol=2, fontsize=8)
                single.tight_layout(rect=(0, 0.12, 1, 1))
                for suffix in ('png', 'pdf'):
                    single.savefig(plots / f'{slug}_{label}.{suffix}', dpi=220, bbox_inches='tight')
                plt.close(single)
            fig.legend(handles=legend, loc='lower center', ncol=4, fontsize=9)
            fig.suptitle('Quality versus measured denoising latency (lower-left is better)', y=0.995)
            fig.tight_layout(rect=(0, 0.065, 1, 0.97))
            for suffix in ('png', 'pdf'):
                fig.savefig(plots / f'{slug}_six_conditions.{suffix}', dpi=220, bbox_inches='tight')
            plt.close(fig)


def build_report(root):
    root = Path(root)
    rows = collect_results(root)
    draw_curves(root, rows)
    return rows
