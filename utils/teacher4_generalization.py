"""CPU-only discovery and reporting for the manual Teacher4 evaluation suite."""

import copy
import hashlib
import math
import os
from pathlib import Path
import re
import statistics

from utils.denoise_benchmark_report import (
    condition_label, read_csv, read_result, summarize, write_csv, write_json)


FAMILIES = ('punet_laplace', 'punet_anisotropic', 'pcnet')
EFFICIENCY_CASES = ((10000, 0.01), (50000, 0.03))
SCENE_PATTERN = re.compile(r'(?<!\d)(\d{4,6})_(?:poisson_)?(0\.\d+)(?=$|[_-])', re.I)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def xyz_directories(root):
    """Follow mounted dataset links while preventing directory-link cycles."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f'Missing noisy data directory: {root}')
    visited = set()

    def fail(error):
        raise error

    for directory, subdirs, files in os.walk(root, followlinks=True, onerror=fail):
        resolved = Path(directory).resolve()
        if resolved in visited:
            subdirs[:] = []
            continue
        visited.add(resolved)
        subdirs.sort()
        if any(name.endswith('.xyz') for name in files):
            yield Path(directory)


def parse_scene(directory, root):
    matches = {match.groups() for part in Path(directory).relative_to(root).parts
               for match in SCENE_PATTERN.finditer(part)}
    if len(matches) != 1:
        raise ValueError(f'Cannot uniquely read resolution/noise from {directory}; '
                         'expected a name such as PUNet_10000_poisson_0.01_cov')
    points, noise = next(iter(matches))
    points, noise = int(points), float(noise)
    if points < 1024 or not 0 < noise < 1:
        raise ValueError(f'Invalid test condition: {directory}')
    return points, noise


def dataset_root(data_root, name, override=None):
    if override is not None:
        candidates = [Path(override).expanduser().resolve()]
    else:
        candidates = [Path(data_root) / name, Path(data_root) / 'ScoreDenoise' / name]
    matches = {p.resolve() for p in candidates
               if (p / 'pointclouds/test').is_dir() and (p / 'meshes/test').is_dir()}
    if len(matches) != 1:
        raise ValueError(f'Expected one {name} root containing pointclouds/test and meshes/test; '
                         f'checked {candidates}. Use --{name.lower()}_root to select it.')
    return matches.pop()


def discover_scenes(data_root, *, other_noise_root=None, pcnet_noisy_root=None,
                    punet_root=None, pcnet_root=None):
    data_root = Path(data_root).expanduser().resolve()
    other = Path(other_noise_root or data_root / 'other-noise').expanduser().resolve()
    pc_noisy = Path(pcnet_noisy_root or data_root / 'PCNet/noisy').expanduser().resolve()
    roots = dict(punet=dataset_root(data_root, 'PUNet', punet_root),
                 pcnet=dataset_root(data_root, 'PCNet', pcnet_root))
    discovered, ignored = [], []
    for directory in xyz_directories(other):
        tags = {m.group(1).lower() for part in directory.relative_to(other).parts
                for m in re.finditer(r'(?:^|[_-])(lap|cov)$', part, re.I)}
        if not tags:
            ignored.append(str(directory))
            continue
        if len(tags) != 1:
            raise ValueError(f'Ambiguous lap/cov dataset labels: {directory}')
        family = 'punet_laplace' if tags == {'lap'} else 'punet_anisotropic'
        discovered.append((family, directory, *parse_scene(directory, other)))
    for directory in xyz_directories(pc_noisy):
        discovered.append(('pcnet', directory, *parse_scene(directory, pc_noisy)))
    scenes, seen, hash_cache = [], set(), {}
    for family, noisy_dir, points, noise in discovered:
        key = family, points, noise
        if key in seen:
            raise ValueError(f'Duplicate test condition {key}; select a unique noisy root')
        seen.add(key)
        source = roots['pcnet' if family == 'pcnet' else 'punet']
        clean_dir = source / 'pointclouds/test' / f'{points}_poisson'
        mesh_dir = source / 'meshes/test'
        if not clean_dir.is_dir():
            raise FileNotFoundError(f'Missing clean point clouds: {clean_dir}')
        clean = {p.stem: p.resolve() for p in clean_dir.glob('*.xyz')}
        noisy = {p.stem: p.resolve() for p in noisy_dir.glob('*.xyz')}
        if not clean or set(clean) != set(noisy):
            raise ValueError(f'Incomplete clean/noisy pairing in {noisy_dir}: '
                             f'missing noisy={sorted(set(clean) - set(noisy))}, '
                             f'missing clean={sorted(set(noisy) - set(clean))}')
        files = list(clean.values()) + list(noisy.values())
        for name in sorted(clean):
            mesh = mesh_dir / f'{name}.off'
            if not mesh.is_file():
                raise FileNotFoundError(f'Missing P2M mesh: {mesh}')
            files.append(mesh.resolve())
        hashes = {}
        for path in files:
            key_path = str(path)
            if key_path not in hash_cache:
                hash_cache[key_path] = file_sha256(path)
            hashes[key_path] = hash_cache[key_path]
        # Preserve the exact noise value in the identifier (no integer-percent rounding).
        label = f'{points}_poisson_{noise:g}'
        scenes.append(dict(dataset=family, condition=f'{family}__{label}', points=points,
                           noise=noise, noisy_dir=str(noisy_dir.resolve()),
                           clean_root=str(clean_dir.parent), mesh_root=str(mesh_dir.parent),
                           expected_clouds=len(clean), expected_names=sorted(clean),
                           input_files_sha256=hashes, evaluate_sor_both=family != 'pcnet',
                           measure_sor_efficiency=family != 'pcnet' and (points, noise) in EFFICIENCY_CASES,
                           measure_efficiency=(points, noise) in EFFICIENCY_CASES))
    for family in FAMILIES:
        for points, noise in EFFICIENCY_CASES:
            if (family, points, noise) not in seen:
                raise ValueError(f'Missing required efficiency scene: {family} '
                                 f'{condition_label(points, noise)}; check the noisy directory names')
    return sorted(scenes, key=lambda s: (FAMILIES.index(s['dataset']), s['points'], s['noise'])), ignored


def scene_job(scene, variant, config):
    job = copy.deepcopy(scene)
    job['variant'] = variant
    job['config'] = copy.deepcopy(config)
    job['config']['dataset']['_base_'].update(
        TEST_NOISY_PATH=job['noisy_dir'], TEST_CLEAN_PATH=job['clean_root'],
        TEST_MESH_ROOT=job['mesh_root'], TEST_RESOLUTION=f'{job["points"]}_poisson',
        TEST_NOISE=job['noise'])
    return job


def verify_inputs(job):
    for path, expected in job['input_files_sha256'].items():
        if file_sha256(path) != expected:
            raise ValueError(f'Data changed after the plan was created: {path}')
    for root in (Path(job['noisy_dir']), Path(job['clean_root']) / f'{job["points"]}_poisson'):
        if {p.stem for p in root.glob('*.xyz')} != set(job['expected_names']):
            raise ValueError(f'Point cloud membership changed: {root}')


def validate_paired_timings(core, with_sor):
    """A SOR interval must extend the matching inference interval, not replace it."""
    index = {(r['name'], int(r['repeat'])): r for r in core}
    keys = [(r['name'], int(r['repeat'])) for r in with_sor]
    if len(index) != len(core) or len(set(keys)) != len(keys) or set(keys) != set(index):
        raise ValueError('SOR and inference timing observations must match one-to-one')
    for row, key in zip(with_sor, keys):
        original = index[key]
        core_ms, total_ms = float(original['latency_ms']), float(row['latency_ms'])
        if not math.isfinite(total_ms) or not math.isfinite(core_ms) or not 0 < core_ms <= total_ms:
            raise ValueError('SOR total time must include the entire paired inference interval')
        if (int(row['allocated_before_bytes']) != int(original['allocated_before_bytes'])
                or int(row['input_points']) != int(original['input_points'])
                or any(int(row[k]) < int(original[k])
                       for k in ('peak_allocated_bytes', 'peak_reserved_bytes'))):
            raise ValueError('Paired SOR memory/input records are inconsistent')
    return index


def read_sor_result(output, job, manifest):
    import json
    quality = read_csv(output / 'test_metrics_sor.csv')
    saved = json.loads((output / 'summary_sor.json').read_text(encoding='utf-8'))
    measured = job.get('measure_sor_efficiency', False)
    timing = read_csv(output / 'timing_runs_sor.csv') if measured else []
    actual = summarize(timing, quality, job['points'], manifest['settings']['repeats'],
                       expected_clouds=job['expected_clouds'],
                       measure_efficiency=measured, allow_filtered_output=True)
    if measured:
        validate_paired_timings(read_csv(output / 'timing_runs.csv'), timing)
        first_counts = {r['name']: int(r['output_points']) for r in timing if int(r['repeat']) == 0}
        if any(int(r['output_points']) != first_counts[r['name']] for r in quality):
            raise ValueError(f'SOR quality must use the actual first timed output: {output}')
    if not saved.get('complete') or saved != dict(complete=True, **actual):
        raise ValueError(f'Incomplete or inconsistent SOR results: {output}')
    if {r['name'] for r in quality} != set(job['expected_names']):
        raise ValueError(f'SOR cloud names differ from the plan: {output}')
    if any(not math.isclose(float(r['sigma0']), job['noise'], abs_tol=1e-7) for r in quality):
        raise ValueError(f'SOR noise level differs from the plan: {output}')
    protocol = manifest['quality_protocol'].get('sor_comparison', {})
    if protocol != dict(modes=['off', 'on'], same_prediction=True, nb_neighbors=20,
                        std_ratio=2.0, timing_includes_sor=measured):
        raise ValueError(f'SOR protocol mismatch: {output}')
    return actual, quality, timing


def validate_result(root, job, settings):
    output = root / job['variant']['id'] / job['condition']
    result = read_result(output, job, settings)
    if set(result[1]['input_sha256']) != set(job['expected_names']):
        raise ValueError(f'Incomplete test set: {output}')
    if job['evaluate_sor_both']:
        read_sor_result(output, job, result[1])
    return result


def build_report(root, plan):
    rows, per_cloud, timings, environments = [], [], [], []
    for job in plan['jobs']:
        summary, manifest, quality, raw_timing = validate_result(root, job, plan['settings'])
        if environments and manifest['environment'] != environments[0]:
            raise ValueError('Hardware/software/precision settings changed between jobs')
        if not environments:
            environments.append(manifest['environment'])
        output = root / job['variant']['id'] / job['condition']
        core_index = {(r['name'], int(r['repeat'])): r for r in raw_timing}
        versions = [(False, summary, quality, raw_timing)]
        if job['evaluate_sor_both']:
            sor_summary, sor_quality, sor_timing = read_sor_result(output, job, manifest)
            versions.append((True, sor_summary, sor_quality, sor_timing))
        for sor, stats, cloud_rows, mode_timing in versions:
            row = dict(dataset=job['dataset'], condition=job['condition'], sor_enabled=sor,
                       variant='teacher_4', points=job['points'], noise=job['noise'], steps=4,
                       checkpoint_epoch=manifest['checkpoint_epoch'], parameters=manifest['parameters'],
                       parameters_m=manifest['parameters'] / 1e6,
                       measure_efficiency=job['measure_efficiency']
                       and (not sor or job.get('measure_sor_efficiency', False)),
                       timing_scope='whole_cloud_inference_plus_sor' if sor else 'whole_cloud_inference')
            row.update({k: v for k, v in stats.items() if k not in ('round_mean_ms', 'complete')})
            measured = row['measure_efficiency']
            row['latency_mean_seconds'] = stats['latency_mean_ms'] / 1000 if measured else None
            row['latency_std_seconds'] = stats['latency_std_ms'] / 1000 if measured else None
            row['latency_mean_std_seconds'] = (
                f'{row["latency_mean_seconds"]:.3f} ± {row["latency_std_seconds"]:.3f}' if measured else '')
            row['core_latency_mean_seconds'] = summary['latency_mean_ms'] / 1000 if measured else None
            row['sor_overhead_mean_seconds'] = (statistics.mean(
                (float(r['latency_ms']) - float(core_index[(r['name'], int(r['repeat']))]['latency_ms'])) / 1000
                for r in mode_timing) if measured else None)
            rows.append(row)
            per_cloud.extend(dict(dataset=job['dataset'], condition=job['condition'],
                                  sor_enabled=sor, variant='teacher_4', **q) for q in cloud_rows)
            for observation in mode_timing:
                core_ms = float(core_index[(observation['name'], int(observation['repeat']))]['latency_ms'])
                timings.append(dict(dataset=job['dataset'], condition=job['condition'],
                                    variant='teacher_4', sor_enabled=sor, sor_in_timing=sor,
                                    core_latency_ms=core_ms,
                                    sor_overhead_ms=float(observation['latency_ms']) - core_ms, **observation))
    efficiency = [r for r in rows if r['measure_efficiency']]
    expected_efficiency = 10 if plan.get('schema_version', 1) >= 2 else 6
    if len(efficiency) != expected_efficiency:
        raise ValueError(f'Expected exactly {expected_efficiency} efficiency results')
    write_csv(root / 'summary.csv', rows)
    write_csv(root / 'efficiency.csv', efficiency)
    write_csv(root / 'per_cloud.csv', per_cloud)
    write_csv(root / 'timing_runs.csv', timings)
    write_json(root / 'summary.json', dict(complete=True, settings=plan['settings'],
                                          environment=environments[0], results=rows))
    lines = ['# Teacher4 generalization', '',
             'CD/P2M are scaled by 1e4 (lower is better). SOR on/off share the same prediction.',
             'SOR-off: whole-cloud inference. SOR-on: inference + SOR including its CPU/GPU transfers.',
             'Metrics/I/O/external transfers are excluded. ± is the population SD of all measured total intervals.',
             'SOR GPU peaks do not include CPU memory used by Open3D.',
             'No Teacher10 is measured in this run; no speedup is inferred.', '',
             '| Dataset | Scene | SOR | Clouds | CD | P2M | Seconds/cloud | Peak allocated GiB |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        peak = f'{row["peak_allocated_gib"]:.4f}' if row['measure_efficiency'] else '—'
        lines.append(f'| {row["dataset"]} | {row["points"]}, {row["noise"]:.1%} | '
                     f'{"on" if row["sor_enabled"] else "off"} | {row["clouds"]} | '
                     f'{row["cd_x1e4"]:.4f} | {row["p2m_x1e4"]:.4f} | '
                     f'{row["latency_mean_std_seconds"] or "—"} | {peak} |')
    (root / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return rows
