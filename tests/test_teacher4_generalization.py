"""Manual CPU checks; no model, training data, checkpoint or CUDA dependency."""

from pathlib import Path
import csv
import importlib.util
import statistics
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from utils.denoise_benchmark_report import read_csv, summarize, write_csv, write_json
from utils.teacher4_generalization import (
    build_report, discover_scenes, validate_paired_timings, verify_inputs)


def fixture(root):
    for family, names in [('PUNet', ('a', 'b')), ('PCNet', ('c',))]:
        source = root / ('ScoreDenoise' if family == 'PUNet' else '') / family
        for points in (10000, 50000):
            clean = source / 'pointclouds/test' / f'{points}_poisson'
            clean.mkdir(parents=True)
            for name in names:
                (clean / f'{name}.xyz').write_text('0 0 0\n', encoding='utf-8')
        meshes = source / 'meshes/test'
        meshes.mkdir(parents=True)
        for name in names:
            (meshes / f'{name}.off').write_text('fixture mesh\n', encoding='utf-8')
    for points in (10000, 50000):
        for noise in (0.01, 0.02, 0.03):
            for tag in ('lap', 'cov', 'pcnet'):
                if tag == 'pcnet':
                    directory = root / 'PCNet/noisy' / f'PCNet_{points}_poisson_{noise:.2f}'
                    names = ('c',)
                else:
                    directory = root / 'other-noise' / f'PUNet_{points}_poisson_{noise:.2f}_{tag}'
                    names = ('a', 'b')
                directory.mkdir(parents=True)
                for name in names:
                    (directory / f'{name}.xyz').write_text('0.01 0 0\n', encoding='utf-8')


class DiscoveryTests(unittest.TestCase):
    def test_all_scenes_and_only_six_timing_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            scenes, ignored = discover_scenes(root)
            self.assertEqual(len(scenes), 18)
            self.assertEqual(sum(s['measure_efficiency'] for s in scenes), 6)
            self.assertEqual(sum(s['measure_sor_efficiency'] for s in scenes), 4)
            self.assertEqual(sum(2 if s['evaluate_sor_both'] else 1 for s in scenes), 30)
            self.assertFalse(ignored)
            self.assertEqual({s['expected_clouds'] for s in scenes if s['dataset'] == 'pcnet'}, {1})
            for scene in scenes:
                verify_inputs(scene)

    def test_missing_pair_is_rejected_instead_of_intersection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            (root / 'other-noise/PUNet_10000_poisson_0.02_lap/b.xyz').unlink()
            with self.assertRaisesRegex(ValueError, 'Incomplete clean/noisy pairing'):
                discover_scenes(root)

    def test_duplicate_scene_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            duplicate = root / 'other-noise/duplicate/PUNet_10000_poisson_0.01_cov'
            duplicate.mkdir(parents=True)
            for name in ('a', 'b'):
                (duplicate / f'{name}.xyz').write_text('0 0 0\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Duplicate test condition'):
                discover_scenes(root)

    def test_missing_mesh_stops_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            (root / 'PCNet/meshes/test/c.off').unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'Missing P2M mesh'):
                discover_scenes(root)

    def test_data_change_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            scenes, _ = discover_scenes(root)
            (Path(scenes[0]['noisy_dir']) / 'a.xyz').write_text('1 1 1\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Data changed'):
                verify_inputs(scenes[0])


class SummaryTests(unittest.TestCase):
    def quality(self, output=10000):
        return [dict(name='a', input_points=10000, output_points=output, cd_x1e4=2, p2m_x1e4=1)]

    def test_quality_only_accepts_sor_deletion_and_leaves_timing_empty(self):
        result = summarize([], self.quality(9900), 10000, 0, expected_clouds=1,
                           measure_efficiency=False, allow_filtered_output=True)
        self.assertEqual(result['clouds'], 1)
        self.assertEqual(result['timing_observations'], 0)
        self.assertIsNone(result['latency_mean_ms'])
        self.assertIsNone(result['peak_allocated_gib'])
        with self.assertRaisesRegex(ValueError, 'point counts'):
            summarize([], self.quality(0), 10000, 0, expected_clouds=1,
                      measure_efficiency=False, allow_filtered_output=True)

    def test_legacy_default_still_requires_twenty_clouds(self):
        with self.assertRaisesRegex(ValueError, '20 unique clouds'):
            summarize([], self.quality(), 10000, 3)

    def test_population_standard_deviation_and_peak(self):
        timing = [dict(name='a', repeat=i, input_points=10000, output_points=10000,
                       latency_ms=ms, allocated_before_bytes=100, peak_allocated_bytes=200 + i,
                       peak_reserved_bytes=300) for i, ms in enumerate((1000, 2000, 3000))]
        result = summarize(timing, self.quality(), 10000, 3, expected_clouds=1)
        self.assertEqual(result['latency_mean_ms'], 2000)
        self.assertAlmostEqual(result['latency_std_ms'], (2000000 / 3) ** 0.5)
        self.assertEqual(result['peak_allocated_gib'], 202 / 1024 ** 3)
        timing[2]['repeat'] = 1
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            summarize(timing, self.quality(), 10000, 3, expected_clouds=1)

    def test_sor_timing_allows_deleted_points_and_uses_total_interval_sd(self):
        core = [dict(name='a', repeat=i, input_points=10000, output_points=10000,
                     latency_ms=ms, allocated_before_bytes=100, peak_allocated_bytes=200,
                     peak_reserved_bytes=300) for i, ms in enumerate((1000, 2000, 3000))]
        timed = [dict(r, output_points=9900 - i, latency_ms=ms, peak_allocated_bytes=220)
                 for i, (r, ms) in enumerate(zip(core, (1100, 2400, 3300)))]
        validate_paired_timings(core, timed)
        result = summarize(timed, self.quality(9900), 10000, 3, expected_clouds=1,
                           allow_filtered_output=True)
        self.assertAlmostEqual(result['latency_mean_ms'], 6800 / 3)
        self.assertAlmostEqual(result['latency_std_ms'], statistics.pstdev((1100, 2400, 3300)))
        self.assertNotAlmostEqual(result['latency_std_ms'],
                                  statistics.pstdev((1000, 2000, 3000)) + statistics.pstdev((100, 400, 300)))
        with self.assertRaisesRegex(ValueError, 'point counts'):
            summarize(timed, self.quality(9900), 10000, 3, expected_clouds=1)
        with self.assertRaisesRegex(ValueError, 'one-to-one'):
            validate_paired_timings(core, timed[:-1])
        timed[0]['latency_ms'] = 100
        with self.assertRaisesRegex(ValueError, 'entire paired inference'):
            validate_paired_timings(core, timed)


class TimingBoundaryTests(unittest.TestCase):
    def test_sor_total_uses_same_start_and_waits_for_gpu_output(self):
        path = Path(__file__).resolve().parents[1] / 'tools/benchmark_teacher10_six.py'
        spec = importlib.util.spec_from_file_location('teacher4_timing_fixture', path)
        benchmark = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(benchmark)
        events, peak = [], [200]

        class Cloud:
            def __init__(self, points):
                self.shape = (points, 3)

            def to(self, device):
                events.append('SOR output to GPU')
                return self

        noisy, prediction, filtered = Cloud(10000), Cloud(10000), Cloud(9900)

        def infer(cloud):
            self.assertIs(cloud, noisy)
            events.append('infer')
            return prediction

        def sor(cloud):
            self.assertIs(cloud, prediction)
            events.append('SOR input to CPU and filtering')
            peak[0] = 220
            return filtered

        ticks = iter((1_000_000, 3_000_000, 3_800_000))

        def clock():
            events.append('clock')
            return next(ticks)

        cuda = SimpleNamespace(synchronize=lambda device: events.append('sync'),
                               memory_allocated=lambda device: 100,
                               reset_peak_memory_stats=lambda device: events.append('reset'),
                               max_memory_allocated=lambda device: peak[0],
                               max_memory_reserved=lambda device: 300)
        with mock.patch.object(benchmark.time, 'perf_counter_ns', side_effect=clock):
            raw, off, kept, on = benchmark.timed_inference(SimpleNamespace(cuda=cuda), infer,
                                                          noisy, 'cuda:0', sor=sor)
        self.assertIs(raw, prediction)
        self.assertIs(kept, filtered)
        self.assertEqual(off['latency_ms'], 2.0)
        self.assertEqual(on['latency_ms'], 2.8)
        self.assertEqual(off['peak_allocated_bytes'], 200)
        self.assertEqual(on['peak_allocated_bytes'], 220)
        self.assertEqual(on['output_points'], 9900)
        self.assertEqual(events, ['sync', 'reset', 'clock', 'infer', 'sync', 'clock',
                                  'SOR input to CPU and filtering', 'SOR output to GPU', 'sync', 'clock'])


class ReportTests(unittest.TestCase):
    def test_full_suite_has_ten_efficiency_rows_and_actual_sor_timings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            jobs, _ = discover_scenes(root)
            result_root = root / 'result'
            settings = dict(repeats=3)
            for job in jobs:
                job['variant'] = dict(id='teacher_4')
                output = result_root / 'teacher_4' / job['condition']
                output.mkdir(parents=True)
                quality = [dict(name=name, sigma0=job['noise'], input_points=job['points'],
                                output_points=job['points'], cd_x1e4=2, p2m_x1e4=1)
                           for name in job['expected_names']]
                core = [dict(name=name, repeat=i, input_points=job['points'], output_points=job['points'],
                             latency_ms=ms, allocated_before_bytes=100, peak_allocated_bytes=200,
                             peak_reserved_bytes=300)
                        for name in job['expected_names'] for i, ms in enumerate((1000, 2000, 3000))
                        if job['measure_efficiency']]
                write_csv(output / 'test_metrics.csv', quality)
                fields = ['name', 'repeat', 'input_points', 'output_points', 'latency_ms',
                          'allocated_before_bytes', 'peak_allocated_bytes', 'peak_reserved_bytes']
                with (output / 'timing_runs.csv').open('w', newline='', encoding='utf-8') as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(core)
                summary = summarize(core, quality, job['points'], 3, expected_clouds=job['expected_clouds'],
                                    measure_efficiency=job['measure_efficiency'])
                write_json(output / 'summary.json', dict(complete=True, **summary))
                manifest = dict(job=job, settings=settings, checkpoint_epoch=1, parameters=360457679,
                                environment=dict(gpu_name='fixture'), quality_protocol={},
                                input_sha256={name: 'input hash' for name in job['expected_names']},
                                mesh_sha256={name: 'mesh hash' for name in job['expected_names']})
                if job['evaluate_sor_both']:
                    sor_quality = [dict(q, output_points=job['points'] - 10, cd_x1e4=1.8) for q in quality]
                    sor_timing = [dict(r, output_points=job['points'] - 10,
                                       latency_ms=(1100, 2400, 3300)[r['repeat']], peak_allocated_bytes=220)
                                  for r in core]
                    write_csv(output / 'test_metrics_sor.csv', sor_quality)
                    if sor_timing:
                        write_csv(output / 'timing_runs_sor.csv', sor_timing)
                    sor_summary = summarize(sor_timing, sor_quality, job['points'], 3,
                                            expected_clouds=job['expected_clouds'],
                                            measure_efficiency=job['measure_sor_efficiency'],
                                            allow_filtered_output=True)
                    write_json(output / 'summary_sor.json', dict(complete=True, **sor_summary))
                    manifest['quality_protocol']['sor_comparison'] = dict(
                        modes=['off', 'on'], same_prediction=True, nb_neighbors=20, std_ratio=2.0,
                        timing_includes_sor=job['measure_sor_efficiency'])
                write_json(output / 'manifest.json', manifest)
            rows = build_report(result_root, dict(schema_version=2, settings=settings, jobs=jobs))
            efficiency = read_csv(result_root / 'efficiency.csv')
            self.assertEqual(len(rows), 30)
            self.assertEqual(len(efficiency), 10)
            self.assertEqual(sum(r['sor_enabled'] == 'True' for r in efficiency), 4)
            for row in rows:
                if row['measure_efficiency'] and row['sor_enabled']:
                    self.assertAlmostEqual(row['latency_mean_seconds'], 6.8 / 3)
                    self.assertAlmostEqual(row['sor_overhead_mean_seconds'], 0.8 / 3)
                    self.assertEqual(row['timing_scope'], 'whole_cloud_inference_plus_sor')
                elif not row['measure_efficiency']:
                    self.assertIsNone(row['latency_mean_seconds'])
            # 4 PUNet scenes x 2 clouds x 3 repeats x 2 modes + 2 PCNet scenes x 1 cloud x 3 repeats.
            self.assertEqual(len(read_csv(result_root / 'timing_runs.csv')), 54)


if __name__ == '__main__':
    unittest.main()
