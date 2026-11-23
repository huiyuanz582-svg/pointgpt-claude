"""Manual CPU checks; no model, training data, checkpoint or CUDA dependency."""

from pathlib import Path
import tempfile
import unittest

from utils.denoise_benchmark_report import summarize
from utils.teacher4_generalization import discover_scenes, verify_inputs


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


if __name__ == '__main__':
    unittest.main()
