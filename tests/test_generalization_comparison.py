"""Manual CPU fixtures for the three-model suite; no model or CUDA dependency."""

import copy
import csv
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from utils.denoise_benchmark_report import read_csv, summarize, write_csv, write_json
from utils.teacher4_generalization import build_report, scene_job, validate_comparison_jobs


ENTRY = Path(__file__).resolve().parents[1] / 'tools/benchmark_generalization_comparison.py'
SPEC = importlib.util.spec_from_file_location('generalization_comparison_fixture', ENTRY)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def fixture_plan():
    variants = [dict(id=name, kind=kind, steps=steps) for name, kind, steps in (
        ('teacher_10', 'teacher', 10), ('student_baseline', 'student', 4),
        ('student_local_endpoint', 'student', 4))]
    jobs = []
    for family in ('punet_laplace', 'punet_anisotropic', 'pcnet'):
        for points in (10000, 50000):
            for noise in ((0.01, 0.02, 0.03) if family == 'pcnet' else (0.01, 0.02, 0.025, 0.03)):
                efficient = (points, noise) in ((10000, 0.01), (50000, 0.03))
                scene = dict(dataset=family, condition=f'{family}__{points}_{noise:g}', points=points,
                             noise=noise, noisy_dir=f'/fixture/{family}/{points}_{noise:g}',
                             clean_root=f'/fixture/{family}/pointclouds/test',
                             mesh_root=f'/fixture/{family}/meshes',
                             expected_names=['c'] if family == 'pcnet' else ['a', 'b'],
                             expected_clouds=1 if family == 'pcnet' else 2,
                             evaluate_sor_both=family != 'pcnet', measure_efficiency=efficient,
                             measure_sor_efficiency=efficient and family != 'pcnet')
                jobs.extend(dict(copy.deepcopy(scene), variant=v, config={}) for v in variants)
    return dict(schema_version=3, benchmark_mode='generalization_comparison',
                settings=dict(repeats=3), variants=variants, jobs=jobs)


def write_results(root, plan):
    fields = ['name', 'repeat', 'input_points', 'output_points', 'latency_ms',
              'allocated_before_bytes', 'peak_allocated_bytes', 'peak_reserved_bytes']
    identities = {'teacher_10': (1.0, 95), 'student_baseline': (3.0, 6),
                  'student_local_endpoint': (2.0, 1)}
    for job in plan['jobs']:
        name = job['variant']['id']
        cd, epoch = identities[name]
        output = root / name / job['condition']
        output.mkdir(parents=True)
        manifest = dict(job=job, settings=plan['settings'], checkpoint_epoch=epoch,
                        parameters=1000 + epoch, environment=dict(gpu_name='fixture'),
                        input_sha256={n: 'hash' for n in job['expected_names']},
                        mesh_sha256={n: 'mesh hash' for n in job['expected_names']}, quality_protocol={})
        for sor in ([False, True] if job['evaluate_sor_both'] else [False]):
            suffix = '_sor' if sor else ''
            quality = [dict(name=n, sigma0=job['noise'], input_points=job['points'],
                            output_points=job['points'] - (10 if sor else 0),
                            cd_x1e4=cd - (0.1 if sor else 0), p2m_x1e4=0.5 * cd)
                       for n in job['expected_names']]
            measured = job['measure_sor_efficiency'] if sor else job['measure_efficiency']
            timing = [dict(name=n, repeat=i, input_points=job['points'],
                           output_points=job['points'] - (10 if sor else 0),
                           latency_ms=cd * (1000, 2000, 3000)[i] + (100 if sor else 0),
                           allocated_before_bytes=100, peak_allocated_bytes=220 if sor else 200,
                           peak_reserved_bytes=300)
                      for n in job['expected_names'] for i in range(3) if measured]
            write_csv(output / f'test_metrics{suffix}.csv', quality)
            if not sor or measured:
                with (output / f'timing_runs{suffix}.csv').open('w', newline='', encoding='utf-8') as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(timing)
            stats = summarize(timing, quality, job['points'], 3, expected_clouds=job['expected_clouds'],
                              measure_efficiency=measured, allow_filtered_output=sor)
            write_json(output / f'summary{suffix}.json', dict(complete=True, **stats))
        if job['evaluate_sor_both']:
            manifest['quality_protocol']['sor_comparison'] = dict(
                modes=['off', 'on'], same_prediction=True, nb_neighbors=20, std_ratio=2.0,
                timing_includes_sor=job['measure_sor_efficiency'])
        write_json(output / 'manifest.json', manifest)


class ComparisonReportTests(unittest.TestCase):
    def test_actual_22_scenes_keep_three_models_and_114_quality_30_efficiency_rows(self):
        plan = fixture_plan()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_results(root, plan)
            rows = build_report(root, plan)
            self.assertEqual(len(plan['jobs']), 66)
            self.assertEqual(len(rows), 114)
            efficiency = read_csv(root / 'efficiency.csv')
            wide = read_csv(root / 'comparison.csv')
            self.assertEqual(len(efficiency), 30)
            self.assertEqual(len(wide), 38)
            self.assertEqual(sum(r['noise'] == '0.025' for r in wide), 8)
            for variant in plan['variants']:
                name = variant['id']
                self.assertEqual(sum(r['variant'] == name for r in efficiency), 10)
                self.assertTrue(all(r['steps'] == variant['steps'] for r in rows if r['variant'] == name))
            for row in wide:
                delta = 0.1 if row['sor_enabled'] == 'True' else 0
                self.assertAlmostEqual(float(row['teacher_10_cd_x1e4']), 1.0 - delta)
                self.assertAlmostEqual(float(row['student_baseline_cd_x1e4']), 3.0 - delta)
                self.assertAlmostEqual(float(row['student_local_endpoint_cd_x1e4']), 2.0 - delta)
                measured = (int(row['points']), float(row['noise'])) in ((10000, 0.01), (50000, 0.03))
                self.assertEqual(bool(row['student_baseline_latency_mean_seconds']), measured)
            self.assertEqual(len(read_csv(root / 'timing_runs.csv')), 162)
            self.assertEqual(len(read_csv(root / 'per_cloud.csv')), 210)

    def test_missing_quality_only_scene_and_changed_data_are_rejected(self):
        plan = fixture_plan()
        plan['jobs'] = [j for j in plan['jobs'] if not (
            j['variant']['id'] == 'student_local_endpoint'
            and j['dataset'] == 'pcnet' and j['points'] == 50000 and j['noise'] == 0.02)]
        with self.assertRaisesRegex(ValueError, 'identical data'):
            validate_comparison_jobs(plan)
        plan = fixture_plan()
        plan['jobs'][1]['expected_names'] = ['different']
        with self.assertRaisesRegex(ValueError, 'identical data'):
            validate_comparison_jobs(plan)

    def test_duplicate_scene_and_wrong_steps_are_rejected(self):
        plan = fixture_plan()
        plan['jobs'].append(copy.deepcopy(plan['jobs'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_comparison_jobs(plan)
        plan = fixture_plan()
        plan['variants'][0]['steps'] = 4
        with self.assertRaisesRegex(ValueError, 'Teacher10'):
            validate_comparison_jobs(plan)

    def test_loaded_input_fingerprint_must_match_across_models(self):
        plan = fixture_plan()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_results(root, plan)
            job = plan['jobs'][1]
            path = root / job['variant']['id'] / job['condition'] / 'manifest.json'
            manifest = comparison.benchmark.load_json(path)
            manifest['input_sha256']['a'] = 'changed normalized input'
            write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, 'Input/mesh data changed'):
                build_report(root, plan)


class CheckpointTests(unittest.TestCase):
    def saved(self, role):
        epoch = 6 if role == 'student_baseline' else 1
        variant = dict(id=role, checkpoint_validation='fixed_10to4', expected_epoch=epoch,
                       model_config={'NAME': 'fixture'})
        saved = dict(epoch=epoch, best_epoch=epoch, model_config=variant['model_config'],
                     distillation=copy.deepcopy(comparison.benchmark.SCHEDULE),
                     selection=dict(metric='val_rollout_score', value=2.0),
                     base_model={'step_condition.fixture': 'not a tensor'})
        if role == 'student_local_endpoint':
            saved.update(endpoint_finetune=dict(local_weight=1.0, endpoint_weight=1.0),
                         endpoint_finetune_protocol=dict(objective='local_endpoint'))
        return saved, variant

    def test_teacher_forced_metadata_may_be_omitted_and_both_roles_are_accepted(self):
        for role in ('student_baseline', 'student_local_endpoint'):
            saved, variant = self.saved(role)
            comparison.benchmark.validate_student(saved, variant)

    def test_wrong_role_schedule_or_missing_trained_condition_is_rejected(self):
        saved, variant = self.saved('student_baseline')
        saved['endpoint_finetune'] = {'local_weight': 1.0}
        with self.assertRaisesRegex(ValueError, 'Ordinary distillation'):
            comparison.benchmark.validate_student(saved, variant)
        saved, variant = self.saved('student_local_endpoint')
        saved['distillation']['teacher_nodes'] = [0, 4, 8, 12, 16]
        with self.assertRaisesRegex(ValueError, 'schedule'):
            comparison.benchmark.validate_student(saved, variant)
        saved, variant = self.saved('student_baseline')
        saved['base_model'] = {'backbone': 'not a tensor'}
        with self.assertRaisesRegex(ValueError, 'trained step condition'):
            comparison.benchmark.validate_student(saved, variant)

    def test_checkpoint_hash_rejects_replaced_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'ckpt-best.pth'
            checkpoint.write_bytes(b'frozen fixture')
            expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            variant = comparison.student_variant(checkpoint, 'student_baseline', expected, 6, {})
            self.assertEqual(variant['checkpoint_sha256'], expected)
            checkpoint.write_bytes(b'replaced')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                comparison.student_variant(checkpoint, 'student_baseline', expected, 6, {})


class PlanTests(unittest.TestCase):
    def test_plan_uses_10_4_4_steps_and_all_scenes_without_deserializing_checkpoints(self):
        full = fixture_plan()
        scenes = [{k: v for k, v in j.items() if k not in ('variant', 'config')}
                  for j in full['jobs'] if j['variant']['id'] == 'teacher_10']
        base = dict(schema_version=2, settings={}, variants=[full['variants'][0]],
                    ignored_other_noise_directories=[],
                    jobs=[scene_job(s, full['variants'][0], dict(model={}, dataset={'_base_': {}}))
                          for s in scenes])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vanilla, method = root / 'vanilla.pth', root / 'method.pth'
            vanilla.write_bytes(b'vanilla'); method.write_bytes(b'method')
            base['variants'][0]['checkpoint_sha256'] = 'teacher identity'
            args = SimpleNamespace(vanilla_ckpt=vanilla, method_ckpt=method,
                                   expected_vanilla_sha256=hashlib.sha256(b'vanilla').hexdigest(),
                                   expected_method_sha256=hashlib.sha256(b'method').hexdigest(), plan_only=True)
            with mock.patch.object(comparison.generalization, 'make_plan', return_value=base), \
                    mock.patch.object(comparison.benchmark, 'check_saved_checkpoints') as deserialize:
                plan = comparison.make_plan(args)
                validate_comparison_jobs(plan)
                comparison.generalization.execute_plan(args, plan, ENTRY, 'fixture', 'unused.md')
                deserialize.assert_not_called()
            self.assertEqual([v['steps'] for v in plan['variants']], [10, 4, 4])
            self.assertEqual(len(plan['jobs']), 66)


if __name__ == '__main__':
    unittest.main()
