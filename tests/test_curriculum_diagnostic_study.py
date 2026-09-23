"""CPU configuration/artifact checks for expanded diagnostics; no model imports."""

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from utils.curriculum_diagnostic_study import resolve_study, seed_configuration, summarize_seeds, write_seed_summary


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'cfgs/PointGPT-L/distill_16to4_rollout_diagnostics.yaml'


def report(seed, value):
    path = [0, 4, 8, 12, 16]
    row = dict(path_id='P0', path=path, overlap_roles=['A', 'A_alias'], noise=.005,
               metric='E_FR', stage=4, start_step=12, target_step=16, normalization='teacher_endpoint',
               aggregation='pooled_patch_distribution', sample_count=16, total_patch_count=16, undefined_count=0,
               mean=value, median=value, p90=value + 10, p95=value + 20, value=None)
    return dict(status='completed', nonfinite_count=0,
                model_state=dict(id='fixed-model', kind='mock', checkpoint='student.pth', teacher_checkpoint='teacher.pth',
                                 seed=0, calibration_seed=seed, current_checkpoint_nodes=path),
                noise_levels=[.005, .01, .02], eps=1e-12,
                paths=[dict(path_id='P0', path=path, path_roles=['A', 'A_alias'])],
                rows=[dict(row, path_role=role) for role in ('A', 'A_alias')])


class DiagnosticStudyTests(unittest.TestCase):
    def setUp(self):
        self.raw = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))

    def test_defaults_and_real_bank_seed_override_do_not_mutate_base(self):
        raw_before = copy.deepcopy(self.raw)
        study = resolve_study(self.raw)
        self.assertEqual(study['calibration_seeds'], [2025, 2026, 2027])
        self.assertEqual(list(study['paths']), ['A', 'B', 'C', 'D'])
        self.assertEqual(study['expected_cost']['student_forwards_per_seed'], 672)
        self.assertEqual(study['expected_cost']['student_forwards_total'], 2016)
        base = yaml.safe_load((ROOT / study['base_config']).read_text(encoding='utf-8'))
        before = copy.deepcopy(base)
        for seed in study['calibration_seeds']:
            cfg, resolved = seed_configuration(base, study, seed)
            self.assertEqual(cfg['curriculum_calibration']['seed'], seed)
            self.assertEqual(resolved['calibration']['seed'], seed)
            self.assertEqual(resolved['calibration']['patches_per_level'], 16)
            self.assertEqual(resolved['search']['patch_batch'], 2)
        self.assertEqual(self.raw, raw_before)
        self.assertEqual(base, before)
        self.assertEqual(base['curriculum_calibration']['patches_per_level'], 2)

    def test_limits_cli_overrides_and_partial_micro_batches(self):
        study = resolve_study(self.raw, calibration_seeds=[7, 8], patches_per_level=17, patch_batch=8)
        self.assertEqual(study['expected_cost']['micro_batches_per_seed'], 9)
        self.assertEqual(study['expected_cost']['student_forwards_total'], 504)
        for key, value in [('calibration_seeds', []), ('calibration_seeds', [7, 7]),
                           ('calibration_seeds', [2 ** 32]), ('calibration_seeds', [True]),
                           ('calibration_seeds', list(range(6))), ('patches_per_level', 33),
                           ('patch_batch', 0), ('patch_batch', 9), ('patches_per_level', True),
                           ('noise_levels', [.005, .01, .03]), ('paths', {})]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                resolve_study(dict(self.raw, **{key: value}))
        with self.assertRaises(ValueError):
            resolve_study(dict(self.raw, typo=1))

    def test_seed_summary_deduplicates_roles_and_does_not_pool_quantiles(self):
        reports = [report(2025, 1.), report(2026, 3.), report(2027, 5.)]
        summary = summarize_seeds(reports, [2025, 2026, 2027])
        self.assertEqual(len(summary['rows']), 1)
        row = summary['rows'][0]
        self.assertEqual(row['seed_count'], 3)
        self.assertEqual(row['mean_across_seeds'], 3.)
        self.assertAlmostEqual(row['std_across_seeds'], (8 / 3) ** .5)
        self.assertEqual([r['sample_count'] for r in row['seed_values']], [16, 16, 16])
        self.assertEqual(row['source_statistic'], 'mean')
        self.assertNotIn('p90', row)
        self.assertNotIn('p95', row)

    def test_incomplete_mismatched_failed_and_nonfinite_runs_fail(self):
        valid = [report(1, 1.), report(2, 3.)]
        cases = [valid[:1], [valid[0], valid[0]]]
        for mutate in (lambda r: r.update(status='failed'),
                       lambda r: r['model_state'].update(checkpoint='different.pth'),
                       lambda r: r.update(rows=[]),
                       lambda r: r['rows'][0].update(mean=99.),
                       lambda r: [row.update(mean=float('inf')) for row in r['rows']]):
            changed = copy.deepcopy(valid)
            mutate(changed[1])
            cases.append(changed)
        for reports in cases:
            with self.subTest(reports=reports), self.assertRaises((ValueError, FloatingPointError)):
                summarize_seeds(reports, [1, 2])

    def test_undefined_shares_stay_null_and_record_defined_seed_counts(self):
        reports = [report(1, 1.), report(2, 3.)]
        for index, entry in enumerate(reports):
            for row in entry['rows']:
                row.update(metric='first_stage_A_TF_share', stage=None, start_step=None, target_step=None,
                           aggregation='summary_of_four_stage_patch_means', mean=None, median=None, p90=None, p95=None,
                           value=None if index == 0 else .5, undefined_count=16 if index == 0 else 0)
        row = summarize_seeds(reports, [1, 2])['rows'][0]
        self.assertEqual(row['source_statistic'], 'value')
        self.assertEqual(row['mean_across_seeds'], .5)
        self.assertEqual(row['defined_seed_count'], 1)
        self.assertEqual(row['undefined_seed_count'], 1)
        for entry in reports:
            for row in entry['rows']:
                row.update(value=None, undefined_count=16)
        row = summarize_seeds(reports, [1, 2])['rows'][0]
        self.assertIsNone(row['mean_across_seeds'])
        self.assertIsNone(row['std_across_seeds'])

    def test_summary_artifacts_preserve_seed_values_and_refuse_overwrite(self):
        summary = summarize_seeds([report(1, 1.), report(2, 3.)], [1, 2])
        with tempfile.TemporaryDirectory() as directory:
            files = write_seed_summary(directory, summary)
            saved = json.loads((Path(directory) / files['json']).read_text(encoding='utf-8'))
            self.assertEqual(saved, summary)
            with (Path(directory) / files['csv']).open(encoding='utf-8', newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(json.loads(rows[0]['seed_values']), summary['rows'][0]['seed_values'])
            with self.assertRaises(FileExistsError):
                write_seed_summary(directory, summary)


if __name__ == '__main__':
    unittest.main()
