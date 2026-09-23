"""Synthetic CPU-only stage diagnostics; no training or real model imports."""
import copy
import csv
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from test_rollout_curriculum import MockStudent, forward, rollout, settings, synthetic_bank
from utils.curriculum_diagnostics import diagnostic_paths, summarize_path, compact_table


REFERENCE = (0, 4, 8, 12, 16)
CURRENT = (0, 7, 10, 13, 16)
SELECTED = (0, 10, 12, 14, 16)
MODEL_STATE = dict(id='synthetic-cpu-state', checkpoint=None, kind='mock')


class StageDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def run_diagnostics(self, *, current=REFERENCE, selected=REFERENCE, bank=None, model=None, output=None):
        return rollout.diagnose_stage_paths(
            model if model is not None else MockStudent(), synthetic_bank() if bank is None else bank,
            settings([REFERENCE]), forward, current=current, selected=selected, epoch=2,
            model_state=MODEL_STATE, output=output, print_table=False)

    def rows(self, report, metric, noise='all_noise', role='fixed_reference', aggregation='pooled_patch_distribution'):
        return [row for row in report['rows'] if row['metric'] == metric and row['noise'] == noise and
                row['path_role'] == role and row['aggregation'] == aggregation]

    def test_three_roles_deduplicated_and_fixed_present_outside_search_candidates(self):
        model = MockStudent()
        report = self.run_diagnostics(model=model)
        self.assertEqual(len(report['paths']), 1)
        self.assertEqual(report['paths'][0]['path_roles'], ['fixed_reference', 'current', 'selected'])
        self.assertEqual(model.calls, 7)
        self.assertEqual(report['tf_forward_calls'], 4)
        self.assertEqual(report['fr_additional_forward_calls'], 3)
        self.assertEqual(report['shared_first_stage_calls'], 1)
        self.assertEqual({r['path_role'] for r in report['rows']}, {'fixed_reference', 'current', 'selected'})
        for metric in ('A_TF', 'M', 'C_TF', 'A_FR', 'E_FR', 'A_FR_minus_A_TF'):
            self.assertEqual([r['mean'] for r in self.rows(report, metric)],
                             [r['mean'] for r in self.rows(report, metric, role='selected')])
        paths = diagnostic_paths(CURRENT, SELECTED)
        self.assertEqual([p['path'] for p in paths], [list(REFERENCE), list(CURRENT), list(SELECTED)])
        config = settings([SELECTED])
        model = MockStudent()
        report = rollout.diagnose_stage_paths(model, synthetic_bank(), config, forward,
                                              current=CURRENT, selected=SELECTED, epoch=2,
                                              model_state=MODEL_STATE, print_table=False)
        self.assertEqual(len(report['paths']), 3)
        self.assertEqual(model.calls, 21)

    def test_stage_formulas_against_independent_manual_tf_and_fr(self):
        bank, model = synthetic_bank(), MockStudent()
        report = self.run_diagnostics(bank=bank, model=model)
        teacher, sigma0, ids = bank[0]
        with rollout.isolated_evaluation(model):
            previous = teacher[0]
            denominator = ((teacher[16] - teacher[0]) ** 2).sum(-1).mean(-1) + 1e-12
            for stage, (start, target) in enumerate(zip(REFERENCE[:-1], REFERENCE[1:]), 1):
                sigma = sigma0 * .95 ** start
                tf = forward(model, teacher[start], sigma, start, target)
                fr = forward(model, previous, sigma, start, target)
                a_tf = ((tf - teacher[target]) ** 2).sum(-1).mean(-1)
                move = ((teacher[start] - teacher[target]) ** 2).sum(-1).mean(-1)
                a_fr = ((fr - teacher[target]) ** 2).sum(-1).mean(-1)
                expected = dict(A_TF=a_tf, M=move, C_TF=a_tf / (move + 1e-12), A_FR=a_fr,
                                E_FR=a_fr / denominator, A_FR_minus_A_TF=a_fr - a_tf)
                for noise, mask in [(s, ids == g) for g, s in enumerate([.005,.01,.02])] + [('all_noise', torch.ones(6,dtype=torch.bool))]:
                    for name, values in expected.items():
                        row = next(r for r in self.rows(report, name, noise) if r['stage'] == stage)
                        values = values[mask].double()
                        for key, value in (('mean', values.mean()), ('median', torch.quantile(values,.5)),
                                           ('p90', torch.quantile(values,.9)), ('p95', torch.quantile(values,.95))):
                            self.assertAlmostEqual(row[key], float(value), places=10)
                        self.assertEqual(row['sample_count'], int(mask.sum()))
                previous = fr
        self.assertEqual(self.rows(report, 'A_FR_minus_A_TF')[0]['mean'], 0.)

    def test_named_study_paths_include_all_four_without_selection_and_share_aliases(self):
        named = dict(A=REFERENCE, B=CURRENT, C=SELECTED, D=(0, 13, 14, 15, 16), C_alias=SELECTED)
        model = MockStudent().train()
        model.bn.eval()
        state = copy.deepcopy(model.state_dict())
        modes = [m.training for m in model.modules()]
        rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
        report = rollout.diagnose_stage_paths(
            model, synthetic_bank(), settings([REFERENCE]), forward, named_paths=named,
            epoch=0, model_state=MODEL_STATE, print_table=False)
        self.assertEqual([p['path'] for p in report['paths']], [list(p) for p in list(named.values())[:4]])
        self.assertEqual(report['paths'][2]['path_roles'], ['C', 'C_alias'])
        self.assertEqual(report['student_forward_calls'], 28)
        self.assertEqual(report['tf_forward_calls'], 16)
        self.assertEqual(report['fr_additional_forward_calls'], 12)
        self.assertNotIn('selected', {r['path_role'] for r in report['rows']})
        self.assertFalse(report['changes_selection'])
        self.assertEqual(modes, [m.training for m in model.modules()])
        for key, value in state.items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(rng[0], torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(rng[1], random.getstate())
        np.testing.assert_equal(rng[2], random.get_state())
        for label, path in list(named.items())[:4]:
            single = self.run_diagnostics(current=path, selected=path)
            expected = self.rows(single, 'E_FR', role='current')
            self.assertEqual([r['mean'] for r in self.rows(report, 'E_FR', role=label)],
                             [r['mean'] for r in expected])

    def test_named_paths_reject_ambiguous_or_invalid_requests(self):
        for paths in ({}, {'A': [0, 4, 4, 12, 16]}, {'A': [0, 4.0, 8, 12, 16]}, {'': REFERENCE}):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                diagnostic_paths(named_paths=paths)
        with self.assertRaises(ValueError):
            diagnostic_paths(REFERENCE, SELECTED, named_paths={'A': REFERENCE})

    def test_summary_aggregation_order_and_pooled_all_noise(self):
        c = torch.tensor([[8.,0.,0.,0.], [0.,8.,0.,0.], [0.,0.,8.,0.]], dtype=torch.float64)
        a = torch.tensor([[1.,1.,1.,1.], [9.,1.,1.,1.], [0.,0.,0.,0.]], dtype=torch.float64)
        values = dict(A_TF=a, M=torch.ones_like(a), C_TF=c, A_FR=a * 2, E_FR=c * 2, A_FR_minus_A_TF=a)
        paths = diagnostic_paths(REFERENCE, REFERENCE)
        rows = summarize_path(paths[0], values, torch.tensor([0,0,1]), [.005,.01], epoch=2, model_state=MODEL_STATE)
        def row(metric, aggregation):
            return next(r for r in rows if r['metric'] == metric and r['aggregation'] == aggregation and
                        r['noise'] == 'all_noise' and r['path_role'] == 'fixed_reference')
        self.assertEqual(row('first_stage_A_TF_share', 'per_patch_four_stage_summary')['mean'], .5)
        self.assertEqual(row('first_stage_A_TF_share', 'per_patch_four_stage_summary')['undefined_count'], 1)
        self.assertAlmostEqual(row('first_stage_A_TF_share', 'summary_of_four_stage_patch_means')['value'], 10 / 16)
        self.assertAlmostEqual(row('C_TF_stage_std', 'per_patch_four_stage_summary')['mean'], float(c.std(-1,unbiased=False).mean()))
        self.assertAlmostEqual(row('C_TF_stage_std', 'summary_of_four_stage_patch_means')['value'], float(c.mean(0).std(unbiased=False)))
        stage1 = next(r for r in rows if r['metric'] == 'A_TF' and r['stage'] == 1 and r['noise'] == 'all_noise')
        self.assertAlmostEqual(stage1['mean'], 10 / 3)  # Pool patches, not equally weighted group means.
        self.assertEqual(stage1['median'], 1.)

    def test_diagnostic_does_not_change_scores_selection_rng_modes_parameters_or_gradients(self):
        model, bank = MockStudent().train(), synthetic_bank()
        model.bn.eval()
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        config = settings([REFERENCE, SELECTED])
        before_search = rollout.search_rollout_paths(model, bank, config, forward)
        frozen_search = copy.deepcopy(before_search)
        state = copy.deepcopy(model.state_dict())
        modes = [m.training for m in model.modules()]
        rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
        self.run_diagnostics(current=CURRENT, selected=tuple(before_search['nodes']), bank=bank, model=model)
        self.assertEqual(before_search, frozen_search)
        self.assertEqual(modes, [m.training for m in model.modules()])
        for key, value in state.items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        for p in model.parameters():
            self.assertTrue(torch.equal(p.grad, torch.ones_like(p)))
        torch.testing.assert_close(rng[0], torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(rng[1], random.getstate())
        np.testing.assert_equal(rng[2], np.random.get_state())
        after_search = rollout.search_rollout_paths(model, bank, config, forward)
        self.assertEqual(before_search['candidates'], after_search['candidates'])
        self.assertEqual(before_search['nodes'], after_search['nodes'])

    def test_json_csv_and_compact_log_contain_roles_all_metrics_and_model_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.run_diagnostics(output=directory)
            saved = json.loads((Path(directory) / 'curriculum_stage_diagnostics_epoch_002.json').read_text(encoding='utf-8'))
            with (Path(directory) / 'curriculum_stage_diagnostics_epoch_002.csv').open(encoding='utf-8',newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(saved['rows']), len(rows))
            self.assertEqual(saved['status'], 'completed')
            self.assertEqual({r['noise'] for r in saved['rows']}, {.005,.01,.02,'all_noise'})
            for row in rows:
                for key in ('epoch','path','path_role','noise','metric','normalization','sample_count','model_state_id'):
                    self.assertNotEqual(row[key], '')
            table = compact_table(report)
            self.assertIn('fixed_reference/current/selected', table)
            self.assertIn('Path | Noise | Metric | Stage1 | Stage2 | Stage3 | Stage4', table)
            for name in ('C_TF','A_TF','E_FR','A_FR'):
                self.assertIn(f'| all_noise | {name} |', table)
            with self.assertRaises(FileExistsError):
                self.run_diagnostics(output=directory)

    def test_failure_restores_mode_and_writes_failed_json_without_partial_csv(self):
        model = MockStudent().train()
        model.bn.eval()
        modes = [m.training for m in model.modules()]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FloatingPointError):
                rollout.diagnose_stage_paths(model, synthetic_bank(), settings([REFERENCE]),
                    lambda model,x,*args: x * float('nan'), current=REFERENCE, selected=REFERENCE,
                    epoch=2, model_state=MODEL_STATE, output=directory, print_table=False)
            report = json.loads((Path(directory) / 'curriculum_stage_diagnostics_epoch_002.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertGreater(report['nonfinite_count'], 0)
            self.assertEqual(report['rows'], [])
            self.assertFalse((Path(directory) / 'curriculum_stage_diagnostics_epoch_002.csv').exists())
        self.assertEqual(modes, [m.training for m in model.modules()])


if __name__ == '__main__':
    unittest.main()
