"""CPU shadow-search boundary and fixed-training isolation regression tests."""

import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_distill_16to4 import ToyEpsilonModel, runner


class RandomBatchNormModel(ToyEpsilonModel):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(3)
        self.dropout = torch.nn.Dropout(.25)

    def forward(self, x, clean=None, type='val', name='', noise_std=None):
        # Consume RNG even in eval, as FPS/grouping may do; shadow must restore it.
        torch.rand(1), np.random.rand(), random.random()
        extra = self.dropout(self.bn(x.transpose(1, 2))).transpose(1, 2)
        return super().forward(x, clean, type, name, noise_std) + .001 * extra


class ShadowSearchTests(unittest.TestCase):
    def setUp(self):
        self.states = torch.stack([torch.full((4, 3), float(t)) for t in range(17)])

    def test_exhaustive_search_minimizes_distance_instead_of_first_crossing(self):
        pred = torch.ones(4, 3, requires_grad=True)
        states = self.states.requires_grad_()
        with patch.object(runner, 'evaluate_candidate_pcd', wraps=runner.evaluate_candidate_pcd) as evaluate:
            result = runner.search_dynamic_teacher_target(states, pred, 0, 3, .3)
        # u=1 -> 0; u=2 -> 1/4; u=3 -> 4/9. Closest is u=2, not first crossing u=3.
        self.assertEqual(evaluate.call_count, 13)
        self.assertEqual(result['selected_target_step'], 2)
        self.assertEqual(result['selected_gap'], 2)
        self.assertEqual(result['stop_reason'], 'closest_pcd_target')
        self.assertAlmostEqual(result['selected_D_move'], 12.)
        self.assertAlmostEqual(result['selected_E_imit'], 3.)
        self.assertAlmostEqual(result['selected_PCD'], .25)
        self.assertAlmostEqual(result['pcd_distance_to_target'], .05)
        self.assertEqual([r['target_step'] for r in result['candidate_metrics']], list(range(1, 14)))
        equal = runner.search_dynamic_teacher_target(states, pred, 0, 3, .25)
        self.assertEqual(equal['selected_target_step'], 2)
        self.assertIsNone(states.grad)
        self.assertIsNone(pred.grad)
        self.assertTrue(torch.is_grad_enabled())
        self.assertTrue(all(not torch.is_tensor(value) for value in result.values()))

    def test_nonmonotonic_candidates_ignore_early_large_pcd_and_visit_all_conditions(self):
        values = {u: .7 for u in range(1, 14)}
        values.update({1: 4.08250, 2: .5, 3: .2, 4: .31462, 5: .1, 6: .32935})
        visited = []
        parameter = torch.nn.Parameter(torch.ones(4, 3))
        def predict(u):
            self.assertFalse(torch.is_grad_enabled())
            visited.append(u)
            return parameter * u
        def evaluate(start, target, prediction):
            u = int(target[0, 0])
            return dict(D_move=torch.tensor(float(u)), E_imit=torch.tensor(values[u] * u),
                        PCD=torch.tensor(values[u]))
        with patch.object(runner, 'evaluate_candidate_pcd', side_effect=evaluate):
            result = runner.search_dynamic_teacher_target(self.states, predict, 0, 3, .3)
        self.assertEqual(visited, list(range(1, 14)))
        self.assertEqual(result['selected_target_step'], 4)
        self.assertAlmostEqual(result['selected_PCD'], .31462, places=6)
        self.assertAlmostEqual(result['pcd_distance_to_target'], .01462, places=6)
        self.assertIsNone(parameter.grad)

    def test_equal_distances_select_smaller_target_and_do_not_skip_candidates(self):
        def evaluate(start, target, prediction):
            u = int(target[0, 0])
            return dict(D_move=torch.tensor(1.), E_imit=torch.tensor(0.),
                        PCD=torch.tensor(.125 if u == 3 else .375 if u == 5 else 2.))
        with patch.object(runner, 'evaluate_candidate_pcd', side_effect=evaluate) as mock:
            result = runner.search_dynamic_teacher_target(self.states, self.states[0], 0, 3, .25)
        self.assertEqual(mock.call_count, 13)
        self.assertEqual(result['selected_target_step'], 3)
        self.assertEqual(result['pcd_distance_to_target'], .125)

    def test_upper_bound_reserves_future_steps_and_final_stage_forces_t16(self):
        start = 0
        selected = [start]
        for stage in range(4):
            remaining = 3 - stage
            result = runner.search_dynamic_teacher_target(
                self.states, self.states[start] + 1, start, remaining, 2.)
            target = result['selected_target_step']
            self.assertLess(start, target)
            self.assertLessEqual(target, 16 - remaining)
            self.assertGreaterEqual(16 - target, remaining)
            self.assertEqual(result['stop_reason'], 'forced_final_target' if remaining == 0 else 'closest_pcd_target')
            self.assertTrue(all(start < r['target_step'] <= 16 - remaining
                                for r in result['candidate_metrics']))
            selected.append(target)
            start = target
        self.assertEqual(selected, [0, 13, 14, 15, 16])
        # Even a perfect PCD match at u=4 cannot override the final T16 constraint.
        last = runner.search_dynamic_teacher_target(self.states, self.states[3] + 1, 3, 0, 0.)
        self.assertEqual(last['selected_target_step'], 16)
        self.assertTrue(last['forced_final_target'])

    def test_invalid_intervals_cannot_overrun_teacher_or_exhaust_remaining_steps(self):
        for start, remaining in ((16, 0), (15, 1), (14, 2), (-1, 3)):
            with self.assertRaises(ValueError):
                runner.search_dynamic_teacher_target(self.states, self.states[0], start, remaining, .3)
        with self.assertRaises(ValueError):
            runner.search_dynamic_teacher_target(self.states, self.states[0], 0, 3, .3, teacher_end=17)
        with self.assertRaises(ValueError):
            runner._shadow_search_config(SimpleNamespace(pcd_dynamic=dict(shadow_enabled=True, threshold=-.1)))

    def test_full_teacher_cache_keeps_fixed_nodes_and_forward_count_unchanged(self):
        teacher = ToyEpsilonModel()
        noisy = torch.randn(3, 6, 3)
        sigma = torch.tensor([.01, .02, .03])
        fixed = runner.capture_teacher(teacher, noisy, sigma)
        count = len(teacher.calls)
        full = runner.capture_teacher(teacher, noisy, sigma, return_full_trajectory=True)
        self.assertEqual(count, 3 * 16)
        self.assertEqual(len(teacher.calls) - count, count)
        self.assertEqual(full.shape, (17, 3, 6, 3))
        self.assertFalse(full.requires_grad)
        torch.testing.assert_close(full[list(runner.TEACHER_NODES)], fixed, rtol=0, atol=0)

    def test_shadow_chains_teacher_starts_and_preserves_modes_gradients_and_rng(self):
        student = RandomBatchNormModel().train()
        student.dropout.eval()  # Preserve mixed module modes too.
        before = copy.deepcopy(student.state_dict())
        modes = [module.training for module in student.modules()]
        for p in student.parameters():
            p.grad = torch.ones_like(p)
        states = self.states[:, None].expand(-1, 2, -1, -1).clone().requires_grad_()
        sigma = torch.tensor([.01, .03])
        torch_rng, np_rng, py_rng = torch.get_rng_state(), np.random.get_state(), random.getstate()
        rows = runner.shadow_search_teacher_targets(student, states, sigma, .3, patch_batch=2)
        self.assertEqual(len(rows), 8)
        self.assertEqual(len(student.calls), 4)
        for stage in range(4):
            stage_rows = [row for row in rows if row['stage'] == stage]
            call = student.calls[stage]
            starts = [row['start_step'] for row in stage_rows]
            expected_x = torch.stack([states[t, i] for i, t in enumerate(starts)])
            torch.testing.assert_close(call['x'], expected_x.detach(), rtol=0, atol=0)
            torch.testing.assert_close(call['sigma'], sigma * torch.tensor([.95 ** t for t in starts]))
            self.assertFalse(call['grad'] or call['training'])
        self.assertEqual([module.training for module in student.modules()], modes)
        for key, value in student.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertIsNone(states.grad)
        self.assertTrue(all(torch.equal(p.grad, torch.ones_like(p)) for p in student.parameters()))
        torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
        np.testing.assert_equal(np.random.get_state(), np_rng)
        self.assertEqual(random.getstate(), py_rng)
        summary = runner.summarize_shadow_search(rows)
        self.assertEqual(summary['closest_pcd_target_count'], [2, 2, 2, 0])
        self.assertEqual(summary['forced_final_target_count'], [0, 0, 0, 2])
        self.assertEqual(len(summary['shadow_debug_samples']), 8)
        for stage in range(4):
            stage_rows = [row for row in rows if row['stage'] == stage]
            self.assertEqual(sum(summary['shadow_target_hist'][stage].values()), 2)
            self.assertEqual(summary['mean_PCD_distance_to_target'][stage],
                             sum(row['pcd_distance_to_target'] for row in stage_rows) / 2)

    def test_shadow_enabled_training_matches_disabled_loss_weights_and_optimizer(self):
        config = SimpleNamespace(model={}, learning_rate=.001, weight_decay=0.,
                                 teacher_patch_batch=1, student_patch_batch=8, total_bs=8,
                                 test_patch_batch=1, epochs=1, grad_norm_clip=1.)
        args = SimpleNamespace(epochs=1, max_shapes=0, max_patch_batches=0)
        points = torch.arange(18, dtype=torch.float32).reshape(6, 3) / 20
        samples = [(points, points, .01, torch.zeros(1, 3), torch.ones(1, 1), 'cloud')] * 16
        loader = torch.utils.data.DataLoader(samples, batch_size=8, shuffle=True, drop_last=True)
        bank = [(torch.stack([points[None] + t * .001 for t in range(5)]), torch.tensor([.01]))]
        builder = SimpleNamespace(model_builder=lambda _: RandomBatchNormModel(), load_model=lambda *_: None)
        with tempfile.TemporaryDirectory() as directory:
            results, logs, gradients = [], [], []
            for enabled in (False, True):
                torch.manual_seed(123)
                np.random.seed(123)
                random.seed(123)
                config.pcd_dynamic = dict(shadow_enabled=enabled, threshold=.3, min_gap=1,
                                          teacher_end=16, student_steps=4)
                output = Path(directory) / str(enabled)
                output.mkdir()
                observed = []
                check = runner.check_gradients
                def check_grads(teacher, student):
                    observed.append([p.grad.clone() if p.grad is not None else None for p in student.parameters()])
                    return check(teacher, student)
                with patch.object(runner, '_train_loader', return_value=loader), \
                        patch.object(runner, '_validation_bank', return_value=(bank, {'split': 'synthetic'})), \
                        patch.object(runner, 'check_gradients', side_effect=check_grads):
                    runner.train(args, config, builder, torch.device('cpu'), Path('teacher.pth'), output)
                results.append(torch.load(output / 'ckpt-last.pth', weights_only=True))
                logs.append(json.loads((output / 'train.jsonl').read_text()))
                gradients.append(observed)
            for key in results[0]['base_model']:
                torch.testing.assert_close(results[0]['base_model'][key], results[1]['base_model'][key], rtol=0, atol=0)
            for first, second in zip(gradients[0], gradients[1]):
                for a, b in zip(first, second):
                    if a is None:
                        self.assertIsNone(b)
                    else:
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
            for key, state in results[0]['optimizer']['state'].items():
                for field, value in state.items():
                    torch.testing.assert_close(value, results[1]['optimizer']['state'][key][field], rtol=0, atol=0)
            for key in logs[0]:
                self.assertEqual(logs[0][key], logs[1][key])
            self.assertEqual(len(logs[1]['mean_PCD_distance_to_target']), 4)
            self.assertTrue(logs[1]['shadow_debug_samples'])
            for histogram in logs[1]['shadow_target_hist']:
                self.assertEqual(sum(histogram.values()), 16)
            self.assertEqual(runner.TEACHER_NODES, (0, 4, 8, 12, 16))


if __name__ == '__main__':
    unittest.main()
