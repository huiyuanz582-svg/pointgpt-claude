"""Exact global path selection and cached, no-grad model evaluation tests."""
import copy
import importlib.util
import math
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_distill_16to4 import ROOT, ToyEpsilonModel, runner
from test_distill_shadow_search import RandomBatchNormModel

spec = importlib.util.spec_from_file_location('shadow_global_search_tests', ROOT / 'tools/shadow_global_search.py')
global_search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(global_search)


def metrics(pcd):
    return dict(PCD=pcd, D_move=1., E_imit=pcd)


class GlobalShadowTests(unittest.TestCase):
    def setUp(self):
        self.cache = {edge: metrics(3.) for edge in global_search.TEACHER_INTERVALS}

    def test_counts_all_complete_paths_and_only_unique_legal_edges(self):
        self.assertEqual(len(global_search.TEACHER_PATHS), 455)
        self.assertEqual(len(global_search.TEACHER_INTERVALS), 130)
        self.assertNotIn((0, 16), global_search.TEACHER_INTERVALS)
        for nodes in global_search.TEACHER_PATHS:
            self.assertEqual((nodes[0], nodes[-1]), (0, 16))
            self.assertTrue(all(t < u for t, u in zip(nodes[:-1], nodes[1:])))

    def test_last_edge_changes_global_choice_instead_of_forced_greedy_tail(self):
        for edge in ((0, 4), (4, 8), (8, 12)):
            self.cache[edge] = metrics(.3)
        self.cache[12, 16] = metrics(2.)
        better = (0, 5, 9, 13, 16)
        for edge in zip(better[:-1], better[1:]):
            self.cache[edge] = metrics(.4)
        result = global_search.search_global_teacher_nodes(self.cache, .3, 1., return_path_scores=True)
        self.assertEqual(result['nodes'], list(better))
        self.assertAlmostEqual(result['path_score'], .1)
        self.assertEqual(len(result['candidate_paths']), 455)
        self.assertLess(result['path_score'], global_search.score_teacher_path((0,4,8,12,16), self.cache, .3, 1.)['path_score'])
        self.cache[12, 16] = metrics(.3)
        self.assertEqual(global_search.search_global_teacher_nodes(self.cache, .3, 1.)['nodes'], [0,4,8,12,16])

    def test_lambda_balance_is_configurable_and_population_std_is_used(self):
        a, b = (0,4,8,12,16), (0,5,9,13,16)
        for edge, value in zip(zip(a[:-1], a[1:]), [.3,.3,.3,.7]):
            self.cache[edge] = metrics(value)
        for edge in zip(b[:-1], b[1:]):
            self.cache[edge] = metrics(.45)
        self.assertEqual(global_search.search_global_teacher_nodes(self.cache, .3, 0.)['nodes'], list(a))
        self.assertEqual(global_search.search_global_teacher_nodes(self.cache, .3, 1.)['nodes'], list(b))
        score = global_search.score_teacher_path(a, self.cache, .3, 1.)
        self.assertAlmostEqual(score['pcd_std'], math.sqrt(.03))
        self.assertAlmostEqual(score['path_score'], .1 + math.sqrt(.03))

    def test_deterministic_ties_missing_edges_and_no_gradient_outputs(self):
        value = torch.tensor(.3, requires_grad=True)
        cache = {edge: metrics(value) for edge in global_search.TEACHER_INTERVALS}
        first = global_search.search_global_teacher_nodes(cache, .3, 1.)
        self.assertEqual(first, global_search.search_global_teacher_nodes(cache, .3, 1.))
        self.assertEqual(first['nodes'], [0,1,2,3,16])
        self.assertIsNone(value.grad)
        self.assertIsInstance(first['path_score'], float)
        cache.pop((3,16))
        with self.assertRaises(ValueError):
            global_search.search_global_teacher_nodes(cache, .3, 1.)
        with self.assertRaises(ValueError):
            global_search.search_global_teacher_nodes(self.cache, .3, -1.)

    def test_interval_cache_preserves_modes_rng_grads_and_avoids_repeat_forwards(self):
        student = RandomBatchNormModel().train()
        student.dropout.eval()
        states = torch.stack([torch.full((2,4,3), t * .01) for t in range(17)]).requires_grad_()
        sigma = torch.tensor([.01,.02])
        before = copy.deepcopy(student.state_dict())
        modes = [m.training for m in student.modules()]
        for p in student.parameters():
            p.grad = torch.ones_like(p)
        cpu_rng, numpy_rng, python_rng = torch.get_rng_state(), np.random.get_state(), random.getstate()
        with patch.object(global_search, 'import_module', return_value=runner):
            cache = global_search.build_interval_pcd_cache(student, states, sigma, patch_batch=2)
            self.assertEqual(cache['forward_calls'], 130)
            self.assertEqual(cache['patch_interval_evaluations'], 260)
            self.assertEqual(len(student.calls), 130)
            for call, (t,u) in zip(student.calls, global_search.TEACHER_INTERVALS):
                torch.testing.assert_close(call['x'], states[t].detach())
                torch.testing.assert_close(call['sigma'], sigma * .95 ** t)
                self.assertFalse(call['training'] or call['grad'])
            for prediction in cache['predictions'].values():
                self.assertFalse(prediction.requires_grad)
            with patch.object(runner, 'search_dynamic_teacher_target', wraps=runner.search_dynamic_teacher_target) as greedy:
                rows = global_search.compare_cached_shadow_paths(states, cache, .3, 1.)
                self.assertEqual(greedy.call_count, 8)
            self.assertEqual(len(student.calls), 130)
        self.assertEqual([m.training for m in student.modules()], modes)
        torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
        np.testing.assert_equal(np.random.get_state(), numpy_rng)
        self.assertEqual(random.getstate(), python_rng)
        self.assertIsNone(states.grad)
        for name, value in before.items():
            torch.testing.assert_close(student.state_dict()[name], value, rtol=0, atol=0)
        self.assertTrue(all(torch.equal(p.grad, torch.ones_like(p)) for p in student.parameters()))
        for row in rows:
            self.assertLessEqual(row['global_dynamic']['path_score'], row['fixed']['path_score'] + 1e-12)
            self.assertLessEqual(row['global_dynamic']['path_score'], row['greedy_dynamic']['path_score'] + 1e-12)
        summary = global_search.summarize_path_comparison(rows)
        for method in summary.values():
            self.assertTrue(all(sum(hist.values())==2 for hist in method['target_histogram']))

    def test_global_shadow_cannot_change_fixed_loss_or_gradient(self):
        teacher = ToyEpsilonModel()
        points = torch.randn(2,6,3)
        full = runner.capture_teacher(teacher, points, .01, return_full_trajectory=True)
        student = RandomBatchNormModel().train()
        reference = copy.deepcopy(student)
        torch.manual_seed(42); np.random.seed(42); random.seed(42)
        with patch.object(global_search, 'import_module', return_value=runner):
            cache = global_search.build_interval_pcd_cache(student, full, .01, 2)
            global_search.compare_cached_shadow_paths(full, cache, .3, 1.)
        losses = runner.backward_stages(student, full[list(runner.TEACHER_NODES)], .01)
        torch.manual_seed(42); np.random.seed(42); random.seed(42)
        expected = runner.backward_stages(reference, full[list(runner.TEACHER_NODES)], .01)
        self.assertEqual(losses, expected)
        for a,b in zip(student.parameters(), reference.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
