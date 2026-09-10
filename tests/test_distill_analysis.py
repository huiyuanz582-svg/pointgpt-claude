import ast
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import runner_distill as runner
import analyze_curriculum_difficulty as difficulty


class EpsilonModel(torch.nn.Module):
    def __init__(self, weight=0.2):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(weight))
        self.calls = []

    def forward(self, x, clean=None, type='val', name='', noise_std=None):
        assert clean is None and type == 'val'
        self.calls.append((x.detach().clone(), noise_std.detach().clone(), torch.is_grad_enabled()))
        return x + noise_std[:, None, None] * self.weight * (x + 1)


def normalize_baseline():
    source = ROOT / 'tools/runner_finetune.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'normalize_unit_sphere')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['normalize_unit_sphere']


class DifficultyTests(unittest.TestCase):
    def test_full_trajectory_is_frozen_and_keeps_all_seventeen_states(self):
        teacher = EpsilonModel()
        noisy = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3).requires_grad_()
        sigma = torch.tensor([0.01, 0.02])
        states = difficulty.capture_full_teacher(teacher, noisy, sigma, patch_batch=1)
        self.assertEqual(tuple(states.shape), (17, 2, 3, 3))
        self.assertEqual(len(teacher.calls), 32)
        self.assertFalse(states.requires_grad)
        self.assertFalse(teacher.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in teacher.parameters()))
        self.assertTrue(all(not call[2] for call in teacher.calls))
        torch.testing.assert_close(states[0], noisy.detach(), rtol=0, atol=0)
        x = noisy.detach().clone()
        for step in range(16):
            x = x + 0.3 * sigma[:, None, None] * 0.95 ** step * 0.2 * (x + 1)
            torch.testing.assert_close(states[step + 1], x)
        # 与训练中只保留五个节点的 Teacher 捕获数值一致。
        nodes = runner.capture_teacher(teacher, noisy, sigma, patch_batch=1)
        torch.testing.assert_close(states[list(runner.TEACHER_NODES)], nodes, rtol=0, atol=0)

    def test_all_stages_are_teacher_forced_with_exact_raw_metrics(self):
        start = torch.arange(1, 19, dtype=torch.float32).reshape(2, 3, 3) / 10
        states = torch.stack([start * (0.9 ** step) for step in range(17)]).requires_grad_()
        clean = torch.zeros_like(start).requires_grad_()
        sigmas = torch.tensor([0.01, 0.02])
        student = EpsilonModel(weight=9.0)  # 强烈偏离 Teacher，暴露误用 Student 连续输出。
        rows, predictions = difficulty.analyze_stages(student, states, clean, sigmas)
        self.assertEqual(len(rows), 8)
        self.assertEqual(tuple(predictions.shape), (4, 2, 3, 3))
        self.assertFalse(predictions.requires_grad)
        self.assertEqual(float(student.weight), 9.0)
        for stage, (first, last) in enumerate(zip((0, 4, 8, 12), (4, 8, 12, 16))):
            call_x, call_sigma, grad = student.calls[stage]
            torch.testing.assert_close(call_x, states[first].detach(), rtol=0, atol=0)
            torch.testing.assert_close(call_sigma, sigmas * .95 ** first)
            self.assertFalse(grad)
            if stage:
                self.assertFalse(torch.allclose(call_x, predictions[stage - 1]))
            for index in range(2):
                row = next(row for row in rows if row['stage'] == stage and row['batch_sample'] == index)
                x, target = states[first, index].detach(), states[last, index].detach()
                expected = x + sigmas[index] * .95 ** first * 9 * (x + 1)
                d_move = (x - target).square().sum(-1).mean().item()
                d_remain = x.square().sum(-1).mean().item()
                e_imit = (expected - target).square().sum(-1).mean().item()
                self.assertAlmostEqual(row['D_move'], d_move, places=6)
                self.assertAlmostEqual(row['D_remain'], d_remain, places=6)
                self.assertAlmostEqual(row['E_imit'], e_imit, places=6)
                self.assertAlmostEqual(row['R_relative'], d_move / (d_remain + 1e-12), places=6)
        self.assertIsNone(states.grad)
        self.assertIsNone(clean.grad)
        self.assertIsNone(student.weight.grad)

    def test_zero_remaining_distance_and_summary_mean_of_individual_ratios(self):
        states = torch.zeros(17, 2, 3, 3)
        rows, _ = difficulty.analyze_stages(EpsilonModel(), states, torch.zeros(2, 3, 3), 0.01)
        self.assertTrue(all(row['D_remain'] == 0 and row['R_relative'] == 0 for row in rows))
        rows[0]['R_relative'], rows[1]['R_relative'] = 1.0, 3.0
        with tempfile.TemporaryDirectory() as directory:
            difficulty.write_stage_summary(Path(directory), rows)
            with (Path(directory) / 'stage_summary.csv').open() as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual(float(summary[0]['mean_R_relative']), 2.0)


class BaselineMetricTests(unittest.TestCase):
    def ops(self):
        calls = []
        def cd(prediction, clean):
            calls.append(('cd', prediction.clone(), clean.clone()))
            distances = torch.cdist(prediction, clean).square()
            return distances.min(-1).values.mean() + distances.min(-2).values.mean()
        def p2m(world, name, split):
            calls.append(('p2m', world.clone(), name, split))
            return world.square().mean()
        def sor(points):
            calls.append(('sor', points.clone()))
            return points[:-1].cpu()
        def project(points, **kwargs):
            calls.append(('project', points.clone(), kwargs))
            return points + 0.02
        return dict(cd=cd, p2m=p2m, normalize=normalize_baseline(), sor=sor, project=project,
                    mesh_root='test_mesh_root'), calls

    def test_normalization_scaling_and_postprocessing_match_baseline_order(self):
        prediction = torch.tensor([[0., 0., 0.], [1., 0., 0.], [10., 10., 10.]])
        clean = torch.tensor([[[-1., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
        center, scale = torch.tensor([[3., -2., 5.]]), torch.tensor([[2.]])
        config = SimpleNamespace(sor_enable=True, surface_projection={'enable': True, 'k': 8,
                                                                     'num_iters': 2, 'blend': 0.6})
        ops, calls = self.ops()
        world, metrics = runner.evaluate_baseline_metrics(prediction, clean, center, scale,
                                                          'shape_a', config, ops)
        expected_world = (prediction[:-1] + 0.02) * scale + center
        torch.testing.assert_close(world, expected_world)
        self.assertEqual([call[0] for call in calls], ['sor', 'project', 'p2m', 'cd'])
        self.assertEqual(calls[1][2], {'k': 8, 'num_iters': 2, 'blend': 0.6})
        self.assertEqual(calls[2][2:], ('shape_a', 'test'))
        clean_world = clean * scale + center
        _, c, s = ops['normalize'](clean_world)
        torch.testing.assert_close(calls[3][1], (expected_world[None] - c) / s)
        torch.testing.assert_close(calls[3][2], (clean_world - c) / s)
        expected_distances = torch.cdist((expected_world[None] - c) / s, (clean_world - c) / s).square()
        expected_cd = (expected_distances.min(-1).values.mean() + expected_distances.min(-2).values.mean()) * 1e4
        self.assertAlmostEqual(metrics['cd_x1e4'], float(expected_cd), places=4)
        self.assertAlmostEqual(metrics['p2m_x1e4'], float(expected_world.square().mean() * 1e4), places=4)

    def test_no_postprocessing_and_equal_cloud_weighting(self):
        prediction = torch.tensor([[0., 0., 0.], [1., 1., 1.]])
        config = SimpleNamespace(sor_enable=False, surface_projection={'enable': False})
        ops, calls = self.ops()
        world, _ = runner.evaluate_baseline_metrics(prediction, prediction[None], torch.zeros(1, 3),
                                                    torch.ones(1, 1), 'shape', config, ops)
        torch.testing.assert_close(world, prediction)
        self.assertEqual([call[0] for call in calls], ['p2m', 'cd'])
        with tempfile.TemporaryDirectory() as directory:
            rows = [dict(cd_x1e4=2., p2m_x1e4=4., output_points=10),
                    dict(cd_x1e4=6., p2m_x1e4=8., output_points=100)]
            summary = runner.write_test_summary(Path(directory), rows, {}, 2, 20)
            self.assertEqual(summary['mean_cd_x1e4'], 4.)
            self.assertEqual(summary['mean_p2m_x1e4'], 6.)
            self.assertTrue(summary['complete'])
            self.assertFalse(summary['full_dataset'])

    def test_test_entry_writes_metrics_and_keeps_raw_trajectory_separate(self):
        class EasyDict(dict):
            __getattr__ = dict.__getitem__
        noisy = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
        loader = [(noisy, noisy.clone(), None, [torch.zeros(1, 3)], [torch.ones(1, 1)], ['shape'])]
        config = SimpleNamespace(model={}, dataset=SimpleNamespace(_base_={'TEST_NOISE': .01}),
                                 inference_patch_size=3, seed_ratio=1, test_patch_batch=1,
                                 fuse_tau_ratio=.5, sor_enable=True, surface_projection={'enable': False})
        builder = SimpleNamespace(model_builder=lambda _: EpsilonModel(), load_model=lambda *_: None,
                                  dataset_builder=lambda *_: (None, loader))
        ops, _ = self.ops()
        trajectory = dict(global_states=noisy.expand(5, -1, -1).clone(),
                          patch_states=noisy.expand(5, -1, -1)[:, None].clone(),
                          patch_idx=torch.tensor([[0, 1, 2]]), fuse_weights=torch.ones(1, 3),
                          sigma_before=torch.tensor([.01 * .95 ** k for k in (0, 4, 8, 12)]))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.dict(sys.modules, {'easydict': SimpleNamespace(EasyDict=EasyDict)}), \
                    patch.object(runner, 'baseline_metric_ops', return_value=ops), \
                    patch.object(runner, 'infer_student', return_value=(noisy[0], trajectory)) as inference:
                runner.test(SimpleNamespace(max_shapes=1, save_trajectory=True), config, builder,
                            torch.device('cpu'), Path('student-best.pth'), output)
            self.assertEqual(inference.call_count, 1)
            summary = json.loads((output / 'test_summary.json').read_text())
            self.assertEqual(summary['processed_shapes'], 1)
            self.assertEqual(summary['protocol']['mode'], 'student_continuous_4_step_rollout')
            self.assertEqual(np.loadtxt(output / 'shape.xyz').shape, (2, 3))
            self.assertEqual(np.loadtxt(output / 'shape_raw.xyz').shape, (3, 3))
            with np.load(output / 'shape_trajectory.npz') as saved:
                self.assertEqual(saved['global_states'].shape[0], 5)
                self.assertFalse(bool(saved['postprocessed']))
            self.assertTrue((output / 'test_metrics.csv').is_file())
            self.assertIn('P2M=', (output / 'test.log').read_text())


if __name__ == '__main__':
    unittest.main()
