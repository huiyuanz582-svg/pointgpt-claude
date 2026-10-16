"""Analytic CPU-only target/gradient checks and a mock training integration.

Run only when authorized. No real PointGPT model, dataset or GPU is loaded.
"""
import importlib.util
import io
import json
import random
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('matched_target_runner', ROOT / 'tools/runner_distill.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ToyModel(torch.nn.Module):
    def __init__(self, consume_rng=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.8, dtype=torch.float64))
        self.bias = torch.nn.Parameter(torch.tensor(.1, dtype=torch.float64))
        self.step_condition = None
        self.consume_rng = consume_rng
        self.calls = []

    def enable_step_condition(self):
        self.step_condition = torch.nn.Identity()
        return self

    def forward(self, points, _clean, mode, _name, noise_std, start_step=None, target_step=None):
        self.calls.append(dict(points=points.detach().clone(), sigma=noise_std.detach().clone(),
                               grad_enabled=torch.is_grad_enabled(), input_requires_grad=points.requires_grad,
                               training=self.training, interval=(start_step, target_step)))
        if self.consume_rng:
            random.random()
            np.random.rand()
            torch.rand(1)
        if self.step_condition is None:
            return points + noise_std[:, None, None] * (self.weight * points + self.bias)
        return self.weight * points + self.bias


def sample_nodes():
    return (torch.arange(90, dtype=torch.float64).reshape(5, 3, 2, 3) / 100,
            torch.tensor([.005, .01, .02], dtype=torch.float64))


def analytic_target(points, sigma0, start, end):
    # Each affine Teacher update preserves the fixed point -b/a.
    factors = torch.stack([1 + .3 * .8 * sigma0 * .95 ** step for step in range(start, end)])
    return (points + .1 / .8) * factors.prod(0)[:, None, None] - .1 / .8


class MatchedTargetTests(unittest.TestCase):
    def test_teacher_interval_matches_closed_form_and_detaches(self):
        nodes, sigmas = sample_nodes()
        points = nodes[2].clone().requires_grad_()
        sigmas.requires_grad_()
        teacher = ToyModel()
        result = runner.teacher_interval_target(teacher, points, sigmas, 7, 10, patch_batch=2)
        torch.testing.assert_close(result, analytic_target(points.detach(), sigmas.detach(), 7, 10))
        self.assertFalse(result.requires_grad)
        self.assertEqual(len(teacher.calls), 6)  # 2 micro-batches x 3 steps.
        for call in teacher.calls:
            self.assertFalse(call['grad_enabled'])
            self.assertFalse(call['input_requires_grad'])
            self.assertFalse(call['training'])
        torch.testing.assert_close(teacher.calls[0]['sigma'], sigmas.detach()[:2] * .95 ** 7)
        torch.testing.assert_close(teacher.calls[2]['sigma'], sigmas.detach()[:2] * .95 ** 9)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in teacher.parameters()))
        self.assertIsNone(points.grad)
        self.assertIsNone(sigmas.grad)

    def test_extra_teacher_calls_restore_all_cpu_rngs_even_on_failure(self):
        nodes, sigmas = sample_nodes()
        py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()
        runner.teacher_interval_target(ToyModel(consume_rng=True), nodes[0], sigmas, 4, 8)
        self.assertEqual(random.getstate(), py_state)
        self.assertEqual(np.random.get_state()[0], np_state[0])
        np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
        self.assertEqual(np.random.get_state()[2:], np_state[2:])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))
        bad_teacher = ToyModel(consume_rng=True)
        def fail_after_random_draws(*args, **kwargs):
            random.random()
            np.random.rand()
            torch.rand(1)
            raise RuntimeError('synthetic failure')
        with patch.object(bad_teacher, 'forward', side_effect=fail_after_random_draws):
            with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                runner.teacher_interval_target(bad_teacher, nodes[0], sigmas, 4, 8)
        self.assertEqual(random.getstate(), py_state)
        np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
        self.assertEqual(np.random.get_state()[2:], np_state[2:])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))

    def test_all_teacher_sources_keep_original_loss_and_make_no_extra_calls(self):
        nodes, sigmas = sample_nodes()
        original, matched = ToyModel().enable_step_condition(), ToyModel().enable_step_condition()
        old_losses = runner.backward_stages(original, nodes, sigmas)
        teacher, diagnostics = ToyModel(), []
        new_losses = runner.backward_student_input_stages(
            matched, nodes, sigmas, torch.ones(4, 3, dtype=torch.bool),
            student_target='teacher_from_student_input', teacher=teacher,
            target_diagnostics=diagnostics)
        self.assertEqual(old_losses, new_losses)
        self.assertEqual(teacher.calls, [])
        self.assertEqual(diagnostics, [[3, 0, 0]] * 4)
        for left, right in zip(original.parameters(), matched.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)

    def test_mixed_sources_use_actual_states_and_analytic_local_gradients(self):
        nodes, sigmas = sample_nodes()
        original_nodes = nodes.clone()
        nodes.requires_grad_()
        sigmas.requires_grad_()
        mask = torch.tensor([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.bool)
        student, teacher, diagnostics = ToyModel().enable_step_condition(), ToyModel(), []
        losses = runner.backward_student_input_stages(
            student, nodes, sigmas, mask, student_target='teacher_from_student_input',
            teacher=teacher, target_diagnostics=diagnostics)
        previous, weight_grad, bias_grad = None, 0., 0.
        for stage, start in enumerate((0, 4, 8, 12)):
            x = nodes[stage].detach() if stage == 0 else torch.where(mask[stage, :, None, None], nodes[stage].detach(), previous)
            local = analytic_target(x, sigmas.detach(), start, start + 4)
            target = torch.where(mask[stage, :, None, None], nodes[stage + 1].detach(), local)
            prediction = .8 * x + .1
            residual = prediction - target
            self.assertAlmostEqual(losses[stage], float(residual.square().sum(-1).mean()))
            weight_grad += (2 * residual * x).sum(-1).mean() / 4
            bias_grad += (2 * residual).sum(-1).mean() / 4
            torch.testing.assert_close(student.calls[stage]['points'], x)
            self.assertFalse(student.calls[stage]['input_requires_grad'])
            previous = prediction
        torch.testing.assert_close(student.weight.grad, weight_grad)
        torch.testing.assert_close(student.bias.grad, bias_grad)
        self.assertIsNone(nodes.grad)
        self.assertIsNone(sigmas.grad)
        self.assertTrue(torch.equal(nodes.detach(), original_nodes))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertEqual(diagnostics, [[3, 0, 0], [1, 2, 8], [1, 2, 8], [1, 2, 8]])
        self.assertEqual(len(student.calls), 4)
        self.assertEqual(len(teacher.calls), 24)

    def test_microbatch_partition_preserves_gradients_and_target_counts(self):
        nodes, sigmas = sample_nodes()
        mask = torch.tensor([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.bool)
        whole, split = ToyModel().enable_step_condition(), ToyModel().enable_step_condition()
        diagnostics = []
        runner.backward_student_input_stages(whole, nodes, sigmas, mask,
            student_target='teacher_from_student_input', teacher=ToyModel(), target_diagnostics=diagnostics)
        summed = np.zeros((4, 3), dtype=np.int64)
        for first, last in ((0, 2), (2, 3)):
            part = []
            runner.backward_student_input_stages(split, nodes[:, first:last], sigmas[first:last], mask[:, first:last],
                loss_scale=(last-first)/3, student_target='teacher_from_student_input', teacher=ToyModel(),
                teacher_patch_batch=2, target_diagnostics=part)
            summed += np.asarray(part)
        for left, right in zip(whole.parameters(), split.parameters()):
            torch.testing.assert_close(left.grad, right.grad)
        np.testing.assert_array_equal(summed, diagnostics)

    def test_invalid_teacher_intervals_or_target_policies_fail(self):
        nodes, sigmas = sample_nodes()
        for start, end, batch in ((4, 4, 1), (-1, 4, 1), (0, 17, 1), (0, 4, 0), (True, 4, 1)):
            with self.subTest(interval=(start, end, batch)), self.assertRaises(ValueError):
                runner.teacher_interval_target(ToyModel(), nodes[0], sigmas, start, end, batch)
        student = ToyModel().enable_step_condition()
        for teacher in (None, student):
            with self.assertRaises(ValueError):
                runner.backward_student_input_stages(student, nodes, sigmas, torch.ones(4, 3, dtype=torch.bool),
                    student_target='teacher_from_student_input', teacher=teacher)
        with self.assertRaises(ValueError):
            runner.backward_student_input_stages(student, nodes, sigmas, torch.ones(4, 3, dtype=torch.bool), student_target='typo')
        with self.assertRaises(ValueError):
            runner.teacher_interval_target(ToyModel(), nodes[0] * float('nan'), sigmas, 4, 8)
        bad_teacher = ToyModel()
        with patch.object(bad_teacher, 'forward', side_effect=lambda points, *args, **kwargs: points * float('nan')):
            with self.assertRaises(FloatingPointError):
                runner.teacher_interval_target(bad_teacher, nodes[0], sigmas, 4, 8)

    def test_new_config_keeps_original_mixed20_controls(self):
        old = yaml.safe_load((ROOT / 'cfgs/PointGPT-L/distill_16to4_input_scheduled.yaml').read_text(encoding='utf-8'))
        new = yaml.safe_load((ROOT / 'cfgs/PointGPT-L/distill_16to4_input_matched.yaml').read_text(encoding='utf-8'))
        self.assertEqual(new['training_input'].pop('student_target'), 'teacher_from_student_input')
        old['rollout_val_interval'] = 1
        self.assertEqual(old, new)

    def test_mock_training_records_targets_and_keeps_checkpoint_inference_format(self):
        nodes, sigmas = sample_nodes()
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            nodes[0], nodes[0], sigmas, torch.zeros(3), torch.ones(3), torch.arange(3)), batch_size=3)
        config = SimpleNamespace(model={}, curriculum_mode='fixed', training_input={
            'type': 'student_rollout', 'student_target': 'teacher_from_student_input'},
            epochs=2, learning_rate=1e-3, weight_decay=0., grad_norm_clip=1.,
            student_patch_batch=2, teacher_patch_batch=1, test_patch_batch=1,
            rollout_val_interval=1, pcd_dynamic={'shadow_enabled': False})
        args = SimpleNamespace(epochs=None, resume=None, max_shapes=0, max_patch_batches=0, stop_after_epoch=None)
        teacher = ToyModel()
        builder = SimpleNamespace(model_builder=lambda _: teacher, load_model=Mock())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(runner, '_train_loader', return_value=loader), \
                    patch.object(runner, '_validation_bank', return_value=([(nodes, sigmas)], {'split': 'mock'})), \
                    patch.object(runner, 'capture_teacher', return_value=nodes), \
                    patch.object(runner, 'validate_trajectory', return_value={'val_loss_traj': .5}), \
                    patch.object(runner, 'validate_rollout', side_effect=[{'val_rollout_score': 2.}, {'val_rollout_score': 3.}]), \
                    redirect_stdout(io.StringIO()):
                runner.train(args, config, builder, torch.device('cpu'), Path('teacher.pth'), output)
            records = [json.loads(line) for line in (output / 'train.jsonl').read_text().splitlines()]
            self.assertEqual(records[-1]['best_epoch'], 1)
            for row in records:
                self.assertEqual(row['teacher_nodes'], [0, 4, 8, 12, 16])
                self.assertEqual(row['loss_traj_target'], 'input_matched_teacher_interval')
                self.assertEqual(row['val_loss_traj_target'], 'teacher_target_node')
                targets = row['training_target_stages']
                self.assertEqual([entry['student_input_teacher_target_count'] for entry in targets], [0, 3, 3, 3])
                self.assertEqual(sum(entry['extra_teacher_sample_steps'] for entry in targets), 36)
            checkpoint = torch.load(output / 'ckpt-best.pth', map_location='cpu')
            self.assertEqual(checkpoint['epoch'], 1)
            self.assertEqual(checkpoint['training_input']['student_target'], 'teacher_from_student_input')
            self.assertEqual(checkpoint['training_input_protocol']['target'], 'input_matched_teacher_interval')
            restored = ToyModel().enable_step_condition()
            runner.load_student_checkpoint(restored, output / 'ckpt-best.pth', builder, expected_nodes=runner.TEACHER_NODES)
            self.assertEqual(restored.distillation_teacher_nodes, runner.TEACHER_NODES)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in teacher.parameters()))


if __name__ == '__main__':
    unittest.main()
