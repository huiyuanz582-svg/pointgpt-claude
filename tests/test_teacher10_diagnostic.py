"""Manual-only checks for diagnostic gradient connectivity and state isolation."""

import importlib.util
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('teacher10_diagnostic', ROOT / 'tools/diagnose_teacher10_checkpoint.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.tensor([0.9, 0.8, 0.7, 0.6], dtype=torch.float64))
        self.bn = torch.nn.BatchNorm1d(3, dtype=torch.float64)

    def forward(self, points, sigma, first, last):
        return points * self.weights[(0, 2, 5, 7).index(first)] + sigma[:, None, None] * 0.01


def forward(student, points, sigma, first, last):
    return student(points, sigma, first, last)


RUNNER = SimpleNamespace(forward_student_interval=forward, TEACHER_DECAY=0.95)


class DiagnosticTests(unittest.TestCase):
    def test_best_last_selection_checks_last_epoch_against_completed_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(epochs=3, training_input=dict(type='scheduled_rollout'))
            schedule = dict(teacher_nodes=[0, 2, 5, 7, 10])
            manifest = dict(config=config, arguments=dict(epochs=None), schedule=schedule)
            (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            rows = [dict(epoch=epoch, val_rollout_score=score) for epoch, score in [(1, 3), (2, 1), (3, 2)]]
            (root / 'train.jsonl').write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
            for name in ('ckpt-best.pth', 'ckpt-last.pth'):
                (root / name).write_bytes(b'fixture')
            def load(path, **kwargs):
                return dict(epoch=2 if path.name == 'ckpt-best.pth' else 3,
                            distillation=schedule, training_input=config['training_input'], best_epoch=2)
            loader = SimpleNamespace(load=load)
            _, best = audit.inspect_run(root, loader)
            _, last = audit.inspect_run(root, loader, 'last')
            self.assertEqual(best['epoch'], 2)
            self.assertEqual(last['epoch'], 3)
            self.assertEqual(last['validation_score'], 2)
            self.assertTrue(last['checkpoint'].endswith('ckpt-last.pth'))
            (root / 'train.jsonl').write_text('\n'.join(json.dumps(row) for row in rows[:-1]), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'completed training log'):
                audit.inspect_run(root, loader, 'last')

    def test_bn_swap_changes_eval_output_and_restores_buffers_after_failure(self):
        model = Toy().eval()
        original = audit.batchnorm_buffers(model, torch)
        swapped = {name: tensor.clone() for name, tensor in original.items()}
        swapped['bn.running_mean'].fill_(0.5)
        swapped['bn.running_var'].fill_(4.0)
        swapped['bn.num_batches_tracked'].fill_(30)
        parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        points = torch.ones((2, 3, 4), dtype=torch.float64)
        before = model.bn(points).detach()
        with self.assertRaisesRegex(RuntimeError, 'probe failure'):
            with audit.isolated_mode(model, 'eval', 42, torch, np):
                audit.copy_batchnorm_buffers(model, swapped, torch)
                with torch.no_grad():
                    actual = model.bn(points)
                expected = torch.full_like(points, 0.5 / (4.0 + model.bn.eps) ** 0.5)
                torch.testing.assert_close(actual, expected)
                self.assertFalse(torch.allclose(actual, before))
                raise RuntimeError('probe failure')
        for name, value in audit.batchnorm_buffers(model, torch).items():
            torch.testing.assert_close(value, original[name])
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, parameters[name])
            self.assertIsNone(parameter.grad)
        self.assertFalse(model.training)

    def test_bn_summary_pairs_four_cases_and_decomposes_interaction(self):
        rows = []
        for (weights, bn), errors in {('best', 'best'): [1.0, 3.0], ('best', 'last'): [3.0, 5.0],
                                      ('last', 'best'): [4.0, 6.0], ('last', 'last'): [8.0, 10.0]}.items():
            for patch, error in enumerate(errors):
                rows.append(dict(condition='10k_1pct', stage=4, name='cloud', patch=patch,
                                 weights_checkpoint=weights, bn_checkpoint=bn, free_to_canonical_mse=error))
        summary = audit.summarize_best_last_bn(rows)[0]
        self.assertEqual(summary['total_last_minus_best'], 7.0)
        self.assertEqual(summary['bn_effect_at_best_weights'], 2.0)
        self.assertEqual(summary['bn_effect_at_last_weights'], 4.0)
        self.assertEqual(summary['weights_effect_at_best_bn'], 3.0)
        self.assertEqual(summary['weights_effect_at_last_bn'], 5.0)
        self.assertEqual(summary['interaction'], 2.0)
        self.assertEqual(summary['patches'], 2)
        with self.assertRaisesRegex(ValueError, 'identical patches'):
            audit.summarize_best_last_bn(rows[:-1])
        with self.assertRaisesRegex(ValueError, 'Duplicate patch'):
            audit.summarize_best_last_bn(rows + rows[:1])

    def test_bias_mask_detects_only_consecutive_conv_bn(self):
        model = torch.nn.Module()
        model.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(3, 4, 1), torch.nn.BatchNorm1d(4),
            torch.nn.Conv1d(4, 4, 1), torch.nn.ReLU(), torch.nn.BatchNorm1d(4))
        self.assertEqual(audit.pre_bn_conv_biases(model, torch), ['encoder.0.bias'])

    def test_virtual_mask_rebalances_remaining_parameters_without_mutation(self):
        bias = 'encoder.first_conv.0.bias'
        local = {bias: torch.tensor([0.0]), 'blocks.weight': torch.tensor([3.0, 4.0])}
        endpoint = {bias: torch.tensor([1000.0]), 'blocks.weight': torch.tensor([3.0, 4.0])}
        before = {key: value.clone() for key, value in endpoint.items()}
        original = audit.virtual_endpoint_balance(local, endpoint, (), 0.1, 1.0, torch)
        masked = audit.virtual_endpoint_balance(local, endpoint, [bias], 0.1, 1.0, torch)
        self.assertAlmostEqual(masked['effective_weight'], 0.1)
        self.assertAlmostEqual(masked['groups']['blocks']['weighted_endpoint_to_local_ratio'], 0.1)
        self.assertLess(original['groups']['blocks']['weighted_endpoint_to_local_ratio'], 0.001)
        self.assertAlmostEqual(masked['combined_norm_relative_to_original_local'], 1.1)
        self.assertAlmostEqual(masked['cosine_to_allowed_local'], 1.0)
        self.assertFalse(masked['update_applied'])
        self.assertTrue(masked['forward_unchanged'])
        for key, value in endpoint.items():
            torch.testing.assert_close(value, before[key])

    def test_full_and_last_only_share_outputs_but_have_different_early_gradients(self):
        model = Toy().eval()
        nodes = torch.arange(180, dtype=torch.float64).reshape(5, 3, 4, 3) / 200
        sigma = torch.tensor([0.005, 0.01, 0.02], dtype=torch.float64)
        full_loss, full_states = audit.endpoint_backward(model, nodes, sigma, 'full', RUNNER, torch)
        full_gradient = model.weights.grad.clone()
        self.assertTrue(torch.all(full_gradient[:3].abs() > 0))
        model.zero_grad(set_to_none=True)
        last_loss, last_states = audit.endpoint_backward(model, nodes, sigma, 'last_only', RUNNER, torch)
        self.assertAlmostEqual(full_loss, last_loss, places=12)
        torch.testing.assert_close(model.weights.grad[:3], torch.zeros(3, dtype=torch.float64))
        torch.testing.assert_close(model.weights.grad[3], full_gradient[3])
        self.assertTrue(all(value > 0 for value in full_states))
        self.assertEqual(last_states[:3], [0, 0, 0])

    def test_mode_buffers_and_rng_restore_after_failure(self):
        model = Toy().train()
        before = {name: buffer.clone() for name, buffer in model.named_buffers()}
        py_state, np_state = random.getstate(), np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, 'probe failure'):
            with audit.isolated_mode(model, 'bn_batch_stats', 42, torch, np):
                self.assertFalse(model.training)
                self.assertTrue(model.bn.training)
                model.bn(torch.ones((2, 3, 4), dtype=torch.float64))
                model.weights.grad = torch.ones_like(model.weights)
                random.random()
                np.random.rand()
                torch.rand(1)
                raise RuntimeError('probe failure')
        self.assertTrue(model.training)
        self.assertTrue(model.bn.training)
        self.assertIsNone(model.weights.grad)
        for name, buffer in model.named_buffers():
            torch.testing.assert_close(buffer, before[name])
        self.assertEqual(random.getstate(), py_state)
        self.assertEqual(np.random.get_state()[0], np_state[0])
        np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
        self.assertEqual(np.random.get_state()[2:], np_state[2:])
        torch.testing.assert_close(torch.random.get_rng_state(), torch_state)


if __name__ == '__main__':
    unittest.main()
