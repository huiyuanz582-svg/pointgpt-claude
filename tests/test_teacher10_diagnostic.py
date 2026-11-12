"""Manual-only checks for diagnostic gradient connectivity and state isolation."""

import importlib.util
from pathlib import Path
import random
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
