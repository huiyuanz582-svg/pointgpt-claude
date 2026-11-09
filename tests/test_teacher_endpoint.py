"""Endpoint gradient-chain, balancing, micro-batch and state-isolation checks.

These tests are provided for manual use only; no PointGPT/CUDA imports.
"""

import copy
import math
import random
import unittest

import numpy as np
import torch

from utils.teacher_endpoint import (accumulate_teacher_endpoint, endpoint_grad_ratio,
                                    gradient_norm, resolve_teacher_endpoint)


def options(patch_batch=1, checkpointing=True):
    return dict(target_grad_ratio=0.1, max_weight=1.0, warmup_epochs=2,
                ramp_epochs=4, patch_batch=patch_batch, gradient_checkpointing=checkpointing)


class ToyStudent(torch.nn.Module):
    def __init__(self, consume_rng=False, batchnorm=False):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.tensor([0.9, 0.8, 0.7, 0.6], dtype=torch.float64))
        self.bias = torch.nn.Parameter(torch.tensor(0.05, dtype=torch.float64))
        self.bn = torch.nn.BatchNorm1d(3, dtype=torch.float64) if batchnorm else torch.nn.Identity()
        self.consume_rng = consume_rng

    def forward(self, points, sigma, start, stop):
        if self.consume_rng:
            random.random()
            np.random.rand()
            torch.rand(1)
        stage = (0, 2, 5, 7).index(start)
        points = self.bn(points.transpose(1, 2)).transpose(1, 2)
        return points * self.weights[stage] + self.bias * sigma[:, None, None]


def forward(model, points, sigma, start, stop):
    return model(points, sigma, start, stop)


def fixtures():
    nodes = torch.arange(180, dtype=torch.float64).reshape(5, 3, 4, 3) / 200
    sigmas = torch.tensor([0.005, 0.01, 0.02], dtype=torch.float64)
    return nodes, sigmas


def set_local_gradients(model):
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.01)


class TeacherEndpointTests(unittest.TestCase):
    def test_balanced_gradient_matches_full_four_step_reference_and_detaches_teacher(self):
        model = ToyStudent()
        reference = copy.deepcopy(model)
        set_local_gradients(model)
        nodes, sigma = fixtures()
        nodes.requires_grad_()
        state = nodes[0].detach()
        for start, stop in zip((0, 2, 5, 7), (2, 5, 7, 10)):
            state = reference(state, sigma * 0.95 ** start, start, stop)
        loss = (state - nodes[-1].detach()).square().sum(-1).mean()
        endpoint = torch.autograd.grad(loss, list(reference.parameters()))
        local_norm = gradient_norm(p.grad for p in model.parameters())
        weight = min(1.0, 0.1 * local_norm / gradient_norm(endpoint))
        result = accumulate_teacher_endpoint(model, nodes, sigma, options(), 6, forward)
        for parameter, gradient in zip(model.parameters(), endpoint):
            torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 0.01) + weight * gradient)
        self.assertTrue(all(abs(float(g)) > 0 for g in endpoint[0]))
        self.assertAlmostEqual(result['loss'], float(loss), places=12)
        self.assertAlmostEqual(result['actual_grad_ratio'], 0.1, places=12)
        self.assertIsNone(nodes.grad)

    def test_microbatching_and_checkpointing_preserve_gradient_and_weight(self):
        nodes, sigma = fixtures()
        results = []
        for batch, checkpointing in ((1, True), (3, True), (1, False), (3, False)):
            model = ToyStudent()
            set_local_gradients(model)
            result = accumulate_teacher_endpoint(model, nodes, sigma, options(batch, checkpointing), 6, forward)
            results.append((result, [p.grad.clone() for p in model.parameters()]))
        for result, gradients in results[1:]:
            self.assertAlmostEqual(result['weight'], results[0][0]['weight'], places=12)
            self.assertAlmostEqual(result['loss'], results[0][0]['loss'], places=12)
            for actual, expected in zip(gradients, results[0][1]):
                torch.testing.assert_close(actual, expected)

    def test_auxiliary_eval_restores_rng_modes_and_batchnorm_buffers(self):
        model = ToyStudent(consume_rng=True, batchnorm=True).train()
        set_local_gradients(model)
        nodes, sigma = fixtures()
        buffers = {name: value.clone() for name, value in model.named_buffers()}
        modes = [m.training for m in model.modules()]
        py_state, np_state = random.getstate(), np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        accumulate_teacher_endpoint(model, nodes, sigma, options(), 6, forward)
        self.assertEqual([m.training for m in model.modules()], modes)
        self.assertEqual(random.getstate(), py_state)
        self.assertEqual(np.random.get_state()[0], np_state[0])
        np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
        self.assertEqual(np.random.get_state()[2:], np_state[2:])
        torch.testing.assert_close(torch.random.get_rng_state(), torch_state)
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, buffers[name])

    def test_weight_cap_and_zero_endpoint_gradient_are_finite(self):
        model = ToyStudent()
        set_local_gradients(model)
        nodes, sigma = fixtures()
        settings = options()
        settings['max_weight'] = 1e-10
        result = accumulate_teacher_endpoint(model, nodes, sigma, settings, 6, forward)
        self.assertEqual(result['weight'], settings['max_weight'])
        self.assertLess(result['actual_grad_ratio'], settings['target_grad_ratio'])
        # An exact endpoint has zero endpoint gradient and must not yield NaN.
        with torch.no_grad():
            state = nodes[0]
            for start, stop in zip((0, 2, 5, 7), (2, 5, 7, 10)):
                state = model(state, sigma * 0.95 ** start, start, stop)
            nodes[-1].copy_(state)
        result = accumulate_teacher_endpoint(model, nodes, sigma, options(3), 6, forward)
        self.assertEqual(result['weight'], 0)
        self.assertTrue(math.isfinite(result['weighted_loss']))

    def test_warmup_skips_auxiliary_forward(self):
        model = ToyStudent()
        set_local_gradients(model)
        nodes, sigma = fixtures()
        gradients = [p.grad.clone() for p in model.parameters()]
        def forbidden(*args):
            raise AssertionError('warmup must not execute an endpoint forward')
        result = accumulate_teacher_endpoint(model, nodes, sigma, options(), 2, forbidden)
        self.assertIsNone(result['loss'])
        for parameter, expected in zip(model.parameters(), gradients):
            torch.testing.assert_close(parameter.grad, expected)
        self.assertEqual([endpoint_grad_ratio(options(), epoch) for epoch in (1, 2, 3, 6, 20)],
                         [0, 0, 0.025, 0.1, 0.1])
        self.assertIsNone(resolve_teacher_endpoint({}))


if __name__ == '__main__':
    unittest.main()
