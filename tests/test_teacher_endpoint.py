"""Endpoint gradient-chain, balancing, micro-batch and state-isolation checks.

These tests are provided for manual use only; no PointGPT/CUDA imports.
"""

import copy
import importlib.util
import math
import random
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from utils.teacher_endpoint import (accumulate_teacher_endpoint, endpoint_grad_ratio,
                                    endpoint_metadata, gradient_norm, pre_bn_conv_biases,
                                    resolve_teacher_endpoint, resolve_endpoint_finetune,
                                    continuous_endpoint_loss, continuous_finetune_losses,
                                    finetune_metadata)


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


class ToyConvStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Conv1d(3, 3, 1), torch.nn.BatchNorm1d(3))
        self.decoder = torch.nn.Linear(3, 3)
        self.double()
        with torch.no_grad():
            self.encoder[0].weight.copy_(torch.eye(3, dtype=torch.float64).unsqueeze(-1) * 0.8)
            self.encoder[0].bias.fill_(0.2)
            self.decoder.weight.copy_(torch.eye(3, dtype=torch.float64) * 0.9)
            self.decoder.bias.fill_(0.01)

    def forward(self, points, sigma, start, stop):
        encoded = self.encoder(points.transpose(1, 2)).transpose(1, 2)
        return self.decoder(encoded) + sigma[:, None, None] * 0.1


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
    def test_finetune_weights_preserve_legacy_and_validate_explicit_objectives(self):
        settings = dict(patch_batch=1, gradient_checkpointing=True, freeze_pre_bn_conv_bias=True)
        config = dict(curriculum_mode='fixed', distillation=dict(teacher_nodes=[0, 2, 5, 7, 10]),
                      endpoint_finetune=settings)
        self.assertEqual(resolve_endpoint_finetune(config), settings)
        self.assertNotIn('loss_weights', finetune_metadata(settings))
        self.assertEqual(finetune_metadata(settings)['loss_weight'], 1.0)
        for local, endpoint, objective in ((1, 1, 'local_endpoint'), (1, 0, 'local'), (0, 1, 'endpoint')):
            supplied = dict(settings, local_weight=local, endpoint_weight=endpoint)
            resolved = resolve_endpoint_finetune(dict(config, endpoint_finetune=supplied))
            self.assertEqual(resolved, supplied)
            metadata = finetune_metadata(resolved)
            self.assertEqual(metadata['objective'], objective)
            self.assertEqual(metadata['loss_weights'], dict(local=local, endpoint=endpoint))
        for extra in (dict(local_weight=1), dict(endpoint_weight=1),
                      dict(local_weight=0, endpoint_weight=0),
                      dict(local_weight=-1, endpoint_weight=1),
                      dict(local_weight=True, endpoint_weight=1),
                      dict(local_weight=1, endpoint_weight=float('nan')),
                      dict(local_weight=float('inf'), endpoint_weight=1)):
            with self.assertRaises(ValueError):
                resolve_endpoint_finetune(dict(config, endpoint_finetune=dict(settings, **extra)))

    def test_local_endpoint_targets_gradients_and_checkpointing_match_connected_reference(self):
        nodes, sigmas = fixtures()
        nodes.requires_grad_()
        sigmas.requires_grad_()
        teacher_weight = torch.nn.Parameter(torch.tensor(0.93, dtype=torch.float64))
        intervals = list(zip((0, 2, 5, 7), (2, 5, 7, 10)))

        def teacher_formula(points, sigma0, start, stop):
            return points * teacher_weight ** (stop - start) + sigma0[:, None, None] * (stop + 1)

        for local_weight, endpoint_weight in ((1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.4, 1.7)):
            reference = ToyStudent().eval()
            state, expected_inputs, local_terms = nodes[0].detach(), [], []
            for start, stop in intervals:
                expected_inputs.append(state.detach().clone())
                with torch.no_grad():
                    target = teacher_formula(state.detach(), sigmas.detach(), start, stop)
                state = forward(reference, state, sigmas.detach() * 0.95 ** start, start, stop)
                local_terms.append((state - target).square().sum(-1).mean())
            local = torch.stack(local_terms).mean()
            endpoint = (state - nodes[-1].detach()).square().sum(-1).mean()
            expected = local_weight * local + endpoint_weight * endpoint
            expected.backward()
            for checkpointing in (False, True):
                # Include an uneven final micro-batch to exercise sample weighting.
                for patch_batch in (1, 2, 3):
                    model = ToyStudent().eval()
                    calls, actual_loss = [], 0.0
                    for offset in range(0, 3, patch_batch):
                        end = min(offset + patch_batch, 3)

                        def target_fn(points, sigma0, start, stop):
                            self.assertGreater(local_weight, 0)
                            self.assertFalse(points.requires_grad)
                            self.assertFalse(sigma0.requires_grad)
                            self.assertFalse(torch.is_grad_enabled())
                            stage = intervals.index((start, stop))
                            torch.testing.assert_close(points, expected_inputs[stage][offset:end])
                            torch.testing.assert_close(sigma0, sigmas.detach()[offset:end])
                            calls.append((start, stop))
                            return teacher_formula(points, sigma0, start, stop)

                        result = continuous_finetune_losses(
                            model, nodes[0, offset:end], nodes[-1, offset:end] if endpoint_weight else None,
                            sigmas[offset:end], forward, teacher_target_fn=target_fn,
                            local_weight=local_weight, endpoint_weight=endpoint_weight,
                            gradient_checkpointing=checkpointing)
                        if local_weight:
                            torch.testing.assert_close(result['local'], torch.stack(result['stages']).mean())
                        else:
                            self.assertIsNone(result['local'])
                            self.assertIsNone(result['stages'])
                        if not endpoint_weight:
                            self.assertIsNone(result['endpoint'])
                        count_before_backward = len(calls)
                        (result['loss'] * ((end - offset) / 3)).backward()
                        # Teacher targets are outside recomputed Student closures.
                        self.assertEqual(len(calls), count_before_backward)
                        actual_loss += float(result['loss'].detach()) * (end - offset) / 3
                    self.assertEqual(calls, intervals * math.ceil(3 / patch_batch) if local_weight else [])
                    self.assertAlmostEqual(actual_loss, float(expected.detach()), places=12)
                    for actual, wanted in zip(model.parameters(), reference.parameters()):
                        torch.testing.assert_close(actual.grad, wanted.grad)
                    self.assertIsNone(teacher_weight.grad)
                    self.assertIsNone(nodes.grad)
                    self.assertIsNone(sigmas.grad)

    def test_endpoint_finetune_config_requires_its_own_objective(self):
        settings = dict(patch_batch=1, gradient_checkpointing=True, freeze_pre_bn_conv_bias=True)
        config = dict(curriculum_mode='fixed', distillation=dict(teacher_nodes=[0, 2, 5, 7, 10]),
                      endpoint_finetune=settings)
        self.assertEqual(resolve_endpoint_finetune(config), settings)
        self.assertIsNone(resolve_endpoint_finetune({}))
        for field, value in [('training_input', dict(type='scheduled_rollout')), ('teacher_endpoint', options())]:
            with self.assertRaises(ValueError):
                resolve_endpoint_finetune(dict(config, **{field: value}))

    def test_direct_endpoint_update_preserves_bias_buffers_and_matches_full_batch_gradient(self):
        nodes, sigmas = fixtures()
        nodes.requires_grad_()
        reference = ToyConvStudent().eval()
        reference.encoder[0].bias.requires_grad_(False)
        state = nodes[0].detach()
        for start, stop in zip((0, 2, 5, 7), (2, 5, 7, 10)):
            state = reference(state, sigmas * 0.95 ** start, start, stop)
        loss = (state - nodes[-1].detach()).square().sum(-1).mean()
        loss.backward()
        for checkpointing in (False, True):
            model = ToyConvStudent().eval()
            model.encoder[0].bias.requires_grad_(False)
            before = {name: value.clone() for name, value in model.named_buffers()}
            bias = model.encoder[0].bias.detach().clone()
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=1e-6, weight_decay=0)
            for index in range(3):
                micro_loss = continuous_endpoint_loss(
                    model, nodes[0, index:index + 1], nodes[-1, index:index + 1], sigmas[index:index + 1],
                    forward, gradient_checkpointing=checkpointing)
                (micro_loss / 3).backward()
            for actual, expected in zip(model.parameters(), reference.parameters()):
                if expected.grad is not None:
                    torch.testing.assert_close(actual.grad, expected.grad)
            self.assertIsNone(model.encoder[0].bias.grad)
            original_weight = model.decoder.weight.detach().clone()
            optimizer.step()
            self.assertFalse(torch.equal(model.decoder.weight, original_weight))
            torch.testing.assert_close(model.encoder[0].bias, bias)
            for name, value in model.named_buffers():
                torch.testing.assert_close(value, before[name])
        self.assertIsNone(nodes.grad)

    def test_epoch_zero_checkpoint_survives_worse_epochs_and_can_be_replaced(self):
        path = Path(__file__).resolve().parents[1] / 'tools/runner_distill.py'
        spec = importlib.util.spec_from_file_location('endpoint_test_runner', path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        settings = dict(patch_batch=1, gradient_checkpointing=True, freeze_pre_bn_conv_bias=True)
        config = SimpleNamespace(curriculum_mode='fixed', distillation=runner.schedule((0, 2, 5, 7, 10)),
                                 model={}, endpoint_finetune=settings)
        student = ToyConvStudent().eval()
        student.encoder[0].bias.requires_grad_(False)
        student.endpoint_finetune_initialization = dict(source_epoch=6, checkpoint_sha256='fixture')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            score, epoch, improved = runner.save_epoch_checkpoints(
                output, student, None, 0, Path('teacher.pth'), config, dict(val_rollout_score=2.0), float('inf'), None)
            self.assertEqual((score, epoch, improved), (2.0, 0, True))
            before = (output / 'ckpt-best.pth').read_bytes()
            score, epoch, improved = runner.save_epoch_checkpoints(
                output, student, None, 1, Path('teacher.pth'), config, dict(val_rollout_score=2.1), score, epoch)
            self.assertEqual((score, epoch, improved), (2.0, 0, False))
            self.assertEqual((output / 'ckpt-best.pth').read_bytes(), before)
            score, epoch, improved = runner.save_epoch_checkpoints(
                output, student, None, 2, Path('teacher.pth'), config, dict(val_rollout_score=1.9), score, epoch)
            saved = torch.load(output / 'ckpt-best.pth', map_location='cpu', weights_only=False)
            self.assertEqual((score, epoch, improved), (1.9, 2, True))
            self.assertEqual(saved['epoch'], 2)
            self.assertEqual(saved['initialization']['source_epoch'], 6)
            self.assertEqual(saved['endpoint_finetune_protocol']['frozen_parameters'], ['encoder.0.bias'])

    def test_optional_mask_preserves_legacy_options_and_rejects_non_bool(self):
        config = dict(curriculum_mode='fixed', distillation=dict(teacher_nodes=[0, 2, 5, 7, 10]),
                      training_input=dict(type='scheduled_rollout', student_target='teacher_from_student_input'),
                      teacher_endpoint=options())
        self.assertEqual(resolve_teacher_endpoint(config), options())
        config['teacher_endpoint']['mask_pre_bn_conv_bias'] = True
        self.assertEqual(resolve_teacher_endpoint(config), config['teacher_endpoint'])
        config['teacher_endpoint']['mask_pre_bn_conv_bias'] = False
        self.assertEqual(resolve_teacher_endpoint(config), config['teacher_endpoint'])
        config['teacher_endpoint']['mask_pre_bn_conv_bias'] = 1
        with self.assertRaisesRegex(ValueError, 'must be bool'):
            resolve_teacher_endpoint(config)

    def test_masked_gradient_matches_reference_and_preserves_excluded_local_gradient(self):
        model = ToyConvStudent().train()
        set_local_gradients(model)
        model.encoder[0].bias.grad.fill_(1e4)
        reference = copy.deepcopy(model).eval()
        nodes, sigma = fixtures()
        state = nodes[0]
        for start, stop in zip((0, 2, 5, 7), (2, 5, 7, 10)):
            state = reference(state, sigma * 0.95 ** start, start, stop)
        loss = (state - nodes[-1]).square().sum(-1).mean()
        endpoint = dict(zip(dict(reference.named_parameters()),
                            torch.autograd.grad(loss, list(reference.parameters()))))
        self.assertGreater(float(endpoint['encoder.0.bias'].norm()), 0)
        excluded = pre_bn_conv_biases(model, torch)
        self.assertEqual(excluded, ['encoder.0.bias'])
        local = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
        values = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        eligible = [name for name in local if name not in excluded]
        local_norm = gradient_norm(local[name] for name in eligible)
        endpoint_norm = gradient_norm(endpoint[name] for name in eligible)
        weight = min(1.0, 0.1 * local_norm / endpoint_norm)
        settings = dict(options(), mask_pre_bn_conv_bias=True)
        original_bias_grad = model.encoder[0].bias.grad
        buffers = {name: value.clone() for name, value in model.named_buffers()}
        result = accumulate_teacher_endpoint(model, nodes, sigma, settings, 6, forward)
        self.assertEqual(result['excluded_parameters'], excluded)
        self.assertIs(model.encoder[0].bias.grad, original_bias_grad)
        self.assertAlmostEqual(result['weight'], weight, places=12)
        self.assertAlmostEqual(result['local_grad_norm'], local_norm, places=12)
        self.assertAlmostEqual(result['endpoint_grad_norm'], endpoint_norm, places=12)
        self.assertAlmostEqual(result['actual_grad_ratio'], 0.1, places=12)
        for name, parameter in model.named_parameters():
            expected = local[name] if name in excluded else local[name] + weight * endpoint[name]
            torch.testing.assert_close(parameter.grad, expected)
            torch.testing.assert_close(parameter, values[name])
            self.assertTrue(parameter.requires_grad)
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, buffers[name])
        self.assertTrue(all(module.training for module in model.modules()))
        expected_dot = sum(float(torch.dot(local[name].reshape(-1), endpoint[name].reshape(-1)))
                           for name in eligible)
        self.assertAlmostEqual(result['grad_cosine'], expected_dot / (local_norm * endpoint_norm), places=12)
        for group, metrics in result['group_diagnostics'].items():
            names = [name for name in eligible if name.split('.')[0] == group]
            left = gradient_norm(local[name] for name in names)
            right = gradient_norm(endpoint[name] for name in names)
            dot = sum(float(torch.dot(local[name].reshape(-1), endpoint[name].reshape(-1))) for name in names)
            self.assertAlmostEqual(metrics['local_norm'], left, places=12)
            self.assertAlmostEqual(metrics['endpoint_norm'], right, places=12)
            self.assertAlmostEqual(metrics['weighted_endpoint_to_local_ratio'], weight * right / left, places=12)
            self.assertAlmostEqual(metrics['cosine_to_local'], dot / (left * right), places=12)
        self.assertAlmostEqual(sum(g['endpoint_squared_norm_fraction']
                                   for g in result['group_diagnostics'].values()), 1.0, places=12)
        self.assertEqual(endpoint_metadata(settings, model)['excluded_parameters'], excluded)

    def test_masked_microbatching_and_checkpointing_preserve_gradient_and_weight(self):
        nodes, sigma = fixtures()
        results = []
        for batch, checkpointing in ((1, True), (3, True), (1, False), (3, False)):
            model = ToyConvStudent().train()
            set_local_gradients(model)
            settings = dict(options(batch, checkpointing), mask_pre_bn_conv_bias=True)
            result = accumulate_teacher_endpoint(model, nodes, sigma, settings, 6, forward)
            results.append((result, [p.grad.clone() for p in model.parameters()]))
        for result, gradients in results[1:]:
            self.assertAlmostEqual(result['weight'], results[0][0]['weight'], places=12)
            self.assertAlmostEqual(result['grad_cosine'], results[0][0]['grad_cosine'], places=12)
            for actual, expected in zip(gradients, results[0][1]):
                torch.testing.assert_close(actual, expected)

    def test_masked_zero_local_norm_logs_undefined_cosine_and_preserves_bias(self):
        model = ToyConvStudent()
        model.encoder[0].bias.grad = torch.full_like(model.encoder[0].bias, 1e4)
        original = model.encoder[0].bias.grad.clone()
        nodes, sigma = fixtures()
        result = accumulate_teacher_endpoint(model, nodes, sigma,
                                             dict(options(), mask_pre_bn_conv_bias=True), 6, forward)
        self.assertEqual(result['local_grad_norm'], 0)
        self.assertEqual(result['weight'], 0)
        self.assertIsNone(result['grad_cosine'])
        for metrics in result['group_diagnostics'].values():
            self.assertIsNone(metrics['cosine_to_local'])
            self.assertIsNone(metrics['weighted_endpoint_to_local_ratio'])
        torch.testing.assert_close(model.encoder[0].bias.grad, original)

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
