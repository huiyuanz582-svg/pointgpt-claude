"""CPU synthetic tensors and mocked data/validation; no real model or CUDA.

Run only when authorized. The integration case exercises the training loop with
a two-parameter model, including optimizer/checkpoint/log dispatch.
"""
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
import yaml

from utils.curriculum_config import resolve_curriculum, require_supported_resume
from utils.training_input import sample_teacher_input_mask, summarize_input_stages


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('training_input_runner', ROOT / 'tools/runner_distill.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ToyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.8, dtype=torch.float64))
        self.bias = torch.nn.Parameter(torch.tensor(.1, dtype=torch.float64))
        self.step_condition = torch.nn.Identity()
        self.calls = []

    def enable_step_condition(self):
        pass

    def forward(self, points, _clean, mode, _name, noise_std, start_step, target_step):
        assert mode == 'val'
        self.calls.append((points.detach().clone(), points.requires_grad, noise_std.detach().clone(),
                           start_step, target_step, self.training))
        return self.weight * points + self.bias


def nodes_and_sigma():
    nodes = torch.arange(90, dtype=torch.float64).reshape(5, 3, 2, 3) / 100
    return nodes, torch.tensor([.005, .01, .02], dtype=torch.float64)


class TrainingInputTests(unittest.TestCase):
    def test_all_teacher_matches_original_losses_and_gradients_exactly(self):
        nodes, sigmas = nodes_and_sigma()
        original, new = ToyStudent(), ToyStudent()
        old_losses = runner.backward_stages(original, nodes, sigmas)
        new_losses = runner.backward_student_input_stages(
            new, nodes, sigmas, sample_teacher_input_mask(3, 1))
        self.assertEqual(old_losses, new_losses)
        for old, updated in zip(original.parameters(), new.parameters()):
            self.assertTrue(torch.equal(old.grad, updated.grad))
        self.assertEqual(len(new.calls), 4)

    def test_continuous_states_detach_and_keep_nominal_sigma_and_interval(self):
        nodes, sigmas = nodes_and_sigma()
        nodes.requires_grad_()
        sigmas.requires_grad_()
        model = ToyStudent()
        diagnostics = []
        path = (0, 7, 10, 13, 16)
        runner.backward_student_input_stages(
            model, nodes, sigmas, sample_teacher_input_mask(3, 0),
            input_diagnostics=diagnostics, teacher_nodes=path)
        state = nodes[0].detach()
        expected_weight_grad, expected_bias_grad = 0., 0.
        for stage, (x, needs_grad, sigma, start, target, training) in enumerate(model.calls):
            self.assertTrue(torch.equal(x, state))
            self.assertFalse(needs_grad)
            self.assertTrue(training)
            self.assertEqual((start, target), path[stage:stage + 2])
            self.assertTrue(torch.equal(sigma, sigmas.detach() * .95 ** start))
            prediction = .8 * state + .1
            residual = prediction - nodes[stage + 1].detach()
            expected_weight_grad += (2 * residual * state).sum(-1).mean() / 4
            expected_bias_grad += (2 * residual).sum(-1).mean() / 4
            state = prediction
        torch.testing.assert_close(model.weight.grad, expected_weight_grad)
        torch.testing.assert_close(model.bias.grad, expected_bias_grad)
        self.assertIsNone(nodes.grad)
        self.assertIsNone(sigmas.grad)
        self.assertEqual([row[:2] for row in diagnostics], [[3, 0], [0, 3], [0, 3], [0, 3]])

    def test_per_patch_mixing_and_resets_use_previous_actual_prediction(self):
        nodes, sigmas = nodes_and_sigma()
        mask = torch.tensor([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.bool)
        model, diagnostics = ToyStudent(), []
        losses = runner.backward_student_input_stages(model, nodes, sigmas, mask,
                                                       input_diagnostics=diagnostics)
        previous = None
        for stage, call in enumerate(model.calls):
            expected = nodes[stage] if stage == 0 else torch.stack([
                nodes[stage, p] if mask[stage, p] else previous[p] for p in range(3)])
            self.assertTrue(torch.equal(call[0], expected))
            previous = .8 * expected + .1
            per_patch_loss = (previous - nodes[stage + 1]).square().sum(-1).mean(-1)
            self.assertAlmostEqual(losses[stage], float(per_patch_loss.mean()))
            self.assertAlmostEqual(diagnostics[stage][2], float(per_patch_loss[mask[stage]].sum()))
            self.assertAlmostEqual(diagnostics[stage][3], float(per_patch_loss[~mask[stage]].sum()))
        self.assertEqual(len(model.calls), 4)

    def test_microbatch_scaling_preserves_gradients_and_count_weighted_statistics(self):
        nodes, sigmas = nodes_and_sigma()
        mask = torch.tensor([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.bool)
        whole, split = ToyStudent(), ToyStudent()
        diagnostics = []
        runner.backward_student_input_stages(whole, nodes, sigmas, mask, input_diagnostics=diagnostics)
        summed = np.zeros((4, 4))
        for first, last in ((0, 2), (2, 3)):
            part = []
            runner.backward_student_input_stages(split, nodes[:, first:last], sigmas[first:last],
                                                  mask[:, first:last], loss_scale=(last - first) / 3,
                                                  input_diagnostics=part)
            summed += np.asarray(part)
        for a, b in zip(whole.parameters(), split.parameters()):
            torch.testing.assert_close(a.grad, b.grad)
        np.testing.assert_allclose(summed, diagnostics)
        self.assertEqual([r['teacher_count'] for r in summarize_input_stages(summed)], [3, 1, 1, 1])

    def test_sampling_is_reproducible_and_does_not_touch_global_rng(self):
        global_state = torch.random.get_rng_state().clone()
        first = torch.Generator(device='cpu').manual_seed(2025)
        second = torch.Generator(device='cpu').manual_seed(2025)
        a = sample_teacher_input_mask(128, .4, first)
        b = sample_teacher_input_mask(128, .4, second)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(a[0].all())
        self.assertTrue(a[1:].any() and (~a[1:]).any())
        self.assertTrue(torch.equal(global_state, torch.random.get_rng_state()))
        before = first.get_state().clone()
        sample_teacher_input_mask(3, 0, first)
        sample_teacher_input_mask(3, 1, first)
        self.assertTrue(torch.equal(before, first.get_state()))
        with self.assertRaises(ValueError):
            sample_teacher_input_mask(3, .5)

    def test_invalid_mask_and_nonfinite_loss_fail(self):
        nodes, sigmas = nodes_and_sigma()
        for mask in (torch.ones(4, 3), torch.ones(5, 3, dtype=torch.bool),
                     torch.zeros(4, 3, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                runner.backward_student_input_stages(ToyStudent(), nodes, sigmas, mask)
        nodes[1] = float('nan')
        with self.assertRaises(FloatingPointError):
            runner.backward_student_input_stages(ToyStudent(), nodes, sigmas,
                                                 sample_teacher_input_mask(3, 0))

    def test_per_patch_probability_mask_is_reproducible_and_extremes_are_exact(self):
        probabilities = [0., 1., .2, .68]
        global_state = torch.random.get_rng_state().clone()
        generator = torch.Generator(device='cpu').manual_seed(2025)
        expected_generator = torch.Generator(device='cpu').manual_seed(2025)
        expected = torch.rand((4, 3), generator=expected_generator) < torch.tensor(probabilities)[:, None]
        mask = sample_teacher_input_mask(4, probabilities, generator)
        self.assertTrue(mask[0].all())
        self.assertFalse(mask[1:, 0].any())
        self.assertTrue(mask[1:, 1].all())
        self.assertTrue(torch.equal(mask[1:], expected.T))
        self.assertTrue(torch.equal(global_state, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(generator.get_state(), expected_generator.get_state()))
        state = generator.get_state().clone()
        sample_teacher_input_mask(4, [0.] * 4, generator)
        sample_teacher_input_mask(4, [1.] * 4, generator)
        self.assertTrue(torch.equal(state, generator.get_state()))
        for invalid in ([.2], [0., 1., .2, float('nan')], [0., 1., .2, True]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                sample_teacher_input_mask(4, invalid, generator)

    def test_noise_anchor_config_keeps_mixed20_controls(self):
        control = yaml.safe_load((ROOT / 'cfgs/PointGPT-L/distill_16to4_input_scheduled.yaml').read_text(encoding='utf-8'))
        candidate = yaml.safe_load((ROOT / 'cfgs/PointGPT-L/distill_16to4_input_noise_anchor.yaml').read_text(encoding='utf-8'))
        resolve_curriculum(candidate)
        self.assertEqual(candidate['distillation'], runner.schedule())
        self.assertEqual(candidate['rollout_val_interval'], 1)
        candidate['training_input'].pop('noise_teacher_anchor')
        control['rollout_val_interval'] = 1
        self.assertEqual(candidate, control)

    def test_noise_anchor_range_is_checked_before_loading_models(self):
        config = SimpleNamespace(curriculum_mode='fixed', training_input={
            'type': 'scheduled_rollout', 'noise_teacher_anchor': dict(
                sigma_start=.01, sigma_end=.03, teacher_prob_high=.68)},
            dataset=SimpleNamespace(_base_=SimpleNamespace(NOISE_MIN=.005, NOISE_MAX=.02)))
        builder = SimpleNamespace(model_builder=Mock())
        with self.assertRaisesRegex(ValueError, 'training noise range'):
            runner.train(SimpleNamespace(), config, builder, torch.device('cpu'), Path('teacher.pth'), Path('unused'))
        builder.model_builder.assert_not_called()

    def test_fixed_controls_differ_only_in_input_policy(self):
        configs = [yaml.safe_load((ROOT / f'cfgs/PointGPT-L/distill_16to4_input_{name}.yaml')
                                  .read_text(encoding='utf-8'))
                   for name in ('teacher', 'scheduled', 'student')]
        for config in configs:
            resolve_curriculum(config)
            config.pop('training_input')
            self.assertEqual(config['curriculum_mode'], 'fixed')
            self.assertEqual(config['distillation'], runner.schedule())
            self.assertFalse(config['pcd_dynamic']['shadow_enabled'])
        self.assertEqual(configs[0], configs[1])
        self.assertEqual(configs[0], configs[2])

    def test_mock_training_dispatch_logs_checkpoint_and_inference_compatibility(self):
        nodes, sigmas = nodes_and_sigma()
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            nodes[0], nodes[0], sigmas, torch.zeros(3), torch.ones(3), torch.arange(3)), batch_size=3)
        for label in ('teacher_forced', 'scheduled_rollout', 'student_rollout', 'noise_anchor'):
            mode = 'scheduled_rollout' if label == 'noise_anchor' else label
            with self.subTest(mode=label), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                policy = {'type': mode}
                if mode == 'scheduled_rollout':
                    policy.update(warmup_epochs=0, decay_epochs=1,
                                  teacher_prob_start=.5, teacher_prob_end=.5)
                if label == 'noise_anchor':
                    policy.update(teacher_prob_start=1., teacher_prob_end=.2,
                                  noise_teacher_anchor=dict(sigma_start=.01, sigma_end=.02,
                                                            teacher_prob_high=.68))
                config = SimpleNamespace(model={}, curriculum_mode='fixed', training_input=policy,
                                         dataset=SimpleNamespace(_base_=SimpleNamespace(NOISE_MIN=.005, NOISE_MAX=.02)),
                                         epochs=2, learning_rate=1e-3, weight_decay=0., grad_norm_clip=1.,
                                         student_patch_batch=2, teacher_patch_batch=1, test_patch_batch=1,
                                         rollout_val_interval=1, pcd_dynamic={'shadow_enabled': False})
                args = SimpleNamespace(epochs=None, resume=None, max_shapes=0, max_patch_batches=0,
                                       stop_after_epoch=None)
                builder = SimpleNamespace(model_builder=lambda _: ToyStudent(), load_model=Mock())
                with patch.object(runner, '_train_loader', return_value=loader), \
                        patch.object(runner, '_validation_bank',
                                     return_value=([(nodes, sigmas)], {'split': 'mock'})), \
                        patch.object(runner, 'capture_teacher', return_value=nodes), \
                        patch.object(runner, 'validate_trajectory', return_value={'val_loss_traj': .5}), \
                        patch.object(runner, 'validate_rollout', side_effect=[
                            {'val_rollout_score': 2.}, {'val_rollout_score': 3.}]), \
                        patch.object(runner, 'curriculum_calibration_bank',
                                     side_effect=AssertionError('fixed mode must not calibrate')), \
                        patch.object(runner, 'backward_stages', wraps=runner.backward_stages) as old, \
                        redirect_stdout(io.StringIO()):
                    runner.train(args, config, builder, torch.device('cpu'), Path('teacher.pth'), output)
                self.assertEqual(old.call_count, 4 if mode == 'teacher_forced' else 0)
                records = [json.loads(line) for line in (output / 'train.jsonl').read_text().splitlines()]
                for row in records:
                    self.assertEqual(row['teacher_nodes'], [0, 4, 8, 12, 16])
                    self.assertEqual(row['training_input'], mode)
                    self.assertEqual(row['val_loss_traj_input'], 'teacher_forced')
                    self.assertEqual(row['training_input_stages'][0]['teacher_fraction'], 1.)
                    if mode != 'teacher_forced':
                        self.assertIsNone(row['mean_PCD'])
                    for stage, stats in enumerate(row['training_input_stages']):
                        self.assertEqual(stats['teacher_count'] + stats['student_count'], 3)
                        combined = sum((stats[source + '_input_loss'] or 0.) * stats[source + '_count']
                                       for source in ('teacher', 'student')) / 3
                        self.assertAlmostEqual(combined, row['stage_losses'][stage])
                    if label == 'noise_anchor':
                        bands = row['training_input_noise_bands']
                        self.assertEqual(sum(band['patches'] for band in bands), 3)
                        self.assertEqual(row['training_input_protocol']['teacher_probability_scope'],
                                         'base_before_noise_anchor')
                        for stage in range(3):
                            self.assertEqual(sum(band['stages'][stage]['teacher_count'] for band in bands),
                                             row['training_input_stages'][stage + 1]['teacher_count'])
                    else:
                        self.assertNotIn('training_input_noise_bands', row)
                last = torch.load(output / 'ckpt-last.pth', map_location='cpu')
                best = torch.load(output / 'ckpt-best.pth', map_location='cpu')
                self.assertEqual((last['epoch'], best['epoch']), (2, 1))
                self.assertEqual(last['distillation'], runner.schedule())
                if mode == 'teacher_forced':
                    self.assertNotIn('training_input', last)
                else:
                    self.assertEqual(last['training_input']['type'], mode)
                    self.assertTrue(last['training_input_protocol']['student_state_detached'])
                    with self.assertRaises(NotImplementedError):
                        require_supported_resume(resolve_curriculum({}), last)
                restored = ToyStudent()
                runner.load_student_checkpoint(restored, output / 'ckpt-last.pth', builder,
                                               expected_nodes=runner.TEACHER_NODES)
                self.assertEqual(restored.distillation_teacher_nodes, runner.TEACHER_NODES)
                for key, value in restored.state_dict().items():
                    self.assertTrue(torch.equal(value, last['base_model'][key]))


if __name__ == '__main__':
    unittest.main()
