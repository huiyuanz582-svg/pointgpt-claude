"""CPU autograd + 原 patch 推理循环检查；FPS 用确定性 CPU 桩替代 CUDA op。"""

import ast
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('distill_runner', ROOT / 'tools/runner_distill.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def original_patch_denoiser():
    # 直接执行生产函数 AST，避免导入无关的 CUDA Chamfer/Open3D 依赖。
    # 迭代更新、分批、KNN、融合、轨迹捕获均使用原函数，未重写其算法。
    path = ROOT / 'tools/runner_finetune.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'patch_based_denoise')
    namespace = dict(torch=torch, np=np, sys=sys, gpu_mem_ratio=lambda: 0.0,
                     misc=SimpleNamespace(fps=lambda points, count: points[:, :count]))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['patch_based_denoise']


class ToyEpsilonModel(torch.nn.Module):
    """与 PointTransformer 相同接口：epsilon head 后返回 x + sigma * epsilon。"""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.2))
        self.bias = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.05]))
        self.calls = []

    def forward(self, x, clean=None, type='val', name='', noise_std=None):
        assert clean is None and type == 'val'
        self.calls.append(dict(x=x.detach().clone(), sigma=noise_std.detach().clone(),
                               grad=torch.is_grad_enabled(), training=self.training,
                               input_requires_grad=x.requires_grad))
        return x + noise_std[:, None, None] * (self.weight * x + self.bias)


class DistillationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.teacher = ToyEpsilonModel()
        self.student = copy.deepcopy(self.teacher)
        self.noisy = torch.randn(8, 3)
        self.sigma0 = 0.02
        self.options = dict(patch_size=4, seed_ratio=2, patch_batch=2, fuse_tau_ratio=0.5)
        self.denoise = original_patch_denoiser()

    def capture(self):
        patches = torch.stack([self.noisy[:4], self.noisy[1:5], self.noisy[2:6], self.noisy[4:8]])
        return runner.capture_teacher(self.teacher, patches, self.sigma0, patch_batch=2)

    def test_teacher_frozen_exact_sixteen_step_nodes(self):
        self.noisy.requires_grad_(True)
        before = copy.deepcopy(self.teacher.state_dict())
        nodes = self.capture()
        self.assertEqual(tuple(nodes.shape), (5, 4, 4, 3))
        self.assertFalse(nodes.requires_grad)
        self.assertFalse(self.teacher.training)
        self.assertEqual(len(self.teacher.calls), 32)  # 2 patch batches * 16 steps
        self.assertTrue(all(not call['grad'] and not call['training']
                            for call in self.teacher.calls))
        self.assertTrue(all(not p.requires_grad and p.grad is None
                            for p in self.teacher.parameters()))
        for name, value in self.teacher.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        # 独立解析式验证节点是每个 patch 的 0/4/8/12/16 状态，非整云融合结果。
        x = nodes[0].clone()
        for step in range(16):
            eps = 0.2 * x + torch.tensor([0.1, -0.2, 0.05])
            x = x + 0.3 * self.sigma0 * 0.95 ** step * eps
            if (step + 1) % 4 == 0:
                torch.testing.assert_close(nodes[(step + 1) // 4], x)

    def test_teacher_forcing_detach_loss_and_gradients(self):
        nodes = self.capture().requires_grad_(True)
        reference = copy.deepcopy(self.student)
        optimizer = torch.optim.SGD(self.student.parameters(), lr=0.1)
        optimizer.zero_grad(set_to_none=True)
        losses = runner.backward_stages(self.student, nodes, self.sigma0)
        self.assertEqual(len(self.student.calls), 4)
        reference_losses = []
        for stage, start in enumerate((0, 4, 8, 12)):
            call = self.student.calls[stage]
            torch.testing.assert_close(call['x'], nodes[stage])
            torch.testing.assert_close(call['sigma'], torch.full((4,), self.sigma0 * 0.95 ** start))
            self.assertTrue(call['grad'])
            self.assertFalse(call['input_requires_grad'])
            x, target = nodes[stage].detach(), nodes[stage + 1].detach()
            predicted = x + self.sigma0 * 0.95 ** start * (reference.weight * x + reference.bias)
            reference_losses.append((predicted - target).square().sum(-1).mean())
        (sum(reference_losses) / 4).backward()
        np.testing.assert_allclose(losses, [float(loss.detach()) for loss in reference_losses], rtol=1e-5)
        for actual, expected in zip(self.student.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad)
        self.assertIsNone(nodes.grad)  # 输入和 target 都被 detach
        self.assertTrue(runner.check_gradients(self.teacher, self.student))
        before = self.student.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, self.student.weight))
        self.assertTrue(all(p.grad is None for p in self.teacher.parameters()))

    def test_pcd_diagnostics_exact_ratios_without_changing_loss_or_gradients(self):
        base = torch.arange(45, dtype=torch.float32).reshape(3, 5, 3) / 50
        moves = torch.tensor([0., .1, .5])[:, None, None]
        nodes = torch.stack([base + step * moves for step in range(5)]).requires_grad_()
        before = nodes.detach().clone()
        sigmas = torch.tensor([.01, .02, .03])
        reference = copy.deepcopy(self.student)
        original_losses = runner.backward_stages(reference, nodes, sigmas)
        diagnostics = []
        losses = runner.backward_stages(self.student, nodes, sigmas, stage_diagnostics=diagnostics)
        self.assertEqual(losses, original_losses)
        self.assertEqual(len(self.student.calls), 4)  # Logging must not add a forward.
        for actual, expected in zip(self.student.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
        self.assertIsNone(nodes.grad)
        torch.testing.assert_close(nodes.detach(), before, rtol=0, atol=0)
        for stage, start in enumerate((0, 4, 8, 12)):
            x, target = before[stage], before[stage + 1]
            prediction = x + sigmas[:, None, None] * .95 ** start * (
                self.student.weight.detach() * x + self.student.bias.detach())
            d_move = (target - x).square().sum(-1).mean(-1)
            e_imit = (prediction - target).square().sum(-1).mean(-1)
            expected = torch.stack((d_move.mean(), e_imit.mean(),
                                    (e_imit / (d_move + runner.PCD_EPS)).mean()))
            np.testing.assert_allclose(diagnostics[stage], expected.numpy(), rtol=1e-5)
            self.assertTrue(all(isinstance(value, float) and math.isfinite(value)
                                for value in diagnostics[stage]))
            self.assertEqual(float(d_move[0]), 0.)  # eps handles a stationary Teacher patch.
            self.assertNotAlmostEqual(diagnostics[stage][2],
                                      float(e_imit.mean() / (d_move.mean() + runner.PCD_EPS)))
        # Unequal micro-batches retain the same per-patch diagnostic averages.
        split_student = ToyEpsilonModel()
        parts = []
        for first, last in ((0, 2), (2, 3)):
            chunk = []
            runner.backward_stages(split_student, nodes[:, first:last], sigmas[first:last],
                                   loss_scale=(last - first) / 3, stage_diagnostics=chunk)
            parts.append(np.asarray(chunk) * (last - first) / 3)
        np.testing.assert_allclose(sum(parts), diagnostics, rtol=1e-5)

    def test_candidate_pcd_repeatable_arbitrary_targets_and_no_gradients(self):
        base = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3)
        states = [(base + .25 * step).requires_grad_() for step in range(17)]
        predictions = []
        values = []
        for t, u in ((1, 2), (1, 3), (1, 16), (15, 16)):
            prediction = (states[t].detach() + .5).requires_grad_()
            predictions.append(prediction)
            first = runner.evaluate_candidate_pcd(states[t], states[u], prediction)
            second = runner.evaluate_candidate_pcd(states[t], states[u], prediction)
            self.assertTrue(torch.is_grad_enabled())  # The helper restores its caller's grad mode.
            with torch.no_grad():
                move = (states[u] - states[t]).square().sum(-1).mean(-1)
                imitation = (prediction - states[u]).square().sum(-1).mean(-1)
                expected = dict(D_move=move, E_imit=imitation, PCD=imitation / (move + 1e-12))
            for key in ('D_move', 'E_imit', 'PCD'):
                self.assertEqual(first[key].shape, (2,))
                torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
                torch.testing.assert_close(first[key], expected[key], rtol=0, atol=0)
                self.assertFalse(first[key].requires_grad)
                self.assertIsNone(first[key].grad_fn)
            values.append(float(first['PCD'][0]))
        self.assertEqual(len(set(values[:3])), 3)
        self.assertTrue(all(p.requires_grad and p.grad is None for p in states + predictions))

    def test_candidate_pcd_single_cloud_zero_movement_and_shared_epsilon(self):
        points = torch.ones(5, 3, dtype=torch.float16, requires_grad=True)
        zero = runner.evaluate_candidate_pcd(points, points, points)
        displaced = runner.evaluate_candidate_pcd(points, points, points + 1)
        self.assertEqual(runner.PCD_EPS, 1e-12)
        for value in zero.values():
            self.assertEqual(value.shape, ())
            self.assertEqual(float(value), 0.)
            self.assertFalse(value.requires_grad)
        self.assertTrue(torch.isfinite(displaced['PCD']))
        torch.testing.assert_close(displaced['PCD'], torch.tensor(3. / runner.PCD_EPS))
        with self.assertRaises(ValueError):
            runner.evaluate_candidate_pcd(points, points[:1], points)

    def test_candidate_pcd_does_not_affect_fixed_baseline_loss(self):
        nodes = self.capture().requires_grad_()
        reference = copy.deepcopy(self.student)
        expected_losses = runner.backward_stages(reference, nodes, self.sigma0)
        forward = self.student.forward
        diagnostics = []
        def forward_with_diagnostic(x, *args, **kwargs):
            prediction = forward(x, *args, **kwargs)
            diagnostics.append(runner.evaluate_candidate_pcd(x, nodes[len(diagnostics) + 1], prediction))
            self.assertTrue(prediction.requires_grad)
            return prediction
        with patch.object(self.student, 'forward', side_effect=forward_with_diagnostic):
            losses = runner.backward_stages(self.student, nodes, self.sigma0)
        self.assertEqual(losses, expected_losses)
        self.assertEqual(len(diagnostics), 4)
        for actual, expected in zip(self.student.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
        self.assertIsNone(nodes.grad)

    def test_student_inference_is_four_chained_steps(self):
        prediction, trajectory = runner.infer_student(
            self.student, self.noisy, self.sigma0, self.options, self.denoise,
            return_trajectory=True)
        self.assertEqual(len(self.student.calls), 8)  # 2 batches * 4，未调用 Teacher
        self.assertEqual(len(self.teacher.calls), 0)
        self.assertEqual(trajectory['global_states'].shape[0], 5)
        self.assertEqual(trajectory['patch_states'].shape[0], 5)
        torch.testing.assert_close(prediction, trajectory['global_states'][-1])
        for batch in range(2):
            previous = None
            for stage in range(4):
                call = self.student.calls[batch * 4 + stage]
                self.assertFalse(call['grad'])
                self.assertFalse(call['training'])
                torch.testing.assert_close(call['sigma'], torch.full((2,), self.sigma0 * 0.95 ** (4 * stage)))
                if previous is not None:
                    torch.testing.assert_close(call['x'], previous)
                previous = call['x'] + call['sigma'][:, None, None] * (
                    self.student.weight.detach() * call['x'] + self.student.bias.detach())
        torch.testing.assert_close(trajectory['sigma_before'], torch.tensor(
            [self.sigma0 * 0.95 ** k for k in (0, 4, 8, 12)], dtype=torch.float64))

    def test_student_four_steps_match_direct_baseline_call(self):
        actual = runner.infer_student(self.student, self.noisy, self.sigma0,
                                      self.options, self.denoise)
        expected = self.denoise(self.student, self.noisy, self.sigma0, **self.options,
                               num_steps=4, step_size=1.0, decay=0.95 ** 4)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_config_preserves_pointgpt_l_structure_and_fixed_schedule(self):
        config = yaml.safe_load((ROOT / 'cfgs/PointGPT-L/distill_16to4.yaml').read_text(encoding='utf-8'))
        baseline = yaml.safe_load((ROOT / 'cfgs/PointGPT-L/finetune_scoredenoise.yaml').read_text(encoding='utf-8'))
        self.assertEqual(config['model'], baseline['model'])
        self.assertEqual(config['distillation'], runner.schedule())

    def test_training_entry_initialization_update_and_checkpoint(self):
        created = []

        def build(_config):
            model = ToyEpsilonModel()
            created.append(model)
            return model

        def load(model, path):
            model.load_state_dict(torch.load(path, weights_only=True)['base_model'], strict=True)

        config = SimpleNamespace(
            model={}, learning_rate=0.01, weight_decay=0.0,
            teacher_patch_batch=2, student_patch_batch=1, total_bs=2,
            test_patch_batch=2,
            inference_patch_size=4, seed_ratio=2, fuse_tau_ratio=0.5,
            epochs=1, grad_norm_clip=1.0)
        args = SimpleNamespace(epochs=1, max_shapes=0, max_patch_batches=1)
        samples = [(self.noisy[:4], self.noisy[:4], sigma, torch.zeros(1, 3),
                    torch.ones(1, 1), 'synthetic') for sigma in (0.01, 0.02)]
        loader = torch.utils.data.DataLoader(samples, batch_size=2, drop_last=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            teacher_path = output / 'teacher.pth'
            with torch.no_grad():
                self.teacher.weight.fill_(0.7)  # 区别于构造默认值，检查确实使用 checkpoint。
            expected = copy.deepcopy(self.teacher.state_dict())
            torch.save(dict(base_model=expected), teacher_path)
            bank = [(runner.capture_teacher(self.teacher, self.noisy[:4][None], self.sigma0),
                     torch.tensor([self.sigma0]))]
            with patch.object(runner, '_train_loader', return_value=loader), \
                    patch.object(runner, '_validation_bank', return_value=(bank, {'split': 'synthetic'})):
                runner.train(args, config, SimpleNamespace(model_builder=build, load_model=load),
                             torch.device('cpu'), teacher_path, output)
            checkpoint = torch.load(output / 'ckpt-last.pth', weights_only=True)
            self.assertEqual(checkpoint['distillation'], runner.schedule())
            self.assertEqual(checkpoint['epoch'], 1)
            self.assertEqual(checkpoint['teacher_checkpoint'], str(teacher_path))
            self.assertIn('optimizer', checkpoint)
            self.assertEqual(checkpoint['best_val_rollout_score'], float('inf'))
            self.assertIsNone(checkpoint['best_epoch'])
            self.assertTrue((output / 'train.jsonl').is_file())
            self.assertEqual(list(output.glob('ckpt-epoch*.pth')), [])
            self.assertFalse((output / 'ckpt-best.pth').exists())  # No rollout before epoch 5.
            self.assertEqual(len(created), 1)  # Student 来自加载后的 Teacher deepcopy。
            for key, value in created[0].state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
            self.assertFalse(torch.equal(checkpoint['base_model']['weight'], expected['weight']))
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in created[0].parameters()))
            # 旧版 last 缺少历史 best 字段也可续训，恢复 optimizer 和 epoch。
            checkpoint.pop('selection', None)
            checkpoint.pop('best_val_rollout_score')
            checkpoint.pop('best_epoch')
            old_last = output / 'old-last.pth'
            torch.save(checkpoint, old_last)
            continued = output / 'continued'
            continued.mkdir()
            args.resume, args.epochs = str(old_last), 2
            with patch.object(runner, '_train_loader', return_value=loader), \
                    patch.object(runner, '_validation_bank', return_value=(bank, {'split': 'synthetic'})):
                runner.train(args, config, SimpleNamespace(model_builder=build, load_model=load),
                             torch.device('cpu'), teacher_path, continued)
            self.assertEqual(torch.load(continued / 'ckpt-last.pth', weights_only=True)['epoch'], 2)
            self.assertEqual({path.name for path in continued.glob('*.pth')},
                             {'ckpt-last.pth'})
            self.assertFalse((continued / 'ckpt-best.pth').exists())
            legacy_resumed = torch.load(continued / 'ckpt-last.pth', weights_only=True)
            self.assertEqual(legacy_resumed['best_val_rollout_score'], float('inf'))
            self.assertIsNone(legacy_resumed['best_epoch'])

    def test_best_selection_saves_only_last_and_best(self):
        config = SimpleNamespace(model={})
        optimizer = torch.optim.AdamW(self.student.parameters())
        best_score, best_epoch = float('inf'), None
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            rollout_scores = {5: 0.3, 10: 0.2, 15: 0.4, 20: 0.2}
            for epoch in range(1, 22):
                with torch.no_grad():
                    self.student.weight.fill_(epoch)
                validation = {'val_loss_traj': 1. / epoch}  # Keeps improving, including non-rollout epochs.
                if epoch in rollout_scores:
                    validation['val_rollout_score'] = rollout_scores[epoch]
                elif epoch % 2 == 0:
                    validation['val_rollout_score'] = None  # Both absent and explicit null must skip.
                best_score, best_epoch, improved = runner.save_epoch_checkpoints(
                    output, self.student, optimizer, epoch, Path('teacher.pth'), config,
                    validation, best_score, best_epoch)
                self.assertEqual(improved, epoch in (5, 10))
                if epoch < 5:
                    self.assertFalse((output / 'ckpt-best.pth').exists())
            best = torch.load(output / 'ckpt-best.pth', weights_only=True)
            last = torch.load(output / 'ckpt-last.pth', weights_only=True)
            self.assertEqual(best['epoch'], 10)
            self.assertEqual(float(best['base_model']['weight']), 10)
            self.assertEqual(last['epoch'], 21)
            self.assertEqual(best['selection']['best_value'], 0.2)
            self.assertEqual(best['selection']['metric'], 'val_rollout_score')
            self.assertIsNone(last['selection']['value'])
            self.assertEqual(last['selection']['best_epoch'], 10)
            self.assertEqual(last['selection']['best_value'], 0.2)
            self.assertEqual(best['best_val_rollout_score'], 0.2)
            self.assertEqual(last['best_val_rollout_score'], 0.2)
            self.assertEqual(best['best_epoch'], 10)
            self.assertEqual(last['best_epoch'], 10)
            self.assertIn('optimizer', last)
            self.assertEqual({path.name for path in output.iterdir()},
                             {'ckpt-last.pth', 'ckpt-best.pth'})
            with self.assertRaises(FloatingPointError):
                runner.save_epoch_checkpoints(output, self.student, optimizer, 22, Path('teacher.pth'),
                                              config, {'val_rollout_score': float('nan')}, best_score, best_epoch)
            self.assertEqual(torch.load(output / 'ckpt-last.pth', weights_only=True)['epoch'], 21)
            self.assertEqual(torch.load(output / 'ckpt-best.pth', weights_only=True)['epoch'], 10)

    def test_resume_preserves_historical_best_without_reevaluation(self):
        config = SimpleNamespace(model={}, learning_rate=.001, weight_decay=0.,
                                 teacher_patch_batch=1, student_patch_batch=8, total_bs=8,
                                 test_patch_batch=1, epochs=16, grad_norm_clip=1.)
        samples = [(self.noisy[:4], self.noisy[:4], .01, torch.zeros(1, 3),
                    torch.ones(1, 1), 'cloud') for _ in range(8)]
        loader = torch.utils.data.DataLoader(samples, batch_size=8, drop_last=True)
        bank = [(self.capture(), torch.full((4,), self.sigma0))]
        builder = SimpleNamespace(model_builder=lambda _: ToyEpsilonModel(), load_model=lambda *_: None)
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=.001, weight_decay=0.)
        with tempfile.TemporaryDirectory() as directory:
            resume_path = Path(directory) / 'previous-last.pth'
            torch.save(dict(base_model=self.student.state_dict(), optimizer=optimizer.state_dict(),
                            epoch=9, distillation=runner.schedule(),
                            best_val_rollout_score=1.2, best_epoch=5), resume_path)
            original_checkpoint = resume_path.read_bytes()
            output = Path(directory) / 'continued'
            output.mkdir()
            args = SimpleNamespace(resume=str(resume_path), epochs=None, max_shapes=0, max_patch_batches=0)
            with patch.object(runner, '_train_loader', return_value=loader), \
                    patch.object(runner, '_validation_bank', return_value=(bank, {'split': 'synthetic'})), \
                    patch.object(runner, 'validate_trajectory', side_effect=[dict(
                        val_loss_traj=1. / epoch, val_stage_losses=[1. / epoch] * 4,
                        val_patches=4) for epoch in range(10, 17)]) as trajectory, \
                    patch.object(runner, 'validate_rollout', side_effect=[dict(
                        val_rollout_cd=cd, val_rollout_p2m=1., val_rollout_score=cd + .3)
                        for cd in (1.5, .5)]) as rollout:
                runner.train(args, config, builder, torch.device('cpu'), Path('teacher.pth'), output)
            self.assertEqual(trajectory.call_count, 7)  # Only new epochs 10..16, no resume evaluation.
            self.assertEqual(rollout.call_count, 2)  # Only scheduled epochs 10 and 15.
            records = [json.loads(line) for line in (output / 'train.jsonl').read_text().splitlines()]
            self.assertEqual([row['epoch'] for row in records], list(range(10, 17)))
            for row in records[:5]:
                self.assertEqual(row['best_val_rollout_score'], 1.2)
                self.assertEqual(row['best_epoch'], 5)
                self.assertFalse(row['is_best'])
            self.assertEqual([row['epoch'] for row in records if row['is_best']], [15])
            for filename in ('ckpt-last.pth', 'ckpt-best.pth'):
                checkpoint = torch.load(output / filename, weights_only=True)
                self.assertEqual(checkpoint['best_val_rollout_score'], .8)
                self.assertEqual(checkpoint['best_epoch'], 15)
            self.assertEqual(resume_path.read_bytes(), original_checkpoint)

    def test_validation_preserves_batchnorm_gradients_and_train_mode(self):
        class BatchNormModel(ToyEpsilonModel):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm1d(3)

            def forward(self, x, clean=None, type='val', name='', noise_std=None):
                x = self.bn(x.transpose(1, 2)).transpose(1, 2)
                return super().forward(x, clean, type, name, noise_std)

        student = BatchNormModel().train()
        bank = [(self.capture(), torch.full((4,), self.sigma0))]
        before = copy.deepcopy(student.state_dict())
        for parameter in student.parameters():
            parameter.grad = torch.ones_like(parameter)
        first = runner.validate_trajectory(student, bank, 2)
        second = runner.validate_trajectory(student, bank, 2)
        self.assertEqual(first, second)
        self.assertTrue(student.training)
        self.assertTrue(all(not call['grad'] and not call['training'] for call in student.calls))
        for name, value in student.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(all(torch.equal(p.grad, torch.ones_like(p)) for p in student.parameters()))
        self.assertEqual(first['val_patches'], 4)

    def test_rollout_validation_continuous_full_cloud_and_rng_restore(self):
        class BatchNormModel(ToyEpsilonModel):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm1d(3)

            def forward(self, x, clean=None, type='val', name='', noise_std=None):
                return (super().forward(x, clean, type, name, noise_std) +
                        .001 * self.bn(x.transpose(1, 2)).transpose(1, 2))

        student = BatchNormModel().train()
        before = copy.deepcopy(student.state_dict())
        for p in student.parameters():
            p.grad = torch.ones_like(p)
        config = SimpleNamespace(inference_patch_size=4, seed_ratio=2, test_patch_batch=2,
                                 fuse_tau_ratio=.5, validation_seed=2024)
        loader = [(points[None], points[None], torch.tensor([sigma]),
                   [torch.zeros(1, 3)], [torch.ones(1, 1)], [name])
                  for points, sigma, name in ((self.noisy, .01, 'first'),
                                              (self.noisy[:5], .03, 'second'))]
        infer = runner.infer_student
        def inference(model, noisy, sigma, options, **kwargs):
            return infer(model, noisy, sigma, options, denoise_fn=self.denoise, **kwargs)
        received = []
        def metrics(prediction, clean, center, scale, name, cfg, ops, mesh_split):
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(student.training)
            self.assertEqual(mesh_split, 'train')
            self.assertEqual(prediction.shape, clean[0].shape)
            received.append((name, len(prediction)))
            random.random(), np.random.rand(), torch.rand(1)
            cd, p2m = (2., 4.) if name == 'first' else (6., 8.)
            return prediction, dict(cd_x1e4=cd, p2m_x1e4=p2m)
        python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        with patch.object(runner, '_rollout_validation_loader', return_value=(loader, 'train')), \
                patch.object(runner, 'baseline_metric_ops', return_value={}), \
                patch.object(runner, 'infer_student', side_effect=inference), \
                patch.object(runner, 'evaluate_baseline_metrics', side_effect=metrics):
            result = runner.validate_rollout(student, config)
        self.assertEqual(received, [('first', 8), ('second', 5)])
        self.assertEqual(result, dict(val_rollout_cd=4., val_rollout_p2m=6.,
                                     val_rollout_score=5.8, val_rollout_clouds=2))
        # Two patch batches for the first cloud, one for the second; four calls each.
        self.assertEqual(len(student.calls), 12)
        for offset, sigma0 in ((0, .01), (4, .01), (8, .03)):
            for stage in range(4):
                call = student.calls[offset + stage]
                self.assertFalse(call['grad'] or call['training'])
                torch.testing.assert_close(call['sigma'], torch.full_like(call['sigma'], sigma0 * .95 ** (4 * stage)))
                if stage < 3:
                    x = call['x']
                    expected = (x + call['sigma'][:, None, None] * (student.weight.detach() * x + student.bias.detach()) +
                                .001 * x / math.sqrt(1 + student.bn.eps))
                    torch.testing.assert_close(student.calls[offset + stage + 1]['x'], expected)
        self.assertTrue(student.training)
        for key, value in student.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertTrue(all(torch.equal(p.grad, torch.ones_like(p)) for p in student.parameters()))
        self.assertEqual(random.getstate(), python_rng)
        np.testing.assert_equal(np.random.get_state(), numpy_rng)
        torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)

    def test_rollout_validation_failure_restores_mode_and_rng(self):
        self.student.train()
        before = torch.get_rng_state()
        config = SimpleNamespace(validation_seed=2024)
        with patch.object(runner, '_rollout_validation_loader', side_effect=RuntimeError('missing mesh')):
            with self.assertRaisesRegex(RuntimeError, 'missing mesh'):
                runner.validate_rollout(self.student, config)
        self.assertTrue(self.student.training)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)

    def test_only_rollout_epochs_update_best_and_all_diagnostics_are_logged(self):
        config = SimpleNamespace(model={}, learning_rate=.001, weight_decay=0.,
                                 teacher_patch_batch=1, student_patch_batch=8, total_bs=8,
                                 test_patch_batch=1, epochs=6, grad_norm_clip=1.)
        args = SimpleNamespace(epochs=None, max_shapes=0, max_patch_batches=0)
        samples = [(self.noisy[:4], self.noisy[:4], .01, torch.zeros(1, 3),
                    torch.ones(1, 1), 'cloud') for _ in range(8)]
        loader = torch.utils.data.DataLoader(samples, batch_size=8, drop_last=True)
        bank = [(self.capture(), torch.full((4,), self.sigma0))]
        builder = SimpleNamespace(model_builder=lambda _: ToyEpsilonModel(), load_model=lambda *_: None)
        diagnostics = [dict(val_loss_traj=float(7-epoch), val_stage_losses=[float(7-epoch)]*4,
                            val_patches=4) for epoch in range(1, 7)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(runner, '_train_loader', return_value=loader), \
                    patch.object(runner, '_validation_bank', return_value=(bank, {'split': 'synthetic'})), \
                    patch.object(runner, 'validate_trajectory', side_effect=diagnostics) as trajectory, \
                    patch.object(runner, 'validate_rollout', return_value=dict(
                        val_rollout_cd=2., val_rollout_p2m=4., val_rollout_score=3.2)) as rollout:
                runner.train(args, config, builder, torch.device('cpu'), Path('teacher.pth'), output)
            self.assertEqual(trajectory.call_count, 6)
            self.assertEqual(rollout.call_count, 1)  # Default interval=5, no forced final rollout.
            records = [json.loads(line) for line in (output / 'train.jsonl').read_text().splitlines()]
            self.assertEqual([row['epoch'] for row in records if row['val_rollout_score'] is not None], [5])
            fields = {'val_loss_traj', 'val_stage_losses', 'val_rollout_cd', 'val_rollout_p2m',
                      'val_rollout_score', 'best_val_rollout_score', 'best_epoch', 'is_best',
                      'mean_D_move', 'mean_E_imit', 'mean_PCD', 'pcd_eps', 'pcd_aggregation'}
            self.assertTrue(all(fields <= row.keys() for row in records))
            for row in records:
                for key in ('mean_D_move', 'mean_E_imit', 'mean_PCD'):
                    self.assertEqual(len(row[key]), 4)
                    self.assertTrue(all(math.isfinite(value) and value >= 0 for value in row[key]))
                np.testing.assert_allclose(row['mean_E_imit'], row['stage_losses'], rtol=1e-5)
                self.assertEqual(row['pcd_eps'], 1e-12)
                self.assertEqual(row['pcd_aggregation'], 'mean_of_per_patch_ratios')
            self.assertEqual([row['epoch'] for row in records if row['is_best']], [5])
            self.assertTrue(all(row['best_val_rollout_score'] is None and row['best_epoch'] is None
                                for row in records[:4]))
            self.assertEqual(records[5]['best_val_rollout_score'], 3.2)
            self.assertEqual(records[5]['best_epoch'], 5)
            best = torch.load(output / 'ckpt-best.pth', weights_only=True)
            self.assertEqual(best['epoch'], 5)  # Epoch 6 improves trajectory loss but cannot replace best.
            self.assertEqual(best['selection']['metric'], 'val_rollout_score')

    def test_same_patch_order_and_individual_sigma_teacher_schedule(self):
        patches = torch.randn(3, 7, 3)
        sigmas = torch.tensor([0.005, 0.01, 0.02])
        nodes = runner.capture_teacher(self.teacher, patches, sigmas, patch_batch=2)
        self.assertEqual(tuple(nodes.shape), (5, 3, 7, 3))
        torch.testing.assert_close(nodes[0], patches, rtol=0, atol=0)
        # 原 patch 内公式与新批量 Teacher 公式一致，且不同样本的 sigma 不串用。
        x = patches.clone()
        for step in range(16):
            eps = 0.2 * x + torch.tensor([0.1, -0.2, 0.05])
            x = x + 0.3 * sigmas[:, None, None] * 0.95 ** step * eps
            if (step + 1) % 4 == 0:
                torch.testing.assert_close(nodes[(step + 1) // 4], x)
        for offset, calls in ((0, self.teacher.calls[:16]), (2, self.teacher.calls[16:])):
            for step, call in enumerate(calls):
                torch.testing.assert_close(call['sigma'], sigmas[offset:offset + 2] * 0.95 ** step)

    def test_microbatch_accumulation_equals_full_batch_with_mixed_sigmas(self):
        sigmas = torch.tensor([0.005, 0.01, 0.02])
        nodes = runner.capture_teacher(self.teacher, torch.randn(3, 5, 3), sigmas)
        reference = copy.deepcopy(self.student)
        runner.backward_stages(reference, nodes, sigmas)
        # 不等长 2+1 micro-batches，不能直接按 micro-batch 个数平均。
        runner.backward_stages(self.student, nodes[:, :2], sigmas[:2], loss_scale=2/3)
        runner.backward_stages(self.student, nodes[:, 2:], sigmas[2:], loss_scale=1/3)
        for actual, expected in zip(self.student.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad)

    def test_baseline_patch_sampling_oversample_and_split_are_reused(self):
        class Compose:
            def __init__(self, transforms):
                self.transforms = transforms

            def __call__(self, sample):
                for transform in self.transforms:
                    sample = transform(sample)
                return sample

        # 执行原数据类及 transform，不导入无关的 CUDA/Lightning 运行依赖。
        namespace = dict(torch=torch, np=np, random=random, math=math, os=os,
                         Dataset=torch.utils.data.Dataset, DataLoader=torch.utils.data.DataLoader,
                         Compose=Compose, pl=SimpleNamespace(LightningDataModule=object))
        for source, names in (
            ('datasets/scoredenoise/transforms.py', {'NormalizeUnitSphere', 'AddNoise'}),
            ('datasets/ScoreDenoiseDataset.py', {'PointCloudDataset', 'PairedPatchDataset',
                                                'ScoreDenoise', 'HeldOutValDataset', 'denoise_collate_fn_test'}),
        ):
            tree = ast.parse((ROOT / source).read_text(encoding='utf-8'))
            nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                     and node.name in names]
            exec(compile(ast.Module(body=nodes, type_ignores=[]), source, 'exec'), namespace)
        cls = namespace['ScoreDenoise']
        with tempfile.TemporaryDirectory() as directory:
            for resolution in ('res_a', 'res_b'):
                folder = Path(directory) / 'PUNet/pointclouds/train' / resolution
                folder.mkdir(parents=True)
                for name in ('one', 'two', 'three'):
                    np.savetxt(folder / f'{name}.xyz', np.random.default_rng(5).normal(size=(1100, 3)))
            cfg = SimpleNamespace(
                ROOT=directory, DATASET='PUNet', RESOLUTIONS=['res_a', 'res_b'],
                NOISE_MIN=0.005, NOISE_MAX=0.02, NOISE_LOG_UNIFORM=True,
                PATCH_SIZE=1024, NUM_PATCHES=1, TRAIN_BATCH_SIZE=32, NUM_WORKERS=0,
                VAL_NOISE=0.01, AUG_ROTATE=True, TRAIN_OVERSAMPLE=50, VAL_NUM=1,
                VAL_RESOLUTION='res_a')
            config = SimpleNamespace(dataset=SimpleNamespace(_base_=cfg), total_bs=8,
                                     teacher_patch_batch=2, validation_patches_per_cloud=2)
            with patch.dict(sys.modules, {'datasets.ScoreDenoiseDataset': SimpleNamespace(ScoreDenoise=cls)}):
                loader = runner._train_loader(config)
            self.assertEqual(cfg.TRAIN_BATCH_SIZE, 32)  # 不修改共享 dataset 配置。
            self.assertEqual(loader.batch_size, 8)
            self.assertTrue(loader.drop_last)
            self.assertEqual(loader.dataset.oversample_factor, 50)
            self.assertEqual(len(loader.dataset), 2 * 2 * 50)  # 每分辨率排除 1 个 val shape。
            self.assertEqual(len(loader) * loader.batch_size, 200)
            self.assertIsInstance(loader.sampler, torch.utils.data.RandomSampler)
            expected_cfg = copy.deepcopy(cfg)
            expected_cfg.TRAIN_BATCH_SIZE = 8
            _, baseline = cls(SimpleNamespace(distributed=False), expected_cfg).train_dataloader()
            random.seed(42)
            torch.manual_seed(42)
            actual = next(iter(loader))
            random.seed(42)
            torch.manual_seed(42)
            expected = next(iter(baseline))
            for i in range(3):
                torch.testing.assert_close(actual[i], expected[i], rtol=0, atol=0)
            self.assertEqual(tuple(actual[0].shape), (8, 1024, 3))
            self.assertEqual(actual[5], expected[5])
            self.assertTrue(((actual[2] >= 0.005) & (actual[2] <= 0.02)).all())
            state_before = torch.get_rng_state().clone()
            with patch.dict(sys.modules, {'datasets.ScoreDenoiseDataset': SimpleNamespace(ScoreDenoise=cls)}):
                bank, metadata = runner._validation_bank(config, self.teacher)
                repeated, repeat_metadata = runner._validation_bank(config, self.teacher)
            torch.testing.assert_close(torch.get_rng_state(), state_before, rtol=0, atol=0)
            self.assertEqual(metadata, repeat_metadata)
            self.assertEqual(metadata['split'], 'held_out_train')
            torch.testing.assert_close(bank[0][0], repeated[0][0], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
