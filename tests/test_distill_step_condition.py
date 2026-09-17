"""CPU checks of the actual PointTransformer forward with small deterministic fixtures."""

import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
import yaml

from test_distill_16to4 import ROOT, original_patch_denoiser, runner


spec = importlib.util.spec_from_file_location('step_condition_test_module', ROOT / 'models/step_condition.py')
condition_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(condition_module)
StepConditionEmbedding = condition_module.StepConditionEmbedding


def point_transformer_class():
    source = ROOT / 'models/PointGPT.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PointTransformer')
    cls.decorator_list = []
    fusion = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'project_patch_scores_weighted')
    namespace = dict(torch=torch, nn=nn, StepConditionEmbedding=StepConditionEmbedding)
    exec(compile(ast.Module(body=[fusion, cls], type_ignores=[]), str(source), 'exec'), namespace)
    old_tree = ast.parse(subprocess.check_output(['git', 'show', '5b3f39795fff63e525500e047d1260dcc2f7426d:models/PointGPT.py'], encoding='utf-8'))
    old_cls = next(n for n in old_tree.body if isinstance(n, ast.ClassDef) and n.name == 'PointTransformer')
    forward = next(n for n in old_cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    forward.name = 'original_forward'
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['PointTransformer'], namespace['original_forward']


PointTransformer, original_forward = point_transformer_class()


class Groups(nn.Module):
    def forward(self, points):
        groups = points.reshape(points.shape[0], 2, 2, 3)
        centers = groups.mean(2)
        indices = torch.arange(4, device=points.device).reshape(1, 2, 2).expand(points.shape[0], -1, -1)
        return groups - centers[:, :, None], centers, indices


class Encoder(nn.Linear):
    def forward(self, groups):
        return super().forward(groups.mean(2))


class Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(1024)

    def forward(self, x, pos, attn_mask=None, classify=False):
        return self.norm(torch.cat((x.mean(1, keepdim=True), x), dim=1) + pos)


class Generator(nn.Linear):
    def forward(self, x, pos, attn_mask=None):
        return super().forward(x + pos)


def model_fixture():
    # Exercise the production forward and condition methods without CUDA FPS or
    # allocating the complete 24-layer PointGPT-L. Backbone fixtures stay fixed.
    model = PointTransformer.__new__(PointTransformer)
    nn.Module.__init__(model)
    model.step_condition = None
    model.encoder_dims = model.trans_dim = 1024
    model.num_group = model.group_size = 2
    model.group_divider = Groups()
    model.encoder = Encoder(3, 1024)
    model.pos_embed = nn.Linear(3, 1024)
    model.sos_pos = nn.Parameter(torch.zeros(1, 1, 1024))
    model.blocks = Extractor()
    model.generator_blocks = Generator(1024, 6)
    model.abl_fc_decoder = model.abl_causal_attn = model.abl_fusion_uniform = False
    return model


class StepConditionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.teacher = model_fixture().eval()
        self.points = torch.randn(2, 4, 3)
        self.sigmas = torch.tensor([.01, .02])

    def test_encoding_normalizes_steps_differs_by_target_and_has_33920_parameters(self):
        embedding = StepConditionEmbedding(1024)
        inputs = []
        hook = embedding.mlp[0].register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach().clone()))
        encoded = embedding(torch.tensor([0, 0]), torch.tensor([4, 6]), 2)
        hook.remove()
        torch.testing.assert_close(inputs[0], torch.tensor([[0, .25, .25], [0, .375, .375]]))
        self.assertFalse(torch.equal(encoded[0], encoded[1]))
        self.assertLess(float(encoded.abs().max().detach()), .01)
        self.assertEqual(sum(p.numel() for p in embedding.parameters()), 33920)
        with self.assertRaises(ValueError):
            embedding(4, 4, 1)

    def test_teacher_unchanged_student_init_preserves_all_old_weights_and_output_shape(self):
        before = copy.deepcopy(self.teacher.state_dict())
        expected = original_forward(self.teacher, self.points, noise_std=self.sigmas)
        actual = self.teacher(self.points, noise_std=self.sigmas)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        student = copy.deepcopy(self.teacher)
        rng = torch.get_rng_state()
        runner.enable_student_condition(student)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for name, value in before.items():
            torch.testing.assert_close(student.state_dict()[name], value, rtol=0, atol=0)
        self.assertIsNone(self.teacher.step_condition)
        self.assertEqual(set(self.teacher.state_dict()), set(before))
        prediction = runner.forward_student_interval(student, self.points, self.sigmas, 0, 4)
        self.assertEqual(prediction.shape, self.points.shape)
        self.assertLess(float((prediction - actual).abs().max().detach()), 1e-4)

    def test_fixed_stages_pass_conditions_and_new_parameters_receive_gradients(self):
        nodes = runner.capture_teacher(self.teacher, self.points, self.sigmas)
        student = runner.enable_student_condition(copy.deepcopy(self.teacher)).train()
        student.requires_grad_(True)
        intervals = []
        hook = student.step_condition.register_forward_pre_hook(
            lambda module, args: intervals.append((args[0], args[1], torch.is_grad_enabled())))
        runner.backward_stages(student, nodes, self.sigmas)
        hook.remove()
        self.assertEqual(intervals, [(0, 4, True), (4, 8, True), (8, 12, True), (12, 16, True)])
        for parameter in student.step_condition.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(int(torch.count_nonzero(parameter.grad)), 0)
        self.assertIsNone(self.teacher.step_condition)

    def test_condition_config_changes_only_the_opt_in_switch(self):
        baseline = yaml.safe_load((ROOT / 'cfgs/PointGPT-L/distill_16to4.yaml').read_text(encoding='utf-8'))
        conditioned = yaml.safe_load((ROOT / 'cfgs/PointGPT-L/distill_16to4_condition.yaml').read_text(encoding='utf-8'))
        self.assertIs(conditioned.pop('student_step_condition'), True)
        self.assertEqual(conditioned, baseline)

    def test_fixed_conditions_sigmas_loss_and_gradients_match_equal_stage_mean(self):
        nodes = runner.capture_teacher(self.teacher, self.points, self.sigmas)
        teacher_before = copy.deepcopy(self.teacher.state_dict())
        student = runner.enable_student_condition(copy.deepcopy(self.teacher)).train().requires_grad_(True)
        reference = copy.deepcopy(student)
        calls, encodings = [], []
        first = student.register_forward_pre_hook(
            lambda _, args, kw: calls.append((args[0].detach().clone(), kw)), with_kwargs=True)
        second = student.step_condition.mlp[0].register_forward_pre_hook(
            lambda _, args: encodings.append(args[0].detach().clone()))
        losses = runner.backward_stages(student, nodes.requires_grad_(), self.sigmas)
        expected = []
        for stage, (start, target) in enumerate(zip(runner.TEACHER_NODES[:-1], runner.TEACHER_NODES[1:])):
            inputs, kwargs = calls[stage]
            torch.testing.assert_close(inputs, nodes[stage])
            self.assertEqual((kwargs['start_step'], kwargs['target_step']), (start, target))
            torch.testing.assert_close(kwargs['noise_std'], self.sigmas * .95 ** start)
            torch.testing.assert_close(encodings[stage], torch.tensor([start, target, 4]).float().expand(2, -1) / 16)
            prediction = reference(nodes[stage].detach(), noise_std=self.sigmas * .95 ** start,
                                   start_step=start, target_step=target)
            expected.append((prediction - nodes[stage + 1].detach()).square().sum(-1).mean())
        torch.stack(expected).mean().backward()
        self.assertEqual(losses, [float(value.detach()) for value in expected])
        for actual, wanted in zip(student.parameters(), reference.parameters()):
            if wanted.grad is not None:
                torch.testing.assert_close(actual.grad, wanted.grad)
        for module in (student.encoder, student.blocks, student.generator_blocks, student.step_condition):
            self.assertGreater(sum(float(p.grad.norm()) for p in module.parameters() if p.grad is not None), 0)
        self.assertIsNone(nodes.grad)
        self.assertFalse(self.teacher.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.teacher.parameters()))
        for key, value in teacher_before.items():
            torch.testing.assert_close(self.teacher.state_dict()[key], value, rtol=0, atol=0)
        first.remove(); second.remove()

    def test_conditioned_rollout_and_trajectory_validation_keep_fixed_intervals(self):
        student = runner.enable_student_condition(copy.deepcopy(self.teacher))
        intervals = []
        hook = student.step_condition.register_forward_pre_hook(lambda module, args: intervals.append(args[:2]))
        options = dict(patch_size=4, seed_ratio=2, patch_batch=2, fuse_tau_ratio=.5)
        prediction = runner.infer_student(student, self.points.reshape(8, 3), .01, options, original_patch_denoiser())
        self.assertEqual(prediction.shape, (8, 3))
        self.assertEqual(intervals, [(0, 4), (4, 8), (8, 12), (12, 16)] * 2)
        intervals.clear()
        nodes = runner.capture_teacher(self.teacher, self.points, self.sigmas)
        runner.validate_trajectory(student, [(nodes, self.sigmas)], 2)
        self.assertEqual(intervals, [(0, 4), (4, 8), (8, 12), (12, 16)])
        hook.remove()


    def test_old_and_conditioned_checkpoint_loading_and_legacy_optimizer_slots(self):
        original = model_fixture()
        old_optimizer = torch.optim.AdamW(original.parameters(), lr=3e-4, weight_decay=.007)
        sum(p.square().sum() for p in original.parameters()).backward()
        old_optimizer.step()
        old_state = copy.deepcopy(original.state_dict())
        student = model_fixture()
        runner.load_student_state(student, old_state)
        for name, value in old_state.items():
            torch.testing.assert_close(student.state_dict()[name], value, rtol=0, atol=0)
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5)
        runner.restore_student_optimizer(optimizer, old_optimizer.state_dict(), student)
        self.assertEqual(optimizer.param_groups[0]['lr'], 3e-4)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], .007)
        for old, new in zip(original.parameters(), student.parameters()):
            for key, value in old_optimizer.state[old].items():
                torch.testing.assert_close(optimizer.state[new][key], value, rtol=0, atol=0)
        self.assertTrue(all(p not in optimizer.state for p in student.step_condition.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'student.pth'
            torch.save({'base_model': old_state}, path)
            legacy = model_fixture()
            runner.load_student_checkpoint(legacy, path, SimpleNamespace())
            self.assertIsNone(legacy.step_condition)
            torch.testing.assert_close(legacy(self.points, noise_std=self.sigmas),
                                       original(self.points, noise_std=self.sigmas), rtol=0, atol=0)
            torch.save({'base_model': student.state_dict()}, path)
            restored = model_fixture()
            runner.load_student_checkpoint(restored, path, SimpleNamespace())
            for name, value in student.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
        missing_backbone = dict(old_state)
        missing_backbone.pop('generator_blocks.weight')
        with self.assertRaises(RuntimeError):
            runner.load_student_state(model_fixture(), missing_backbone)

    def test_training_opt_in_best_reload_rollout_and_test_entry(self):
        samples = [(points, points, sigma, torch.zeros(1, 3), torch.ones(1, 1), 'cloud')
                   for points, sigma in zip(self.points, self.sigmas)]
        loader = torch.utils.data.DataLoader(samples, batch_size=2)
        nodes = runner.capture_teacher(self.teacher, self.points, self.sigmas)
        config = SimpleNamespace(model={}, learning_rate=1e-5, weight_decay=0., grad_norm_clip=1.,
                                 total_bs=2, teacher_patch_batch=1, student_patch_batch=2,
                                 test_patch_batch=2, epochs=1, rollout_val_interval=1,
                                 inference_patch_size=4, seed_ratio=1, fuse_tau_ratio=.5,
                                 dataset=SimpleNamespace(_base_={}))
        args = SimpleNamespace(epochs=1, max_shapes=0, max_patch_batches=1, resume=None)
        original_backward = runner.backward_stages
        teacher_state = copy.deepcopy(self.teacher.state_dict())
        for enabled in (False, True):
            with self.subTest(condition=enabled), tempfile.TemporaryDirectory() as directory:
                config.student_step_condition = enabled
                output = Path(directory)
                teacher_path = output / 'teacher.pth'
                torch.save({'base_model': teacher_state}, teacher_path)
                created = []
                def build(_):
                    model = model_fixture()
                    created.append(model)
                    return model
                def load(model, path):
                    model.load_state_dict(torch.load(path, weights_only=True)['base_model'], strict=True)
                def backward(student, *values, **kwargs):
                    self.assertEqual(student.step_condition is not None, enabled)
                    for key, value in teacher_state.items():
                        torch.testing.assert_close(student.state_dict()[key], value, rtol=0, atol=0)
                    return original_backward(student, *values, **kwargs)
                builder = SimpleNamespace(model_builder=build, load_model=load)
                cloud = self.points.reshape(8, 3)
                whole_loader = [(cloud[None], cloud[None], torch.tensor([.01]),
                                 [torch.zeros(1, 3)], [torch.ones(1, 1)], ['cloud'])]
                original_infer = runner.infer_student
                def infer(model, points, sigma, options, **kwargs):
                    return original_infer(model, points, sigma, options, original_patch_denoiser(), **kwargs)
                with patch.object(runner, '_train_loader', return_value=loader), \
                        patch.object(runner, '_validation_bank', return_value=([(nodes, self.sigmas)], {'split': 'fixture'})), \
                        patch.object(runner, '_rollout_validation_loader', return_value=(whole_loader, 'test')), \
                        patch.object(runner, 'baseline_metric_ops', return_value={'mesh_root': 'fixture'}), \
                        patch.object(runner, 'evaluate_baseline_metrics', side_effect=lambda prediction, *a, **kw:
                                     (prediction, dict(cd_x1e4=1., p2m_x1e4=2.))), \
                        patch.object(runner, 'infer_student', side_effect=infer), \
                        patch.object(runner, 'backward_stages', side_effect=backward):
                    runner.train(args, config, builder, torch.device('cpu'), teacher_path, output)
                self.assertFalse(created[0].training)
                self.assertIsNone(created[0].step_condition)
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in created[0].parameters()))
                for key, value in teacher_state.items():
                    torch.testing.assert_close(created[0].state_dict()[key], value, rtol=0, atol=0)
                best = torch.load(output / 'ckpt-best.pth', weights_only=True)
                last = torch.load(output / 'ckpt-last.pth', weights_only=True)
                self.assertEqual(best['distillation'], runner.schedule())
                self.assertEqual(best['selection']['metric'], 'val_rollout_score')
                self.assertEqual(best['best_val_rollout_score'], 1.6)
                self.assertEqual(best['best_epoch'], 1)
                self.assertIn('optimizer', last)
                self.assertEqual(any(key.startswith('step_condition.') for key in best['base_model']), enabled)
                restored = model_fixture()
                runner.load_student_checkpoint(restored, output / 'ckpt-best.pth', builder)
                for key, value in best['base_model'].items():
                    torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
                intervals, rollout_inputs, rollout_outputs, rollout_sigmas = [], [], [], []
                def inspect(model, call_args, kwargs):
                    intervals.append((kwargs.get('start_step'), kwargs.get('target_step')))
                    rollout_inputs.append(call_args[0].detach().clone())
                    rollout_sigmas.append(kwargs['noise_std'].detach().clone())
                pre = restored.register_forward_pre_hook(inspect, with_kwargs=True)
                post = restored.register_forward_hook(lambda _, a, value: rollout_outputs.append(value.detach().clone()))
                test_output = output / 'test'; test_output.mkdir()
                test_builder = SimpleNamespace(model_builder=lambda _: restored,
                                               dataset_builder=lambda *_: (None, whole_loader), load_model=load)
                with patch.object(runner, 'baseline_metric_ops', return_value={'mesh_root': 'fixture'}), \
                        patch.object(runner, 'evaluate_baseline_metrics', side_effect=lambda prediction, *a, **kw:
                                     (prediction, dict(cd_x1e4=1., p2m_x1e4=2.))), \
                        patch.object(runner, 'infer_student', side_effect=infer):
                    runner.test(SimpleNamespace(max_shapes=0, save_trajectory=False), config, test_builder,
                                torch.device('cpu'), output / 'ckpt-best.pth', test_output)
                self.assertEqual(intervals, [(0, 4), (4, 8), (8, 12), (12, 16)] if enabled else [(None, None)] * 4)
                for stage, start in enumerate(runner.TEACHER_NODES[:-1]):
                    torch.testing.assert_close(rollout_sigmas[stage], torch.full_like(rollout_sigmas[stage], .01 * .95 ** start))
                    if stage:
                        torch.testing.assert_close(rollout_inputs[stage], rollout_outputs[stage - 1])
                summary = json.loads((test_output / 'test_summary.json').read_text())
                self.assertTrue(summary['complete'] and summary['full_dataset'])
                self.assertEqual((summary['mean_cd_x1e4'], summary['mean_p2m_x1e4']), (1., 2.))
                pre.remove(); post.remove()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable for the small fixture smoke')
    def test_cuda_one_batch_fixture_smoke(self):
        # Actual production forward/loss/condition/rollout on CUDA, with the small
        # backbone and deterministic grouping fixture above. Not a full-L smoke.
        device = torch.device('cuda:0')
        teacher = copy.deepcopy(self.teacher).to(device)
        student = runner.enable_student_condition(copy.deepcopy(teacher)).train().requires_grad_(True)
        teacher_before = {key: value.detach().cpu().clone() for key, value in teacher.state_dict().items()}
        student_before = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5, weight_decay=0.)
        points = self.points.to(device)
        nodes = runner.capture_teacher(teacher, points, self.sigmas)
        intervals = []
        hook = student.step_condition.register_forward_pre_hook(lambda _, args: intervals.append(args[:2]))
        losses = runner.backward_stages(student, nodes, self.sigmas)
        self.assertEqual(intervals, [(0, 4), (4, 8), (8, 12), (12, 16)])
        self.assertTrue(all(torch.isfinite(torch.tensor(losses))))
        self.assertTrue(runner.check_gradients(teacher, student))
        for module in (student.encoder, student.blocks, student.generator_blocks, student.step_condition):
            self.assertGreater(sum(float(p.grad.norm()) for p in module.parameters() if p.grad is not None), 0)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        self.assertTrue(all(int(state['step']) == 1 for state in optimizer.state.values()))
        self.assertTrue(any(not torch.equal(value.detach().cpu(), student_before[key])
                            for key, value in student.state_dict().items() if key.startswith('step_condition.')))
        self.assertTrue(any(not torch.equal(value.detach().cpu(), student_before[key])
                            for key, value in student.state_dict().items() if key.startswith('encoder.')))
        validation = runner.validate_trajectory(student, [(nodes, self.sigmas)], 2)
        self.assertTrue(torch.isfinite(torch.tensor(validation['val_loss_traj'])))
        intervals.clear()
        prediction = runner.infer_student(student, points.reshape(8, 3), .01,
                                          dict(patch_size=4, seed_ratio=1, patch_batch=2, fuse_tau_ratio=.5),
                                          original_patch_denoiser())
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertEqual(intervals, [(0, 4), (4, 8), (8, 12), (12, 16)])
        for key, value in teacher_before.items():
            torch.testing.assert_close(teacher.state_dict()[key].cpu(), value, rtol=0, atol=0)
        torch.cuda.synchronize(device)
        hook.remove()


if __name__ == '__main__':
    unittest.main()
