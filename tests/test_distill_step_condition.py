"""CPU checks of the actual PointTransformer forward with small deterministic fixtures."""

import ast
import copy
import importlib.util
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

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
    old_tree = ast.parse(subprocess.check_output(['git', 'show', 'HEAD:models/PointGPT.py'], encoding='utf-8'))
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
        indices = torch.arange(4).reshape(1, 2, 2).expand(points.shape[0], -1, -1)
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

    def test_shadow_evaluates_each_candidate_condition_without_training_on_it(self):
        full = runner.capture_teacher(self.teacher, self.points[:1], .01, return_full_trajectory=True)
        student = runner.enable_student_condition(copy.deepcopy(self.teacher)).train()
        student.requires_grad_(True)
        reference = copy.deepcopy(student)
        before = copy.deepcopy(student.state_dict())
        intervals = []
        hook = student.step_condition.register_forward_pre_hook(lambda module, args: intervals.append(
            (args[0], args[1], torch.is_grad_enabled(), module.training)))
        rows = runner.shadow_search_teacher_targets(student, full, .01, .3, 1)
        shadow_intervals = list(intervals)
        self.assertEqual(rows[0]['start_step'], 0)
        for previous, current in zip(rows, rows[1:]):
            self.assertEqual(previous['selected_target_step'], current['start_step'])
        self.assertEqual(rows[-1]['selected_target_step'], 16)
        for stage, row in enumerate(rows):
            self.assertEqual([candidate['target_step'] for candidate in row['candidate_metrics']],
                             list(range(row['start_step'] + 1, 17 - (3 - stage))))
            if stage < 3:
                closest = min(row['candidate_metrics'], key=lambda c: c['pcd_distance_to_target'])
                self.assertEqual(row['selected_target_step'], closest['target_step'])
        self.assertEqual([(t, u) for t, u, _, _ in shadow_intervals[:13]], [(0, u) for u in range(1, 14)])
        self.assertTrue(all(not grad and not training for _, _, grad, training in shadow_intervals))
        for name, value in before.items():
            torch.testing.assert_close(student.state_dict()[name], value, rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in student.parameters()))
        intervals.clear()
        losses = runner.backward_stages(student, full[list(runner.TEACHER_NODES)], .01)
        hook.remove()
        self.assertEqual([(t, u) for t, u, _, _ in intervals], [(0, 4), (4, 8), (8, 12), (12, 16)])
        expected_losses = runner.backward_stages(reference, full[list(runner.TEACHER_NODES)], .01)
        self.assertEqual(losses, expected_losses)
        for actual, expected in zip(student.parameters(), reference.parameters()):
            if expected.grad is not None:
                torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)

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


if __name__ == '__main__':
    unittest.main()
