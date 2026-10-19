"""Mixed + dynamic-node regression cases. Do not run without user authorization.

Uses only synthetic tensors and the small ToyStudent fixture, never PointGPT/CUDA.
"""
import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import yaml

from test_training_input import ROOT, ToyStudent, runner
from utils.curriculum_config import resolve_curriculum, require_supported_resume


INITIAL = (0, 4, 8, 12, 16)
SELECTED = (0, 6, 9, 13, 16)
LATER = (0, 5, 10, 14, 16)


def fixture_config():
    return SimpleNamespace(
        model={}, curriculum_mode='dynamic_pcd', distillation=runner.schedule(),
        curriculum_metric={'type': 'original_pcd'},
        dynamic_pcd=dict(initial_nodes=list(INITIAL), target=.3, lambda_balance=1.,
                         update_every_epochs=1, calibration_patches=3, calibration_seed=2025,
                         interval_patch_batch=2),
        training_input=dict(type='scheduled_rollout', warmup_epochs=0, decay_epochs=1,
                            teacher_prob_start=.5, teacher_prob_end=.5, sampling_seed=2025),
        pcd_dynamic={'shadow_enabled': False}, epochs=2, learning_rate=1e-3,
        weight_decay=0., grad_norm_clip=1., student_patch_batch=2, teacher_patch_batch=1,
        test_patch_batch=1, rollout_val_interval=1,
    )


def search_result(nodes):
    return dict(nodes=list(nodes), stage_PCD=[.3] * 4, pcd_std=0.,
                pcd_max_minus_min=0., mean_target_error=0., path_score=0.,
                student_forward_calls=260, paths_evaluated=455, unique_intervals=130)


class MixedDynamicTests(unittest.TestCase):
    def setUp(self):
        self.full = torch.arange(17 * 3 * 2 * 3, dtype=torch.float64).reshape(17, 3, 2, 3) / 1000
        self.sigmas = torch.tensor([.005, .01, .02], dtype=torch.float64)
        self.mask = torch.tensor([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.bool)

    def test_configuration_preserves_mixed_controls_and_uses_original_global_pcd(self):
        fixed = yaml.safe_load((ROOT / 'cfgs/PointGPT-L/distill_16to4_input_scheduled.yaml')
                               .read_text(encoding='utf-8'))
        dynamic = yaml.safe_load((ROOT / 'cfgs/PointGPT-L/distill_16to4_input_scheduled_dynamic.yaml')
                                 .read_text(encoding='utf-8'))
        resolved = resolve_curriculum(dynamic)
        self.assertEqual(resolved['metric'], {'type': 'original_pcd'})
        self.assertEqual(dynamic['training_input'], fixed['training_input'])
        self.assertEqual(dynamic['rollout_val_interval'], 1)
        self.assertEqual(dynamic['dynamic_pcd']['initial_nodes'], list(INITIAL))
        self.assertEqual(dynamic['dynamic_pcd']['calibration_patches'], 64)
        self.assertEqual(dynamic['dynamic_pcd']['update_every_epochs'], 1)
        self.assertEqual(runner.configured_teacher_nodes(SimpleNamespace(**dynamic)), INITIAL)
        comparable = copy.deepcopy(dynamic)
        comparable.pop('dynamic_pcd')
        comparable.pop('curriculum_metric')
        comparable['curriculum_mode'] = 'fixed'
        comparable['rollout_val_interval'] = fixed['rollout_val_interval']
        self.assertEqual(comparable, fixed)
        with self.assertRaises(NotImplementedError):
            require_supported_resume(resolved)

    def test_nonuniform_mixed_states_conditions_and_canonical_targets(self):
        student, stats = ToyStudent(), []
        nodes = self.full[list(SELECTED)]
        losses = runner.backward_student_input_stages(
            student, nodes, self.sigmas, self.mask,
            input_diagnostics=stats, teacher_nodes=SELECTED)
        previous = None
        for stage, call in enumerate(student.calls):
            expected = nodes[stage] if stage == 0 else torch.where(
                self.mask[stage, :, None, None], nodes[stage], previous)
            self.assertTrue(torch.equal(call[0], expected))
            self.assertFalse(call[1])
            torch.testing.assert_close(call[2], self.sigmas * .95 ** SELECTED[stage])
            self.assertEqual(call[3:5], SELECTED[stage:stage + 2])
            previous = .8 * expected + .1
            per_patch = (previous - nodes[stage + 1]).square().sum(-1).mean(-1)
            self.assertAlmostEqual(losses[stage], float(per_patch.mean()))
            self.assertEqual(stats[stage][:2], [3, 0] if stage == 0 else [1, 2])
            self.assertAlmostEqual(stats[stage][2], float(per_patch[self.mask[stage]].sum()))
            self.assertAlmostEqual(stats[stage][3], float(per_patch[~self.mask[stage]].sum()))

    def test_epoch_adoption_validation_checkpoint_binding_and_reload(self):
        config = fixture_config()
        samples = torch.utils.data.TensorDataset(
            self.full[0], self.full[0], self.sigmas,
            torch.zeros(3), torch.ones(3), torch.arange(3))
        loader = torch.utils.data.DataLoader(samples, batch_size=3)
        builder = SimpleNamespace(model_builder=lambda _: ToyStudent(), load_model=Mock())
        captured = []

        def capture(_teacher, _noisy, _sigmas, _batch, *, teacher_nodes):
            captured.append(tuple(teacher_nodes))
            return self.full[list(teacher_nodes)].clone()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(runner, '_train_loader', return_value=loader), \
                    patch.object(runner, '_validation_bank',
                                 return_value=([(self.full, self.sigmas)], {'split': 'synthetic'})), \
                    patch.object(runner, 'curriculum_calibration_bank',
                                 return_value=([(self.full, self.sigmas)], {'split': 'train', 'id': 'bank'})), \
                    patch.object(runner, 'capture_teacher', side_effect=capture), \
                    patch.object(runner, 'sample_teacher_input_mask', return_value=self.mask) as masks, \
                    patch.object(runner, 'backward_stages', side_effect=AssertionError('Mixed uses its own backward')), \
                    patch.object(runner, 'backward_student_input_stages',
                                 wraps=runner.backward_student_input_stages) as backward, \
                    patch.object(runner, 'validate_trajectory', return_value={'val_loss_traj': .5}) as trajectory, \
                    patch.object(runner, 'validate_rollout',
                                 side_effect=[{'val_rollout_score': 2.}, {'val_rollout_score': 3.}]) as rollout, \
                    patch.object(runner, 'update_dynamic_curriculum',
                                 side_effect=[search_result(SELECTED), search_result(LATER)]) as search, \
                    redirect_stdout(io.StringIO()):
                runner.train(SimpleNamespace(epochs=None, resume=None, max_shapes=0,
                                             max_patch_batches=0, stop_after_epoch=None),
                             config, builder, torch.device('cpu'), Path('teacher.pth'), output)
            self.assertEqual(captured, [INITIAL, SELECTED])
            self.assertEqual(search.call_count, 2)
            self.assertEqual(masks.call_count, 2)  # One full-batch mask, before micro-batching.
            self.assertEqual([call.args[2].initial_seed() for call in masks.call_args_list], [2025, 2026])
            self.assertEqual([call.kwargs['teacher_nodes'] for call in backward.call_args_list],
                             [INITIAL, INITIAL, SELECTED, SELECTED])
            for epoch, path in enumerate((INITIAL, SELECTED)):
                for micro, (first, last) in enumerate(((0, 2), (2, 3))):
                    call = backward.call_args_list[epoch * 2 + micro]
                    torch.testing.assert_close(call.args[1], self.full[list(path), first:last])
                    self.assertTrue(torch.equal(call.args[3], self.mask[:, first:last]))
                    self.assertAlmostEqual(call.kwargs['loss_scale'], (last - first) / 3)
            for operation in (trajectory, rollout):
                self.assertEqual([call.kwargs['teacher_nodes'] for call in operation.call_args_list],
                                 [INITIAL, SELECTED])
            rows = [json.loads(line) for line in (output / 'train.jsonl').read_text().splitlines()]
            for row, used, next_nodes in zip(rows, (INITIAL, SELECTED), (SELECTED, LATER)):
                self.assertEqual(row['nodes_used_this_epoch'], list(used))
                self.assertEqual(row['checkpoint_teacher_nodes'], list(used))
                self.assertEqual(row['next_teacher_nodes'], list(next_nodes))
                self.assertEqual(row['training_input'], 'scheduled_rollout')
                self.assertEqual(row['stage_gaps'], [u - t for t, u in zip(used[:-1], used[1:])])
                self.assertIsNone(row['mean_PCD'])
                self.assertIsNone(row['train_PCD_std'])
                self.assertEqual(row['search_stage_PCD'], [.3] * 4)
                self.assertEqual([s['teacher_count'] for s in row['training_input_stages']], [3, 1, 1, 1])
            best = torch.load(output / 'ckpt-best.pth', map_location='cpu', weights_only=True)
            last = torch.load(output / 'ckpt-last.pth', map_location='cpu', weights_only=True)
            self.assertEqual((best['epoch'], last['epoch']), (1, 2))
            for checkpoint, used, next_nodes in ((best, INITIAL, SELECTED), (last, SELECTED, LATER)):
                self.assertEqual(checkpoint['distillation'], runner.schedule(used))
                self.assertEqual(checkpoint['current_teacher_nodes'], list(used))
                self.assertEqual(checkpoint['next_teacher_nodes'], list(next_nodes))
                self.assertEqual(checkpoint['training_input']['type'], 'scheduled_rollout')
            for filename, expected in (('ckpt-best.pth', INITIAL), ('ckpt-last.pth', SELECTED)):
                loaded = ToyStudent()
                runner.load_student_checkpoint(loaded, output / filename, builder)
                self.assertEqual(loaded.distillation_teacher_nodes, expected)


if __name__ == '__main__':
    unittest.main()
