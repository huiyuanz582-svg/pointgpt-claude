"""Epoch curriculum state, real global search, schedule/gradient and resume regression tests."""
import ast
import copy
import json
import math
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from test_distill_16to4 import ROOT, ToyEpsilonModel, original_patch_denoiser, runner
from test_distill_global_search import global_search
from test_distill_shadow_search import RandomBatchNormModel
from test_distill_step_condition import model_fixture


NODES = (0, 6, 9, 13, 16)
LATER = (0, 5, 10, 14, 16)


def config_fixture():
    return SimpleNamespace(
        curriculum_mode='dynamic_pcd', distillation=runner.schedule(), model={},
        dynamic_pcd=dict(initial_nodes=list(runner.TEACHER_NODES), target=.3, lambda_balance=1.,
                         update_every_epochs=1, calibration_patches=3, calibration_seed=2025,
                         interval_patch_batch=2),
        pcd_dynamic={'shadow_enabled': False}, total_bs=2, student_patch_batch=2,
        teacher_patch_batch=1, test_patch_batch=2, learning_rate=1e-5, weight_decay=0.,
        grad_norm_clip=1., epochs=2, rollout_val_interval=1,
        inference_patch_size=4, seed_ratio=1, fuse_tau_ratio=.5,
        dataset=SimpleNamespace(_base_={}))


def search_result(nodes):
    pcd = [.31, .34, .35, .37]
    metrics = {edge: dict(PCD=value, D_move=1., E_imit=value)
               for edge, value in zip(zip(nodes[:-1], nodes[1:]), pcd)}
    return dict(global_search.score_teacher_path(nodes, metrics, .3, 1.), student_forward_calls=260)


def paired_dataset():
    """Execute production normalization/noise/paired KNN code with small CPU clouds."""
    namespace = dict(torch=torch, np=np, random=random, math=math, Dataset=torch.utils.data.Dataset)
    for file, names in (
        ('datasets/scoredenoise/transforms.py', {'NormalizeUnitSphere', 'AddNoise'}),
        ('datasets/ScoreDenoiseDataset.py', {'PairedPatchDataset'}),
    ):
        tree = ast.parse((ROOT / file).read_text(encoding='utf-8'))
        nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), file, 'exec'), namespace)
    normalize = namespace['NormalizeUnitSphere']()
    noise = namespace['AddNoise'](.005, .02, log_uniform=True)
    def transform(sample):
        return noise(normalize(sample))
    clouds = [dict(pcl_clean=torch.randn(12, 3) + i, name=f'cloud{i}') for i in range(3)]
    return namespace['PairedPatchDataset']([clouds], 1, patch_size=4, num_patches=1,
                                          transform=transform, flag='train', oversample_factor=4)


class DynamicTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.config = config_fixture()
        self.teacher = model_fixture().eval()
        self.student = runner.enable_student_condition(copy.deepcopy(self.teacher)).train()
        self.points = torch.randn(3, 4, 3) * .03
        self.sigmas = torch.tensor([.005, .01, .02])
        self.full = runner.capture_teacher(self.teacher, self.points, self.sigmas, return_full_trajectory=True)
        self.bank = [(self.full[:, :2], self.sigmas[:2]), (self.full[:, 2:], self.sigmas[2:])]

    def search_context(self):
        return patch.dict(sys.modules, {'tools.shadow_global_search': global_search})

    def run_train(self, output, *, epochs=2, resume=None, search=None, scores=None, metadata=None, stop_after_epoch=None):
        samples = [(x, x, s, torch.zeros(1, 3), torch.ones(1, 1), 'cloud')
                   for x, s in zip(self.points[:2], self.sigmas[:2])]
        loader = torch.utils.data.DataLoader(samples, batch_size=2)
        builder = SimpleNamespace(model_builder=lambda _: copy.deepcopy(self.teacher), load_model=lambda *_: None)
        results = search if search is not None else [search_result(NODES), search_result(LATER)]
        with patch.object(runner, '_train_loader', return_value=loader), \
                patch.object(runner, '_validation_bank', return_value=(self.bank, {'split': 'synthetic'})), \
                patch.object(runner, 'curriculum_calibration_bank', return_value=(self.bank, metadata or {'id': 'bank'})), \
                patch.object(runner, 'update_dynamic_curriculum', side_effect=results) as update, \
                patch.object(runner, 'validate_rollout', side_effect=scores or [dict(val_rollout_score=1.), dict(val_rollout_score=2.)]) as rollout, \
                patch.object(runner, 'validate_trajectory', wraps=runner.validate_trajectory) as validation, \
                patch.object(runner, 'backward_stages', wraps=runner.backward_stages) as backward, \
                patch.object(runner, 'shadow_search_teacher_targets', side_effect=AssertionError('No per-batch search')):
            runner.train(SimpleNamespace(epochs=epochs, max_shapes=0, max_patch_batches=1, resume=resume,
                                         stop_after_epoch=stop_after_epoch),
                         self.config, builder, torch.device('cpu'), Path('teacher.pth'), Path(output))
        return update, rollout, validation, backward

    def test_yaml_initial_nodes_and_protocol_unchanged(self):
        fixed = yaml.safe_load((ROOT / 'cfgs/PointGPT-L/distill_16to4.yaml').read_text(encoding='utf-8'))
        dynamic = yaml.safe_load((ROOT / 'cfgs/PointGPT-L/distill_16to4_dynamic.yaml').read_text(encoding='utf-8'))
        self.assertEqual(runner.configured_teacher_nodes(SimpleNamespace(**dynamic)), runner.TEACHER_NODES)
        self.assertEqual(dynamic['distillation'], fixed['distillation'])
        self.assertEqual(dynamic['model'], fixed['model'])
        self.assertEqual(dynamic['dynamic_pcd']['calibration_patches'], 64)
        self.assertFalse(dynamic['pcd_dynamic']['shadow_enabled'])
        # Initial schedule is configurable, not hardcoded by the runner.
        self.config.dynamic_pcd['initial_nodes'] = list(NODES)
        self.config.distillation = runner.schedule(NODES)
        self.assertEqual(runner.configured_teacher_nodes(self.config), NODES)

    def test_invalid_nodes_and_dynamic_settings_rejected(self):
        for nodes in ((0, 6, 6, 13, 16), (0, 6., 9, 13, 16), (0, 6, 9, 13, 15), (0, 6, 9, 17, 16)):
            with self.assertRaises(ValueError):
                runner.schedule(nodes)
        for key, value in (('target', float('nan')), ('lambda_balance', -1), ('update_every_epochs', 0),
                           ('calibration_patches', 0), ('interval_patch_batch', 1.5)):
            config = copy.deepcopy(self.config)
            config.dynamic_pcd[key] = value
            with self.assertRaises(ValueError):
                runner.configured_teacher_nodes(config)
        self.config.pcd_dynamic['shadow_enabled'] = True
        with self.assertRaises(ValueError):
            runner.configured_teacher_nodes(self.config)

    def test_fixed_modes_still_reject_arbitrary_schedules(self):
        for mode in ('fixed', 'fixed_nonuniform'):
            self.config.curriculum_mode = mode
            self.config.distillation = runner.schedule(NODES)
            with self.assertRaises(ValueError):
                runner.configured_teacher_nodes(self.config)

    def test_calibration_reuses_paired_training_patches_repeatably_and_restores_rng(self):
        dataset = paired_dataset()
        torch_state, numpy_state, python_state = torch.get_rng_state(), np.random.get_state(), random.getstate()
        before = copy.deepcopy(self.teacher.state_dict())
        bank, metadata = runner.curriculum_calibration_bank(self.config, self.teacher, dataset)
        repeated, same_metadata = runner.curriculum_calibration_bank(self.config, self.teacher, dataset)
        self.assertEqual(metadata, same_metadata)
        self.assertEqual(metadata['patches'], 3)
        self.assertEqual(metadata['split'], 'train')
        torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
        np.testing.assert_equal(np.random.get_state(), numpy_state)
        self.assertEqual(random.getstate(), python_state)
        for (states, sigmas), (expected, _) in zip(bank, repeated):
            self.assertEqual(states.shape[0], 17)
            self.assertFalse(states.requires_grad)
            torch.testing.assert_close(states, expected, rtol=0, atol=0)
            self.assertTrue(((sigmas >= .005) & (sigmas <= .02)).all())
        for key, value in self.teacher.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.teacher.parameters()))
        # Reproduce the exact underlying dataset calls; no alternate normalization/noise/KNN.
        with torch.random.fork_rng():
            random.seed(2025)
            np.random.seed(2025)
            torch.manual_seed(2025)
            expected = [dataset[row['dataset_index']] for row in metadata['samples']]
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        actual = torch.cat([states[0] for states, _ in bank])
        torch.testing.assert_close(actual, torch.stack([row['pcl_noisy'] for row in expected]), rtol=0, atol=0)

    def test_calibration_rejects_wrong_patch_count(self):
        self.config.dynamic_pcd['calibration_patches'] = 13
        with self.assertRaises(ValueError):
            runner.curriculum_calibration_bank(self.config, self.teacher, paired_dataset())

    def test_global_update_no_grad_no_parameter_buffer_gradient_or_rng_change(self):
        student = RandomBatchNormModel().train()
        student.bn.eval()  # Preserve mixed module modes too.
        for p in student.parameters():
            p.grad = torch.ones_like(p)
        before, modes = copy.deepcopy(student.state_dict()), [m.training for m in student.modules()]
        rng, numpy_rng, python_rng = torch.get_rng_state(), np.random.get_state(), random.getstate()
        with self.search_context(), patch.object(global_search, 'import_module', return_value=runner):
            result = runner.update_dynamic_curriculum(student, self.bank, self.config)
        self.assertEqual(result['paths_evaluated'], 455)
        self.assertEqual(result['student_forward_calls'], 260)
        self.assertEqual(result['patch_interval_evaluations'], 390)
        self.assertEqual(runner._teacher_nodes(result['nodes']), tuple(result['nodes']))
        self.assertTrue(all(not call['grad'] and not call['training'] for call in student.calls))
        self.assertEqual(modes, [m.training for m in student.modules()])
        for key, value in student.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertTrue(all(torch.equal(p.grad, torch.ones_like(p)) for p in student.parameters()))
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        np.testing.assert_equal(np.random.get_state(), numpy_rng)
        self.assertEqual(random.getstate(), python_rng)

    def test_search_uses_every_interval_condition_and_start_sigma(self):
        calls = []
        handle = self.student.register_forward_pre_hook(lambda _, args, kw: calls.append(
            (args[0].clone(), kw['noise_std'].clone(), kw['start_step'], kw['target_step'], torch.is_grad_enabled())),
            with_kwargs=True)
        with self.search_context(), patch.object(global_search, 'import_module', return_value=runner):
            runner.update_dynamic_curriculum(self.student, self.bank, self.config)
        handle.remove()
        self.assertEqual(len(calls), 260)
        for batch_index, (states, sigmas) in enumerate(self.bank):
            for call, (t, u) in zip(calls[batch_index * 130:(batch_index + 1) * 130], global_search.TEACHER_INTERVALS):
                x, sigma, start, target, grad = call
                self.assertEqual((start, target, grad), (t, u, False))
                torch.testing.assert_close(x, states[t])
                torch.testing.assert_close(sigma, sigmas * .95 ** t)

    def test_aggregation_is_mean_patch_ratio_weighted_across_unequal_batches(self):
        def cache(_, states, sigmas, patch_batch):
            size = states.shape[1]
            move, imit = (torch.tensor([1., 10.]), torch.tensor([.1, 9.])) if size == 2 else (torch.tensor([100.]), torch.tensor([20.]))
            return dict(metrics={edge: dict(D_move=move, E_imit=imit, PCD=imit / move)
                                 for edge in global_search.TEACHER_INTERVALS}, forward_calls=130)
        with self.search_context(), patch.object(global_search, 'build_interval_pcd_cache', side_effect=cache), \
                patch.object(global_search, 'search_global_teacher_nodes', wraps=global_search.search_global_teacher_nodes) as search:
            runner.update_dynamic_curriculum(self.student, self.bank, self.config)
            means = search.call_args.args[0]
        for values in means.values():
            self.assertAlmostEqual(values['PCD'], .4, places=6)
            self.assertNotAlmostEqual(values['PCD'], (29.1 / 111), places=3)

    def test_arbitrary_current_nodes_drive_teacher_targets_condition_sigma_and_exact_loss(self):
        nodes = runner.capture_teacher(self.teacher, self.points, self.sigmas, teacher_nodes=NODES)
        torch.testing.assert_close(nodes, self.full[list(NODES)], rtol=0, atol=0)
        reference = copy.deepcopy(self.student)
        calls, encodings = [], []
        a = self.student.register_forward_pre_hook(lambda _, args, kw: calls.append((args[0].clone(), kw)), with_kwargs=True)
        b = self.student.step_condition.mlp[0].register_forward_pre_hook(lambda _, args: encodings.append(args[0].detach().clone()))
        diagnostics = []
        losses = runner.backward_stages(self.student, nodes.requires_grad_(), self.sigmas,
                                        teacher_nodes=NODES, stage_diagnostics=diagnostics)
        expected = []
        for k, (t, u) in enumerate(zip(NODES[:-1], NODES[1:])):
            x, kw = calls[k]
            torch.testing.assert_close(x, self.full[t])
            self.assertEqual((kw['start_step'], kw['target_step']), (t, u))
            torch.testing.assert_close(kw['noise_std'], self.sigmas * .95 ** t)
            torch.testing.assert_close(encodings[k], torch.tensor([t, u, u-t]).float().expand(3, -1) / 16)
            prediction = runner.forward_student_interval(reference, self.full[t], self.sigmas * .95 ** t, t, u)
            expected.append((prediction - self.full[u]).square().sum(-1).mean())
        torch.stack(expected).mean().backward()
        np.testing.assert_allclose(losses, [float(value.detach()) for value in expected])
        for a_grad, b_grad in zip(self.student.parameters(), reference.parameters()):
            if a_grad.grad is not None:
                torch.testing.assert_close(a_grad.grad, b_grad.grad)
        self.assertIsNone(nodes.grad)
        for module in (self.student.step_condition, self.student.encoder, self.student.blocks):
            self.assertGreater(sum(float(p.grad.norm()) for p in module.parameters() if p.grad is not None), 0)
        a.remove(); b.remove()

    def test_validation_selects_current_targets_from_full_teacher_bank(self):
        actual = runner.validate_trajectory(self.student, self.bank, 2, teacher_nodes=NODES)
        expected = runner.validate_trajectory(self.student, [(self.full[list(NODES)], self.sigmas)], 2, teacher_nodes=NODES)
        self.assertEqual(actual, expected)
        self.assertTrue(self.student.training)

    def test_rollout_uses_current_nodes_sigmas_and_previous_student_outputs(self):
        calls, outputs = [], []
        a = self.student.register_forward_pre_hook(lambda _, args, kw: calls.append((args[0].clone(), kw)), with_kwargs=True)
        b = self.student.register_forward_hook(lambda _, args, output: outputs.append(output.detach().clone()))
        _, trajectory = runner.infer_student(self.student, self.points[0], .02,
                                             dict(patch_size=4, seed_ratio=1, patch_batch=1, fuse_tau_ratio=.5),
                                             original_patch_denoiser(), True, NODES)
        for k, (t, u) in enumerate(zip(NODES[:-1], NODES[1:])):
            x, kw = calls[k]
            self.assertEqual((kw['start_step'], kw['target_step']), (t, u))
            torch.testing.assert_close(kw['noise_std'], torch.full_like(kw['noise_std'], .02 * .95 ** t))
            if k:
                torch.testing.assert_close(x, outputs[k-1], rtol=0, atol=2e-8)
        torch.testing.assert_close(trajectory['sigma_before'], torch.tensor([.02 * .95 ** t for t in NODES[:-1]], dtype=torch.float64))
        a.remove(); b.remove()

    def test_rollout_validation_receives_explicit_current_schedule(self):
        cloud = self.points[0]
        loader = [(cloud[None], cloud[None], torch.tensor([.01]), [torch.zeros(1, 3)], [torch.ones(1, 1)], ['cloud'])]
        with patch.object(runner, '_rollout_validation_loader', return_value=(loader, 'test')), \
                patch.object(runner, 'baseline_metric_ops', return_value={}), \
                patch.object(runner, 'evaluate_baseline_metrics', return_value=(cloud, dict(cd_x1e4=2., p2m_x1e4=1.))), \
                patch.object(runner, 'infer_student', return_value=cloud) as infer:
            runner.validate_rollout(self.student, self.config, teacher_nodes=NODES)
        self.assertEqual(infer.call_args.kwargs['teacher_nodes'], NODES)

    def test_epoch1_initial_nodes_epoch2_search_result_and_best_keeps_epoch1_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            update, rollout, validation, backward = self.run_train(directory)
            self.assertEqual(update.call_count, 2)
            for operation in (rollout, validation, backward):
                self.assertEqual([c.kwargs['teacher_nodes'] for c in operation.call_args_list], [runner.TEACHER_NODES, NODES])
            torch.testing.assert_close(backward.call_args_list[1].args[1], self.full[list(NODES), :2])
            rows = [json.loads(line) for line in (Path(directory) / 'train.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['teacher_nodes_used'], list(runner.TEACHER_NODES))
            self.assertEqual(rows[1]['teacher_nodes_used'], rows[0]['next_teacher_nodes'])
            self.assertEqual(rows[1]['stage_gaps'], [6, 3, 4, 3])
            best = torch.load(Path(directory) / 'ckpt-best.pth', weights_only=True)
            last = torch.load(Path(directory) / 'ckpt-last.pth', weights_only=True)
            self.assertEqual(best['best_epoch'], 1)
            self.assertEqual(best['distillation'], runner.schedule())
            self.assertEqual(best['next_teacher_nodes'], list(NODES))
            self.assertEqual(last['distillation'], runner.schedule(NODES))
            self.assertEqual(last['next_teacher_nodes'], list(LATER))
            self.assertEqual(len(best['curriculum_history']), 1)
            self.assertEqual(len(last['curriculum_history']), 2)

    def test_update_occurs_after_training_and_both_validations(self):
        events = []
        original_backward, original_validate = runner.backward_stages, runner.validate_trajectory
        original_step, original_save = torch.optim.AdamW.step, runner.save_epoch_checkpoints
        def backward(*args, **kwargs):
            events.append('backward')
            return original_backward(*args, **kwargs)
        def step(*args, **kwargs):
            events.append('optimizer_step')
            return original_step(*args, **kwargs)
        def validate(*args, **kwargs):
            events.append('trajectory')
            return original_validate(*args, **kwargs)
        def rollout(*args, **kwargs):
            events.append('rollout')
            return dict(val_rollout_score=1.)
        def select(*args):
            events.append('search')
            self.assertTrue(any(p.grad is not None for p in args[0].step_condition.parameters()))
            return search_result(NODES)
        def save(*args, **kwargs):
            events.append('checkpoint')
            return original_save(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, 'backward_stages', side_effect=backward), \
                patch.object(torch.optim.AdamW, 'step', step), \
                patch.object(runner, 'validate_trajectory', side_effect=validate), \
                patch.object(runner, 'save_epoch_checkpoints', side_effect=save):
            self.run_train(directory, search=select, scores=rollout)
        self.assertEqual(events, ['backward', 'optimizer_step', 'trajectory', 'rollout', 'search', 'checkpoint'] * 2)

    def test_update_cadence_skips_epochs_without_changing_nodes(self):
        self.config.dynamic_pcd['update_every_epochs'] = 2
        with tempfile.TemporaryDirectory() as directory:
            update, _, _, backward = self.run_train(directory, epochs=3, search=[search_result(NODES)],
                                                   scores=[dict(val_rollout_score=1.)] * 3)
            self.assertEqual(update.call_count, 1)
            self.assertEqual([c.kwargs['teacher_nodes'] for c in backward.call_args_list],
                             [runner.TEACHER_NODES, runner.TEACHER_NODES, NODES])
            rows = [json.loads(line) for line in (Path(directory) / 'train.jsonl').read_text().splitlines()]
            self.assertEqual([r['dynamic_update_performed'] for r in rows], [False, True, False])
            self.assertIsNone(rows[0]['search_stage_PCD'])

    def test_test_entry_loads_best_checkpoint_nodes_instead_of_initial_or_latest_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_train(root, scores=[dict(val_rollout_score=2.), dict(val_rollout_score=1.)])
            # Epoch 2 best used NODES, has LATER as next nodes, and YAML initial is uniform.
            cloud = self.points[0]
            loader = [(cloud[None], cloud[None], torch.tensor([.01]), [torch.zeros(1, 3)], [torch.ones(1, 1)], ['cloud'])]
            builder = SimpleNamespace(model_builder=lambda _: model_fixture(), dataset_builder=lambda *_: (None, loader))
            output = root / 'test'; output.mkdir()
            with patch.object(runner, 'baseline_metric_ops', return_value={'mesh_root': 'fixture'}), \
                    patch.object(runner, 'evaluate_baseline_metrics', return_value=(cloud, dict(cd_x1e4=1., p2m_x1e4=2.))), \
                    patch.object(runner, 'infer_student', return_value=cloud) as infer:
                runner.test(SimpleNamespace(max_shapes=1, save_trajectory=False), self.config, builder,
                            torch.device('cpu'), root / 'ckpt-best.pth', output)
            self.assertEqual(infer.call_args.kwargs['teacher_nodes'], NODES)
            self.assertEqual(infer.call_args.args[0].distillation_teacher_nodes, NODES)
            summary = json.loads((output / 'test_summary.json').read_text())
            self.assertEqual(summary['protocol']['teacher_nodes'], list(NODES))

    def test_resume_restores_next_nodes_history_and_historical_best(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            self.run_train(first, epochs=1, search=[search_result(NODES)], scores=[dict(val_rollout_score=1.)])
            _, _, _, backward = self.run_train(second, epochs=2, resume=str(Path(first) / 'ckpt-last.pth'),
                                               search=[search_result(LATER)], scores=[dict(val_rollout_score=2.)])
            self.assertEqual(backward.call_args.kwargs['teacher_nodes'], NODES)
            saved = torch.load(Path(second) / 'ckpt-last.pth', weights_only=True)
            self.assertEqual(saved['epoch'], 2)
            self.assertEqual(saved['current_teacher_nodes'], list(NODES))
            self.assertEqual(saved['next_teacher_nodes'], list(LATER))
            self.assertEqual(saved['best_epoch'], 1)
            self.assertEqual(saved['best_val_rollout_score'], 1.)
            self.assertEqual([r['epoch'] for r in saved['curriculum_history']], [1, 2])

    def test_resume_rejects_changed_config_or_calibration(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            self.run_train(first, epochs=1, search=[search_result(NODES)], scores=[dict(val_rollout_score=1.)])
            checkpoint = Path(first) / 'ckpt-last.pth'
            saved = torch.load(checkpoint, weights_only=True)
            self.config.dynamic_pcd['target'] = .4
            with self.assertRaisesRegex(ValueError, 'same dynamic_pcd'):
                runner.restore_dynamic_curriculum(saved, self.config)
            self.config.dynamic_pcd['target'] = .3
            with self.assertRaisesRegex(ValueError, 'calibration bank differs'):
                self.run_train(second, resume=str(checkpoint), metadata={'id': 'changed'})

    def test_two_mini_epochs_with_actual_global_search(self):
        with tempfile.TemporaryDirectory() as directory, self.search_context(), \
                patch.object(global_search, 'import_module', return_value=runner):
            actual_update = runner.update_dynamic_curriculum
            self.run_train(directory, search=actual_update)
            rows = [json.loads(line) for line in (Path(directory) / 'train.jsonl').read_text().splitlines()]
            self.assertEqual(rows[1]['teacher_nodes_used'], rows[0]['next_teacher_nodes'])
            self.assertEqual(rows[0]['search_student_forward_calls'], 260)
            self.assertTrue(all(math.isfinite(r['loss_traj']) for r in rows))
            self.assertTrue(all(len(r['search_stage_PCD']) == 4 for r in rows))

    def test_pilot_stops_after_epoch5_save_with_total_plan20_and_can_restore_epoch6(self):
        self.config.epochs = 20
        self.config.rollout_val_interval = 5
        before = copy.deepcopy(self.config)
        with tempfile.TemporaryDirectory() as directory:
            update, rollout, _, backward = self.run_train(
                directory, epochs=None, stop_after_epoch=5,
                search=[search_result(NODES)] * 5, scores=[dict(val_rollout_score=1.)])
            rows = [json.loads(line) for line in (Path(directory) / 'train.jsonl').read_text().splitlines()]
            self.assertEqual([r['epoch'] for r in rows], [1, 2, 3, 4, 5])
            self.assertEqual(update.call_count, 5)
            self.assertEqual(backward.call_count, 5)
            self.assertEqual(rollout.call_count, 1)
            self.assertEqual(self.config, before)
            for row in rows:
                self.assertEqual(row['total_planned_epochs'], 20)
                self.assertEqual(row['stop_after_epoch'], 5)
                self.assertEqual(row['checkpoint_teacher_nodes'], row['nodes_used_this_epoch'])
                self.assertAlmostEqual(row['train_PCD_std'], float(np.std(row['mean_PCD'])))
                self.assertAlmostEqual(row['train_PCD_range'], max(row['mean_PCD']) - min(row['mean_PCD']))
                for key in ('train_seconds', 'trajectory_validation_seconds', 'rollout_validation_seconds',
                            'curriculum_search_seconds', 'checkpoint_seconds', 'epoch_seconds',
                            'setup_seconds', 'run_elapsed_seconds'):
                    self.assertTrue(math.isfinite(row[key]) and row[key] >= 0)
            last = torch.load(Path(directory) / 'ckpt-last.pth', weights_only=True)
            best = torch.load(Path(directory) / 'ckpt-best.pth', weights_only=True)
            self.assertEqual(last['epoch'], 5)
            self.assertEqual(best['epoch'], 5)
            self.assertEqual(best['best_epoch'], 5)
            self.assertEqual(best['distillation'], runner.schedule(NODES))
            self.assertNotIn('stop_after_epoch', last)  # Do not change the checkpoint schema.
            next_nodes, history, _ = runner.restore_dynamic_curriculum(last, self.config)
            self.assertEqual(next_nodes, NODES)
            self.assertEqual(len(history), 5)
            restored = runner.enable_student_condition(model_fixture())
            runner.load_student_state(restored, last['base_model'])
            optimizer = torch.optim.AdamW(restored.parameters(), lr=self.config.learning_rate)
            runner.restore_student_optimizer(optimizer, last['optimizer'], restored)
            self.assertEqual(optimizer.param_groups[0]['lr'], self.config.learning_rate)
            self.assertTrue(all(int(state['step']) == 5 for state in optimizer.state.values()))

    def test_stop_boundary_rejects_outside_plan_and_before_resumed_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            for stop in (0, 21):
                with self.assertRaisesRegex(ValueError, 'unchanged total epoch plan'):
                    self.run_train(directory, epochs=20, stop_after_epoch=stop)
            self.run_train(directory, epochs=20, stop_after_epoch=1,
                           search=[search_result(NODES)], scores=[dict(val_rollout_score=1.)])
            with tempfile.TemporaryDirectory() as resumed:
                with self.assertRaisesRegex(ValueError, 'first resumed epoch'):
                    self.run_train(resumed, epochs=20, stop_after_epoch=1,
                                   resume=str(Path(directory) / 'ckpt-last.pth'))


if __name__ == '__main__':
    unittest.main()
