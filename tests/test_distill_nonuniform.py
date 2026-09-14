"""Non-uniform schedule wiring; shared fixed loss and whole-cloud pipeline."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from test_distill_16to4 import ROOT, runner, original_patch_denoiser
from test_distill_step_condition import model_fixture

NODES = (0, 7, 10, 13, 16)
INTERVALS = list(zip(NODES[:-1], NODES[1:]))


class NonuniformTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        self.teacher = model_fixture().eval()
        self.student = runner.enable_student_condition(copy.deepcopy(self.teacher)).train()
        self.points = torch.randn(8, 4, 3) * .03
        self.sigma = torch.linspace(.005, .02, 8)
        self.config = SimpleNamespace(curriculum_mode='fixed_nonuniform', distillation=runner.schedule(NODES),
                                      model={}, total_bs=8, student_patch_batch=8, teacher_patch_batch=1,
                                      test_patch_batch=8, learning_rate=1e-5, weight_decay=0., grad_norm_clip=1.,
                                      epochs=20, pcd_dynamic={'shadow_enabled': False})

    def test_yaml_only_changes_schedule_and_disables_online_shadow(self):
        fixed = yaml.safe_load((ROOT / 'cfgs/PointGPT-L/distill_16to4.yaml').read_text(encoding='utf-8'))
        nonuniform = yaml.safe_load((ROOT / 'cfgs/PointGPT-L/distill_16to4_nonuniform.yaml').read_text(encoding='utf-8'))
        expected = copy.deepcopy(fixed)
        expected['curriculum_mode'] = 'fixed_nonuniform'
        expected['distillation'] = runner.schedule(NODES)
        expected['pcd_dynamic']['shadow_enabled'] = False
        self.assertEqual(nonuniform, expected)
        self.assertEqual(runner.configured_teacher_nodes(self.config), NODES)
        self.assertEqual(runner.configured_teacher_nodes(SimpleNamespace(**fixed)), runner.TEACHER_NODES)
        self.config.curriculum_mode = 'fixed'
        with self.assertRaises(ValueError):
            runner.configured_teacher_nodes(self.config)
        for nodes in ((0,7,10,13,15), (0,7,7,13,16), (0,7.,10,13,16)):
            with self.assertRaises(ValueError):
                runner.schedule(nodes)

    def test_teacher_targets_equal_full_original_trajectory(self):
        full = runner.capture_teacher(self.teacher, self.points, self.sigma, return_full_trajectory=True)
        actual = runner.capture_teacher(self.teacher, self.points, self.sigma, teacher_nodes=NODES)
        torch.testing.assert_close(actual, full[list(NODES)], rtol=0, atol=0)
        self.assertFalse(actual.requires_grad)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.teacher.parameters()))

    def test_true_batch_conditions_sigmas_teacher_forcing_loss_and_backward(self):
        nodes = runner.capture_teacher(self.teacher, self.points, self.sigma, teacher_nodes=NODES).requires_grad_()
        reference = copy.deepcopy(self.student)
        calls, encodings = [], []
        def hook(model, inputs, kwargs):
            calls.append((inputs[0].detach().clone(), kwargs['noise_std'].detach().clone(),
                          kwargs['start_step'], kwargs['target_step']))
        handles = [self.student.register_forward_pre_hook(hook, with_kwargs=True),
                   self.student.step_condition.mlp[0].register_forward_pre_hook(
                       lambda _, inputs: encodings.append(inputs[0].detach().clone()))]
        diagnostics = []
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=1e-5, weight_decay=0.)
        losses = runner.backward_stages(self.student, nodes, self.sigma,
                                        stage_diagnostics=diagnostics, teacher_nodes=NODES)
        expected = []
        for k, (t,u) in enumerate(INTERVALS):
            x, sigma, start, end = calls[k]
            self.assertEqual((start,end), (t,u))
            self.assertEqual(x.shape[0], 8)
            torch.testing.assert_close(x, nodes[k], rtol=0, atol=0)
            torch.testing.assert_close(sigma, self.sigma * .95 ** t, rtol=0, atol=0)
            torch.testing.assert_close(encodings[k], torch.tensor([t,u,u-t]).float().expand(8,-1)/16)
            pred = runner.forward_student_interval(reference, nodes[k].detach(), sigma, t, u)
            expected.append((pred - nodes[k+1].detach()).square().sum(-1).mean())
            torch.testing.assert_close(torch.tensor(diagnostics[k][1]), torch.tensor(losses[k]), rtol=1e-6, atol=1e-12)
        torch.stack(expected).mean().backward()
        self.assertEqual(losses, [float(v.detach()) for v in expected])
        self.assertIsNone(nodes.grad)
        for a,b in zip(self.student.parameters(), reference.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad)
        for module in (self.student.step_condition, self.student.encoder, self.student.blocks):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.norm()) for g in grads), 0)
        # This small fixture encodes centered group means (approximately zero),
        # so its encoder bias carries the meaningful backbone update.
        before = self.student.encoder.bias.detach().clone()
        runner.check_gradients(self.teacher, self.student)
        optimizer.step()
        self.assertFalse(torch.equal(before, self.student.encoder.bias))
        for handle in handles: handle.remove()

    def test_rollout_uses_previous_student_output_sigma_and_original_fusion(self):
        calls, outputs = [], []
        def pre(model, inputs, kwargs):
            calls.append((inputs[0].detach().clone(), kwargs['noise_std'].clone(),
                          kwargs['start_step'], kwargs['target_step']))
        a = self.student.register_forward_pre_hook(pre, with_kwargs=True)
        b = self.student.register_forward_hook(lambda _, inputs, output: outputs.append(output.detach().clone()))
        options = dict(patch_size=4, seed_ratio=2, patch_batch=3, fuse_tau_ratio=.5)
        cloud = self.points[0:2].reshape(8,3)
        denoised, trajectory = runner.infer_student(self.student, cloud, .02, options,
                                                   original_patch_denoiser(), True, NODES)
        self.assertEqual(denoised.shape, cloud.shape)
        self.assertEqual(trajectory['patch_batches'], 2)
        self.assertEqual(len(calls), 8)
        for k,(x,sigma,t,u) in enumerate(calls):
            self.assertEqual((t,u), INTERVALS[k%4])
            torch.testing.assert_close(sigma, torch.full_like(sigma, .02*.95**t))
            if k%4: torch.testing.assert_close(x, outputs[k-1], rtol=0, atol=2e-8)
        for key, steps in [('sigma_before',NODES[:-1]), ('sigma_after',NODES)]:
            torch.testing.assert_close(trajectory[key], torch.tensor([.02*.95**t for t in steps],dtype=torch.float64))
        torch.testing.assert_close(trajectory['global_states'][-1], denoised)
        a.remove(); b.remove()

    def test_checkpoint_load_matches_schedule_and_trajectory_validation(self):
        nodes = runner.capture_teacher(self.teacher, self.points, self.sigma, teacher_nodes=NODES)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'student.pth'
            runner._save_checkpoint(path, self.student, None, 5, 'teacher.pth', self.config)
            restored = model_fixture()
            runner.load_student_checkpoint(restored, path, SimpleNamespace(), expected_nodes=NODES)
            self.assertEqual(restored.distillation_teacher_nodes, NODES)
            calls = []
            handle = restored.step_condition.register_forward_pre_hook(lambda _, inputs: calls.append(inputs[:2]))
            metrics = runner.validate_trajectory(restored, [(nodes,self.sigma)], 8, teacher_nodes=NODES)
            self.assertEqual(calls, INTERVALS)
            self.assertTrue(all(torch.isfinite(torch.tensor(metrics['val_stage_losses']))))
            calls.clear()
            runner.infer_student(restored, self.points[0], .01,
                                 dict(patch_size=4,seed_ratio=1,patch_batch=1,fuse_tau_ratio=.5),
                                 original_patch_denoiser())  # Infer nodes from new checkpoint.
            self.assertEqual(calls, INTERVALS)
            handle.remove()
            with self.assertRaisesRegex(ValueError, 'does not match'):
                runner.load_student_checkpoint(model_fixture(), path, SimpleNamespace(), expected_nodes=runner.TEACHER_NODES)

    def test_whole_cloud_rollout_validation_uses_configured_intervals(self):
        self.config.inference_patch_size = 4
        self.config.seed_ratio = 1
        self.config.fuse_tau_ratio = .5
        cloud = self.points[0]
        loader = [(cloud[None], cloud[None], torch.tensor([.02]),
                   [torch.zeros(1,3)], [torch.ones(1,1)], ['cloud'])]
        calls = []
        handle = self.student.step_condition.register_forward_pre_hook(lambda _, inputs: calls.append(inputs[:2]))
        inference = runner.infer_student
        def infer(model, noisy, sigma0, options, **kwargs):
            return inference(model, noisy, sigma0, options, denoise_fn=original_patch_denoiser(), **kwargs)
        with patch.object(runner, '_rollout_validation_loader', return_value=(loader, 'test')), \
                patch.object(runner, 'infer_student', side_effect=infer), \
                patch.object(runner, 'baseline_metric_ops', return_value={}), \
                patch.object(runner, 'evaluate_baseline_metrics', return_value=(cloud, dict(cd_x1e4=3.,p2m_x1e4=2.))):
            values = runner.validate_rollout(self.student, self.config)
        handle.remove()
        self.assertEqual(calls, INTERVALS)
        self.assertEqual(values['val_rollout_score'], 3.6)
        self.assertTrue(self.student.training)

    def test_train_entry_selects_nonuniform_bank_and_logs_without_search(self):
        samples=[(x,x,s,torch.zeros(1,3),torch.ones(1,1),'cloud') for x,s in zip(self.points,self.sigma)]
        loader=torch.utils.data.DataLoader(samples,batch_size=8,drop_last=True)
        bank=[(runner.capture_teacher(self.teacher,self.points,self.sigma,teacher_nodes=NODES),self.sigma)]
        builder=SimpleNamespace(model_builder=lambda _:model_fixture(),load_model=lambda *_:None)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner,'_train_loader',return_value=loader), \
                patch.object(runner,'_validation_bank',return_value=(bank,{'split':'synthetic'})), \
                patch.object(runner,'shadow_search_teacher_targets',side_effect=AssertionError('No online search')) as shadow, \
                patch.object(runner,'backward_stages',wraps=runner.backward_stages) as backward:
            runner.train(SimpleNamespace(epochs=1,max_shapes=0,max_patch_batches=1),self.config,builder,
                         torch.device('cpu'),Path('teacher.pth'),Path(directory))
            self.assertEqual(backward.call_count,1)
            self.assertEqual(backward.call_args.kwargs['teacher_nodes'],NODES)
            self.assertEqual(backward.call_args.args[1].shape[1],8)
            shadow.assert_not_called()
            row=json.loads((Path(directory)/'train.jsonl').read_text())
            self.assertEqual(row['teacher_nodes'],list(NODES))
            self.assertEqual(row['stage_gaps'],[7,3,3,3])
            self.assertTrue(all(len(row[k])==4 for k in ('mean_D_move','mean_E_imit','mean_PCD')))
            self.assertFalse(row['is_best'])
            checkpoint=torch.load(Path(directory)/'ckpt-last.pth',weights_only=True)
            self.assertEqual(checkpoint['distillation'],runner.schedule(NODES))


if __name__ == '__main__':
    unittest.main()
