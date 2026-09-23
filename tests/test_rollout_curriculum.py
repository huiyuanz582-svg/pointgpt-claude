"""CPU synthetic tensors/mock forwards only; never starts training or CUDA."""
import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.curriculum_config import ALL_PATHS, resolve_curriculum


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rollout = load_module('rollout_cpu_tests', 'tools/rollout_curriculum.py')
runner = load_module('rollout_runner_cpu_tests', 'tools/runner_distill.py')


def settings(paths=None, batch=8):
    config = dict(curriculum_mode='dynamic_pcd', curriculum_metric={'type': 'rollout_aware'},
                  curriculum_search={'patch_batch': batch})
    if paths is not None:
        config['curriculum_search'].update(candidate_set='explicit', candidate_paths=paths)
    return resolve_curriculum(config)


def synthetic_bank(count=6):
    # Unequal groups in the 64-patch case exercise equal-group, not pooled averaging.
    ids = torch.arange(count) % 3
    sigma = torch.tensor([.005, .01, .02])[ids]
    x = torch.arange(count * 12, dtype=torch.float32).reshape(count, 4, 3) * .001
    states = torch.stack([x + sigma[:, None, None] * t * .1 for t in range(17)])
    return [(states, sigma, ids)]


class MockStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.91))
        self.bn = torch.nn.BatchNorm1d(3)
        self.dropout = torch.nn.Dropout(.2)
        self.calls = 0

    def forward(self, x, sigma, start, target):
        if self.training or torch.is_grad_enabled():
            raise AssertionError('Search must use eval/no_grad')
        self.calls += 1
        # Consume all CPU RNGs; outputs remain deterministic for reference scoring.
        torch.rand(1)
        random.random()
        np.random.rand()
        return x * self.weight + sigma[:, None, None] * (target - start) * .13 + target * .0001


def forward(model, x, sigma, start, target):
    return model(x, sigma, start, target)


class RolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_plus_formula_and_max_before_patch_mean(self):
        errors = torch.tensor([[8., 0., 0., 2.], [0., 8., 0., 2.]])
        components = rollout.path_components(errors)
        self.assertEqual(float(components[:, 2].mean()), 8.)
        self.assertEqual(float(errors.mean(0).max()), 4.)
        score = rollout.weighted_score(components, settings()['metric'])
        torch.testing.assert_close(score, torch.tensor([4.625, 4.625]))

    def test_all_prefix_counts_and_actual_8112_calls_with_64_patches(self):
        self.assertEqual(len(ALL_PATHS), 455)
        self.assertEqual(rollout.prefix_count(ALL_PATHS), 1014)
        model = MockStudent()
        report = rollout.search_rollout_paths(model, synthetic_bank(64), settings(), forward)
        self.assertEqual(model.calls, 8112)
        self.assertEqual(report['student_forward_calls'], 8112)
        self.assertEqual(report['theoretical_forward_calls'], 8112)
        self.assertEqual(report['patch_forward_evaluations'], 1014 * 64)
        self.assertEqual(len(report['candidates']), 455)
        self.assertEqual(report['nonfinite_count'], 0)
        print('MOCK all_455: theory=8112 actual=8112; unique prefixes=1014; CPU only')

    def test_prefix_matches_naive_rollout_and_robust_group_reduction(self):
        paths = [(0,4,8,12,16), (0,4,8,14,16), (0,7,8,14,16)]
        options = settings(paths, batch=3)
        model, bank = MockStudent(), synthetic_bank(7)
        report = rollout.search_rollout_paths(model, bank, options, forward)
        self.assertEqual(model.calls, rollout.prefix_count(paths) * 3)
        states, sigma, ids = bank[0]
        expected = {}
        with rollout.isolated_evaluation(model):
            for path in paths:
                x, errors = states[0], []
                denominator = ((states[0] - states[16]) ** 2).sum(-1).mean(-1) + 1e-12
                for start, target in zip(path[:-1], path[1:]):
                    x = forward(model, x, sigma * .95 ** start, start, target)
                    errors.append(((x - states[target]) ** 2).sum(-1).mean(-1) / denominator)
                errors = torch.stack(errors, -1)
                per_patch = errors[:, 3] + .25 * errors.mean(-1) + .25 * errors.max(-1).values
                group_scores = [float(per_patch[ids == group].double().mean()) for group in range(3)]
                expected[path] = sum(group_scores) / 3 + .25 * max(group_scores)
        for row in report['candidates']:
            self.assertAlmostEqual(row['Jrobust'], expected[tuple(row['nodes'])], places=5)
        self.assertEqual(report['nodes'], list(min(paths, key=lambda p: (expected[p], p))))
        self.assertAlmostEqual(report['runner_up_gap'],
                               report['candidates'][1]['Jrobust'] - report['candidates'][0]['Jrobust'])
        self.assertEqual([g['valid_patches'] for g in report['noise_groups']], [3, 2, 2])

    def test_rng_modes_parameters_buffers_and_gradients_restored(self):
        model = MockStudent().train()
        model.bn.eval()
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        modes = [m.training for m in model.modules()]
        before = copy.deepcopy(model.state_dict())
        rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
        rollout.search_rollout_paths(model, synthetic_bank(), settings([(0,4,8,12,16)]), forward)
        self.assertEqual(modes, [m.training for m in model.modules()])
        for key, value in before.items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        for parameter in model.parameters():
            self.assertTrue(torch.equal(parameter.grad, torch.ones_like(parameter)))
        torch.testing.assert_close(rng[0], torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(rng[1], random.getstate())
        np.testing.assert_equal(rng[2], np.random.get_state())

    def test_tiny_denominator_warns_without_clamping_and_failure_is_logged(self):
        bank = synthetic_bank()
        bank[0][0][16].copy_(bank[0][0][0])
        options = settings([(0,4,8,12,16)])
        with self.assertWarns(RuntimeWarning):
            report = rollout.search_rollout_paths(MockStudent(), bank, options, forward)
        self.assertAlmostEqual(report['D']['min'], 1e-12, delta=1e-19)
        self.assertEqual(report['small_denominator_count'], 6)
        model = MockStudent().train()
        model.bn.eval()
        modes = [m.training for m in model.modules()]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'failed.json'
            bad = lambda model, x, *args: x * float('nan')
            with self.assertRaises(FloatingPointError):
                rollout.search_rollout_paths(model, synthetic_bank(), options, bad, report_path=path)
            report = json.loads(path.read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertGreater(report['nonfinite_count'], 0)
        self.assertEqual(modes, [m.training for m in model.modules()])

    def test_empty_group_rejected_and_partial_batches_counted(self):
        with self.assertRaisesRegex(ValueError, 'Every configured noise group'):
            rollout.search_rollout_paths(MockStudent(), synthetic_bank(2),
                                         settings([(0,4,8,12,16)]), forward)

    def test_original_config_dispatches_to_old_cache_and_selector(self):
        config = SimpleNamespace(curriculum_mode='dynamic_pcd',
            dynamic_pcd=dict(initial_nodes=[0,4,8,12,16], target=.3, lambda_balance=1.,
                             update_every_epochs=1, calibration_patches=2, calibration_seed=2025,
                             interval_patch_batch=8), pcd_dynamic={})
        bank = [(torch.zeros(17,2,4,3), torch.tensor([.005,.01]))]
        cache = Mock(return_value=dict(metrics={(0,4): {key: torch.tensor([1.,3.])
                             for key in ('D_move','E_imit','PCD')}}, forward_calls=130))
        select = Mock(return_value={'nodes': [0,4,8,12,16]})
        fake = SimpleNamespace(build_interval_pcd_cache=cache, search_global_teacher_nodes=select)
        with patch.dict(sys.modules, {'tools.shadow_global_search': fake}):
            implicit = runner.update_dynamic_curriculum(None, bank, config)
            config.curriculum_metric = {'type': 'original_pcd'}
            config.training_input = {'type': 'teacher_forced'}
            explicit = runner.update_dynamic_curriculum(None, bank, config)
        self.assertEqual(implicit, explicit)
        self.assertEqual(cache.call_count, 2)
        self.assertEqual(select.call_count, 2)
        self.assertEqual(select.call_args.args[0][(0,4)]['PCD'], 2.)

    def test_epoch_cadence_saves_reports_without_training(self):
        options = settings([(0,4,8,12,16)])
        model, bank, history = MockStudent(), synthetic_bank(), []
        updater = lambda student, bank, config, **kw: rollout.search_rollout_paths(student, bank, options, forward, **kw)
        used = (0,4,8,12,16)
        with tempfile.TemporaryDirectory() as directory:
            same, report, state = rollout.update_rollout_epoch(model, bank, None, options, 1, used, history,
                                                               {}, directory, updater)
            self.assertIsNone(report)
            self.assertEqual(model.calls, 0)
            _, report, state = rollout.update_rollout_epoch(model, bank, None, options, 2, used, history,
                                                            {}, directory, updater)
            self.assertTrue((Path(directory) / 'curriculum_search_epoch0002.json').is_file())
            self.assertEqual(len(history), 1)
            self.assertFalse(state['rollout_curriculum']['implementation']['ema_enabled'])
            self.assertEqual(state['nodes_used_this_epoch'], list(used))

    def test_experiment_and_smoke_yaml_counts_with_mock_only(self):
        for filename, expected in (('distill_16to4_rollout_aware.yaml', 6084),
                                   ('distill_16to4_rollout_smoke.yaml', 30)):
            config = yaml.safe_load((ROOT / 'cfgs/PointGPT-L' / filename).read_text(encoding='utf-8'))
            options = resolve_curriculum(config)
            count = options['calibration']['patches_per_level'] * 3
            report = rollout.search_rollout_paths(MockStudent(), synthetic_bank(count), options, forward)
            self.assertEqual(report['student_forward_calls'], expected)
            self.assertEqual([g['valid_patches'] for g in report['noise_groups']], [count // 3] * 3)
            print(f'MOCK {filename}: theory={expected} actual={report["student_forward_calls"]}; CPU only')

    def test_new_dispatch_uses_existing_interval_forward_adapter(self):
        config = SimpleNamespace(curriculum_mode='dynamic_pcd', curriculum_metric={'type':'rollout_aware'},
                                 curriculum_search=dict(candidate_set='explicit', candidate_paths=[[0,4,8,12,16]]))
        with patch.dict(sys.modules, {'tools.rollout_curriculum': rollout}), \
                patch.object(runner, 'forward_student_interval', side_effect=forward) as adapter:
            result = runner.update_dynamic_curriculum(MockStudent(), synthetic_bank(), config)
        self.assertEqual(adapter.call_count, 4)
        self.assertEqual(result['student_forward_calls'], 4)

    def test_legacy_checkpoint_without_new_fields_and_new_checkpoint_inference(self):
        class TinyStudent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))
                self.step_condition = None

            def enable_step_condition(self):
                if self.step_condition is None:
                    self.step_condition = torch.nn.Linear(3, 1)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.pth'
            original = TinyStudent()
            torch.save(dict(base_model=original.state_dict(), distillation=runner.schedule()), path)
            loaded = TinyStudent()
            runner.load_student_checkpoint(loaded, path, None)
            self.assertIsNone(loaded.step_condition)
            self.assertEqual(loaded.distillation_teacher_nodes, (0,4,8,12,16))
            used, next_nodes = [0,4,8,12,16], [0,10,12,14,16]
            original.enable_step_condition()
            path = Path(directory) / 'rollout.pth'
            torch.save(dict(base_model=original.state_dict(), distillation=runner.schedule(used),
                            curriculum_mode='dynamic_pcd', nodes_used_this_epoch=used,
                            current_teacher_nodes=used, next_teacher_nodes=next_nodes,
                            rollout_curriculum=settings()), path)
            loaded = TinyStudent()
            runner.load_student_checkpoint(loaded, path, None)
            self.assertEqual(loaded.distillation_teacher_nodes, tuple(used))
            for key,value in original.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)

    def test_stratified_bank_uses_actual_production_normalization_noise_and_knn(self):
        namespace = dict(torch=torch, np=np, random=random, math=math, Dataset=torch.utils.data.Dataset)
        for file, names in (('datasets/scoredenoise/transforms.py', {'NormalizeUnitSphere', 'AddNoise'}),
                            ('datasets/ScoreDenoiseDataset.py', {'PairedPatchDataset'})):
            tree = ast.parse((ROOT / file).read_text(encoding='utf-8'))
            nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
            exec(compile(ast.Module(body=nodes, type_ignores=[]), file, 'exec'), namespace)
        def factory(sigma):
            normal, noise = namespace['NormalizeUnitSphere'](), namespace['AddNoise'](sigma, sigma)
            return lambda sample: noise(normal(sample))
        clouds = [dict(pcl_clean=torch.arange(36, dtype=torch.float32).reshape(12,3) + i, name=str(i))
                  for i in range(3)]
        dataset = namespace['PairedPatchDataset']([clouds], 1, patch_size=4, num_patches=1,
                     noise_min=.005, noise_max=.02, transform=factory(.01), flag='train', oversample_factor=2)
        original_transform = dataset.transform
        options = settings([(0,4,8,12,16)], batch=2)
        options['calibration']['patches_per_level'] = 2
        teacher = MockStudent().eval()
        capture = lambda model, x, sigma, batch, **kw: torch.stack([x + k * sigma[:,None,None] for k in range(17)])
        config = SimpleNamespace(teacher_patch_batch=1)
        rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
        first, meta = rollout.build_stratified_bank(config, teacher, dataset, options, capture, transform_factory=factory)
        second, repeated = rollout.build_stratified_bank(config, teacher, dataset, options, capture, transform_factory=factory)
        self.assertEqual(meta, repeated)
        self.assertEqual(meta['patches'], 6)
        self.assertIs(dataset.transform, original_transform)
        for (a,s,g), (b,_,_) in zip(first, second):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertTrue(torch.allclose(s, torch.full_like(s, options['calibration']['noise_levels'][int(g[0])])))
        torch.testing.assert_close(rng[0], torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(rng[1], random.getstate())
        np.testing.assert_equal(rng[2], np.random.get_state())


if __name__ == '__main__':
    unittest.main()
