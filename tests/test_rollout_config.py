"""CPU-only stdlib configuration checks; no model or training imports."""
import copy
import unittest

from utils.curriculum_config import candidate_paths, resolve_curriculum, require_supported_resume


def config():
    return dict(curriculum_mode='dynamic_pcd', curriculum_metric={'type': 'rollout_aware'})


class ConfigTests(unittest.TestCase):
    def test_missing_and_explicit_legacy_are_identical_and_do_not_mutate(self):
        old = {'dynamic_pcd': {'target': 0.3}}
        before = copy.deepcopy(old)
        expected = dict(metric={'type': 'original_pcd'}, training_input={'type': 'teacher_forced'})
        self.assertEqual(resolve_curriculum(old), expected)
        self.assertEqual(old, before)
        self.assertEqual(resolve_curriculum(dict(old, curriculum_metric={'type': 'original_pcd'},
                                               training_input={'type': 'teacher_forced'})), expected)
        require_supported_resume(expected, {'epoch': 5})

    def test_rollout_defaults_and_inactive_features(self):
        resolved = resolve_curriculum(config())
        self.assertEqual(resolved['calibration']['noise_levels'], [.005, .01, .02])
        self.assertFalse(resolved['implementation']['ema_enabled'])
        self.assertFalse(resolved['implementation']['switch_suppression_enabled'])
        self.assertEqual(len(candidate_paths(resolved['search'])), 455)

    def test_weights_reject_negative_and_nonfinite(self):
        for key in ('alpha', 'beta', 'gamma', 'lambda_worst'):
            for value in (-.1, float('nan'), float('inf'), True):
                cfg = config()
                cfg['curriculum_metric'][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    resolve_curriculum(cfg)
        cfg = config()
        cfg['curriculum_metric'].update(alpha=0, beta=0, gamma=0)
        with self.assertRaises(ValueError):
            resolve_curriculum(cfg)

    def test_explicit_candidates_and_rejections(self):
        a, b = [0, 4, 8, 12, 16], [0, 7, 10, 13, 16]
        cfg = config()
        cfg['curriculum_search'] = dict(candidate_set='explicit', candidate_paths=[b, a])
        self.assertEqual(candidate_paths(resolve_curriculum(cfg)['search']), (tuple(a), tuple(b)))
        for paths in ([], [a, a], [[0, 4, 4, 12, 16]], [[0, 4.0, 8, 12, 16]]):
            cfg['curriculum_search']['candidate_paths'] = paths
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                resolve_curriculum(cfg)

    def test_no_3_percent_or_scheduled_training_or_silent_settings(self):
        cfg = config()
        cfg['curriculum_calibration'] = {'noise_levels': [.005, .01, .03]}
        with self.assertRaises(ValueError):
            resolve_curriculum(cfg)
        with self.assertRaises(NotImplementedError):
            resolve_curriculum({'training_input': {'type': 'scheduled_rollout'}})
        with self.assertRaises(ValueError):
            resolve_curriculum({'curriculum_search': {'update_every_epochs': 2}})
        with self.assertRaises(NotImplementedError):
            require_supported_resume(resolve_curriculum(config()))
        with self.assertRaises(NotImplementedError):
            require_supported_resume(resolve_curriculum({}), {'rollout_curriculum': {}})


if __name__ == '__main__':
    unittest.main()
