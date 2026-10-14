"""Configuration/schedule regression cases; stdlib only."""
import copy
import unittest

from utils.curriculum_config import resolve_curriculum, require_supported_resume
from utils.training_input import (resolve_training_input, teacher_input_probability,
                                  summarize_input_stages, training_input_metadata,
                                  noise_anchored_teacher_probabilities,
                                  new_noise_anchor_statistics, accumulate_noise_anchor_statistics,
                                  summarize_noise_anchor_statistics)


class TrainingInputConfigTests(unittest.TestCase):
    def test_default_and_schedule_boundaries(self):
        self.assertEqual(resolve_training_input({}), {'type': 'teacher_forced'})
        training = resolve_training_input({'training_input': {'type': 'scheduled_rollout'}})
        expected = [1, 1, 1, .84, .68, .52, .36, .2, .2, .2]
        for epoch, probability in enumerate(expected, 1):
            self.assertAlmostEqual(teacher_input_probability(training, epoch), probability)
        for mode, probability in [('teacher_forced', 1), ('student_rollout', 0)]:
            self.assertEqual(teacher_input_probability({'type': mode}, 3), probability)
        instant = dict(training, warmup_epochs=0, decay_epochs=1)
        self.assertEqual(teacher_input_probability(instant, 1), .2)
        constant = dict(training, teacher_prob_start=.5, teacher_prob_end=.5)
        self.assertEqual(teacher_input_probability(constant, 20), .5)
        for epoch in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                teacher_input_probability(training, epoch)

    def test_policy_is_independent_of_path_metric_and_does_not_mutate(self):
        for metric in ('original_pcd', 'rollout_aware'):
            for mode in ('teacher_forced', 'scheduled_rollout', 'student_rollout'):
                config = dict(curriculum_mode='dynamic_pcd', curriculum_metric={'type': metric},
                              training_input={'type': mode})
                original = copy.deepcopy(config)
                resolved = resolve_curriculum(config)
                self.assertEqual(config, original)
                self.assertEqual(resolved['metric']['type'], metric)
                self.assertEqual(resolved['training_input']['type'], mode)
                if metric == 'rollout_aware':
                    self.assertEqual(resolved['implementation']['scheduled_training_enabled'],
                                     mode == 'scheduled_rollout')
                    self.assertEqual(resolved['implementation']['student_rollout_training_enabled'],
                                     mode == 'student_rollout')

    def test_invalid_or_ignored_settings_are_rejected(self):
        cases = [None, [], {'type': 'typo'}, {'type': 'teacher_forced', 'teacher_prob_end': .2},
                 {'type': 'student_rollout', 'sampling_seed': 0},
                 {'type': 'scheduled_rollout', 'detach': False}]
        for key, bad in (('warmup_epochs', -1), ('warmup_epochs', True), ('decay_epochs', 0),
                         ('sampling_seed', 2 ** 32), ('sampling_seed', 1.5),
                         ('teacher_prob_end', float('nan')), ('teacher_prob_start', float('inf')),
                         ('teacher_prob_end', -.1), ('teacher_prob_end', 1.1),
                         ('teacher_prob_end', True), ('teacher_prob_end', '0.2')):
            cases.append({'type': 'scheduled_rollout', key: bad})
        cases.append(dict(type='scheduled_rollout', teacher_prob_start=.2, teacher_prob_end=.8))
        for supplied in cases:
            with self.subTest(supplied=supplied), self.assertRaises(ValueError):
                resolve_training_input({'training_input': supplied})

    def test_new_resume_and_both_migration_directions_are_rejected(self):
        legacy = resolve_curriculum({})
        require_supported_resume(legacy, {'epoch': 5})
        for mode in ('scheduled_rollout', 'student_rollout'):
            new = resolve_curriculum({'training_input': {'type': mode}})
            with self.assertRaises(NotImplementedError):
                require_supported_resume(new, {'epoch': 5})
            with self.assertRaises(NotImplementedError):
                require_supported_resume(legacy, {'training_input': new['training_input']})

    def test_source_loss_means_use_counts_and_absent_sources_are_null(self):
        rows = summarize_input_stages([[3, 1, 6., 9.], [4, 0, 8., 0.], [0, 4, 0., 12.]])
        self.assertEqual(rows[0]['teacher_input_loss'], 2.)
        self.assertEqual(rows[0]['student_input_loss'], 9.)
        self.assertEqual(rows[0]['teacher_fraction'], .75)
        self.assertIsNone(rows[1]['student_input_loss'])
        self.assertIsNone(rows[2]['teacher_input_loss'])

    def test_noise_anchor_preserves_warmup_and_low_noise_schedule(self):
        supplied = {'type': 'scheduled_rollout', 'noise_teacher_anchor': {
            'sigma_start': .01, 'sigma_end': .02, 'teacher_prob_high': .68}}
        original = copy.deepcopy(supplied)
        training = resolve_training_input({'training_input': supplied})
        sigmas = [.005, .01, (.01 * .02) ** .5, .02, .03]
        expected = [.2, .2, .44, .68, .68]
        for actual, value in zip(noise_anchored_teacher_probabilities(training, 8, sigmas), expected):
            self.assertAlmostEqual(actual, value)
        self.assertEqual(noise_anchored_teacher_probabilities(training, 1, sigmas), [1.] * 5)
        self.assertEqual(noise_anchored_teacher_probabilities(training, 4, sigmas), [.84] * 5)
        self.assertEqual(supplied, original)
        metadata = training_input_metadata(training, 8)
        self.assertEqual(metadata['teacher_probability'], .2)
        self.assertEqual(metadata['teacher_probability_scope'], 'base_before_noise_anchor')
        self.assertEqual(metadata['noise_probability_basis'], 'initial_sigma0')
        legacy = resolve_training_input({'training_input': {'type': 'scheduled_rollout'}})
        self.assertEqual(noise_anchored_teacher_probabilities(legacy, 8, sigmas), [.2] * 5)
        self.assertNotIn('noise_teacher_anchor', legacy)
        self.assertNotIn('teacher_probability_scope', training_input_metadata(legacy, 8))
        for bad in (0, -1, True, float('nan'), float('inf')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                noise_anchored_teacher_probabilities(training, 8, [bad])

    def test_noise_anchor_configuration_rejects_ignored_or_invalid_options(self):
        valid = dict(sigma_start=.01, sigma_end=.02, teacher_prob_high=.68)
        invalid = [None, {}, [], dict(valid, unknown=1),
                   dict(valid, sigma_start=0), dict(valid, sigma_start=.02),
                   dict(valid, sigma_end=.009), dict(valid, teacher_prob_high=.1),
                   dict(valid, teacher_prob_high=1.1), dict(valid, sigma_start=True),
                   dict(valid, sigma_end=float('inf')), dict(valid, teacher_prob_high='0.68')]
        for anchor in invalid:
            with self.subTest(anchor=anchor), self.assertRaises(ValueError):
                resolve_training_input({'training_input': {'type': 'scheduled_rollout',
                                                         'noise_teacher_anchor': anchor}})
        for mode in ('teacher_forced', 'student_rollout'):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                resolve_training_input({'training_input': {'type': mode, 'noise_teacher_anchor': valid}})
        training = resolve_training_input({'training_input': {'type': 'scheduled_rollout',
                                                             'noise_teacher_anchor': valid}})
        with self.assertRaises(NotImplementedError):
            require_supported_resume(resolve_curriculum({'training_input': training}))

    def test_noise_band_counts_use_actual_stage_masks(self):
        training = resolve_training_input({'training_input': {'type': 'scheduled_rollout',
            'noise_teacher_anchor': dict(sigma_start=.01, sigma_end=.02, teacher_prob_high=.68)}})
        sigmas, probabilities = [.005, .01, .02], [.2, .2, .68]
        mask = [[True, True, True], [False, True, True], [True, False, True], [False, False, True]]
        totals = new_noise_anchor_statistics()
        accumulate_noise_anchor_statistics(totals, training, sigmas, probabilities, mask)
        low, high = summarize_noise_anchor_statistics(totals, training)
        self.assertEqual((low['patches'], high['patches']), (2, 1))
        self.assertEqual([row['teacher_count'] for row in low['stages']], [1, 1, 0])
        self.assertEqual([row['teacher_count'] for row in high['stages']], [1, 1, 1])
        self.assertAlmostEqual(low['planned_teacher_probability_mean'], .2)
        self.assertAlmostEqual(high['planned_teacher_probability_mean'], .68)
        empty = summarize_noise_anchor_statistics(new_noise_anchor_statistics(), training)
        self.assertIsNone(empty[0]['planned_teacher_probability_mean'])
        self.assertIsNone(empty[1]['stages'][0]['teacher_fraction'])
        with self.assertRaises(ValueError):
            accumulate_noise_anchor_statistics(totals, training, sigmas[:-1], probabilities, mask)


if __name__ == '__main__':
    unittest.main()
