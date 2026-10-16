"""Configuration/schedule regression cases; stdlib only."""
import copy
import unittest

from utils.curriculum_config import resolve_curriculum, require_supported_resume
from utils.training_input import (resolve_training_input, teacher_input_probability,
                                  summarize_input_stages, training_input_metadata,
                                  summarize_training_targets)


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

    def test_matched_target_is_explicit_and_preserves_original_schedule(self):
        original = resolve_training_input({'training_input': {'type': 'scheduled_rollout'}})
        self.assertNotIn('student_target', original)
        self.assertEqual(training_input_metadata(original, 8)['target'], 'teacher_target_node')
        for mode in ('scheduled_rollout', 'student_rollout'):
            supplied = dict(type=mode, student_target='teacher_from_student_input')
            before = copy.deepcopy(supplied)
            resolved = resolve_curriculum({'training_input': supplied})
            self.assertEqual(supplied, before)
            training = resolved['training_input']
            metadata = training_input_metadata(training, 8)
            self.assertEqual(metadata['target'], 'input_matched_teacher_interval')
            self.assertTrue(metadata['teacher_target_detached'])
            self.assertTrue(metadata['extra_teacher_rng_isolated'])
            self.assertEqual(metadata['teacher_probability'], .2 if mode == 'scheduled_rollout' else 0.)
            with self.assertRaises(NotImplementedError):
                require_supported_resume(resolved)
        explicit_old = resolve_training_input({'training_input': dict(
            type='scheduled_rollout', student_target='teacher_target_node')})
        self.assertEqual(training_input_metadata(explicit_old, 8)['target'], 'teacher_target_node')

    def test_retired_noise_policy_and_invalid_target_settings_are_rejected(self):
        cases = [dict(type='scheduled_rollout', noise_teacher_anchor=dict(
                     sigma_start=.01, sigma_end=.02, teacher_prob_high=.68)),
                 dict(type='teacher_forced', student_target='teacher_from_student_input')]
        cases.extend(dict(type='scheduled_rollout', student_target=value)
                     for value in (None, True, .5, 'typo', {}))
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                resolve_training_input({'training_input': case})

    def test_target_counts_are_sample_steps_not_microbatch_calls(self):
        rows = summarize_training_targets([[8, 0, 0], [2, 6, 24], [3, 5, 20], [1, 7, 28]])
        self.assertEqual(rows[0]['teacher_node_target_count'], 8)
        self.assertEqual(rows[1]['student_input_teacher_target_count'], 6)
        self.assertEqual(sum(row['extra_teacher_sample_steps'] for row in rows), 72)


if __name__ == '__main__':
    unittest.main()
