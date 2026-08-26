import unittest

import numpy as np

from utils.trajectory_metrics import (
    analyze_trajectory,
    bidirectional_distance_metrics,
    validate_budgets,
)


class ValidateBudgetsTests(unittest.TestCase):
    def test_preserves_valid_budget_order(self):
        self.assertEqual(validate_budgets([4, 1, 8], max_steps=8), [4, 1, 8])

    def test_rejects_empty_nonpositive_duplicate_and_out_of_range_budgets(self):
        invalid_cases = (
            ([], 30),
            ([0, 1], 30),
            ([-1, 1], 30),
            ([1, 1], 30),
            ([1, 31], 30),
        )
        for budgets, max_steps in invalid_cases:
            with self.subTest(budgets=budgets, max_steps=max_steps):
                with self.assertRaises(ValueError):
                    validate_budgets(budgets, max_steps=max_steps)

    def test_rejects_fractional_budget_instead_of_truncating_it(self):
        with self.assertRaises((TypeError, ValueError)):
            validate_budgets([1.5, 4], max_steps=4)


class BidirectionalDistanceTests(unittest.TestCase):
    def test_identical_clouds_have_zero_distance(self):
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=np.float64,
        )

        metrics = bidirectional_distance_metrics(points, points.copy())

        for key, value in metrics.items():
            with self.subTest(metric=key):
                self.assertAlmostEqual(value, 0.0, places=12)

    def test_single_point_translation_has_expected_euclidean_and_squared_distances(self):
        clean = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        pred = np.array([[1.0, 2.0, 2.0]], dtype=np.float64)

        metrics = bidirectional_distance_metrics(pred, clean)

        self.assertAlmostEqual(metrics['hd_euclidean'], 3.0, places=12)
        self.assertAlmostEqual(metrics['hd95_euclidean'], 3.0, places=12)
        self.assertAlmostEqual(metrics['hd_sq_x1e4'], 9.0e4, places=7)
        self.assertAlmostEqual(metrics['hd95_sq_x1e4'], 9.0e4, places=7)
        # Symmetric squared Chamfer is the sum of the two directional means.
        self.assertAlmostEqual(metrics['cpu_cd_sq_x1e4'], 18.0e4, places=7)


class AnalyzeTrajectoryTests(unittest.TestCase):
    def test_planar_normal_translation_is_decomposed_as_pure_normal_motion(self):
        clean = np.array(
            [
                [-1.0, -1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        translated = clean + np.array([0.0, 0.0, 0.25])
        states = np.stack([clean, translated], axis=0)

        rows, diagnostics = analyze_trajectory(
            states,
            clean,
            sigma0=0.01,
            step_size=0.3,
            decay=0.95,
            sample_points=len(clean),
            knn_k=16,
            normal_confidence_threshold=0.05,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(diagnostics['reference_normals'].shape, (4, 3))
        self.assertEqual(diagnostics['clean_neighbors'].shape, (4, 3))
        self.assertAlmostEqual(rows[1]['step_disp_mean'], 0.25, places=12)
        self.assertAlmostEqual(rows[1]['step_disp_rms'], 0.25, places=12)
        self.assertAlmostEqual(rows[1]['normal_disp_abs_mean'], 0.25, places=12)
        self.assertAlmostEqual(rows[1]['tangent_disp_mean'], 0.0, places=12)
        self.assertAlmostEqual(rows[1]['normal_energy_ratio'], 1.0, places=12)
        self.assertAlmostEqual(rows[1]['normal_valid_fraction'], 1.0, places=12)
        self.assertAlmostEqual(rows[1]['knn_churn_prev'], 0.0, places=12)
        self.assertAlmostEqual(rows[1]['local_edge_rel_change_prev'], 0.0, places=12)

    def test_single_point_cloud_is_safe_when_knn_k_exceeds_point_count(self):
        clean = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        states = np.stack(
            [clean, clean + np.array([0.0, 0.0, 0.1])],
            axis=0,
        )

        rows, diagnostics = analyze_trajectory(
            states,
            clean,
            sigma0=0.01,
            step_size=0.3,
            decay=0.95,
            sample_points=2048,
            knn_k=16,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(diagnostics['sample_indices'].tolist(), [0])
        self.assertEqual(diagnostics['clean_neighbors'].shape, (1, 0))
        self.assertEqual(diagnostics['reference_normals'].shape, (1, 3))
        self.assertAlmostEqual(rows[1]['step_disp_mean'], 0.1, places=12)
        self.assertEqual(rows[1]['normal_valid_fraction'], 0.0)
        self.assertTrue(np.isnan(rows[1]['normal_disp_abs_mean']))
        self.assertTrue(np.isnan(rows[1]['tangent_disp_mean']))

    def test_knn_k_is_capped_at_n_minus_one_for_a_small_cloud(self):
        clean = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        states = clean[None, ...]

        rows, diagnostics = analyze_trajectory(
            states,
            clean,
            sigma0=0.01,
            step_size=0.3,
            decay=0.95,
            sample_points=2048,
            knn_k=16,
        )

        self.assertEqual(diagnostics['clean_neighbors'].shape, (3, 2))
        self.assertAlmostEqual(rows[0]['knn_retention_clean'], 1.0, places=12)
        self.assertAlmostEqual(rows[0]['knn_churn_prev'], 0.0, places=12)


if __name__ == '__main__':
    unittest.main()
