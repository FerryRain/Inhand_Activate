#!/usr/bin/env python3
import unittest

import numpy as np

from pose_error_experiment import (
    cloud_metrics,
    correlated_errors,
    perturb_poses,
    voxel_downsample,
)


class PoseErrorExperimentTests(unittest.TestCase):
    def test_correlated_error_has_requested_rms_and_anchor(self):
        rotation, translation = correlated_errors(
            frame_count=100,
            rotation_rms_deg=5.0,
            translation_rms_mm=3.0,
            temporal_correlation=0.85,
            seed=7,
            anchor_first_frame=True,
        )
        self.assertTrue(np.allclose(rotation[0], 0.0))
        self.assertTrue(np.allclose(translation[0], 0.0))
        rotation_rms = np.sqrt(np.mean(np.sum(rotation[1:] ** 2, axis=1)))
        translation_rms = np.sqrt(np.mean(np.sum(translation[1:] ** 2, axis=1)))
        self.assertAlmostEqual(np.degrees(rotation_rms), 5.0, places=10)
        self.assertAlmostEqual(1000.0 * translation_rms, 3.0, places=10)

    def test_perturb_pose_keeps_translation_and_rotation_errors_independent(self):
        poses = np.repeat(np.eye(4)[None, ...], 2, axis=0)
        poses[:, :3, 3] = [0.1, -0.2, 0.5]
        rotation = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2.0]])
        translation = np.array([[0.0, 0.0, 0.0], [0.001, 0.002, 0.003]])
        perturbed = perturb_poses(poses, rotation, translation)
        self.assertTrue(np.allclose(perturbed[0], poses[0]))
        self.assertTrue(
            np.allclose(perturbed[1, :3, 3] - poses[1, :3, 3], translation[1])
        )
        self.assertTrue(np.allclose(perturbed[1, :3, :3] @ [1, 0, 0], [0, 1, 0]))

    def test_identity_cloud_metrics_are_perfect(self):
        rng = np.random.RandomState(4)
        points = rng.uniform(-0.05, 0.05, size=(1000, 3))
        metrics = cloud_metrics(points, points, [2.0, 5.0, 10.0], 30000, seed=3)
        self.assertAlmostEqual(metrics["fscore_2mm"], 1.0)
        self.assertAlmostEqual(metrics["fscore_5mm"], 1.0)
        self.assertAlmostEqual(metrics["chamfer_l1_mm"], 0.0)

    def test_shifted_cloud_degrades_at_five_millimeters(self):
        points = np.column_stack(
            (np.arange(20, dtype=float) * 0.02, np.zeros(20), np.zeros(20))
        )
        shifted = points + np.array([0.0, 0.006, 0.0])
        metrics = cloud_metrics(shifted, points, [5.0, 10.0], 100, seed=3)
        self.assertAlmostEqual(metrics["fscore_5mm"], 0.0)
        self.assertAlmostEqual(metrics["fscore_10mm"], 1.0)
        self.assertAlmostEqual(metrics["chamfer_l1_mm"], 6.0)

    def test_voxel_downsample_averages_points(self):
        points = np.array(
            [[0.001, 0.001, 0.001], [0.003, 0.001, 0.001], [0.02, 0, 0]]
        )
        result = voxel_downsample(points, 0.01)
        self.assertEqual(result.shape, (2, 3))
        self.assertTrue(
            any(np.allclose(point, [0.002, 0.001, 0.001]) for point in result)
        )


if __name__ == "__main__":
    unittest.main()
