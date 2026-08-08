from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

REBUTTAL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = REBUTTAL_ROOT.parent
if str(REBUTTAL_ROOT) not in sys.path:
    sys.path.insert(0, str(REBUTTAL_ROOT))

from rebuttal.benchmark.assets import load_mesh
from rebuttal.benchmark.config import load_config
from rebuttal.benchmark.environment import KinematicRGBDEnv
from rebuttal.benchmark.evaluation import ReconstructionEvaluator
from rebuttal.benchmark.fusion import PointCloudFusion
from rebuttal.benchmark.geometry import (
    ACTIONS,
    action_rotation,
    candidate_action_assignments,
    fibonacci_sphere,
    map_nbv_to_action,
    next_view_direction,
    rotation_error_deg,
)
from rebuttal.benchmark.planners.pose_novelty import PoseNoveltyPlanner


class CoreBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(str(REBUTTAL_ROOT / "configs" / "base.yaml"))
        cls.cfg["camera"].update({
            "width": 160,
            "height": 90,
            "intrinsics": [
                [75.94186, 0.0, 80.14806],
                [0.0, 75.92567, 45.91211],
                [0.0, 0.0, 1.0],
            ],
        })
        cls.cfg["candidates"]["count"] = 64
        cls.cfg["evaluation"]["gt_sample_count"] = 5000
        cls.manifest = {
            name: str(REBUTTAL_ROOT / "assets" / "objects" / (name + ".ply"))
            for name in cls.cfg["assets"]["objects"]
        }

    def test_all_assets_are_watertight_and_scaled(self):
        for name, path in self.manifest.items():
            mesh = load_mesh(path)
            self.assertTrue(mesh.is_watertight(), name)
            diagonal = np.linalg.norm(mesh.get_axis_aligned_bounding_box().get_extent())
            self.assertAlmostEqual(diagonal, 0.12, places=4, msg=name)

    def test_action_mapper_recovers_each_reachable_view(self):
        env = KinematicRGBDEnv(self.cfg, self.manifest["Cube"], seed=0)
        observation = env.reset(0)
        for action in ACTIONS:
            desired = next_view_direction(observation.executed_pose, action, env.angles_deg)
            selected, _ = map_nbv_to_action(desired, observation.executed_pose, env.angles_deg)
            self.assertEqual(selected, action)

    def test_pose_novelty_scores_executable_actions_from_pose_history(self):
        env = KinematicRGBDEnv(self.cfg, self.manifest["Cube"], seed=0)
        observation = env.reset(0)
        fusion = PointCloudFusion(self.cfg)
        fusion.update(observation)
        planner = PoseNoveltyPlanner(self.cfg)
        planner.reset(observation, fusion)
        candidates = fibonacci_sphere(64)
        scores = planner.score_views(candidates)
        assignments = candidate_action_assignments(
            candidates, observation.executed_pose, env.angles_deg
        )
        self.assertEqual(scores.shape, (64,))
        self.assertTrue(np.all(np.isfinite(scores)))
        for index, action in enumerate(ACTIONS):
            self.assertTrue(np.allclose(scores[assignments == index], planner.last_action_scores[action]))
        self.assertFalse(planner.diagnostics()["uses_reconstruction"])

    def test_render_fusion_and_evaluation(self):
        env = KinematicRGBDEnv(self.cfg, self.manifest["Cube"], seed=0)
        env.reset(0)
        observations = env.bootstrap_observations(15.0)
        fusion = PointCloudFusion(self.cfg)
        for observation in observations:
            fusion.update(observation)
        self.assertGreater(len(fusion.points), 50)
        self.assertTrue(np.all(np.isfinite(fusion.points)))
        mesh = load_mesh(self.manifest["Cube"])
        evaluator = ReconstructionEvaluator(mesh, self.cfg, seed=7)
        evaluator_repeat = ReconstructionEvaluator(mesh, self.cfg, seed=7)
        np.testing.assert_array_equal(evaluator.gt_points, evaluator_repeat.gt_points)
        metrics = evaluator.evaluate(fusion.points)
        self.assertGreater(metrics["f@5"], 0.0)
        self.assertLessEqual(metrics["f@5"], 1.0)

    def test_sixview_protocol_has_one_initial_observation(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["environment"]["bootstrap_offsets_deg"] = [0.0]
        env = KinematicRGBDEnv(cfg, self.manifest["Cube"], seed=0)
        env.reset(0)
        observations = env.bootstrap_observations(15.0)
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].step, 0)

    def test_realistic_noise_is_seeded_and_affects_observation(self):
        cfg = load_config(str(REBUTTAL_ROOT / "configs" / "ablation_realistic.yaml"))
        cfg["camera"] = copy.deepcopy(self.cfg["camera"])
        first = KinematicRGBDEnv(cfg, self.manifest["Cube"], seed=2026)
        second = KinematicRGBDEnv(cfg, self.manifest["Cube"], seed=2026)
        first.reset(0)
        second.reset(0)
        observation = first.step("minus_x")
        repeated = second.step("minus_x")
        np.testing.assert_array_equal(observation.mask, repeated.mask)
        np.testing.assert_allclose(observation.depth, repeated.depth)
        np.testing.assert_allclose(observation.executed_pose, repeated.executed_pose)
        self.assertEqual(len(first.action_residuals["minus_x"]), 6)
        self.assertLess(observation.visibility_ratio, 0.65)
        self.assertGreater(observation.pose_error_deg, 0.0)
        self.assertGreater(observation.translation_error_m, 0.0)

    def test_action_sequence_contains_intermediate_orientations(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["environment"]["level"] = "clean"
        env = KinematicRGBDEnv(cfg, self.manifest["Cube"], seed=0)
        initial = env.reset(0)
        observations = env.step_sequence("minus_y", 5)
        self.assertEqual(len(observations), 5)
        per_frame = [
            np.linalg.norm(observations[index].gt_pose[:3, :3] - observations[index - 1].gt_pose[:3, :3])
            for index in range(1, len(observations))
        ]
        self.assertTrue(all(value > 1e-3 for value in per_frame))
        expected = action_rotation("minus_y", env.angles_deg) @ initial.gt_pose[:3, :3]
        np.testing.assert_allclose(observations[-1].gt_pose[:3, :3], expected, atol=1e-7)

    def test_continuous_video_sequence_records_rate_and_uses_exact_pose(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["environment"].update({
            "level": "realistic",
            "action_duration_s": 0.4,
            "sensor_rate_hz": 5.0,
            "frames_per_action": 2,
            "stress": {"name": "continuous_test"},
        })
        cfg["environment"]["robustness"].update({
            "pose_jitter_deg": 0.0,
            "drift_per_step_deg": 0.0,
            "translation_jitter_mm": 0.0,
            "translation_drift_per_step_mm": 0.0,
            "pose_outlier_probability": 0.0,
            "occlusion_fraction": 0.25,
            "occlusion_fraction_std": 0.05,
            "action_progress_scale_range": [0.7, 0.7],
            "action_stall_probability": 0.0,
            "grip_slip_probability": 0.0,
        })
        env = KinematicRGBDEnv(cfg, self.manifest["Cube"], seed=2026)
        env.reset(2)
        clone = env.clone()
        observations = env.step_sequence("minus_x", 2)
        repeated = clone.step_sequence("minus_x", 2)
        self.assertEqual(len(observations), 2)
        self.assertEqual(env.last_trajectory_metadata, clone.last_trajectory_metadata)
        self.assertEqual(env.last_trajectory_metadata["replay_frames"], 2)
        self.assertEqual(env.last_trajectory_metadata["replay_rate_hz"], 5.0)
        self.assertEqual(env.last_trajectory_metadata["duration_s"], 0.4)
        self.assertTrue(env.last_trajectory_metadata["continuous_rendering"])
        for observation, repeat in zip(observations, repeated):
            self.assertAlmostEqual(observation.pose_error_deg, 0.0, places=8)
            self.assertAlmostEqual(observation.translation_error_m, 0.0, places=8)
            np.testing.assert_array_equal(observation.mask, repeat.mask)
            np.testing.assert_allclose(observation.depth, repeat.depth)

    def test_stall_and_slip_persist_across_future_actions_and_clones(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["environment"]["level"] = "realistic"
        cfg["environment"]["stress"] = {"name": "persistent_action_mismatch"}
        cfg["environment"]["robustness"].update({
            "action_progress_scale_range": [1.0, 1.0],
            "action_stall_probability": 1.0,
            "action_stall_scale_range": [0.2, 0.2],
            "grip_slip_probability": 1.0,
            "grip_slip_angle_deg_range": [20.0, 20.0],
            "slip_axis_drift_deg_range": [25.0, 25.0],
            "slip_axis_drift_max_deg": 50.0,
        })
        env = KinematicRGBDEnv(cfg, self.manifest["Cube"], seed=2026)
        initial = env.reset(3)
        first = env.step("minus_x")
        self.assertIn("action_stall", first.fault_tags)
        self.assertIn("grip_slip", first.fault_tags)
        self.assertAlmostEqual(env.last_action_progress_scale, 0.2, places=8)
        self.assertEqual(env.slip_event_count, 1)
        self.assertGreater(
            np.linalg.norm(env.action_axis_frame - np.eye(3)), 0.1
        )
        realized = first.gt_pose[:3, :3] @ initial.gt_pose[:3, :3].T
        self.assertGreater(
            rotation_error_deg(action_rotation("minus_x", env.angles_deg), realized),
            20.0,
        )

        cloned = env.clone()
        original_next = env.step("plus_z")
        cloned_next = cloned.step("plus_z")
        np.testing.assert_allclose(original_next.gt_pose, cloned_next.gt_pose)
        self.assertEqual(original_next.fault_tags, cloned_next.fault_tags)
        self.assertIn("persistent_axis_misalignment", original_next.fault_tags)

    def test_empirical_six_second_trajectory_replay_is_paired(self):
        cfg = load_config(
            str(REBUTTAL_ROOT / "configs" / "continuous_empirical_one_step_motion.yaml")
        )
        cfg["camera"] = copy.deepcopy(self.cfg["camera"])
        cfg["episode"] = {
            "object": "Cube",
            "initial_pose_seed": 4,
            "planner": "shared_one_step",
        }
        env = KinematicRGBDEnv(cfg, self.manifest["Cube"], seed=2030)
        initial = env.reset(4)
        clone = env.clone()
        observations = env.step_sequence("minus_x", 30)
        cloned_observations = clone.step_sequence("minus_x", 30)
        self.assertEqual(len(observations), 30)
        self.assertEqual(len(cloned_observations), 30)
        self.assertEqual(env.last_trajectory_metadata, clone.last_trajectory_metadata)
        self.assertAlmostEqual(env.last_trajectory_metadata["duration_s"], 6.0)
        self.assertGreater(env.last_trajectory_metadata["first_second_error_deg"], 0.0)
        np.testing.assert_allclose(observations[-1].gt_pose, cloned_observations[-1].gt_pose)
        self.assertGreater(
            rotation_error_deg(
                action_rotation("minus_x", env.angles_deg),
                observations[-1].gt_pose[:3, :3] @ initial.gt_pose[:3, :3].T,
            ),
            1.0,
        )

    def test_sparse_stress_marks_removed_surface(self):
        cfg = load_config(str(REBUTTAL_ROOT / "configs" / "stress_sparse.yaml"))
        cfg["camera"] = copy.deepcopy(self.cfg["camera"])
        env = KinematicRGBDEnv(cfg, self.manifest["Cube"], seed=2041)
        env.reset(15)
        observation = env.bootstrap_observations(15.0)[0]
        self.assertIn("structured_dropout", observation.fault_tags)
        self.assertIsNotNone(observation.target_points_object)
        self.assertGreater(len(observation.target_points_object), 20)
        self.assertLess(observation.visibility_ratio, 0.75)

    def test_scheduled_outlier_is_exact_and_paired(self):
        cfg = load_config(str(REBUTTAL_ROOT / "configs" / "stress_ghost.yaml"))
        cfg["camera"] = copy.deepcopy(self.cfg["camera"])
        first = KinematicRGBDEnv(cfg, self.manifest["Cube"], seed=2041)
        second = KinematicRGBDEnv(cfg, self.manifest["Cube"], seed=2041)
        first.reset(15)
        second.reset(15)
        observation = first.step("minus_x")
        repeated = second.step("minus_x")
        self.assertIn("scheduled_pose_outlier", observation.fault_tags)
        self.assertAlmostEqual(observation.pose_error_deg, 15.0, places=6)
        self.assertAlmostEqual(observation.translation_error_m, 0.008, places=8)
        np.testing.assert_allclose(observation.executed_pose, repeated.executed_pose)

    def test_occlusion_schedule_is_step_specific(self):
        cfg = load_config(str(REBUTTAL_ROOT / "configs" / "stress_hole.yaml"))
        cfg["camera"] = copy.deepcopy(self.cfg["camera"])
        env = KinematicRGBDEnv(cfg, self.manifest["Cube"], seed=2041)
        env.reset(15)
        initial = env.bootstrap_observations(15.0)[0]
        first = env.step("minus_x")
        second = env.step("minus_x")
        self.assertAlmostEqual(initial.visibility_ratio, 0.55, places=2)
        self.assertAlmostEqual(first.visibility_ratio, 0.55, places=2)
        self.assertAlmostEqual(second.visibility_ratio, 0.70, places=2)

    def test_contiguous_registration_gap_is_local_and_paired(self):
        cfg = load_config(
            str(REBUTTAL_ROOT / "configs" / "visited_registration_gap.yaml")
        )
        cfg["camera"] = copy.deepcopy(self.cfg["camera"])
        cfg["episode"] = {
            "object": "Cube",
            "initial_pose_seed": 15,
            "planner": "shared_one_step",
        }
        first = KinematicRGBDEnv(cfg, self.manifest["Cube"], seed=2041)
        first.reset(15)
        second = first.clone()
        observation = first.step("minus_x")
        repeated = second.step("minus_x")
        self.assertIn(
            "contiguous_depth_registration_failure", observation.fault_tags
        )
        self.assertIsNotNone(observation.target_points_object)
        self.assertGreater(len(observation.target_points_object), 20)
        self.assertLess(observation.visibility_ratio, 0.35)
        np.testing.assert_array_equal(observation.mask, repeated.mask)
        np.testing.assert_allclose(
            observation.target_points_object, repeated.target_points_object
        )


if __name__ == "__main__":
    unittest.main()
