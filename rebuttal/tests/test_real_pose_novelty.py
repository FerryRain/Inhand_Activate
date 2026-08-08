from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from Active.pose_novelty import (
    PoseNoveltyAxisPlanner,
    predict_pose_after_axis,
)


def make_pose(rotation=None, translation=None):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.eye(3) if rotation is None else np.asarray(rotation)
    pose[:3, 3] = [0.0, 0.0, 1.0] if translation is None else translation
    return pose


def rotation_z(degrees):
    radians = np.radians(float(degrees))
    cosine, sine = np.cos(radians), np.sin(radians)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


class RealPoseNoveltyTest(unittest.TestCase):
    def setUp(self):
        self.angles = {"x": 90.0, "y": 90.0, "z": 90.0}
        self.initial = make_pose()

    def test_selects_view_farthest_from_pose_history(self):
        planner = PoseNoveltyAxisPlanner(self.angles)
        planner.add_pose(self.initial, "initial")

        first_axis, first = planner.select_axis(self.initial, self.initial)
        self.assertEqual(first_axis, "x")
        self.assertAlmostEqual(first["scores_deg"]["x"], 90.0, places=6)
        self.assertAlmostEqual(first["scores_deg"]["y"], 90.0, places=6)
        self.assertAlmostEqual(first["scores_deg"]["z"], 0.0, places=6)

        visited_x = predict_pose_after_axis(self.initial, self.initial, "x", self.angles)
        planner.add_pose(visited_x, "visited_x")
        second_axis, second = planner.select_axis(self.initial, self.initial)
        self.assertEqual(second_axis, "y")
        self.assertGreater(second["scores_deg"]["y"], second["scores_deg"]["x"])

    def test_scores_are_invariant_to_a_common_camera_rotation(self):
        planner = PoseNoveltyAxisPlanner(self.angles)
        planner.add_pose(self.initial, "initial")
        visited = predict_pose_after_axis(self.initial, self.initial, "x", self.angles)
        planner.add_pose(visited, "visited")
        scores, _ = planner.score_axes(self.initial, self.initial)

        camera_rotation = rotation_z(37.0)

        def change_camera_frame(pose):
            transformed = pose.copy()
            transformed[:3, :3] = camera_rotation @ pose[:3, :3]
            transformed[:3, 3] = camera_rotation @ pose[:3, 3]
            return transformed

        transformed_initial = change_camera_frame(self.initial)
        transformed_visited = change_camera_frame(visited)
        transformed_planner = PoseNoveltyAxisPlanner(self.angles)
        transformed_planner.add_pose(transformed_initial, "initial")
        transformed_planner.add_pose(transformed_visited, "visited")
        transformed_scores, _ = transformed_planner.score_axes(
            transformed_initial,
            transformed_initial,
        )

        for axis in ("x", "y", "z"):
            self.assertAlmostEqual(scores[axis], transformed_scores[axis], places=10)

    def test_keyframe_history_uses_only_fusion_valid_modalities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ("poses", "rgb_full", "depth", "mask"):
                (root / folder).mkdir()

            np.savetxt(root / "poses/000000.txt", self.initial)
            (root / "rgb_full/000000.jpg").touch()
            (root / "depth/000000.png").touch()
            (root / "mask/000000.png").touch()

            np.savetxt(root / "poses/000001.txt", self.initial)
            (root / "rgb_full/000001.jpg").touch()
            (root / "depth/000001.png").touch()

            planner = PoseNoveltyAxisPlanner(self.angles)
            self.assertEqual(planner.update_from_keyframes(root), 1)
            self.assertEqual(planner.history_size, 1)
            self.assertEqual(planner.update_from_keyframes(root), 0)

    def test_state_round_trip_preserves_decisions(self):
        planner = PoseNoveltyAxisPlanner(self.angles)
        planner.add_pose(self.initial, "initial")
        planner.add_pose(
            predict_pose_after_axis(self.initial, self.initial, "x", self.angles),
            "visited_x",
        )
        expected_axis, expected = planner.select_axis(self.initial, self.initial)

        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "pose_novelty_state.json"
            planner.save_state(state_path)
            restored = PoseNoveltyAxisPlanner(self.angles)
            self.assertTrue(restored.load_state(state_path))
            actual_axis, actual = restored.select_axis(self.initial, self.initial)

        self.assertEqual(expected_axis, actual_axis)
        self.assertEqual(expected["scores_deg"], actual["scores_deg"])


if __name__ == "__main__":
    unittest.main()
