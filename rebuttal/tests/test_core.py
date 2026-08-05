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

from benchmark.assets import load_mesh
from benchmark.config import load_config
from benchmark.environment import KinematicRGBDEnv
from benchmark.evaluation import ReconstructionEvaluator
from benchmark.fusion import PointCloudFusion
from benchmark.geometry import ACTIONS, map_nbv_to_action, next_view_direction


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


if __name__ == "__main__":
    unittest.main()
