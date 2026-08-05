from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import open3d as o3d
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial.transform import Rotation

from .assets import load_mesh
from .geometry import ACTION_AXES, action_rotation, axis_angle, random_rotation, rotation_error_deg


@dataclass
class Observation:
    rgb: np.ndarray
    depth: np.ndarray
    mask: np.ndarray
    gt_pose: np.ndarray
    executed_pose: np.ndarray
    camera_intrinsics: np.ndarray
    step: int
    pose_error_deg: float = 0.0
    visibility_ratio: float = 1.0


class KinematicRGBDEnv:
    """Fixed-camera object reorientation environment using Open3D CPU ray casting."""

    def __init__(self, cfg: Dict, mesh_path: str, seed: int = 0):
        self.cfg = cfg
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)
        self.mesh_path = mesh_path
        self.mesh_object = load_mesh(mesh_path)
        self.height = int(cfg["camera"]["height"])
        self.width = int(cfg["camera"]["width"])
        self.K = np.asarray(cfg["camera"]["intrinsics"], dtype=np.float64).reshape(3, 3)
        self.distance = float(cfg["camera"]["distance"])
        self.angles_deg = cfg["actions"]["angles_deg"]
        self.level = cfg["environment"].get("level", "clean")
        self.pose_co = np.eye(4, dtype=np.float64)
        self.gt_pose_co = np.eye(4, dtype=np.float64)
        self.step_index = 0
        self.pose_drift_rotvec = np.zeros(3, dtype=np.float64)
        self._rays = self._make_camera_rays()

    def _make_camera_rays(self) -> np.ndarray:
        ys, xs = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing="ij")
        directions = np.stack(
            [
                (xs + 0.5 - self.K[0, 2]) / self.K[0, 0],
                (ys + 0.5 - self.K[1, 2]) / self.K[1, 1],
                np.ones_like(xs, dtype=np.float64),
            ],
            axis=-1,
        )
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        origins = np.zeros_like(directions)
        return np.concatenate([origins, directions], axis=-1).astype(np.float32)

    def reset(self, initial_seed: int) -> Observation:
        self.step_index = 0
        self.pose_drift_rotvec = np.zeros(3, dtype=np.float64)
        self.gt_pose_co = np.eye(4, dtype=np.float64)
        self.gt_pose_co[:3, :3] = random_rotation(int(initial_seed))
        self.gt_pose_co[:3, 3] = np.array([0.0, 0.0, self.distance], dtype=np.float64)
        self.pose_co = self.gt_pose_co.copy()
        return self.render()

    def bootstrap_observations(self, degrees: float) -> List[Observation]:
        original_gt = self.gt_pose_co.copy()
        original_executed = self.pose_co.copy()
        result = []
        offsets = self.cfg["environment"].get("bootstrap_offsets_deg")
        if offsets is None:
            offsets = (0.0, float(degrees), -float(degrees))
        if not offsets:
            raise ValueError("environment.bootstrap_offsets_deg must contain at least one view")
        for angle in offsets:
            rotation = axis_angle(np.array([0.0, 1.0, 0.0]), np.radians(angle))
            self.gt_pose_co[:3, :3] = rotation @ original_gt[:3, :3]
            self.pose_co[:3, :3] = rotation @ original_executed[:3, :3]
            result.append(self.render(step_override=0))
        self.gt_pose_co = original_gt
        self.pose_co = original_executed
        return result

    def clone(self) -> "KinematicRGBDEnv":
        clone = object.__new__(KinematicRGBDEnv)
        clone.cfg = self.cfg
        clone.seed = self.seed
        clone.rng = np.random.RandomState()
        clone.rng.set_state(self.rng.get_state())
        clone.mesh_path = self.mesh_path
        clone.mesh_object = self.mesh_object
        clone.height = self.height
        clone.width = self.width
        clone.K = self.K.copy()
        clone.distance = self.distance
        clone.angles_deg = self.angles_deg
        clone.level = self.level
        clone.pose_co = self.pose_co.copy()
        clone.gt_pose_co = self.gt_pose_co.copy()
        clone.step_index = self.step_index
        clone.pose_drift_rotvec = self.pose_drift_rotvec.copy()
        clone._rays = self._rays
        return clone

    def step(self, action: str) -> Observation:
        self.step_index += 1
        action_rotation_matrix = action_rotation(action, self.angles_deg)
        if self.level != "clean":
            robustness = self.cfg["environment"].get("robustness", {})
            axis_noise = self.rng.normal(
                0.0, np.tan(np.radians(float(robustness.get("axis_deviation_deg", 0.0)))), size=3
            )
            actual_axis = ACTION_AXES[action] + axis_noise
            actual_angle = np.radians(
                float(self.angles_deg[action])
                + self.rng.normal(0.0, float(robustness.get("magnitude_error_deg", 0.0)))
            )
            action_rotation_matrix = axis_angle(actual_axis, actual_angle)
            drift_std = np.radians(float(robustness.get("drift_per_step_deg", 0.0)))
            if drift_std > 0.0:
                self.pose_drift_rotvec += self.rng.normal(0.0, drift_std, size=3)
        self.gt_pose_co[:3, :3] = action_rotation_matrix @ self.gt_pose_co[:3, :3]
        self.pose_co = self.gt_pose_co.copy()
        return self.render()

    def render(self, step_override: Optional[int] = None) -> Observation:
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh_object)
        mesh.transform(o3d.core.Tensor(self.gt_pose_co.astype(np.float32)))
        scene = o3d.t.geometry.RaycastingScene()
        object_id = scene.add_triangles(mesh)
        result = scene.cast_rays(o3d.core.Tensor(self._rays))
        t_hit = result["t_hit"].numpy()
        geometry_ids = result["geometry_ids"].numpy()
        mask = np.isfinite(t_hit) & (geometry_ids == object_id)
        direction_z = self._rays[..., 5]
        depth = np.zeros((self.height, self.width), dtype=np.float32)
        depth[mask] = t_hit[mask] * direction_z[mask]

        points_object = np.zeros((self.height, self.width, 3), dtype=np.float64)
        points_camera_hit = self._rays[..., 3:][mask] * t_hit[mask][:, None]
        points_object[mask] = points_camera_hit @ self.gt_pose_co[:3, :3] - (
            self.gt_pose_co[:3, :3].T @ self.gt_pose_co[:3, 3]
        )[None, :]
        extent = np.asarray(self.mesh_object.get_axis_aligned_bounding_box().get_extent())
        normalized = 0.5 + points_object / np.maximum(extent[None, None, :], 1e-6)
        normals = result["primitive_normals"].numpy()
        lighting = np.clip(0.35 + 0.65 * np.abs(normals[..., 2:3]), 0.0, 1.0)
        rgb = np.full((self.height, self.width, 3), 18, dtype=np.uint8)
        color = np.clip(255.0 * normalized * lighting, 0.0, 255.0).astype(np.uint8)
        rgb[mask] = color[mask]

        observed_pose = self.pose_co.copy()
        observed_mask = mask.copy()
        visibility_ratio = 1.0
        if self.level != "clean":
            robustness = self.cfg["environment"].get("robustness", {})
            if np.linalg.norm(self.pose_drift_rotvec) > 0.0:
                observed_pose[:3, :3] = Rotation.from_rotvec(self.pose_drift_rotvec).as_matrix() @ observed_pose[:3, :3]
            pose_jitter = np.radians(float(robustness.get("pose_jitter_deg", 0.0)))
            if pose_jitter > 0:
                jitter = self.rng.normal(0.0, pose_jitter, size=3)
                observed_pose[:3, :3] = axis_angle(jitter, np.linalg.norm(jitter)) @ observed_pose[:3, :3]
            if self.rng.rand() < float(robustness.get("pose_outlier_probability", 0.0)):
                outlier = self.rng.normal(
                    0.0, np.radians(float(robustness.get("pose_outlier_deg", 0.0))), size=3
                )
                observed_pose[:3, :3] = axis_angle(outlier, np.linalg.norm(outlier)) @ observed_pose[:3, :3]
            pixels = int(robustness.get("mask_morphology_px", 0))
            if pixels > 0:
                observed_mask = binary_erosion(observed_mask, iterations=pixels)
            elif pixels < 0:
                observed_mask = binary_dilation(observed_mask, iterations=-pixels)
                false_positive = observed_mask & ~mask
                if false_positive.any():
                    depth[false_positive] = np.maximum(
                        0.05,
                        self.distance + self.rng.normal(0.0, 0.02, size=int(false_positive.sum())),
                    )
            occlusion_fraction = float(robustness.get("occlusion_fraction", 0.0))
            if occlusion_fraction > 0.0 and observed_mask.any():
                rows, cols = np.nonzero(observed_mask)
                box_height = max(1, int((rows.max() - rows.min() + 1) * min(occlusion_fraction, 0.9)))
                top_min, top_max = int(rows.min()), max(int(rows.min()), int(rows.max()) - box_height + 1)
                top = self.rng.randint(top_min, top_max + 1)
                observed_mask[top:top + box_height, :] = False
            visibility_ratio = float(observed_mask.sum() / max(mask.sum(), 1))

        return Observation(
            rgb=rgb,
            depth=depth,
            mask=observed_mask.astype(bool),
            gt_pose=self.gt_pose_co.copy(),
            executed_pose=observed_pose,
            camera_intrinsics=self.K.copy(),
            step=self.step_index if step_override is None else int(step_override),
            pose_error_deg=rotation_error_deg(self.gt_pose_co[:3, :3], observed_pose[:3, :3]),
            visibility_ratio=visibility_ratio,
        )
