from __future__ import annotations

import copy
from typing import Dict, List

import numpy as np
import open3d as o3d

from .environment import Observation
from .geometry import current_view_direction_object


def backproject_object(observation: Observation, stride: int = 1) -> np.ndarray:
    depth = observation.depth[::stride, ::stride]
    mask = observation.mask[::stride, ::stride] & (depth > 0.0)
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    original_rows = rows * stride
    original_cols = cols * stride
    z = depth[rows, cols].astype(np.float64)
    K = observation.camera_intrinsics
    x = (original_cols + 0.5 - K[0, 2]) * z / K[0, 0]
    y = (original_rows + 0.5 - K[1, 2]) * z / K[1, 1]
    points_camera = np.stack([x, y, z], axis=1)
    rotation = observation.executed_pose[:3, :3]
    translation = observation.executed_pose[:3, 3]
    points_object = (points_camera - translation[None]) @ rotation
    return points_object.astype(np.float32)


class PointCloudFusion:
    def __init__(self, cfg: Dict):
        self.voxel_size = float(cfg["fusion"]["voxel_size"])
        self.depth_stride = int(cfg["fusion"].get("depth_stride", 1))
        self.filter_mode = str(cfg["fusion"].get("filter_mode", "all"))
        self.min_view_change_deg = float(cfg["fusion"].get("min_view_change_deg", 0.0))
        self.min_visibility_ratio = float(cfg["fusion"].get("min_visibility_ratio", 0.0))
        self.max_pose_error_deg = float(cfg["fusion"].get("max_pose_error_deg", float("inf")))
        self.cloud = o3d.geometry.PointCloud()
        self.frame_points: List[np.ndarray] = []
        self.observations: List[Observation] = []
        self.decisions: List[Dict] = []

    def clone(self) -> "PointCloudFusion":
        out = object.__new__(PointCloudFusion)
        out.voxel_size = self.voxel_size
        out.depth_stride = self.depth_stride
        out.filter_mode = self.filter_mode
        out.min_view_change_deg = self.min_view_change_deg
        out.min_visibility_ratio = self.min_visibility_ratio
        out.max_pose_error_deg = self.max_pose_error_deg
        out.cloud = copy.deepcopy(self.cloud)
        out.frame_points = [points.copy() for points in self.frame_points]
        out.observations = list(self.observations)
        out.decisions = [dict(item) for item in self.decisions]
        return out

    @property
    def points(self) -> np.ndarray:
        return np.asarray(self.cloud.points).astype(np.float32)

    @property
    def last_frame_points(self) -> np.ndarray:
        if not self.frame_points:
            return np.zeros((0, 3), dtype=np.float32)
        return self.frame_points[-1]

    def update(self, observation: Observation) -> np.ndarray:
        accepted, reason = self._accept(observation)
        self.decisions.append({
            "step": int(observation.step),
            "accepted": bool(accepted),
            "reason": reason,
            "pose_error_deg": float(observation.pose_error_deg),
            "visibility_ratio": float(observation.visibility_ratio),
        })
        if not accepted:
            return np.zeros((0, 3), dtype=np.float32)
        points = backproject_object(observation, self.depth_stride)
        self.frame_points.append(points)
        self.observations.append(observation)
        if len(points):
            frame = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
            self.cloud += frame
            if self.voxel_size > 0:
                self.cloud = self.cloud.voxel_down_sample(self.voxel_size)
        return points

    def _accept(self, observation: Observation):
        if self.filter_mode == "all" or not self.observations:
            return True, "accepted"
        previous = current_view_direction_object(self.observations[-1].executed_pose)
        current = current_view_direction_object(observation.executed_pose)
        view_change = float(np.degrees(np.arccos(np.clip(np.dot(previous, current), -1.0, 1.0))))
        if view_change < self.min_view_change_deg:
            return False, "view_change"
        if self.filter_mode == "motion_only":
            return True, "accepted"
        if observation.visibility_ratio < self.min_visibility_ratio:
            return False, "visibility"
        if observation.pose_error_deg > self.max_pose_error_deg:
            return False, "pose_error"
        return True, "accepted"
