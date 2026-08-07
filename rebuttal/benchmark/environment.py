from __future__ import annotations

import copy
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import open3d as o3d
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial.transform import Rotation, Slerp

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
    translation_error_m: float = 0.0
    visibility_ratio: float = 1.0
    target_points_object: Optional[np.ndarray] = None
    fault_tags: tuple = ()


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
        self.stress = cfg["environment"].get("stress", {})
        self.object_name = str(cfg.get("episode", {}).get("object", Path(mesh_path).stem))
        self.initial_pose_seed = 0
        self.pose_co = np.eye(4, dtype=np.float64)
        self.gt_pose_co = np.eye(4, dtype=np.float64)
        self.step_index = 0
        self.pose_drift_rotvec = np.zeros(3, dtype=np.float64)
        self.pose_translation_drift_m = np.zeros(3, dtype=np.float64)
        # Persistent mapping from nominal hand/controller axes to the axes that
        # actually act on the slipped object.  A grip-slip event updates this
        # state, so subsequent primitives remain miscalibrated.
        self.action_axis_frame = np.eye(3, dtype=np.float64)
        self.last_action_fault_tags = ()
        self.last_action_progress_scale = 1.0
        self.slip_event_count = 0
        self.action_residuals = self._load_action_residuals()
        self.action_trajectories = self._load_action_trajectories()
        self.last_trajectory_metadata = {}
        self._rays = self._make_camera_rays()

    def _load_action_residuals(self):
        path = self.cfg["environment"].get("action_calibration_path")
        if not path:
            return {}
        path = Path(path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        payload = json.loads(path.read_text(encoding="utf-8"))
        residuals = {}
        for action, action_payload in payload["actions"].items():
            nominal_axis = ACTION_AXES[action] / np.linalg.norm(ACTION_AXES[action])
            nominal_angle = float(self.angles_deg[action])
            values = []
            for episode in action_payload["episodes"]:
                rotvec = np.asarray(episode["rotvec"], dtype=np.float64)
                norm = float(np.linalg.norm(rotvec))
                if norm <= 1e-12:
                    continue
                observed_axis = rotvec / norm
                # The tracking/export convention can flip the labelled axis;
                # retain the measured cone deviation while enforcing the
                # commanded sign in the simulator.
                cosine = float(np.clip(abs(np.dot(observed_axis, nominal_axis)), 0.0, 1.0))
                values.append({
                    "axis_deviation_deg": float(np.degrees(np.arccos(cosine))),
                    "magnitude_residual_deg": float(episode["angle_deg"]) - nominal_angle,
                })
            residuals[action] = values
        return residuals

    def _load_action_trajectories(self):
        path = self.cfg["environment"].get("action_trajectory_path")
        if not path:
            return {}
        path = Path(path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        payload = json.loads(path.read_text(encoding="utf-8"))
        trajectories = {}
        for action, action_payload in payload["actions"].items():
            values = []
            for item in action_payload["trajectories"]:
                values.append({
                    **item,
                    "anchor_times_s": np.asarray(item["anchor_times_s"], dtype=np.float64),
                    "anchor_relative_rotations": np.asarray(
                        item["anchor_relative_rotations"], dtype=np.float64
                    ),
                })
            trajectories[action] = values
        return trajectories

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
        self.initial_pose_seed = int(initial_seed)
        self.step_index = 0
        self.pose_drift_rotvec = np.zeros(3, dtype=np.float64)
        self.pose_translation_drift_m = np.zeros(3, dtype=np.float64)
        self.action_axis_frame = np.eye(3, dtype=np.float64)
        self.last_action_fault_tags = ()
        self.last_action_progress_scale = 1.0
        self.slip_event_count = 0
        self.last_trajectory_metadata = {}
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
        clone.stress = self.stress
        clone.object_name = self.object_name
        clone.initial_pose_seed = self.initial_pose_seed
        clone.pose_co = self.pose_co.copy()
        clone.gt_pose_co = self.gt_pose_co.copy()
        clone.step_index = self.step_index
        clone.pose_drift_rotvec = self.pose_drift_rotvec.copy()
        clone.pose_translation_drift_m = self.pose_translation_drift_m.copy()
        clone.action_axis_frame = self.action_axis_frame.copy()
        clone.last_action_fault_tags = tuple(self.last_action_fault_tags)
        clone.last_action_progress_scale = float(self.last_action_progress_scale)
        clone.slip_event_count = int(self.slip_event_count)
        clone.action_residuals = self.action_residuals
        clone.action_trajectories = self.action_trajectories
        clone.last_trajectory_metadata = copy.deepcopy(self.last_trajectory_metadata)
        clone._rays = self._rays
        return clone

    def _fault_rng(self, label: str, step: Optional[int] = None) -> np.random.RandomState:
        """Planner-independent RNG for paired stress realizations."""
        key = "%d|%s|%d|%d|%s" % (
            int(self.cfg["seed"]),
            self.object_name,
            int(self.initial_pose_seed),
            int(self.step_index if step is None else step),
            str(label),
        )
        return np.random.RandomState(zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF)

    def _sample_action_rotation(
        self,
        action: str,
        robustness: Dict,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        rng = self.rng if rng is None else rng
        nominal_axis = ACTION_AXES[action] / np.linalg.norm(ACTION_AXES[action])
        nominal_angle = float(self.angles_deg[action])
        if robustness.get("action_noise_mode") == "empirical" and self.action_residuals.get(action):
            scale = float(robustness.get("action_noise_scale", 1.0))
            residual = self.action_residuals[action][rng.randint(len(self.action_residuals[action]))]
            deviation = np.radians(scale * float(residual["axis_deviation_deg"]))
            tangent = rng.normal(size=3)
            tangent -= np.dot(tangent, nominal_axis) * nominal_axis
            tangent /= np.linalg.norm(tangent) + 1e-12
            actual_axis = np.cos(deviation) * nominal_axis + np.sin(deviation) * tangent
            actual_angle = np.radians(
                max(1.0, nominal_angle + scale * float(residual["magnitude_residual_deg"]))
            )
            return axis_angle(actual_axis, actual_angle)
        axis_noise = rng.normal(
            0.0,
            np.tan(np.radians(float(robustness.get("axis_deviation_deg", 0.0)))),
            size=3,
        )
        actual_axis = nominal_axis + axis_noise
        actual_angle = np.radians(
            nominal_angle
            + rng.normal(0.0, float(robustness.get("magnitude_error_deg", 0.0)))
        )
        return axis_angle(actual_axis, actual_angle)

    @staticmethod
    def _configured_range(config: Dict, key: str, default):
        values = config.get(key, default)
        if np.isscalar(values):
            return float(values), float(values)
        if len(values) != 2:
            raise ValueError("%s must be a scalar or [min, max]" % key)
        low, high = float(values[0]), float(values[1])
        if high < low:
            raise ValueError("%s has max < min" % key)
        return low, high

    def _apply_persistent_action_mismatch(
        self,
        action_rotation_matrix: np.ndarray,
        robustness: Dict,
    ) -> np.ndarray:
        """Apply contact stall and persistent grip-slip dynamics.

        Unlike independent axis noise, a slip rotates the effective control
        frame stored in ``action_axis_frame``.  All later primitives therefore
        execute about biased axes until the episode ends.  Event draws are
        object/pose/step deterministic, so paired planners face the same event
        schedule even when their selected action labels differ.
        """
        enabled = any(
            key in robustness
            for key in (
                "action_progress_scale_range",
                "action_stall_probability",
                "grip_slip_probability",
            )
        )
        if not enabled:
            self.last_action_fault_tags = ()
            self.last_action_progress_scale = 1.0
            return action_rotation_matrix

        rng = self._fault_rng("persistent_action_mismatch") if self.stress else self.rng
        tags = []
        base_rotvec = Rotation.from_matrix(action_rotation_matrix).as_rotvec()
        base_angle = float(np.linalg.norm(base_rotvec))
        base_axis = base_rotvec / (base_angle + 1e-12)

        progress_low, progress_high = self._configured_range(
            robustness, "action_progress_scale_range", [1.0, 1.0]
        )
        progress = float(rng.uniform(progress_low, progress_high))
        if progress < 0.999:
            tags.append("action_underrotation")
        if rng.rand() < float(robustness.get("action_stall_probability", 0.0)):
            stall_low, stall_high = self._configured_range(
                robustness, "action_stall_scale_range", [0.1, 0.35]
            )
            progress = float(rng.uniform(stall_low, stall_high))
            tags.append("action_stall")

        frame_error_deg = float(
            np.degrees(Rotation.from_matrix(self.action_axis_frame).magnitude())
        )
        if frame_error_deg > 1e-6:
            tags.append("persistent_axis_misalignment")
        actual_axis = self.action_axis_frame @ base_axis
        commanded = axis_angle(actual_axis, base_angle * progress)

        slip_rotation = np.eye(3, dtype=np.float64)
        if rng.rand() < float(robustness.get("grip_slip_probability", 0.0)):
            slip_axis = rng.normal(size=3)
            slip_axis /= np.linalg.norm(slip_axis) + 1e-12
            slip_low, slip_high = self._configured_range(
                robustness, "grip_slip_angle_deg_range", [10.0, 25.0]
            )
            slip_angle = np.radians(float(rng.uniform(slip_low, slip_high)))
            if rng.rand() < 0.5:
                slip_angle *= -1.0
            slip_rotation = axis_angle(slip_axis, slip_angle)

            drift_axis = rng.normal(size=3)
            drift_axis /= np.linalg.norm(drift_axis) + 1e-12
            drift_low, drift_high = self._configured_range(
                robustness, "slip_axis_drift_deg_range", [12.0, 28.0]
            )
            drift_angle = np.radians(float(rng.uniform(drift_low, drift_high)))
            if rng.rand() < 0.5:
                drift_angle *= -1.0
            updated_frame = axis_angle(drift_axis, drift_angle) @ self.action_axis_frame
            max_frame_deg = float(robustness.get("slip_axis_drift_max_deg", 45.0))
            updated_rotvec = Rotation.from_matrix(updated_frame).as_rotvec()
            updated_angle = float(np.linalg.norm(updated_rotvec))
            max_frame_rad = np.radians(max_frame_deg)
            if updated_angle > max_frame_rad > 0.0:
                updated_rotvec *= max_frame_rad / updated_angle
                updated_frame = Rotation.from_rotvec(updated_rotvec).as_matrix()
            self.action_axis_frame = updated_frame
            self.slip_event_count += 1
            tags.append("grip_slip")

        self.last_action_fault_tags = tuple(tags)
        self.last_action_progress_scale = float(progress)
        return slip_rotation @ commanded

    def _hand_occluder(
        self,
        object_mask: np.ndarray,
        fraction: float,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        rng = self.rng if rng is None else rng
        occluder = np.zeros_like(object_mask, dtype=bool)
        rows, cols = np.nonzero(object_mask)
        if not len(rows) or fraction <= 0.0:
            return occluder
        row_min, row_max = int(rows.min()), int(rows.max())
        col_min, col_max = int(cols.min()), int(cols.max())
        target = max(1, int(round(float(object_mask.sum()) * min(fraction, 0.9))))

        # Palm: a bottom-anchored block with small frame-to-frame vertical jitter.
        palm_target = max(1, int(round(0.7 * target)))
        sorted_rows = np.sort(rows)
        threshold_index = max(0, len(sorted_rows) - palm_target)
        palm_top = int(sorted_rows[threshold_index])
        jitter = max(1, int(0.04 * (row_max - row_min + 1)))
        palm_top = int(np.clip(palm_top + rng.randint(-jitter, jitter + 1), row_min, row_max))
        occluder[palm_top:row_max + 1, col_min:col_max + 1] = True

        # Two finger-like bands enter from the sides at different heights.
        box_width = col_max - col_min + 1
        box_height = row_max - row_min + 1
        finger_width = max(2, int(box_width * (0.025 + 0.10 * fraction)))
        for side in (0, 1):
            center = col_min + int(box_width * (0.22 if side == 0 else 0.78))
            center += rng.randint(-finger_width, finger_width + 1)
            top = row_min + rng.randint(0, max(1, int(0.45 * box_height)))
            bottom = min(row_max + 1, top + max(2, int((0.35 + fraction) * box_height)))
            left = max(col_min, center - finger_width // 2)
            right = min(col_max + 1, left + finger_width)
            occluder[top:bottom, left:right] = True
        occluder &= object_mask
        selected = np.flatnonzero(occluder)
        if len(selected) > target:
            selected_rows = np.unravel_index(selected, object_mask.shape)[0]
            keep = selected[np.argsort(-selected_rows, kind="stable")[:target]]
            occluder[:] = False
            occluder.flat[keep] = True
        elif len(selected) < target:
            remaining = np.flatnonzero(object_mask & ~occluder)
            remaining_rows = np.unravel_index(remaining, object_mask.shape)[0]
            add = remaining[np.argsort(-remaining_rows, kind="stable")[:target - len(selected)]]
            occluder.flat[add] = True
        return occluder

    def _structured_dropout(
        self,
        object_mask: np.ndarray,
        fraction: float,
        patch_range,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Seeded correlated dropout with retained samples inside each patch."""
        dropped = np.zeros_like(object_mask, dtype=bool)
        rows, cols = np.nonzero(object_mask)
        if not len(rows) or fraction <= 0.0:
            return dropped
        target = min(len(rows) - 1, max(1, int(round(fraction * len(rows)))))
        row_min, row_max = int(rows.min()), int(rows.max())
        col_min, col_max = int(cols.min()), int(cols.max())
        box_h = max(1, row_max - row_min + 1)
        box_w = max(1, col_max - col_min + 1)
        low, high = int(patch_range[0]), int(patch_range[1])
        count = rng.randint(low, high + 1)
        yy, xx = np.indices(object_mask.shape)
        candidate = np.zeros_like(object_mask, dtype=bool)
        for _ in range(count):
            center_row = rng.randint(row_min, row_max + 1)
            center_col = rng.randint(col_min, col_max + 1)
            radius_row = max(3, int(box_h * rng.uniform(0.20, 0.38)))
            radius_col = max(3, int(box_w * rng.uniform(0.20, 0.38)))
            ellipse = (
                ((yy - center_row) / float(radius_row)) ** 2
                + ((xx - center_col) / float(radius_col)) ** 2
                <= 1.0
            )
            # A seeded two-pixel lattice leaves local retained observations,
            # so this stress is sparse rather than a contiguous unseen hole.
            cell = 4
            phase_row = rng.randint(0, 2 * cell)
            phase_col = rng.randint(0, 2 * cell)
            lattice = (
                ((yy + phase_row) // cell + (xx + phase_col) // cell) % 2
            ) == 0
            candidate |= ellipse & lattice & object_mask
        indices = np.flatnonzero(candidate)
        rng.shuffle(indices)
        selected = indices[:target]
        if len(selected) < target:
            remaining = np.flatnonzero(object_mask & ~candidate)
            rng.shuffle(remaining)
            selected = np.concatenate([selected, remaining[: target - len(selected)]])
        dropped.flat[selected] = True
        return dropped

    def _contiguous_dropout(
        self,
        object_mask: np.ndarray,
        fraction: float,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Remove one deterministic, spatially contiguous depth region.

        This models a local RGB--depth registration failure: tracking can
        still report that the view was visited while one coherent surface
        patch never enters the fused reconstruction.
        """
        dropped = np.zeros_like(object_mask, dtype=bool)
        rows, cols = np.nonzero(object_mask)
        if not len(rows) or fraction <= 0.0:
            return dropped
        target = min(len(rows) - 1, max(1, int(round(fraction * len(rows)))))
        row_min, row_max = int(rows.min()), int(rows.max())
        col_min, col_max = int(cols.min()), int(cols.max())
        row_span = max(1.0, float(row_max - row_min + 1))
        col_span = max(1.0, float(col_max - col_min + 1))
        center_row = rng.uniform(row_min + 0.2 * row_span, row_max - 0.2 * row_span)
        center_col = rng.uniform(col_min + 0.2 * col_span, col_max - 0.2 * col_span)
        distances = (
            ((rows - center_row) / row_span) ** 2
            + ((cols - center_col) / col_span) ** 2
        )
        selected = np.argsort(distances, kind="stable")[:target]
        dropped[rows[selected], cols[selected]] = True
        return dropped

    def _begin_action(self, action: str) -> np.ndarray:
        self.step_index += 1
        self.last_action_fault_tags = ()
        self.last_action_progress_scale = 1.0
        action_rotation_matrix = action_rotation(action, self.angles_deg)
        if self.level != "clean":
            robustness = self.cfg["environment"].get("robustness", {})
            action_rng = self._fault_rng("action:%s" % action) if self.stress else self.rng
            action_rotation_matrix = self._sample_action_rotation(action, robustness, action_rng)
            action_rotation_matrix = self._apply_persistent_action_mismatch(
                action_rotation_matrix, robustness
            )
            drift_std = np.radians(float(robustness.get("drift_per_step_deg", 0.0)))
            if drift_std > 0.0:
                self.pose_drift_rotvec += self.rng.normal(0.0, drift_std, size=3)
            translation_drift_std = 0.001 * float(
                robustness.get("translation_drift_per_step_mm", 0.0)
            )
            if translation_drift_std > 0.0:
                self.pose_translation_drift_m += self.rng.normal(
                    0.0, translation_drift_std, size=3
                )
        return action_rotation_matrix

    def step(self, action: str) -> Observation:
        action_rotation_matrix = self._begin_action(action)
        self.gt_pose_co[:3, :3] = action_rotation_matrix @ self.gt_pose_co[:3, :3]
        self.pose_co = self.gt_pose_co.copy()
        return self.render()

    def step_sequence(self, action: str, frames: int) -> List[Observation]:
        if str(self.cfg["environment"].get("action_trajectory_mode", "")) == "empirical_replay":
            return self._step_empirical_trajectory(action)
        frames = int(max(1, frames))
        if frames == 1:
            return [self.step(action)]
        duration_s = float(self.cfg["environment"].get("action_duration_s", 0.0))
        sensor_rate_hz = float(self.cfg["environment"].get("sensor_rate_hz", 0.0))
        if duration_s > 0.0 and sensor_rate_hz > 0.0:
            expected_frames = int(round(duration_s * sensor_rate_hz))
            if frames != expected_frames:
                raise ValueError(
                    "frames_per_action=%d, but action_duration_s * sensor_rate_hz=%d"
                    % (frames, expected_frames)
                )
        action_rotation_matrix = self._begin_action(action)
        action_rotvec = Rotation.from_matrix(action_rotation_matrix).as_rotvec()
        increment = Rotation.from_rotvec(action_rotvec / float(frames)).as_matrix()
        observations = []
        for frame_index in range(1, frames + 1):
            self.gt_pose_co[:3, :3] = increment @ self.gt_pose_co[:3, :3]
            self.pose_co = self.gt_pose_co.copy()
            observations.append(self.render(substep_override=frame_index))
        self.last_trajectory_metadata = {
            "action": action,
            "duration_s": duration_s,
            "replay_rate_hz": sensor_rate_hz,
            "replay_frames": frames,
            "endpoint_angle_deg": float(np.degrees(np.linalg.norm(action_rotvec))),
            "progress_scale": float(self.last_action_progress_scale),
            "fault_tags": list(self.last_action_fault_tags),
            "continuous_rendering": True,
        }
        return observations

    def _step_empirical_trajectory(self, action: str) -> List[Observation]:
        if action not in self.action_trajectories or not self.action_trajectories[action]:
            raise RuntimeError("No empirical trajectory is configured for %s" % action)
        self.step_index += 1
        rng = self._fault_rng("empirical_trajectory:%s" % action)
        trajectory_index = int(rng.randint(len(self.action_trajectories[action])))
        trajectory = self.action_trajectories[action][trajectory_index]
        times = trajectory["anchor_times_s"]
        rotations = trajectory["anchor_relative_rotations"]
        if len(times) < 2 or np.any(np.diff(times) <= 0.0):
            raise ValueError("Empirical trajectory anchors must have increasing times")
        rate_hz = float(self.cfg["environment"].get("trajectory_replay_rate_hz", 5.0))
        if rate_hz <= 0.0:
            raise ValueError("trajectory_replay_rate_hz must be positive")
        duration = float(times[-1])
        frame_count = int(round(duration * rate_hz))
        query_times = np.linspace(1.0 / rate_hz, duration, frame_count)
        relative_rotations = Slerp(times, Rotation.from_matrix(rotations))(
            query_times
        ).as_matrix()
        start_rotation = self.gt_pose_co[:3, :3].copy()
        self.last_action_fault_tags = ()
        self.last_action_progress_scale = float(
            trajectory["endpoint_angle_deg"] / max(float(self.angles_deg[action]), 1e-12)
        )
        self.last_trajectory_metadata = {
            "action": action,
            "trajectory_index": trajectory_index,
            "source_object": trajectory["source_object"],
            "source_log": trajectory["log"],
            "duration_s": duration,
            "replay_rate_hz": rate_hz,
            "replay_frames": frame_count,
            "first_second_error_deg": float(trajectory["first_second_error_deg"]),
            "endpoint_nominal_error_deg": float(trajectory["endpoint_nominal_error_deg"]),
            "endpoint_angle_deg": float(trajectory["endpoint_angle_deg"]),
        }
        def scheduled_value(name: str, default: float) -> float:
            schedule = self.stress.get(name, {})
            value = schedule.get(str(self.step_index), schedule.get(self.step_index, default))
            return float(value)

        lag_frames = int(round(scheduled_value("trajectory_pose_lag_frames_schedule", 0.0)))
        camera_bias_deg = scheduled_value("camera_rotation_bias_deg_schedule", 0.0)
        camera_bias_mm = scheduled_value("camera_translation_bias_mm_schedule", 0.0)
        registration_probability = scheduled_value(
            "registration_failure_probability_schedule", 0.0
        )
        registration_dropout = scheduled_value(
            "registration_dropout_fraction_schedule", 0.0
        )
        bias_rng = self._fault_rng("trajectory_camera_bias")
        bias_axis = bias_rng.normal(size=3)
        bias_axis /= np.linalg.norm(bias_axis) + 1e-12
        bias_rotation = axis_angle(bias_axis, np.radians(camera_bias_deg))
        bias_translation = bias_rng.normal(size=3)
        bias_translation /= np.linalg.norm(bias_translation) + 1e-12
        bias_translation *= 0.001 * camera_bias_mm
        registration_failures = 0
        observations = []
        for frame_index, relative_rotation in enumerate(relative_rotations, start=1):
            self.gt_pose_co[:3, :3] = relative_rotation @ start_rotation
            self.pose_co = self.gt_pose_co.copy()
            observation = self.render(substep_override=frame_index)
            tags = list(observation.fault_tags)
            if lag_frames > 0:
                lagged_index = frame_index - 1 - lag_frames
                lagged_relative = (
                    np.eye(3, dtype=np.float64)
                    if lagged_index < 0 else relative_rotations[lagged_index]
                )
                observation.executed_pose[:3, :3] = lagged_relative @ start_rotation
                tags.append("trajectory_pose_lag")
            if camera_bias_deg > 0.0 or camera_bias_mm > 0.0:
                observation.executed_pose[:3, :3] = (
                    bias_rotation @ observation.executed_pose[:3, :3]
                )
                observation.executed_pose[:3, 3] += bias_translation
                tags.append("camera_extrinsic_bias")
            observation.pose_error_deg = rotation_error_deg(
                observation.gt_pose[:3, :3], observation.executed_pose[:3, :3]
            )
            observation.translation_error_m = float(
                np.linalg.norm(observation.gt_pose[:3, 3] - observation.executed_pose[:3, 3])
            )
            registration_rng = self._fault_rng(
                "trajectory_registration:%03d" % frame_index
            )
            if (
                registration_probability > 0.0
                and registration_dropout > 0.0
                and registration_rng.rand() < registration_probability
                and observation.mask.any()
            ):
                valid = np.flatnonzero(observation.mask)
                count = min(
                    len(valid), int(round(registration_dropout * len(valid)))
                )
                dropped = registration_rng.choice(valid, size=count, replace=False)
                observation.mask.flat[dropped] = False
                observation.depth.flat[dropped] = 0.0
                observation.visibility_ratio *= float(1.0 - count / max(len(valid), 1))
                tags.append("depth_registration_failure")
                registration_failures += 1
            observation.fault_tags = tuple(tags)
            observations.append(observation)
        self.last_trajectory_metadata.update({
            "pose_lag_frames": lag_frames,
            "camera_rotation_bias_deg": camera_bias_deg,
            "camera_translation_bias_mm": camera_bias_mm,
            "registration_failure_probability": registration_probability,
            "registration_dropout_fraction": registration_dropout,
            "registration_failure_frames": int(registration_failures),
        })
        return observations

    def render(
        self,
        step_override: Optional[int] = None,
        substep_override: Optional[int] = None,
    ) -> Observation:
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
        target_points_object = None
        fault_tags = list(self.last_action_fault_tags)
        if self.level != "clean":
            robustness = self.cfg["environment"].get("robustness", {})
            render_label = (
                "render:%03d" % int(substep_override)
                if substep_override is not None else "render"
            )
            render_rng = self._fault_rng(render_label) if self.stress else self.rng
            if np.linalg.norm(self.pose_drift_rotvec) > 0.0:
                observed_pose[:3, :3] = Rotation.from_rotvec(self.pose_drift_rotvec).as_matrix() @ observed_pose[:3, :3]
            observed_pose[:3, 3] += self.pose_translation_drift_m
            pose_jitter = np.radians(float(robustness.get("pose_jitter_deg", 0.0)))
            if pose_jitter > 0:
                jitter = render_rng.normal(0.0, pose_jitter, size=3)
                observed_pose[:3, :3] = axis_angle(jitter, np.linalg.norm(jitter)) @ observed_pose[:3, :3]
            translation_jitter_std = 0.001 * float(
                robustness.get("translation_jitter_mm", 0.0)
            )
            if translation_jitter_std > 0.0:
                observed_pose[:3, 3] += render_rng.normal(0.0, translation_jitter_std, size=3)
            scheduled_outlier_steps = {
                int(value) for value in self.stress.get("pose_outlier_steps", [])
            }
            is_scheduled_outlier = self.step_index in scheduled_outlier_steps
            is_pose_outlier = is_scheduled_outlier or render_rng.rand() < float(
                robustness.get("pose_outlier_probability", 0.0)
            )
            if is_pose_outlier:
                if is_scheduled_outlier:
                    outlier_rng = self._fault_rng("scheduled_pose_outlier")
                    outlier_axis = outlier_rng.normal(size=3)
                    outlier_deg = float(self.stress.get("pose_outlier_deg", 15.0))
                    observed_pose[:3, :3] = axis_angle(
                        outlier_axis, np.radians(outlier_deg)
                    ) @ observed_pose[:3, :3]
                    translation_mm = float(self.stress.get("translation_outlier_mm", 8.0))
                    if translation_mm > 0.0:
                        direction = outlier_rng.normal(size=3)
                        direction /= np.linalg.norm(direction) + 1e-12
                        observed_pose[:3, 3] += 0.001 * translation_mm * direction
                    fault_tags.append("scheduled_pose_outlier")
                else:
                    outlier = render_rng.normal(
                        0.0,
                        np.radians(float(robustness.get("pose_outlier_deg", 0.0))),
                        size=3,
                    )
                    observed_pose[:3, :3] = axis_angle(
                        outlier, np.linalg.norm(outlier)
                    ) @ observed_pose[:3, :3]
                    translation_outlier_std = 0.001 * float(
                        robustness.get("translation_outlier_mm", 0.0)
                    )
                    if translation_outlier_std > 0.0:
                        observed_pose[:3, 3] += render_rng.normal(
                            0.0, translation_outlier_std, size=3
                        )
                    fault_tags.append("random_pose_outlier")
            pixels = int(robustness.get("mask_morphology_px", 0))
            if pixels > 0:
                observed_mask = binary_erosion(observed_mask, iterations=pixels)
            elif pixels < 0:
                observed_mask = binary_dilation(observed_mask, iterations=-pixels)
                false_positive = observed_mask & ~mask
                if false_positive.any():
                    depth[false_positive] = np.maximum(
                        0.05,
                        self.distance + render_rng.normal(0.0, 0.02, size=int(false_positive.sum())),
                    )
            depth_noise_std = 0.001 * float(robustness.get("depth_noise_mm", 0.0))
            if depth_noise_std > 0.0 and observed_mask.any():
                depth[observed_mask] = np.maximum(
                    0.05,
                    depth[observed_mask]
                    + render_rng.normal(0.0, depth_noise_std, size=int(observed_mask.sum())),
                )
            dropout_fraction = float(robustness.get("depth_dropout_fraction", 0.0))
            if dropout_fraction > 0.0 and observed_mask.any():
                valid_flat = np.flatnonzero(observed_mask)
                count = min(len(valid_flat), int(round(dropout_fraction * len(valid_flat))))
                if count:
                    dropped = render_rng.choice(valid_flat, size=count, replace=False)
                    observed_mask.flat[dropped] = False
                    depth.flat[dropped] = 0.0
            structured_steps = {
                int(value) for value in self.stress.get("structured_dropout_steps", [])
            }
            if self.step_index in structured_steps:
                structured = self._structured_dropout(
                    observed_mask,
                    float(self.stress.get("structured_dropout_fraction", 0.0)),
                    self.stress.get("structured_dropout_patches", [4, 8]),
                    self._fault_rng("structured_dropout"),
                )
                if structured.any():
                    target_points_object = points_object[structured].astype(np.float32)
                    observed_mask[structured] = False
                    depth[structured] = 0.0
                    fault_tags.append("structured_dropout")
            contiguous_steps = {
                int(value) for value in self.stress.get("contiguous_dropout_steps", [])
            }
            if self.step_index in contiguous_steps:
                contiguous = self._contiguous_dropout(
                    observed_mask,
                    float(self.stress.get("contiguous_dropout_fraction", 0.0)),
                    self._fault_rng("contiguous_dropout"),
                )
                if contiguous.any():
                    removed = points_object[contiguous].astype(np.float32)
                    target_points_object = (
                        removed
                        if target_points_object is None
                        else np.concatenate([target_points_object, removed], axis=0)
                    )
                    observed_mask[contiguous] = False
                    depth[contiguous] = 0.0
                    fault_tags.append("contiguous_depth_registration_failure")
            occlusion_schedule = self.stress.get("occlusion_schedule", {})
            scheduled_fraction = occlusion_schedule.get(
                str(self.step_index), occlusion_schedule.get(self.step_index)
            )
            if scheduled_fraction is None:
                scheduled_fraction = self.stress.get("occlusion_default")
            occlusion_fraction = float(
                robustness.get("occlusion_fraction", 0.0)
                if scheduled_fraction is None
                else scheduled_fraction
            )
            occlusion_std = float(robustness.get("occlusion_fraction_std", 0.0))
            if occlusion_std > 0.0 and scheduled_fraction is None:
                occlusion_fraction += render_rng.normal(0.0, occlusion_std)
            if scheduled_fraction is None and render_rng.rand() < float(
                robustness.get("occlusion_outlier_probability", 0.0)
            ):
                occlusion_fraction += float(
                    robustness.get("occlusion_outlier_fraction", 0.0)
                )
            occlusion_fraction = float(np.clip(occlusion_fraction, 0.0, 0.85))
            if occlusion_fraction > 0.0 and observed_mask.any():
                hand = self._hand_occluder(observed_mask, occlusion_fraction, render_rng)
                if hand.any():
                    observed_mask[hand] = False
                    rgb[hand] = np.array([105, 87, 72], dtype=np.uint8)
                    depth[hand] = np.maximum(
                        0.05,
                        depth[hand] - render_rng.uniform(0.015, 0.035, size=int(hand.sum())),
                    )
                    fault_tags.append("hand_occlusion")
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
            translation_error_m=float(
                np.linalg.norm(self.gt_pose_co[:3, 3] - observed_pose[:3, 3])
            ),
            visibility_ratio=visibility_ratio,
            target_points_object=target_points_object,
            fault_tags=tuple(fault_tags),
        )
