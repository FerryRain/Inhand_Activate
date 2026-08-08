"""Pose-conditioned viewpoint-novelty planning for real deployment.

The planner uses only tracked object poses.  For each executable hand policy
(`x`, `y`, or `z`), it predicts the viewing direction after one calibrated
six-second primitive and selects the direction with the largest minimum
geodesic distance to the previously observed directions.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


AXES = ("x", "y", "z")
AXIS_TO_CALIBRATION_ACTION = {
    "x": "minus_x",
    "y": "minus_y",
    "z": "plus_z",
}
ACTION_AXES_IN_INITIAL_OBJECT = {
    "x": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "y": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
}
DEFAULT_ANGLES_DEG = {
    "x": 106.47934768006431,
    "y": 55.76306834268152,
    "z": 82.38086258986442,
}


def _as_pose(pose: np.ndarray) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64)
    if value.shape == (3, 4):
        value = np.vstack([value, [0.0, 0.0, 0.0, 1.0]])
    if value.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 (or 3x4) pose, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("Pose contains non-finite values")
    return value.copy()


def _project_rotation(rotation: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    projected = u @ vh
    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vh
    return projected


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero-length vector")
    return value / norm


def _axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    unit = _normalize(axis)
    x, y, z = unit
    skew = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    sine = math.sin(float(angle_rad))
    cosine = math.cos(float(angle_rad))
    return np.eye(3, dtype=np.float64) + sine * skew + (1.0 - cosine) * (skew @ skew)


def view_direction_object(pose_co: np.ndarray) -> np.ndarray:
    """Return the object-to-camera unit direction in object coordinates."""

    pose = _as_pose(pose_co)
    rotation_co = _project_rotation(pose[:3, :3])
    return _normalize(rotation_co.T @ (-pose[:3, 3]))


def predict_pose_after_axis(
    current_pose_co: np.ndarray,
    initial_pose_co: np.ndarray,
    axis: str,
    angles_deg: Dict[str, float],
) -> np.ndarray:
    """Predict a world/initial-object-frame primitive in camera coordinates."""

    if axis not in AXES:
        raise KeyError(f"Unknown rotation axis: {axis}")
    current = _as_pose(current_pose_co)
    initial = _as_pose(initial_pose_co)
    rotation_current = _project_rotation(current[:3, :3])
    rotation_camera_world = _project_rotation(initial[:3, :3])
    rotation_world = _axis_angle(
        ACTION_AXES_IN_INITIAL_OBJECT[axis],
        math.radians(float(angles_deg[axis])),
    )

    predicted = current.copy()
    predicted[:3, :3] = (
        rotation_camera_world
        @ rotation_world
        @ rotation_camera_world.T
        @ rotation_current
    )
    return predicted


def load_calibrated_angles(path: os.PathLike) -> Dict[str, float]:
    """Load the real six-second median rotations produced by calibration."""

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    actions = payload.get("actions", {})
    angles = {}
    for axis in AXES:
        action = AXIS_TO_CALIBRATION_ACTION[axis]
        value = actions.get(action, {}).get("median_angle_deg")
        if value is None:
            raise ValueError(f"Missing median_angle_deg for {action} in {path}")
        angles[axis] = float(value)
    return angles


def _pose_file_sort_key(path: Path) -> Tuple[int, object]:
    try:
        return 0, int(path.stem)
    except ValueError:
        return 1, path.stem


def _has_matching_keyframe_modalities(keyframe_root: Path, frame_id: str) -> bool:
    candidates = {
        "rgb_full": ("jpg", "jpeg", "png"),
        "depth": ("png", "exr"),
        "mask": ("png", "jpg", "jpeg"),
    }
    return all(
        any((keyframe_root / folder / f"{frame_id}.{suffix}").is_file() for suffix in suffixes)
        for folder, suffixes in candidates.items()
    )


class PoseNoveltyAxisPlanner:
    """Select an executable axis using only tracked-pose viewpoint coverage."""

    def __init__(self, angles_deg: Optional[Dict[str, float]] = None):
        values = DEFAULT_ANGLES_DEG if angles_deg is None else angles_deg
        self.angles_deg = {axis: float(values[axis]) for axis in AXES}
        self._records = []
        self._seen_ids = set()

    @property
    def history_size(self) -> int:
        return len(self._records)

    @property
    def reference_pose(self) -> Optional[np.ndarray]:
        if not self._records:
            return None
        return np.asarray(self._records[0]["pose_co"], dtype=np.float64).copy()

    @property
    def history_directions(self) -> np.ndarray:
        if not self._records:
            return np.empty((0, 3), dtype=np.float64)
        return np.stack([record["view_direction"] for record in self._records], axis=0)

    def add_pose(self, pose_co: np.ndarray, pose_id: Optional[str] = None) -> bool:
        identifier = str(pose_id) if pose_id is not None else f"memory/{len(self._records):06d}"
        if identifier in self._seen_ids:
            return False
        pose = _as_pose(pose_co)
        direction = view_direction_object(pose)
        self._records.append(
            {
                "id": identifier,
                "pose_co": pose,
                "view_direction": direction,
            }
        )
        self._seen_ids.add(identifier)
        return True

    def update_from_keyframes(self, keyframe_root: os.PathLike) -> int:
        """Add only keyframes that contain the same modalities used by fusion."""

        root = Path(keyframe_root)
        pose_dir = root / "poses"
        if not pose_dir.is_dir():
            return 0
        added = 0
        for pose_path in sorted(pose_dir.glob("*.txt"), key=_pose_file_sort_key):
            frame_id = pose_path.stem
            if not _has_matching_keyframe_modalities(root, frame_id):
                continue
            try:
                pose = np.loadtxt(pose_path, dtype=np.float64)
                added += int(self.add_pose(pose, pose_id=f"keyframe/{frame_id}"))
            except (OSError, ValueError) as error:
                print(f"[PoseNovelty][WARN] Skip invalid pose {pose_path}: {error}")
        return added

    def score_axes(
        self,
        current_pose_co: np.ndarray,
        initial_pose_co: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        if not self._records:
            raise RuntimeError("Pose-Novelty needs at least one observed pose")
        reference = self.reference_pose if initial_pose_co is None else _as_pose(initial_pose_co)
        if reference is None:
            raise RuntimeError("Pose-Novelty has no initial/reference pose")

        history = self.history_directions
        scores = {}
        predicted_directions = {}
        for axis in AXES:
            predicted_pose = predict_pose_after_axis(
                current_pose_co,
                reference,
                axis,
                self.angles_deg,
            )
            predicted_direction = view_direction_object(predicted_pose)
            cosines = np.clip(history @ predicted_direction, -1.0, 1.0)
            scores[axis] = float(np.min(np.arccos(cosines)))
            predicted_directions[axis] = predicted_direction
        return scores, predicted_directions

    def select_axis(
        self,
        current_pose_co: np.ndarray,
        initial_pose_co: Optional[np.ndarray] = None,
    ) -> Tuple[str, Dict[str, object]]:
        current = _as_pose(current_pose_co)
        scores, predicted = self.score_axes(current, initial_pose_co)
        selected = max(AXES, key=lambda axis: scores[axis])
        decision = {
            "selected_axis": selected,
            "selected_action": AXIS_TO_CALIBRATION_ACTION[selected],
            "history_size": self.history_size,
            "angles_deg": dict(self.angles_deg),
            "scores_rad": {axis: float(scores[axis]) for axis in AXES},
            "scores_deg": {axis: float(math.degrees(scores[axis])) for axis in AXES},
            "current_pose_co": current.tolist(),
            "current_view_direction": view_direction_object(current).tolist(),
            "predicted_view_directions": {
                axis: predicted[axis].tolist() for axis in AXES
            },
        }
        return selected, decision

    def state_dict(self) -> Dict[str, object]:
        return {
            "version": 1,
            "angles_deg": dict(self.angles_deg),
            "records": [
                {
                    "id": record["id"],
                    "pose_co": np.asarray(record["pose_co"]).tolist(),
                }
                for record in self._records
            ],
        }

    def save_state(self, path: os.PathLike) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state_dict(), indent=2), encoding="utf-8")
        os.replace(temporary, destination)

    def load_state(self, path: os.PathLike) -> bool:
        source = Path(path)
        if not source.is_file():
            return False
        payload = json.loads(source.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        self._records = []
        self._seen_ids = set()
        for record in records:
            self.add_pose(record["pose_co"], pose_id=record["id"])
        return True
