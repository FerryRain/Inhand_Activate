from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


ACTIONS = ("minus_x", "minus_y", "plus_z")
ACTION_AXES = {
    "minus_x": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "minus_y": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "plus_z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
}


def fibonacci_sphere(count: int) -> np.ndarray:
    if count < 1:
        raise ValueError("count must be positive")
    if count == 1:
        return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    i = np.arange(count, dtype=np.float64)
    y = 1.0 - 2.0 * i / (count - 1)
    radius = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    angle = math.pi * (3.0 - math.sqrt(5.0)) * i
    directions = np.stack([np.cos(angle) * radius, y, np.sin(angle) * radius], axis=1)
    return directions.astype(np.float32)


def axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis) + 1e-12
    return Rotation.from_rotvec(axis * angle_rad).as_matrix()


def action_rotation(action: str, angles_deg: Dict[str, float]) -> np.ndarray:
    if action not in ACTION_AXES:
        raise KeyError("Unknown action: %s" % action)
    return axis_angle(ACTION_AXES[action], math.radians(float(angles_deg[action])))


def apply_action(rotation_co: np.ndarray, action: str, angles_deg: Dict[str, float]) -> np.ndarray:
    """Apply a hand/world-frame primitive to the object-to-camera rotation."""
    return action_rotation(action, angles_deg) @ np.asarray(rotation_co, dtype=np.float64)


def current_view_direction_object(pose_co: np.ndarray) -> np.ndarray:
    """Unit vector from object center to the fixed camera, expressed in object coordinates."""
    rotation_co = pose_co[:3, :3]
    translation_co = pose_co[:3, 3]
    direction = rotation_co.T @ (-translation_co)
    return direction / (np.linalg.norm(direction) + 1e-12)


def next_view_direction(pose_co: np.ndarray, action: str, angles_deg: Dict[str, float]) -> np.ndarray:
    next_pose = np.asarray(pose_co, dtype=np.float64).copy()
    next_pose[:3, :3] = apply_action(next_pose[:3, :3], action, angles_deg)
    return current_view_direction_object(next_pose)


def map_nbv_to_action(
    desired_direction: np.ndarray,
    pose_co: np.ndarray,
    angles_deg: Dict[str, float],
) -> Tuple[str, Dict[str, float]]:
    desired = np.asarray(desired_direction, dtype=np.float64)
    desired /= np.linalg.norm(desired) + 1e-12
    similarities = {
        action: float(np.dot(desired, next_view_direction(pose_co, action, angles_deg)))
        for action in ACTIONS
    }
    selected = max(similarities, key=similarities.get)
    return selected, similarities


def candidate_action_assignments(
    directions: np.ndarray,
    pose_co: np.ndarray,
    angles_deg: Dict[str, float],
) -> np.ndarray:
    next_directions = np.stack(
        [next_view_direction(pose_co, action, angles_deg) for action in ACTIONS], axis=0
    )
    return np.argmax(np.asarray(directions) @ next_directions.T, axis=1).astype(np.int32)


def look_at_basis(camera_position: np.ndarray, target: np.ndarray = None) -> np.ndarray:
    """Camera-to-world/object basis with +z forward, +x right and +y down."""
    camera_position = np.asarray(camera_position, dtype=np.float64)
    target = np.zeros(3, dtype=np.float64) if target is None else np.asarray(target, dtype=np.float64)
    forward = target - camera_position
    forward /= np.linalg.norm(forward) + 1e-12
    up_hint = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(forward, up_hint))) > 0.95:
        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(up_hint, forward)
    right /= np.linalg.norm(right) + 1e-12
    down = np.cross(forward, right)
    down /= np.linalg.norm(down) + 1e-12
    return np.stack([right, down, forward], axis=1)


def virtual_camera_rays(
    direction: np.ndarray,
    distance: float,
    height: int,
    width: int,
    fov_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    origin = np.asarray(direction, dtype=np.float64)
    origin /= np.linalg.norm(origin) + 1e-12
    origin *= float(distance)
    basis = look_at_basis(origin)
    focal = 0.5 * width / math.tan(0.5 * math.radians(fov_deg))
    xs = (np.arange(width, dtype=np.float64) + 0.5 - width / 2.0) / focal
    ys = (np.arange(height, dtype=np.float64) + 0.5 - height / 2.0) / focal
    xx, yy = np.meshgrid(xs, ys)
    rays_camera = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    rays_camera /= np.linalg.norm(rays_camera, axis=-1, keepdims=True)
    rays_object = rays_camera @ basis.T
    origins = np.broadcast_to(origin, rays_object.shape).copy()
    return origins.astype(np.float32), rays_object.astype(np.float32)


def random_rotation(seed: int) -> np.ndarray:
    return Rotation.random(random_state=np.random.RandomState(seed)).as_matrix()


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    return points @ transform[:3, :3].T + transform[:3, 3]


def inverse_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rotation.T
    out[:3, 3] = -rotation.T @ translation
    return out


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(a.T @ b).magnitude()))
