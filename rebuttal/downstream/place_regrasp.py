from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


@dataclass
class TrialResult:
    object_name: str
    method: str
    reconstruction_seed: int
    trial_seed: int
    placement_success: bool
    regrasp_success: bool
    conditional_regrasp_success: bool
    sequential_success: bool
    placement_margin_m: float
    placement_tilt_deg: float
    placement_tilt_limit_deg: float
    predicted_grasp_width_m: float
    actual_grasp_width_m: float
    contact_error_m: float
    antipodal_margin: float
    failure_reason: str

    def to_dict(self):
        return asdict(self)


def load_mesh(path: Path | str) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise RuntimeError(f"Empty mesh scene: {path}")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    mesh = trimesh.Trimesh(
        vertices=np.asarray(loaded.vertices, dtype=np.float64),
        faces=np.asarray(loaded.faces, dtype=np.int64),
        process=True,
    )
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertices) < 4 or len(mesh.faces) < 4:
        raise RuntimeError(f"Mesh is too small for downstream evaluation: {path}")
    return mesh


def _center_of_mass(mesh: trimesh.Trimesh) -> np.ndarray:
    center = np.asarray(mesh.center_mass if mesh.is_watertight else mesh.centroid)
    if not np.all(np.isfinite(center)):
        center = np.asarray(mesh.vertices).mean(axis=0)
    return center.astype(np.float64)


def stable_pose_candidates(mesh: trimesh.Trimesh, maximum: int = 6):
    # Static support is determined by the convex hull.  Computing poses on the
    # hull is both physically equivalent for support contacts and avoids a
    # large cost that otherwise depends on reconstruction tessellation density.
    support_mesh = mesh.convex_hull
    try:
        transforms, probabilities = trimesh.poses.compute_stable_poses(
            support_mesh,
            center_mass=_center_of_mass(mesh),
            sigma=0.0,
            n_samples=1,
        )
    except Exception:
        # Degenerate open reconstructions occasionally need the hull centroid.
        transforms, probabilities = trimesh.poses.compute_stable_poses(
            support_mesh,
            center_mass=_center_of_mass(support_mesh),
            sigma=0.0,
            n_samples=1,
        )
    order = np.argsort(-np.asarray(probabilities, dtype=np.float64))[:maximum]
    return np.asarray(transforms)[order], np.asarray(probabilities)[order]


def _support_margin(mesh: trimesh.Trimesh, rotation: np.ndarray):
    vertices = np.asarray(mesh.vertices) @ rotation.T
    center = rotation @ _center_of_mass(mesh)
    height = float(vertices[:, 2].max() - vertices[:, 2].min())
    tolerance = max(0.0015, 0.012 * height)
    support = vertices[vertices[:, 2] <= vertices[:, 2].min() + tolerance, :2]
    if len(support) < 3:
        return float("-inf"), max(float(center[2] - vertices[:, 2].min()), 1e-6), support
    try:
        hull = ConvexHull(support)
    except Exception:
        return float("-inf"), max(float(center[2] - vertices[:, 2].min()), 1e-6), support
    polygon = support[hull.vertices]
    signed_area = 0.5 * np.sum(
        polygon[:, 0] * np.roll(polygon[:, 1], -1)
        - polygon[:, 1] * np.roll(polygon[:, 0], -1)
    )
    if signed_area < 0.0:
        polygon = polygon[::-1]
    point = center[:2]
    edge = np.roll(polygon, -1, axis=0) - polygon
    relative = point[None, :] - polygon
    cross = edge[:, 0] * relative[:, 1] - edge[:, 1] * relative[:, 0]
    distances = cross / np.maximum(np.linalg.norm(edge, axis=1), 1e-12)
    margin = float(np.min(distances))
    com_height = max(float(center[2] - vertices[:, 2].min()), 1e-6)
    return margin, com_height, polygon


def _o3d_scene(mesh: trimesh.Trimesh):
    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    legacy.compute_triangle_normals()
    tensor = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor)
    return scene


def build_ray_scene(mesh: trimesh.Trimesh):
    return _o3d_scene(mesh)


def _cast(scene, origins: np.ndarray, directions: np.ndarray):
    rays = np.concatenate([origins, directions], axis=1).astype(np.float32)
    result = scene.cast_rays(o3d.core.Tensor(rays))
    distance = result["t_hit"].numpy().astype(np.float64)
    normals = result["primitive_normals"].numpy().astype(np.float64)
    points = np.full_like(origins, np.nan, dtype=np.float64)
    finite = np.isfinite(distance)
    points[finite] = origins[finite] + directions[finite] * distance[finite, None]
    return points, normals, distance


def _transform_mesh(mesh: trimesh.Trimesh, rotation: np.ndarray) -> trimesh.Trimesh:
    transformed = mesh.copy()
    transformed.vertices = np.asarray(mesh.vertices) @ rotation.T
    transformed.vertices[:, 2] -= transformed.vertices[:, 2].min()
    return transformed


def _transform_mesh_with_frame(mesh: trimesh.Trimesh, rotation: np.ndarray):
    transformed = mesh.copy()
    vertices = np.asarray(mesh.vertices) @ rotation.T
    translation = np.array([0.0, 0.0, -vertices[:, 2].min()], dtype=np.float64)
    transformed.vertices = vertices + translation
    return transformed, (np.asarray(rotation, dtype=np.float64), translation)


def plan_parallel_grasp(
    mesh: trimesh.Trimesh,
    max_width_m: float = 0.120,
    friction: float = 0.8,
    scene=None,
    original_to_mesh_frame=None,
) -> Optional[dict]:
    if scene is None:
        scene = _o3d_scene(mesh)
    if original_to_mesh_frame is None:
        frame_rotation = np.eye(3)
        frame_translation = np.zeros(3)
    else:
        frame_rotation, frame_translation = original_to_mesh_frame
    bounds = np.asarray(mesh.bounds)
    extent = bounds[1] - bounds[0]
    center = 0.5 * (bounds[0] + bounds[1])
    reach = 2.0 * float(np.linalg.norm(extent)) + 0.05
    friction_cos = float(np.cos(np.arctan(friction)))
    origins_left = []
    origins_right = []
    directions_left = []
    directions_right = []
    metadata = []
    for angle in np.linspace(0.0, np.pi, 24, endpoint=False):
        closing = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float64)
        lateral = np.array([-closing[1], closing[0], 0.0], dtype=np.float64)
        lateral_scale = 0.16 * max(float(extent[0]), float(extent[1]))
        for height_fraction in (0.35, 0.50, 0.65, 0.78):
            height = bounds[0, 2] + height_fraction * extent[2]
            for offset_fraction in (-1.0, 0.0, 1.0):
                line_center = center.copy()
                line_center[2] = height
                line_center += offset_fraction * lateral_scale * lateral
                origins_left.append(line_center - reach * closing)
                origins_right.append(line_center + reach * closing)
                directions_left.append(closing)
                directions_right.append(-closing)
                metadata.append((closing, lateral, line_center))

    origins_left = np.asarray(origins_left)
    origins_right = np.asarray(origins_right)
    directions_left = np.asarray(directions_left)
    directions_right = np.asarray(directions_right)
    # Reuse one ray scene for all stable-pose candidates.  Rays are expressed
    # in the placed frame above, then transformed back to the original mesh
    # frame for intersection and returned to the placed frame afterwards.
    query_origins_left = (origins_left - frame_translation) @ frame_rotation
    query_origins_right = (origins_right - frame_translation) @ frame_rotation
    query_directions_left = directions_left @ frame_rotation
    query_directions_right = directions_right @ frame_rotation
    left, normal_left, hit_left = _cast(
        scene, query_origins_left, query_directions_left
    )
    right, normal_right, hit_right = _cast(
        scene, query_origins_right, query_directions_right
    )
    left = left @ frame_rotation.T + frame_translation
    right = right @ frame_rotation.T + frame_translation
    normal_left = normal_left @ frame_rotation.T
    normal_right = normal_right @ frame_rotation.T

    candidates = []
    for index, (closing, lateral, line_center) in enumerate(metadata):
        if not np.isfinite(hit_left[index]) or not np.isfinite(hit_right[index]):
            continue
        width = float(np.dot(right[index] - left[index], closing))
        if width < 0.010 or width > max_width_m:
            continue
        cosine_left = float(np.dot(normal_left[index], -closing))
        cosine_right = float(np.dot(normal_right[index], closing))
        antipodal = min(cosine_left, cosine_right)
        if antipodal < friction_cos:
            continue
        midpoint = 0.5 * (left[index] + right[index])
        transverse_error = float(abs(np.dot(midpoint - line_center, lateral)))
        if transverse_error > 0.004:
            continue
        height_score = float(midpoint[2] / max(bounds[1, 2], 1e-6))
        center_score = float(np.linalg.norm(midpoint[:2] - center[:2]))
        score = antipodal + 0.12 * height_score - 2.5 * center_score
        candidates.append({
            "score": score,
            "closing": closing,
            "lateral": lateral,
            "line_center": line_center,
            "left": left[index],
            "right": right[index],
            "midpoint": midpoint,
            "width": width,
            "antipodal": antipodal,
            "friction_cos": friction_cos,
        })
    return max(candidates, key=lambda item: item["score"]) if candidates else None


def validate_grasp(
    grasp: dict,
    ground_truth: trimesh.Trimesh,
    max_width_m: float = 0.120,
    pad_compliance_m: float = 0.008,
    friction: float = 0.8,
    scene=None,
    original_to_ground_truth_frame=None,
):
    if scene is None:
        scene = _o3d_scene(ground_truth)
    if original_to_ground_truth_frame is None:
        frame_rotation = np.eye(3)
        frame_translation = np.zeros(3)
    else:
        frame_rotation, frame_translation = original_to_ground_truth_frame
    closing = grasp["closing"]
    line_center = grasp["line_center"]
    reach = 2.0 * float(np.linalg.norm(ground_truth.extents)) + 0.05
    origins = np.stack([line_center - reach * closing, line_center + reach * closing])
    directions = np.stack([closing, -closing])
    query_origins = (origins - frame_translation) @ frame_rotation
    query_directions = directions @ frame_rotation
    points, normals, distances = _cast(scene, query_origins, query_directions)
    points = points @ frame_rotation.T + frame_translation
    normals = normals @ frame_rotation.T
    if not np.all(np.isfinite(distances)):
        return False, float("nan"), float("inf"), -1.0, "no_gt_contact"
    width = float(np.dot(points[1] - points[0], closing))
    friction_cos = float(np.cos(np.arctan(friction)))
    antipodal = min(
        float(np.dot(normals[0], -closing)),
        float(np.dot(normals[1], closing)),
    )
    contact_error = float(
        max(
            np.linalg.norm(points[0] - grasp["left"]),
            np.linalg.norm(points[1] - grasp["right"]),
        )
    )
    actual_midpoint = 0.5 * (points[0] + points[1])
    center = _center_of_mass(ground_truth)
    lateral_offset = float(abs(np.dot(center - actual_midpoint, grasp["lateral"])))
    vertical_offset = float(abs(center[2] - actual_midpoint[2]))
    if width <= 0.010 or width > max_width_m:
        return False, width, contact_error, antipodal, "jaw_width"
    if antipodal < friction_cos:
        return False, width, contact_error, antipodal, "not_antipodal"
    if contact_error > pad_compliance_m:
        return False, width, contact_error, antipodal, "contact_mismatch"
    vertical_limit = max(0.035, 0.40 * float(ground_truth.extents[2]))
    if lateral_offset > 0.030 or vertical_offset > vertical_limit:
        return False, width, contact_error, antipodal, "grasp_torque"
    return True, width, contact_error, antipodal, "success"


def plan_sequential_task(predicted_mesh: trimesh.Trimesh):
    """Jointly choose a stable support pose and a reachable top-down regrasp."""
    transforms, probabilities = stable_pose_candidates(predicted_mesh)
    scene = _o3d_scene(predicted_mesh)
    feasible = []
    for transform, probability in zip(transforms, probabilities):
        rotation = np.asarray(transform[:3, :3], dtype=np.float64)
        placed, frame = _transform_mesh_with_frame(predicted_mesh, rotation)
        grasp = plan_parallel_grasp(
            placed, scene=scene, original_to_mesh_frame=frame
        )
        if grasp is None:
            continue
        margin, com_height, _ = _support_margin(predicted_mesh, rotation)
        predicted_tilt_limit = float(
            np.degrees(np.arctan2(max(margin, 0.0), com_height))
        )
        # This is a sequential place-then-regrasp task: robust support is the
        # primary objective, while grasp quality and stable-pose probability
        # break near-ties.  All terms use only the reconstructed mesh.
        score = float(
            predicted_tilt_limit
            + 0.10 * grasp["score"]
            + 0.02 * np.log(max(float(probability), 1e-9))
        )
        feasible.append((score, rotation, grasp))
    if feasible:
        _, rotation, grasp = max(feasible, key=lambda item: item[0])
        return rotation, grasp
    rotation = np.asarray(transforms[0, :3, :3], dtype=np.float64)
    return rotation, None


def _yaw_grasp(grasp: Optional[dict], yaw: np.ndarray):
    if grasp is None:
        return None
    rotated = dict(grasp)
    for key in ("closing", "lateral", "line_center", "left", "right", "midpoint"):
        rotated[key] = yaw @ np.asarray(grasp[key], dtype=np.float64)
    return rotated


def _perturb_grasp(grasp: Optional[dict], rng, translation_std_m, yaw_std_deg):
    if grasp is None:
        return None
    perturbed = dict(grasp)
    delta_rotation = Rotation.from_euler(
        "z", rng.normal(0.0, yaw_std_deg), degrees=True
    ).as_matrix()
    translation = rng.normal(0.0, translation_std_m, size=3)
    translation[2] *= 0.67
    center = np.asarray(grasp["line_center"], dtype=np.float64)
    for key in ("closing", "lateral"):
        perturbed[key] = delta_rotation @ np.asarray(grasp[key], dtype=np.float64)
    for key in ("left", "right", "midpoint"):
        point = np.asarray(grasp[key], dtype=np.float64)
        perturbed[key] = delta_rotation @ (point - center) + center + translation
    perturbed["line_center"] = center + translation
    return perturbed


def validate_task_grasp(task_plan, ground_truth_mesh: trimesh.Trimesh, scene=None):
    """Validate the planned regrasp once; random table yaw is invariant."""
    base_rotation, base_grasp = task_plan
    if base_grasp is None:
        return False, float("nan"), float("inf"), -1.0, "no_predicted_grasp"
    placed_ground_truth, frame = _transform_mesh_with_frame(
        ground_truth_mesh, base_rotation
    )
    return validate_grasp(
        base_grasp,
        placed_ground_truth,
        scene=scene,
        original_to_ground_truth_frame=frame,
    )


def evaluate_trial(
    predicted_mesh: trimesh.Trimesh,
    ground_truth_mesh: trimesh.Trimesh,
    object_name: str,
    method: str,
    reconstruction_seed: int,
    trial_seed: int,
    placement_tilt_std_deg: float = 2.5,
    placement_tilt_clip_deg: float = 7.0,
    table_half_extent_m: float = 0.14,
    task_plan=None,
    grasp_validation=None,
    ground_truth_scene=None,
    grasp_translation_std_m: float = 0.0,
    grasp_yaw_std_deg: float = 0.0,
) -> TrialResult:
    rng = np.random.RandomState(int(trial_seed))
    base_rotation, base_grasp = (
        plan_sequential_task(predicted_mesh) if task_plan is None else task_plan
    )
    yaw = Rotation.from_euler("z", rng.uniform(-180.0, 180.0), degrees=True).as_matrix()
    intended = yaw @ base_rotation
    planned_grasp = _yaw_grasp(base_grasp, yaw)
    planned_grasp = _perturb_grasp(
        planned_grasp,
        rng,
        float(grasp_translation_std_m),
        float(grasp_yaw_std_deg),
    )

    margin, com_height, _ = _support_margin(ground_truth_mesh, intended)
    tilt_xy = np.clip(
        rng.normal(0.0, placement_tilt_std_deg, size=2),
        -placement_tilt_clip_deg,
        placement_tilt_clip_deg,
    )
    tilt_deg = float(np.linalg.norm(tilt_xy))
    tilt_limit_deg = float(np.degrees(np.arctan2(max(margin, 0.0), com_height)))
    placed_gt_nominal, ground_truth_frame = _transform_mesh_with_frame(
        ground_truth_mesh, intended
    )
    footprint = np.ptp(np.asarray(placed_gt_nominal.vertices)[:, :2], axis=0)
    target_offset = rng.uniform(-0.006, 0.006, size=2)
    within_target = bool(np.all(0.5 * footprint + np.abs(target_offset) < table_half_extent_m))
    placement_success = bool(
        np.isfinite(margin)
        and margin > 0.001
        and tilt_deg <= tilt_limit_deg
        and within_target
    )

    # The object settles at the planned stable orientation after a successful
    # placement.  Regrasp planning sees only the reconstructed mesh; execution
    # is validated against the GT geometry in the same placed pose.
    placed_ground_truth = placed_gt_nominal
    predicted_width = (
        float("nan") if planned_grasp is None else float(planned_grasp["width"])
    )
    if planned_grasp is None:
        grasp_success = False
        actual_width = float("nan")
        contact_error = float("inf")
        antipodal = -1.0
        reason = "no_predicted_grasp"
    elif (
        grasp_validation is not None
        and grasp_translation_std_m == 0.0
        and grasp_yaw_std_deg == 0.0
    ):
        grasp_success, actual_width, contact_error, antipodal, reason = grasp_validation
    else:
        grasp_success, actual_width, contact_error, antipodal, reason = validate_grasp(
            planned_grasp,
            placed_ground_truth,
            scene=ground_truth_scene,
            original_to_ground_truth_frame=ground_truth_frame,
        )
    conditional = bool(grasp_success)
    sequential = bool(placement_success and grasp_success)
    if not placement_success:
        reason = "placement_unstable" if margin <= 0.001 else "placement_perturbation"

    return TrialResult(
        object_name=object_name,
        method=method,
        reconstruction_seed=int(reconstruction_seed),
        trial_seed=int(trial_seed),
        placement_success=placement_success,
        regrasp_success=sequential,
        conditional_regrasp_success=conditional,
        sequential_success=sequential,
        placement_margin_m=float(margin),
        placement_tilt_deg=tilt_deg,
        placement_tilt_limit_deg=tilt_limit_deg,
        predicted_grasp_width_m=predicted_width,
        actual_grasp_width_m=float(actual_width),
        contact_error_m=float(contact_error),
        antipodal_margin=float(antipodal),
        failure_reason=reason,
    )


def evaluate_mesh_pair(
    predicted_mesh_path: Path | str,
    ground_truth_mesh_path: Path | str,
    object_name: str,
    method: str,
    reconstruction_seed: int,
    trial_seeds: Iterable[int],
):
    predicted = load_mesh(predicted_mesh_path)
    ground_truth = load_mesh(ground_truth_mesh_path)
    task_plan = plan_sequential_task(predicted)
    return [
        evaluate_trial(
            predicted,
            ground_truth,
            object_name,
            method,
            reconstruction_seed,
            int(seed),
            task_plan=task_plan,
        )
        for seed in trial_seeds
    ]
