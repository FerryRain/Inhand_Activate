#!/usr/bin/env python3
"""Controlled 6D pose-error study on previously captured in-hand RGB-D data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


@dataclass
class CapturedSequence:
    name: str
    frame_ids: List[str]
    camera_points: List[np.ndarray]
    poses_co: np.ndarray
    clean_cloud: np.ndarray


def resolve_path(value: str, repo_root: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def portable_path(path: Path, repo_root: Path = REPO_ROOT) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def load_pose(path: Path) -> np.ndarray:
    pose = np.loadtxt(str(path), dtype=np.float64)
    if pose.shape == (3, 4):
        pose = np.vstack((pose, np.array([0.0, 0.0, 0.0, 1.0])))
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("Invalid 4x4 pose: %s" % path)
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("Invalid homogeneous row: %s" % path)
    return pose


def common_frame_ids(keyframes_dir: Path) -> List[str]:
    layouts = {
        "rgb_full": {path.stem for path in (keyframes_dir / "rgb_full").glob("*")},
        "depth": {path.stem for path in (keyframes_dir / "depth").glob("*")},
        "mask": {path.stem for path in (keyframes_dir / "mask").glob("*")},
        "poses": {path.stem for path in (keyframes_dir / "poses").glob("*.txt")},
    }
    missing = [name for name, ids in layouts.items() if not ids]
    if missing:
        raise FileNotFoundError(
            "%s is missing non-empty keyframe folders: %s"
            % (keyframes_dir, ", ".join(missing))
        )
    common = set.intersection(*layouts.values())
    if not common:
        raise RuntimeError("No matched RGB/depth/mask/pose frames under %s" % keyframes_dir)
    return sorted(common, key=lambda value: int(value) if value.isdigit() else value)


def find_image(folder: Path, frame_id: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".tiff", ".tif"):
        path = folder / (frame_id + suffix)
        if path.exists():
            return path
    raise FileNotFoundError("No image for frame %s under %s" % (frame_id, folder))


def voxel_downsample(points: np.ndarray, voxel_m: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0 or voxel_m <= 0:
        return points.copy()
    keys = np.floor(points / float(voxel_m)).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    sums = np.column_stack(
        [np.bincount(inverse, weights=points[:, axis]) for axis in range(3)]
    )
    return sums / counts[:, None]


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def backproject_frame(
    depth_path: Path,
    mask_path: Path,
    intrinsics: np.ndarray,
    fusion_cfg: Dict[str, object],
    seed: int,
) -> np.ndarray:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if depth is None or mask is None:
        raise RuntimeError("Could not read %s or %s" % (depth_path, mask_path))
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.shape != mask.shape:
        raise ValueError("Depth/mask shape mismatch for %s" % depth_path.stem)

    mask_binary = (mask > 0).astype(np.uint8)
    erosion = int(fusion_cfg["mask_erode_px"])
    if erosion > 0:
        kernel_size = 2 * erosion + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask_binary = cv2.erode(mask_binary, kernel, iterations=1)

    z = depth.astype(np.float64) / float(fusion_cfg["depth_scale"])
    valid = (
        (mask_binary > 0)
        & np.isfinite(z)
        & (z >= float(fusion_cfg["depth_min_m"]))
        & (z <= float(fusion_cfg["depth_max_m"]))
    )
    rows, columns = np.nonzero(valid)
    if len(rows) == 0:
        return np.empty((0, 3), dtype=np.float64)

    max_points = int(fusion_cfg["max_points_per_frame"])
    if max_points > 0 and len(rows) > max_points:
        rng = np.random.RandomState(seed)
        selected = np.sort(rng.choice(len(rows), size=max_points, replace=False))
        rows, columns = rows[selected], columns[selected]

    depths = z[rows, columns]
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    points = np.column_stack(
        (
            (columns.astype(np.float64) - cx) * depths / fx,
            (rows.astype(np.float64) - cy) * depths / fy,
            depths,
        )
    )
    return voxel_downsample(points, float(fusion_cfg["per_frame_voxel_m"]))


def fuse_cloud(
    camera_points: Sequence[np.ndarray],
    poses_co: np.ndarray,
    fusion_voxel_m: float,
) -> np.ndarray:
    transformed = []
    for points, pose_co in zip(camera_points, poses_co):
        if len(points):
            transformed.append(transform_points(points, np.linalg.inv(pose_co)))
    if not transformed:
        raise RuntimeError("No valid RGB-D points were available for fusion")
    return voxel_downsample(np.concatenate(transformed, axis=0), fusion_voxel_m)


def load_sequence(
    item: Dict[str, object],
    intrinsics: np.ndarray,
    fusion_cfg: Dict[str, object],
    base_seed: int,
) -> CapturedSequence:
    name = str(item["name"])
    keyframes_dir = resolve_path(str(item["keyframes_dir"]))
    frame_ids = common_frame_ids(keyframes_dir)
    camera_points: List[np.ndarray] = []
    poses = []
    for index, frame_id in enumerate(frame_ids):
        points = backproject_frame(
            find_image(keyframes_dir / "depth", frame_id),
            find_image(keyframes_dir / "mask", frame_id),
            intrinsics,
            fusion_cfg,
            stable_seed(base_seed, name, frame_id, "sampling"),
        )
        camera_points.append(points)
        poses.append(load_pose(keyframes_dir / "poses" / (frame_id + ".txt")))
        if (index + 1) % 50 == 0 or index + 1 == len(frame_ids):
            print("[%s] loaded %d/%d keyframes" % (name, index + 1, len(frame_ids)))
    poses_co = np.stack(poses, axis=0)
    clean_cloud = fuse_cloud(
        camera_points, poses_co, float(fusion_cfg["fusion_voxel_m"])
    )
    if len(clean_cloud) < 100:
        raise RuntimeError("Clean fused cloud is unexpectedly small for %s" % name)
    return CapturedSequence(name, frame_ids, camera_points, poses_co, clean_cloud)


def correlated_errors(
    frame_count: int,
    rotation_rms_deg: float,
    translation_rms_mm: float,
    temporal_correlation: float,
    seed: int,
    anchor_first_frame: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate an anchored Gauss-Markov trace with exact vector-magnitude RMS."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if not 0.0 <= temporal_correlation < 1.0:
        raise ValueError("temporal_correlation must be in [0, 1)")
    rng = np.random.RandomState(seed)
    rotation = np.zeros((frame_count, 3), dtype=np.float64)
    translation = np.zeros((frame_count, 3), dtype=np.float64)
    innovation_scale = math.sqrt(1.0 - temporal_correlation ** 2)
    start = 1 if anchor_first_frame else 0
    for index in range(start, frame_count):
        previous_rotation = rotation[index - 1] if index > 0 else 0.0
        previous_translation = translation[index - 1] if index > 0 else 0.0
        rotation[index] = (
            temporal_correlation * previous_rotation
            + innovation_scale * rng.normal(size=3)
        )
        translation[index] = (
            temporal_correlation * previous_translation
            + innovation_scale * rng.normal(size=3)
        )

    measured = slice(start, None)
    targets = (math.radians(rotation_rms_deg), translation_rms_mm / 1000.0)
    for values, target in ((rotation, targets[0]), (translation, targets[1])):
        if target == 0.0:
            values[:] = 0.0
            continue
        current = float(np.sqrt(np.mean(np.sum(values[measured] ** 2, axis=1))))
        if not np.isfinite(current) or current <= 0.0:
            raise RuntimeError("Could not normalize generated pose errors")
        values *= target / current
    if anchor_first_frame:
        rotation[0] = 0.0
        translation[0] = 0.0
    return rotation, translation


def perturb_poses(
    poses_co: np.ndarray, rotation_vectors: np.ndarray, translations: np.ndarray
) -> np.ndarray:
    if poses_co.shape[0] != rotation_vectors.shape[0] or poses_co.shape[0] != translations.shape[0]:
        raise ValueError("Pose and perturbation lengths differ")
    perturbed = poses_co.copy()
    delta_rotations = Rotation.from_rotvec(rotation_vectors).as_matrix()
    perturbed[:, :3, :3] = np.einsum(
        "nij,njk->nik", delta_rotations, poses_co[:, :3, :3]
    )
    # Translation is perturbed directly so its configured magnitude is the
    # error of the tracked object origin, independent of the rotation error.
    perturbed[:, :3, 3] = poses_co[:, :3, 3] + translations
    return perturbed


def deterministic_sample(points: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0 or len(points) <= maximum:
        return points
    rng = np.random.RandomState(seed)
    return points[np.sort(rng.choice(len(points), size=maximum, replace=False))]


def cloud_metrics(
    reconstruction: np.ndarray,
    reference: np.ndarray,
    thresholds_mm: Iterable[float],
    max_points: int,
    seed: int,
) -> Dict[str, float]:
    reconstruction = deterministic_sample(reconstruction, max_points, stable_seed(seed, "pred"))
    reference = deterministic_sample(reference, max_points, stable_seed(seed, "ref"))
    if len(reconstruction) == 0 or len(reference) == 0:
        raise ValueError("Cannot evaluate an empty cloud")
    reference_tree = cKDTree(reference)
    reconstruction_tree = cKDTree(reconstruction)
    pred_to_ref = reference_tree.query(reconstruction, k=1, workers=-1)[0]
    ref_to_pred = reconstruction_tree.query(reference, k=1, workers=-1)[0]
    metrics: Dict[str, float] = {
        "chamfer_l1_mm": 500.0 * float(pred_to_ref.mean() + ref_to_pred.mean()),
        "pred_to_ref_mean_mm": 1000.0 * float(pred_to_ref.mean()),
        "ref_to_pred_mean_mm": 1000.0 * float(ref_to_pred.mean()),
    }
    for threshold_mm in thresholds_mm:
        threshold_m = float(threshold_mm) / 1000.0
        precision = float(np.mean(pred_to_ref <= threshold_m))
        recall = float(np.mean(ref_to_pred <= threshold_m))
        fscore = 2.0 * precision * recall / max(precision + recall, 1e-12)
        tag = "%g" % float(threshold_mm)
        metrics["precision_%smm" % tag] = precision
        metrics["recall_%smm" % tag] = recall
        metrics["fscore_%smm" % tag] = fscore
    return metrics


def actual_error_statistics(
    rotation_vectors: np.ndarray,
    translations: np.ndarray,
    skip_first: bool = False,
) -> Dict[str, float]:
    if skip_first:
        rotation_vectors = rotation_vectors[1:]
        translations = translations[1:]
    rotation_deg = np.degrees(np.linalg.norm(rotation_vectors, axis=1))
    translation_mm = 1000.0 * np.linalg.norm(translations, axis=1)
    return {
        "actual_rotation_rms_deg": float(np.sqrt(np.mean(rotation_deg ** 2))),
        "actual_rotation_p95_deg": float(np.percentile(rotation_deg, 95)),
        "actual_translation_rms_mm": float(np.sqrt(np.mean(translation_mm ** 2))),
        "actual_translation_p95_mm": float(np.percentile(translation_mm, 95)),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(
    rows: Sequence[Dict[str, object]], conditions: Sequence[Dict[str, object]]
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    aggregate: List[Dict[str, object]] = []
    per_sequence: List[Dict[str, object]] = []
    metric_names = ["fscore_2mm", "fscore_5mm", "fscore_10mm", "chamfer_l1_mm"]
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition["name"]]
        if not selected:
            continue
        output: Dict[str, object] = {
            "condition": condition["name"],
            "rotation_rms_deg": float(condition["rotation_rms_deg"]),
            "translation_rms_mm": float(condition["translation_rms_mm"]),
            "samples": len(selected),
            "sequences": len({str(row["sequence"]) for row in selected}),
        }
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            output[metric] = float(values.mean())
            output[metric + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            output[metric + "_ci95"] = (
                1.96 * output[metric + "_std"] / math.sqrt(len(values))
            )
        aggregate.append(output)

        for sequence_name in sorted({str(row["sequence"]) for row in selected}):
            sequence_rows = [row for row in selected if row["sequence"] == sequence_name]
            per_sequence.append(
                {
                    "condition": condition["name"],
                    "sequence": sequence_name,
                    "samples": len(sequence_rows),
                    **{
                        metric: float(
                            np.mean([float(row[metric]) for row in sequence_rows])
                        )
                        for metric in metric_names
                    },
                }
            )
    return aggregate, per_sequence


def plot_results(aggregate: Sequence[Dict[str, object]], output_path: Path) -> None:
    # Import plotting only when needed so numerical/unit-test use has no GUI or
    # Matplotlib cache dependency.
    import os
    import tempfile

    mpl_cache = Path(tempfile.gettempdir()) / "aurora_pose_error_matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rotation = np.asarray([row["rotation_rms_deg"] for row in aggregate], dtype=float)
    fscore = 100.0 * np.asarray([row["fscore_5mm"] for row in aggregate], dtype=float)
    fscore_ci = 100.0 * np.asarray([row["fscore_5mm_ci95"] for row in aggregate], dtype=float)
    chamfer = np.asarray([row["chamfer_l1_mm"] for row in aggregate], dtype=float)
    chamfer_ci = np.asarray([row["chamfer_l1_mm_ci95"] for row in aggregate], dtype=float)

    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.15))
    axes[0].errorbar(rotation, fscore, yerr=fscore_ci, marker="o", capsize=3, color="#2673b8")
    axes[0].set_xlabel("Pose-error level (deg / mm RMS)")
    axes[0].set_ylabel("F-score at 5 mm (%)")
    lower_limit = max(0.0, 5.0 * math.floor((float(fscore.min()) - 5.0) / 5.0))
    axes[0].set_ylim(lower_limit, 101.0)
    axes[0].grid(alpha=0.25)
    axes[1].errorbar(rotation, chamfer, yerr=chamfer_ci, marker="o", capsize=3, color="#d55e00")
    axes[1].set_xlabel("Pose-error level (deg / mm RMS)")
    axes[1].set_ylabel("Symmetric mean distance (mm)")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), dpi=220, bbox_inches="tight")
    plt.close(figure)


def rebuttal_text(
    aggregate: Sequence[Dict[str, object]], sequence_count: int, keyframe_count: int, trials: int
) -> str:
    by_name = {str(row["condition"]): row for row in aggregate}
    severe = by_name.get("5deg_5mm")
    extreme = by_name.get("10deg_10mm")
    if severe is None:
        raise KeyError("The report expects a 5deg_5mm condition")
    lines = [
        "# Real-capture pose-error result",
        "",
        "## Reviewer response",
        "",
        (
            "We thank the reviewer for this suggestion. We added a controlled 6D pose-tracking "
            "error study on %d previously captured real in-hand RGB-D sequences (%d BundleTrack "
            "keyframes). We keep the observations, masks, and executed trajectories fixed and "
            "perturb only the tracked object-to-camera transform before AURORA's object-centric "
            "fusion. The first frame anchors the object coordinate system, and subsequent errors "
            "follow a temporally correlated SE(3) process; each nonzero level uses %d deterministic "
            "trials per sequence. At 5 deg / 5 mm RMS error, AURORA retains %.1f%% F-score at the "
            "5 mm threshold, with a %.2f mm symmetric mean distance to the clean-pose reconstruction."
        )
        % (
            sequence_count,
            keyframe_count,
            trials,
            100.0 * float(severe["fscore_5mm"]),
            float(severe["chamfer_l1_mm"]),
        ),
    ]
    if extreme is not None:
        lines[-1] += (
            " Even at 10 deg / 10 mm RMS, it retains %.1f%% F-score (%.2f mm symmetric distance)."
            % (100.0 * float(extreme["fscore_5mm"]), float(extreme["chamfer_l1_mm"]))
        )
    lines[-1] += (
        " This real-data sensitivity curve shows graceful degradation under moderate tracking "
        "noise while making clear that large pose drift remains a failure mode."
    )
    lines += [
        "",
        "## Aggregate table",
        "",
        "| Pose error (deg/mm RMS) | F@2 (%) | F@5 (%) | F@10 (%) | Sym. distance (mm) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| %.0f / %.0f | %.1f | %.1f | %.1f | %.2f |"
            % (
                float(row["rotation_rms_deg"]),
                float(row["translation_rms_mm"]),
                100.0 * float(row["fscore_2mm"]),
                100.0 * float(row["fscore_5mm"]),
                100.0 * float(row["fscore_10mm"]),
                float(row["chamfer_l1_mm"]),
            )
        )
    lines += [
        "",
        "F-scores and symmetric distances are measured against the reconstruction obtained from "
        "the same real RGB-D keyframes using the unperturbed BundleTrack poses. This isolates pose "
        "sensitivity; it is not an absolute CAD-ground-truth accuracy claim. Error bars in the plot "
        "are 95% normal confidence intervals over sequence/trial perturbation realizations.",
        "",
    ]
    return "\n".join(lines)


def validate_results(
    rows: Sequence[Dict[str, object]],
    aggregate: Sequence[Dict[str, object]],
    sequences: Sequence[CapturedSequence],
    conditions: Sequence[Dict[str, object]],
    trials: int,
) -> Dict[str, object]:
    expected_rows = len(sequences) * (
        1
        + trials
        * sum(
            float(item["rotation_rms_deg"]) != 0.0
            or float(item["translation_rms_mm"]) != 0.0
            for item in conditions
        )
    )
    keys = {(row["sequence"], row["condition"], row["trial"]) for row in rows}
    finite_fields = (
        "actual_rotation_rms_deg",
        "actual_translation_rms_mm",
        "fscore_2mm",
        "fscore_5mm",
        "fscore_10mm",
        "chamfer_l1_mm",
    )
    finite = all(
        np.isfinite(float(row[field])) for row in rows for field in finite_fields
    )
    metric_ranges = all(
        0.0 <= float(row[field]) <= 1.0
        for row in rows
        for field in ("fscore_2mm", "fscore_5mm", "fscore_10mm")
    ) and all(float(row["chamfer_l1_mm"]) >= 0.0 for row in rows)
    target_errors = all(
        abs(float(row["actual_rotation_rms_deg"]) - float(row["target_rotation_rms_deg"]))
        <= 1e-9
        and abs(
            float(row["actual_translation_rms_mm"])
            - float(row["target_translation_rms_mm"])
        )
        <= 1e-9
        for row in rows
    )
    clean_rows = [row for row in rows if row["condition"] == "clean"]
    clean_identity = len(clean_rows) == len(sequences) and all(
        float(row["fscore_2mm"]) == 1.0
        and float(row["fscore_5mm"]) == 1.0
        and float(row["fscore_10mm"]) == 1.0
        and float(row["chamfer_l1_mm"]) == 0.0
        for row in clean_rows
    )
    f5_curve = [float(row["fscore_5mm"]) for row in aggregate]
    distance_curve = [float(row["chamfer_l1_mm"]) for row in aggregate]
    checks = {
        "expected_sample_count": len(rows) == expected_rows,
        "unique_sequence_condition_trial_keys": len(keys) == len(rows),
        "finite_metrics": finite,
        "metric_ranges": metric_ranges,
        "realized_rms_matches_target": target_errors,
        "clean_reference_identity": clean_identity,
        "aggregate_f5_nonincreasing": all(
            right <= left + 1e-12 for left, right in zip(f5_curve, f5_curve[1:])
        ),
        "aggregate_distance_nondecreasing": all(
            right + 1e-12 >= left
            for left, right in zip(distance_curve, distance_curve[1:])
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "expected_samples": expected_rows,
        "observed_samples": len(rows),
        "checks": checks,
    }


def run(config_path: Path, trials_override: int | None = None) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    intrinsics_path = resolve_path(str(config["intrinsics_path"]))
    intrinsics = np.loadtxt(str(intrinsics_path), dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise ValueError("Intrinsics must be 3x3: %s" % intrinsics_path)
    output_dir = resolve_path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    trials = int(trials_override if trials_override is not None else config["trials"])
    if trials <= 0:
        raise ValueError("trials must be positive")

    sequences = [
        load_sequence(item, intrinsics, config["fusion"], int(config["seed"]))
        for item in config["sequences"]
    ]
    conditions = list(config["conditions"])
    thresholds = [float(value) for value in config["evaluation"]["thresholds_mm"]]
    evaluation_max = int(config["evaluation"]["max_points"])
    rows: List[Dict[str, object]] = []

    for sequence in sequences:
        for condition in conditions:
            is_clean = (
                float(condition["rotation_rms_deg"]) == 0.0
                and float(condition["translation_rms_mm"]) == 0.0
            )
            condition_trials = 1 if is_clean else trials
            for trial in range(condition_trials):
                trace_seed = stable_seed(
                    int(config["seed"]), sequence.name, condition["name"], trial
                )
                rotation_vectors, translations = correlated_errors(
                    len(sequence.frame_ids),
                    float(condition["rotation_rms_deg"]),
                    float(condition["translation_rms_mm"]),
                    float(config["noise"]["temporal_correlation"]),
                    trace_seed,
                    bool(config["noise"]["anchor_first_frame"]),
                )
                poses = perturb_poses(sequence.poses_co, rotation_vectors, translations)
                cloud = (
                    sequence.clean_cloud
                    if is_clean
                    else fuse_cloud(
                        sequence.camera_points,
                        poses,
                        float(config["fusion"]["fusion_voxel_m"]),
                    )
                )
                metrics = cloud_metrics(
                    cloud,
                    sequence.clean_cloud,
                    thresholds,
                    evaluation_max,
                    stable_seed(trace_seed, "metrics"),
                )
                error_stats = actual_error_statistics(
                    rotation_vectors,
                    translations,
                    skip_first=bool(config["noise"]["anchor_first_frame"]),
                )
                rows.append(
                    {
                        "sequence": sequence.name,
                        "keyframes": len(sequence.frame_ids),
                        "condition": condition["name"],
                        "trial": trial,
                        "target_rotation_rms_deg": float(condition["rotation_rms_deg"]),
                        "target_translation_rms_mm": float(condition["translation_rms_mm"]),
                        **error_stats,
                        "clean_points": len(sequence.clean_cloud),
                        "reconstruction_points": len(cloud),
                        **metrics,
                    }
                )
                print(
                    "[%s] %s trial %d/%d: F@5=%.4f, dist=%.3f mm"
                    % (
                        sequence.name,
                        condition["name"],
                        trial + 1,
                        condition_trials,
                        metrics["fscore_5mm"],
                        metrics["chamfer_l1_mm"],
                    ),
                    flush=True,
                )

    aggregate, per_sequence = summarize_rows(rows, conditions)
    write_csv(output_dir / "pose_error_samples.csv", rows)
    write_csv(output_dir / "pose_error_aggregate.csv", aggregate)
    write_csv(output_dir / "pose_error_per_sequence.csv", per_sequence)
    validation = validate_results(rows, aggregate, sequences, conditions, trials)
    (output_dir / "validation_report.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    if validation["status"] != "pass":
        raise RuntimeError("Pose-error result validation failed: %s" % validation)
    plot_results(aggregate, output_dir / "pose_error_curve.png")
    total_keyframes = sum(len(sequence.frame_ids) for sequence in sequences)
    (output_dir / "SUMMARY.md").write_text(
        rebuttal_text(aggregate, len(sequences), total_keyframes, trials), encoding="utf-8"
    )
    metadata = {
        "experiment_name": config["experiment_name"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": portable_path(config_path),
        "intrinsics_path": portable_path(intrinsics_path),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "scipy_rotation_convention": "R_noisy = Exp(delta_r) @ R_CO; t_noisy = t_CO + delta_t",
        "pose_convention": "T_CO maps object-frame points into the camera frame",
        "sequences": [
            {
                "name": sequence.name,
                "keyframes": len(sequence.frame_ids),
                "first_frame": sequence.frame_ids[0],
                "last_frame": sequence.frame_ids[-1],
                "clean_points": len(sequence.clean_cloud),
            }
            for sequence in sequences
        ],
        "trials_per_nonzero_condition": trials,
        "config": config,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("Results written to %s" % output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure AURORA reconstruction sensitivity to 6D tracking errors."
    )
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.json")
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Override trials per nonzero pose-error level (useful for a quick check).",
    )
    args = parser.parse_args()
    run(args.config.resolve(), args.trials)


if __name__ == "__main__":
    main()
