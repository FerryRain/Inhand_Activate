from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import open3d as o3d


def jsonable(value: Any):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def write_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True, allow_nan=False)


def write_cloud(path: Path, points: np.ndarray):
    cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    )
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False):
        raise RuntimeError("Failed to write point cloud: %s" % path)


def save_step(
    step_dir: Path,
    observation,
    fused_points: np.ndarray,
    candidate_directions: np.ndarray,
    planner_scores: np.ndarray,
    metrics: Dict,
    selected_nbv: Optional[np.ndarray],
    action_payload: Dict,
    counterfactual: Optional[Dict[str, Dict]] = None,
    save_images: bool = True,
):
    step_dir.mkdir(parents=True, exist_ok=True)
    if save_images:
        cv2.imwrite(str(step_dir / "rgb.png"), cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(step_dir / "mask.png"), observation.mask.astype(np.uint8) * 255)
    np.save(str(step_dir / "depth.npy"), observation.depth.astype(np.float32))
    np.save(str(step_dir / "gt_pose.npy"), observation.gt_pose.astype(np.float64))
    np.save(str(step_dir / "executed_pose.npy"), observation.executed_pose.astype(np.float64))
    np.save(str(step_dir / "candidate_directions.npy"), candidate_directions.astype(np.float32))
    np.save(str(step_dir / "planner_scores.npy"), np.asarray(planner_scores, dtype=np.float32))
    selected = np.full((3,), np.nan, dtype=np.float32) if selected_nbv is None else np.asarray(selected_nbv, dtype=np.float32)
    np.save(str(step_dir / "selected_nbv.npy"), selected)
    write_cloud(step_dir / "fused_cloud.ply", fused_points)
    write_json(step_dir / "metrics.json", metrics)
    write_json(step_dir / "selected_action.json", action_payload)
    if counterfactual:
        counterfactual_dir = step_dir / "counterfactual"
        for action, action_metrics in counterfactual.items():
            write_json(counterfactual_dir / (action + "_metrics.json"), action_metrics)
