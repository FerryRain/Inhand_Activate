#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from benchmark.assets import load_mesh
from benchmark.config import load_config, resolved_path
from benchmark.geometry import action_rotation, axis_angle, random_rotation


def camera_rays(cfg):
    height = int(cfg["camera"]["height"])
    width = int(cfg["camera"]["width"])
    intrinsic = np.asarray(cfg["camera"]["intrinsics"], dtype=np.float64)
    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    directions = np.stack([
        (xs + 0.5 - intrinsic[0, 2]) / intrinsic[0, 0],
        (ys + 0.5 - intrinsic[1, 2]) / intrinsic[1, 1],
        np.ones_like(xs, dtype=np.float64),
    ], axis=-1)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    origins = np.zeros_like(directions)
    return np.concatenate([origins, directions], axis=-1).astype(np.float32)


def triangle_areas(mesh):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    corners = vertices[triangles]
    return 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    )


def visible_triangles(mesh, pose_co, rays):
    transformed = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    transformed.transform(o3d.core.Tensor(np.asarray(pose_co, dtype=np.float32)))
    scene = o3d.t.geometry.RaycastingScene()
    object_id = scene.add_triangles(transformed)
    result = scene.cast_rays(o3d.core.Tensor(rays))
    geometry_ids = result["geometry_ids"].numpy()
    primitive_ids = result["primitive_ids"].numpy()
    valid = geometry_ids == object_id
    return np.unique(primitive_ids[valid].astype(np.int64))


def pose(rotation, distance):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = np.array([0.0, 0.0, float(distance)], dtype=np.float64)
    return result


def coverage_curve(cfg, mesh, initial_pose_seed, actions, rays):
    areas = triangle_areas(mesh)
    total_area = float(areas.sum())
    initial = random_rotation(int(initial_pose_seed))
    bootstrap_degrees = float(cfg["environment"]["bootstrap_degrees"])
    bootstrap_offsets = cfg["environment"].get(
        "bootstrap_offsets_deg", (0.0, bootstrap_degrees, -bootstrap_degrees)
    )
    distance = float(cfg["camera"]["distance"])
    observed = set()
    for degrees in bootstrap_offsets:
        rotation = axis_angle(np.array([0.0, 1.0, 0.0]), np.radians(degrees)) @ initial
        observed.update(visible_triangles(mesh, pose(rotation, distance), rays).tolist())
    curve = [float(areas[list(observed)].sum() / total_area) if observed else 0.0]
    current = initial.copy()
    for action in actions:
        current = action_rotation(action, cfg["actions"]["angles_deg"]) @ current
        observed.update(visible_triangles(mesh, pose(current, distance), rays).tolist())
        curve.append(float(areas[list(observed)].sum() / total_area) if observed else 0.0)
    return {
        "definition": "cumulative visible GT triangle area / total GT mesh area",
        "coverage_curve": curve,
        "coverage_auc": float(np.mean(0.5 * (np.asarray(curve[:-1]) + np.asarray(curve[1:])))),
        "final_coverage": float(curve[-1]),
        "triangle_count": int(len(areas)),
        "total_surface_area_m2": total_area,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rebuttal/configs/formal_120_sixview_gpu.yaml")
    parser.add_argument("--planners", default="", help="Optional comma-separated planner filter")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    planner_filter = {item.strip() for item in args.planners.split(",") if item.strip()}
    root = resolved_path(cfg["experiment"]["output_dir"])
    rays = camera_rays(cfg)
    mesh_cache = {}
    result_cache = {}
    completed = 0
    summaries = {}
    for summary_path in sorted(root.glob("*/*/pose_*/episode_summary.json")):
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if planner_filter and summary["planner"] not in planner_filter:
            continue
        summary.setdefault("planner_seed", 0)
        key = (summary["planner"], summary["object"], summary["initial_pose_seed"], summary["planner_seed"])
        if key not in summaries or "_seed_" in summary_path.parent.name:
            summaries[key] = (summary_path, summary)
    for summary_path, summary in sorted(summaries.values(), key=lambda item: str(item[0])):
        output_path = summary_path.parent / "visibility_coverage.json"
        if output_path.exists() and not args.overwrite:
            completed += 1
            continue
        config_path = summary_path.parent / "config.json"
        with config_path.open("r", encoding="utf-8") as handle:
            episode_cfg = json.load(handle)
        mesh_path = episode_cfg["episode"]["mesh_path"]
        if mesh_path not in mesh_cache:
            mesh_cache[mesh_path] = load_mesh(mesh_path)
        key = (
            mesh_path,
            int(summary["initial_pose_seed"]),
            tuple(summary["actions"]),
        )
        if key not in result_cache:
            result_cache[key] = coverage_curve(
                cfg,
                mesh_cache[mesh_path],
                int(summary["initial_pose_seed"]),
                summary["actions"],
                rays,
            )
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result_cache[key], handle, indent=2)
        completed += 1
        if completed % 20 == 0:
            print("[coverage] %d episodes" % completed, flush=True)
    print("[coverage] complete: %d episodes, %d unique trajectories" % (completed, len(result_cache)))


if __name__ == "__main__":
    main()
