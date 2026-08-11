#!/usr/bin/env python3
"""Evaluate FaCE + NKSR meshes against YCB ground-truth surfaces.

The default and light reconstructions of each object share one coordinate
frame.  A single rigid transform is therefore estimated from the more complete
light FaCE point cloud and then frozen for both NKSR meshes.  No scale fitting
is permitted.  This avoids independently optimizing the alignment of either
reconstruction variant.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import permutations, product
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


CASES = {
    "pitcher": {
        "gt": "/home/ferry/Code/Research/InHand/neuralfeels/data/assets/gt_models/ycb/019_pitcher_base/google_16k/nontextured.ply",
    },
    "mustard": {
        "gt": "/home/ferry/Code/Research/InHand/neuralfeels/data/assets/gt_models/ycb/006_mustard_bottle/google_16k/nontextured.ply",
    },
}


def load_cloud(path: Path) -> np.ndarray:
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float64)
    if len(points) < 20 or not np.isfinite(points).all():
        raise RuntimeError(f"Invalid point cloud: {path}")
    return points


def load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if len(mesh.vertices) < 4 or len(mesh.triangles) < 4:
        raise RuntimeError(f"Invalid triangle mesh: {path}")
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    return mesh


def sample_surface(
    mesh: o3d.geometry.TriangleMesh, count: int, seed: int
) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    xyz = vertices[triangles]
    areas = 0.5 * np.linalg.norm(
        np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1
    )
    valid = areas > 1e-14
    xyz = xyz[valid]
    areas = areas[valid]
    if len(xyz) == 0:
        raise RuntimeError("Mesh has no non-degenerate surface area")
    rng = np.random.RandomState(int(seed))
    indices = rng.choice(len(xyz), size=int(count), p=areas / areas.sum())
    chosen = xyz[indices]
    uv = rng.rand(int(count), 2)
    root = np.sqrt(uv[:, :1])
    return (
        (1.0 - root) * chosen[:, 0]
        + root * (1.0 - uv[:, 1:2]) * chosen[:, 1]
        + root * uv[:, 1:2] * chosen[:, 2]
    )


def pca_frame(points: np.ndarray):
    center = points.mean(axis=0)
    centered = points - center
    values, vectors = np.linalg.eigh(centered.T @ centered)
    frame = vectors[:, np.argsort(values)[::-1]]
    if np.linalg.det(frame) < 0.0:
        frame[:, 2] *= -1.0
    return center, frame


def rotation_hypotheses(source_frame: np.ndarray, target_frame: np.ndarray):
    for order in permutations(range(3)):
        permuted = source_frame[:, order]
        for signs in product((-1.0, 1.0), repeat=3):
            candidate = permuted @ np.diag(signs)
            rotation = target_frame @ candidate.T
            if np.linalg.det(rotation) > 0.999:
                yield rotation


def rigid_fit(source: np.ndarray, target: np.ndarray):
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right = np.linalg.svd(covariance)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right[-1] *= -1.0
        rotation = right.T @ left.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def trimmed_icp(
    source: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
    levels=((0.080, 0.95, 60), (0.040, 0.92, 80),
            (0.020, 0.90, 100), (0.010, 0.85, 120)),
):
    tree = cKDTree(target)
    transform = initial.copy()
    for maximum, quantile, iterations in levels:
        previous = float("inf")
        for _ in range(int(iterations)):
            transformed = source @ transform[:3, :3].T + transform[:3, 3]
            distances, indices = tree.query(transformed, k=1, workers=-1)
            cutoff = min(float(maximum), float(np.quantile(distances, quantile)))
            keep = distances <= max(cutoff, 1e-9)
            if np.count_nonzero(keep) < 20:
                break
            rotation, translation = rigid_fit(
                transformed[keep], target[indices[keep]]
            )
            update = np.eye(4, dtype=np.float64)
            update[:3, :3] = rotation
            update[:3, 3] = translation
            transform = update @ transform
            error = float(distances[keep].mean())
            if abs(previous - error) < 1e-8:
                break
            previous = error
    return transform


def make_scene(mesh: o3d.geometry.TriangleMesh):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def scene_distance(scene, points: np.ndarray) -> np.ndarray:
    return scene.compute_distance(
        o3d.core.Tensor(points.astype(np.float32))
    ).numpy().astype(np.float64)


def estimate_shared_rigid_transform(
    reference: np.ndarray,
    gt_mesh: o3d.geometry.TriangleMesh,
    alignment_samples: int,
):
    target = sample_surface(gt_mesh, alignment_samples, seed=1123)
    if len(reference) > 10000:
        rng = np.random.RandomState(1124)
        source = reference[rng.choice(len(reference), 10000, replace=False)]
    else:
        source = reference
    _, source_frame = pca_frame(source)
    _, target_frame = pca_frame(target)
    source_aabb_center = 0.5 * (source.min(axis=0) + source.max(axis=0))
    target_aabb_center = 0.5 * (target.min(axis=0) + target.max(axis=0))
    gt_scene = make_scene(gt_mesh)
    candidates = []
    for index, rotation in enumerate(
        rotation_hypotheses(source_frame, target_frame)
    ):
        initial = np.eye(4, dtype=np.float64)
        initial[:3, :3] = rotation
        initial[:3, 3] = target_aabb_center - rotation @ source_aabb_center
        transform = trimmed_icp(source, target, initial)
        transformed = source @ transform[:3, :3].T + transform[:3, 3]
        distances = scene_distance(gt_scene, transformed)
        cutoff = np.quantile(distances, 0.90)
        trimmed_mean = float(distances[distances <= cutoff].mean())
        candidates.append(
            {
                "index": index,
                "trimmed_mean_m": trimmed_mean,
                "median_m": float(np.median(distances)),
                "precision_at_5mm": float(np.mean(distances < 0.005)),
                "transform": transform,
            }
        )
    candidates.sort(
        key=lambda row: (row["trimmed_mean_m"], -row["precision_at_5mm"])
    )
    return candidates[0]["transform"], candidates


def evaluate_mesh(
    reconstructed: o3d.geometry.TriangleMesh,
    gt_mesh: o3d.geometry.TriangleMesh,
    samples: int,
    threshold_m: float,
    seed: int,
):
    rec_points = sample_surface(reconstructed, samples, seed=seed)
    gt_points = sample_surface(gt_mesh, samples, seed=seed + 1)
    gt_scene = make_scene(gt_mesh)
    rec_scene = make_scene(reconstructed)
    rec_to_gt = scene_distance(gt_scene, rec_points)
    gt_to_rec = scene_distance(rec_scene, gt_points)
    precision = float(np.mean(rec_to_gt < threshold_m))
    recall = float(np.mean(gt_to_rec < threshold_m))
    fscore = (
        0.0
        if precision + recall == 0.0
        else float(2.0 * precision * recall / (precision + recall))
    )
    return {
        "precision_at_5mm": precision,
        "recall_at_5mm": recall,
        "fscore_at_5mm": fscore,
        "accuracy_mm": float(1000.0 * rec_to_gt.mean()),
        "completeness_mm": float(1000.0 * gt_to_rec.mean()),
        "chamfer_l1_mm": float(1000.0 * (rec_to_gt.mean() + gt_to_rec.mean())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default="rebuttal/results/nksr_fused_cloud_eval_20260811"
    )
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--alignment-samples", type=int, default=50000)
    parser.add_argument("--sampling-repeats", type=int, default=5)
    parser.add_argument("--threshold-mm", type=float, default=5.0)
    parser.add_argument("--reuse-transforms", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    aligned_root = root / "meshes_aligned_shared_rigid"
    aligned_root.mkdir(parents=True, exist_ok=True)
    results = []
    metadata = {
        "alignment": "one shared light-derived rigid transform per object",
        "scale_fitting": False,
        "threshold_mm": float(args.threshold_mm),
        "surface_samples_per_direction": int(args.samples),
        "sampling_repeats": int(args.sampling_repeats),
        "objects": {},
    }
    for object_index, (object_name, case) in enumerate(CASES.items()):
        gt_path = Path(case["gt"])
        gt_mesh = load_mesh(gt_path)
        reference_path = root / "face_normals" / f"{object_name}_light_face.ply"
        reference = load_cloud(reference_path)
        transform_path = root / f"{object_name}_shared_rigid_T.npy"
        if args.reuse_transforms and transform_path.is_file():
            transform = np.load(transform_path)
            candidates = []
        else:
            transform, candidates = estimate_shared_rigid_transform(
                reference, gt_mesh, int(args.alignment_samples)
            )
        object_meta = {
            "gt_mesh": str(gt_path),
            "alignment_reference": str(reference_path),
            "transform_gt_from_reconstruction": transform.tolist(),
            "top_alignment_candidates": [
                {key: value for key, value in row.items() if key != "transform"}
                for row in candidates[:5]
            ],
        }
        metadata["objects"][object_name] = object_meta
        np.save(transform_path, transform)
        for variant_index, variant in enumerate(("default", "light")):
            mesh_path = (
                root
                / "mesh_nksr"
                / f"{object_name}_{variant}_face_nksr_detail0.4.ply"
            )
            reconstructed = load_mesh(mesh_path)
            reconstructed.transform(transform)
            output = aligned_root / f"{object_name}_{variant}_nksr.ply"
            if not o3d.io.write_triangle_mesh(str(output), reconstructed):
                raise RuntimeError(f"Failed to write {output}")
            repeated_metrics = [
                evaluate_mesh(
                    reconstructed,
                    gt_mesh,
                    int(args.samples),
                    0.001 * float(args.threshold_mm),
                    seed=(
                        20260811
                        + 1000 * repeat
                        + 10 * object_index
                        + variant_index
                    ),
                )
                for repeat in range(int(args.sampling_repeats))
            ]
            metrics = {
                key: float(np.mean([item[key] for item in repeated_metrics]))
                for key in repeated_metrics[0]
            }
            metric_stds = {
                f"{key}_std": float(
                    np.std([item[key] for item in repeated_metrics], ddof=1)
                )
                if len(repeated_metrics) > 1
                else 0.0
                for key in repeated_metrics[0]
            }
            row = {
                "object": object_name,
                "variant": variant,
                "face_points": int(
                    len(load_cloud(root / "face_normals" / f"{object_name}_{variant}_face.ply"))
                ),
                "nksr_vertices": int(len(reconstructed.vertices)),
                "nksr_triangles": int(len(reconstructed.triangles)),
                **metrics,
                **metric_stds,
            }
            results.append(row)
            print(
                f"{object_name}/{variant}: P@5={row['precision_at_5mm']:.4f} "
                f"R@5={row['recall_at_5mm']:.4f} F@5={row['fscore_at_5mm']:.4f}"
            )
    with (root / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    metadata["results"] = results
    (root / "evaluation.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
