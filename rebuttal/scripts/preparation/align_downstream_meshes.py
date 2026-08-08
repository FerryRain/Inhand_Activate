#!/usr/bin/env python3
"""Similarity-align generated single-view meshes for downstream evaluation.

The submitted paper already evaluates TRELLIS.2/SPAR3D after post-hoc GT scale
alignment.  This script deliberately gives those methods the same favourable,
oracle alignment for the downstream shape-usefulness test.  The transform is
isotropic (no shape-warping); the aligned mesh is then planned on exactly like
all other reconstructions.
"""

from __future__ import annotations

import argparse
import json
from itertools import permutations, product
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[3]


def load(path: Path):
    value = trimesh.load(path, force="scene", process=False)
    if isinstance(value, trimesh.Scene):
        if not value.geometry:
            raise RuntimeError(f"Empty mesh scene: {path}")
        value = trimesh.util.concatenate(tuple(value.geometry.values()))
    mesh = trimesh.Trimesh(value.vertices, value.faces, process=True)
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) < 4:
        raise RuntimeError(f"Invalid mesh: {path}")
    return mesh


def metric_gt(path: Path):
    mesh = load(path)
    if np.linalg.norm(mesh.extents) > 5.0:
        mesh.apply_scale(0.001)
    return mesh


def sample(mesh, count, seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        points, _ = trimesh.sample.sample_surface(mesh, int(count))
    finally:
        np.random.set_state(state)
    return np.asarray(points, dtype=np.float64)


def pca(points):
    center = np.mean(points, axis=0)
    covariance = np.cov((points - center).T)
    values, vectors = np.linalg.eigh(covariance)
    frame = vectors[:, np.argsort(values)[::-1]]
    if np.linalg.det(frame) < 0.0:
        frame[:, 2] *= -1.0
    return center, frame


def rotations(source_frame, target_frame):
    for permutation in permutations(range(3)):
        permuted = source_frame[:, permutation]
        for signs in product((-1.0, 1.0), repeat=3):
            candidate = permuted @ np.diag(signs)
            rotation = target_frame @ candidate.T
            if np.linalg.det(rotation) > 0.999:
                yield rotation


def symmetric_distance(source, target, maximum=8000):
    source = source[:maximum]
    target = target[:maximum]
    return 0.5 * (
        cKDTree(target).query(source, k=1, workers=-1)[0].mean()
        + cKDTree(source).query(target, k=1, workers=-1)[0].mean()
    )


def rigid_fit(source, target):
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right = np.linalg.svd(covariance)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right[-1] *= -1.0
        rotation = right.T @ left.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def icp(source, target, initial, maximum_distance, iterations):
    target_tree = cKDTree(target)
    transform = initial.copy()
    previous = float("inf")
    for _ in range(int(iterations)):
        transformed = source @ transform[:3, :3].T + transform[:3, 3]
        distances, indices = target_tree.query(transformed, k=1, workers=-1)
        cutoff = min(float(maximum_distance), float(np.quantile(distances, 0.85)))
        keep = distances <= max(cutoff, 1e-9)
        if np.count_nonzero(keep) < 20:
            break
        rotation, translation = rigid_fit(transformed[keep], target[indices[keep]])
        update = np.eye(4)
        update[:3, :3] = rotation
        update[:3, 3] = translation
        transform = update @ transform
        error = float(np.mean(distances[keep]))
        if abs(previous - error) < 1e-7:
            break
        previous = error
    return transform


def align_similarity(source_mesh, target_mesh, count=6000, allow_scale=True):
    source = sample(source_mesh, count, 7)
    target = sample(target_mesh, count, 11)
    source_center, source_frame = pca(source)
    target_center, target_frame = pca(target)
    source_diag = np.linalg.norm(np.ptp(source, axis=0))
    target_diag = np.linalg.norm(np.ptp(target, axis=0))
    scale = target_diag / max(source_diag, 1e-12) if allow_scale else 1.0

    base = np.eye(4)
    base[:3, :3] *= scale
    base[:3, 3] = -scale * source_center
    scaled_source = (source - source_center) * scale
    diagonal = float(np.linalg.norm(target_mesh.extents))

    candidates = []
    for rotation in rotations(source_frame, target_frame):
        init = np.eye(4)
        init[:3, :3] = rotation
        init[:3, 3] = target_center
        coarse = icp(
            scaled_source,
            target,
            init,
            maximum_distance=max(0.030, 0.30 * diagonal),
            iterations=45,
        )
        transformed = scaled_source @ coarse[:3, :3].T + coarse[:3, 3]
        error = symmetric_distance(transformed, target)
        candidates.append((error, coarse))

    candidates.sort(key=lambda item: item[0])
    best_error, transform = candidates[0]
    for distance, iterations in ((max(0.012, 0.12 * diagonal), 90), (max(0.006, 0.06 * diagonal), 120)):
        transform = icp(
            scaled_source,
            target,
            transform,
            maximum_distance=distance,
            iterations=iterations,
        )
    transformed = scaled_source @ transform[:3, :3].T + transform[:3, 3]
    best_error = symmetric_distance(transformed, target)
    return transform @ base, float(scale), float(best_error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root", default="rebuttal/results/downstream_single_view_raw"
    )
    parser.add_argument("--output-root", default="rebuttal/results/downstream_ycb_meshes")
    parser.add_argument("--gt-root", default="rebuttal/assets/ycb/models")
    parser.add_argument("--methods", default="trellis2,spar3d")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    raw_root = ROOT / args.raw_root
    output_root = ROOT / args.output_root
    gt_root = ROOT / args.gt_root
    for method in [x.strip() for x in args.methods.split(",") if x.strip()]:
        for path in sorted((raw_root / method).glob("*/pose_*.ply")):
            object_name = path.parent.name
            output = output_root / method / object_name / path.name
            metadata = output.with_suffix(".alignment.json")
            if output.exists() and not args.overwrite:
                continue
            source = load(path)
            target = metric_gt(gt_root / object_name / "google_16k/nontextured.ply")
            transform, scale, error = align_similarity(source, target, allow_scale=True)
            source.apply_transform(transform)
            output.parent.mkdir(parents=True, exist_ok=True)
            source.export(output)
            metadata.write_text(
                json.dumps(
                    {
                        "alignment": "oracle_isotropic_similarity",
                        "scale": scale,
                        "symmetric_alignment_error_m": error,
                        "transform": transform.tolist(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[align] {method}/{object_name}/{path.name}: scale={scale:.5f}, error={error * 1000:.2f} mm", flush=True)


if __name__ == "__main__":
    main()
