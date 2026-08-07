#!/usr/bin/env python3
"""Visualize the six existing real AURORA meshes against scanner GT."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from visualize_downstream_meshes import draw_mesh, load


ROOT = Path(__file__).resolve().parents[1]
OBJECTS = (
    "Big_Cylinder",
    "cube_obj_01",
    "cube_obj_02",
    "cube_purple",
    "green_Pepper",
    "tetraprism",
)


def canonical_pair(gt, predicted):
    """Put a registered pair in the same GT-derived PCA display frame."""
    gt = gt.copy()
    predicted = predicted.copy()
    center = np.asarray(gt.vertices).mean(axis=0)
    centered = np.asarray(gt.vertices) - center
    covariance = centered.T @ centered / max(len(centered), 1)
    _, basis = np.linalg.eigh(covariance)
    basis = basis[:, ::-1]
    for column in range(3):
        dominant = int(np.argmax(np.abs(basis[:, column])))
        if basis[dominant, column] < 0:
            basis[:, column] *= -1
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    gt.vertices = (np.asarray(gt.vertices) - center) @ basis
    predicted.vertices = (np.asarray(predicted.vertices) - center) @ basis
    return gt, predicted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mesh-root", default="rebuttal/results/downstream_real_meshes/aurora_real"
    )
    parser.add_argument("--gt-root", default="reconstruction/offline/GT_data")
    parser.add_argument(
        "--output", default="rebuttal/results/downstream_real_task/mesh_montage.png"
    )
    args = parser.parse_args()
    mesh_root, gt_root = ROOT / args.mesh_root, ROOT / args.gt_root
    figure = plt.figure(figsize=(5.0, 1.85 * len(OBJECTS)))
    for row, object_name in enumerate(OBJECTS):
        gt = load(gt_root / object_name / "GT/mesh/GT_mesh.stl")
        # Scanner STL files are triangle soups. Weld them before Open3D
        # decimation; otherwise simplification leaves only scattered facets.
        gt.merge_vertices()
        gt.remove_unreferenced_vertices()
        if np.linalg.norm(gt.extents) > 5.0:
            gt.apply_scale(0.001)
        predicted = load(mesh_root / object_name / "pose_000.ply")
        gt, predicted = canonical_pair(gt, predicted)
        center = np.zeros(3)
        radius = 0.62 * float(np.max(gt.extents))
        for column, (mesh, title, color) in enumerate(
            ((gt, "Scanner GT", "#999999"), (predicted, "Real AURORA mesh", "#d45555"))
        ):
            axis = figure.add_subplot(
                len(OBJECTS), 2, 2 * row + column + 1, projection="3d"
            )
            draw_mesh(axis, mesh, center, radius, color, 100 + row + column)
            if row == 0:
                axis.set_title(title, fontsize=11, pad=0)
            if column == 0:
                axis.text2D(
                    -0.05,
                    0.5,
                    object_name.replace("_", " "),
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=8,
                )
    figure.subplots_adjust(left=0.09, right=0.99, top=0.965, bottom=0.01, wspace=0.01, hspace=0.01)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
