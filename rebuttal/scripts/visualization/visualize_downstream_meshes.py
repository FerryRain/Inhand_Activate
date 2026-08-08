#!/usr/bin/env python3
"""Create a compact mesh montage for the downstream rebuttal experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ROOT = Path(__file__).resolve().parents[3]
METHODS = (
    ("gt", "GT"),
    ("single_view_depth", "Single RGB-D"),
    ("spar3d", "SPAR3D"),
    ("trellis2", "TRELLIS.2"),
    ("fixed", "Fixed"),
    ("ray_gpis", "Ray-GPIS"),
)
OBJECTS = (
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "021_bleach_cleanser",
    "025_mug",
)


def load(path):
    value = trimesh.load(path, force="scene", process=False)
    if isinstance(value, trimesh.Scene):
        value = trimesh.util.concatenate(tuple(value.geometry.values()))
    return trimesh.Trimesh(value.vertices, value.faces, process=False)


def draw_mesh(axis, mesh, center, radius, color, seed):
    if len(mesh.faces) > 12000:
        simplified = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
            o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
        ).simplify_quadric_decimation(12000)
        mesh = trimesh.Trimesh(
            np.asarray(simplified.vertices),
            np.asarray(simplified.triangles),
            process=False,
        )
    faces = np.asarray(mesh.faces)
    triangles = np.asarray(mesh.vertices)[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = np.array([0.3, -0.5, 0.82])
    light /= np.linalg.norm(light)
    shade = np.clip(0.45 + 0.55 * np.abs(normals @ light), 0.25, 1.0)
    base = np.asarray(matplotlib.colors.to_rgb(color))
    colors = np.clip(base[None] * shade[:, None], 0.0, 1.0)
    collection = Poly3DCollection(
        triangles,
        facecolors=colors,
        edgecolors="none",
        linewidths=0.0,
        antialiaseds=False,
    )
    axis.add_collection3d(collection)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=22, azim=-52)
    axis.set_proj_type("ortho")
    axis.set_axis_off()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-root", default="rebuttal/results/downstream_ycb_meshes")
    parser.add_argument("--gt-root", default="rebuttal/assets/ycb/models")
    parser.add_argument(
        "--output",
        default="rebuttal/results/downstream_ycb_task/mesh_montage.png",
    )
    args = parser.parse_args()
    mesh_root = ROOT / args.mesh_root
    gt_root = ROOT / args.gt_root
    available = []
    for method, label in METHODS:
        if method == "gt" or all(
            (mesh_root / method / object_name / "pose_000.ply").exists()
            for object_name in OBJECTS
        ):
            available.append((method, label))
    figure = plt.figure(figsize=(2.25 * len(available), 2.15 * len(OBJECTS)))
    palette = {
        "gt": "#999999",
        "single_view_depth": "#d9a441",
        "spar3d": "#9d78c4",
        "trellis2": "#5a9bd4",
        "fixed": "#64ad78",
        "ray_gpis": "#d45555",
    }
    for row, object_name in enumerate(OBJECTS):
        gt = load(gt_root / object_name / "google_16k/nontextured.ply")
        center = np.asarray(gt.bounds).mean(axis=0)
        radius = 0.62 * float(np.max(gt.extents))
        for column, (method, label) in enumerate(available):
            axis = figure.add_subplot(
                len(OBJECTS), len(available), row * len(available) + column + 1,
                projection="3d",
            )
            mesh = gt if method == "gt" else load(
                mesh_root / method / object_name / "pose_000.ply"
            )
            draw_mesh(axis, mesh, center, radius, palette[method], 17 + row + column)
            if row == 0:
                axis.set_title(label, fontsize=11, pad=0)
            if column == 0:
                axis.text2D(
                    -0.06,
                    0.5,
                    object_name.replace("_", " "),
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=9,
                )
    figure.subplots_adjust(left=0.04, right=0.995, top=0.95, bottom=0.01, wspace=0.01, hspace=0.02)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
