#!/usr/bin/env python3
"""Render clean textured YCB inputs at the paired initial AURORA poses."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]


def camera_rays(width: int, height: int, intrinsics: np.ndarray):
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    direction = np.stack(
        [
            (cols + 0.5 - intrinsics[0, 2]) / intrinsics[0, 0],
            (rows + 0.5 - intrinsics[1, 2]) / intrinsics[1, 1],
            np.ones_like(cols),
        ],
        axis=-1,
    )
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    return np.concatenate([np.zeros_like(direction), direction], axis=-1).astype(np.float32)


def crop_foreground(rgba: np.ndarray, ratio: float = 1.30, size: int = 512):
    mask = rgba[..., 3] > 0
    rows, cols = np.nonzero(mask)
    if not len(rows):
        raise RuntimeError("Rendered YCB view has no foreground pixels")
    cy = 0.5 * (rows.min() + rows.max())
    cx = 0.5 * (cols.min() + cols.max())
    side = ratio * max(rows.max() - rows.min() + 1, cols.max() - cols.min() + 1)
    side = int(np.ceil(side))
    y0 = int(np.floor(cy - 0.5 * side))
    x0 = int(np.floor(cx - 0.5 * side))
    canvas = np.zeros((side, side, 4), dtype=np.uint8)
    src_y0, src_y1 = max(y0, 0), min(y0 + side, rgba.shape[0])
    src_x0, src_x1 = max(x0, 0), min(x0 + side, rgba.shape[1])
    dst_y0, dst_x0 = src_y0 - y0, src_x0 - x0
    canvas[
        dst_y0 : dst_y0 + src_y1 - src_y0,
        dst_x0 : dst_x0 + src_x1 - src_x0,
    ] = rgba[src_y0:src_y1, src_x0:src_x1]
    return np.asarray(Image.fromarray(canvas).resize((size, size), Image.Resampling.LANCZOS))


def render(mesh_path: Path, pose: np.ndarray, width=800, height=450):
    source = trimesh.load(str(mesh_path), force="mesh", process=False)
    vertices = np.asarray(source.vertices, dtype=np.float64)
    faces = np.asarray(source.faces, dtype=np.int32)
    uv = np.asarray(source.visual.uv, dtype=np.float64)
    texture = np.asarray(source.visual.material.image.convert("RGB"))

    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces)
    )
    tensor = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
    tensor.transform(o3d.core.Tensor(pose.astype(np.float32)))
    scene = o3d.t.geometry.RaycastingScene()
    object_id = scene.add_triangles(tensor)
    scale_x, scale_y = width / 400.0, height / 225.0
    intrinsics = np.array(
        [
            [189.85466 * scale_x, 0.0, 200.37016 * scale_x],
            [0.0, 189.81417 * scale_y, 114.78028 * scale_y],
            [0.0, 0.0, 1.0],
        ]
    )
    result = scene.cast_rays(o3d.core.Tensor(camera_rays(width, height, intrinsics)))
    geometry = result["geometry_ids"].numpy()
    valid = geometry == object_id
    primitive = result["primitive_ids"].numpy()
    barycentric = result["primitive_uvs"].numpy()
    normals = result["primitive_normals"].numpy()
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    if valid.any():
        face_index = primitive[valid].astype(np.int64)
        bary12 = barycentric[valid]
        weights = np.column_stack(
            [1.0 - bary12[:, 0] - bary12[:, 1], bary12[:, 0], bary12[:, 1]]
        )
        texture_uv = np.sum(uv[faces[face_index]] * weights[:, :, None], axis=1)
        texture_uv = np.mod(texture_uv, 1.0)
        tx = np.clip(np.rint(texture_uv[:, 0] * (texture.shape[1] - 1)), 0, texture.shape[1] - 1).astype(int)
        ty = np.clip(np.rint((1.0 - texture_uv[:, 1]) * (texture.shape[0] - 1)), 0, texture.shape[0] - 1).astype(int)
        color = texture[ty, tx].astype(np.float64)
        normal = normals[valid]
        light = np.array([0.35, -0.40, -0.85], dtype=np.float64)
        light /= np.linalg.norm(light)
        shade = np.clip(0.45 + 0.55 * np.abs(normal @ light), 0.35, 1.0)
        rgba[valid, :3] = np.clip(color * shade[:, None], 0, 255).astype(np.uint8)
        rgba[valid, 3] = 255
    return crop_foreground(rgba)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode-root", default="rebuttal/results/downstream_ycb_reconstruction/ray_gpis"
    )
    parser.add_argument("--output", default="rebuttal/assets/ycb/single_view_inputs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    episode_root = ROOT / args.episode_root
    output_root = ROOT / args.output
    for pose_path in sorted(episode_root.glob("*/pose_*/step_000/gt_pose.npy")):
        episode = pose_path.parents[1]
        object_name = episode.parent.name
        pose_name = episode.name
        output = output_root / object_name / f"{pose_name}.png"
        if output.exists() and not args.overwrite:
            continue
        mesh = (
            ROOT
            / "rebuttal/assets/ycb/models"
            / object_name
            / "google_16k/textured.obj"
        )
        image = render(mesh, np.load(pose_path))
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image, mode="RGBA").save(output)
        print(f"[input] {output}", flush=True)


if __name__ == "__main__":
    main()
