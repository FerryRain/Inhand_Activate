"""
@FileName：nksr_recon_by_normal.py
@Description：
@Author：Ferry
@Time：2026 1/21/26 5:25 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch NKSR reconstruction from point clouds with normals.

- Input: <root>/normal/*.ply (or other extensions)
- Output: <out_dir>/*.ply meshes

Two modes (choose one):
  1) voxel mode  : reconstructor.reconstruct(..., voxel_size=...)
  2) detail mode : reconstructor.reconstruct(..., detail_level=...)

Example:
  python recon_batch_nksr.py \
    --root /home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/result/offline_tracking/cube_purple/001_ICP \
    --mode detail --detail_level 0.4
"""

import os
import glob
import argparse
import traceback
from typing import List, Tuple

import numpy as np
import torch
import nksr

# Prefer standard Open3D; fall back to open3d_pycg if your env uses that.
try:
    import open3d as o3d  # type: ignore
except Exception:
    import open3d_pycg as o3d  # type: ignore

# Optional warning helper (matches your example style)
try:
    from examples.common import warning_on_low_memory  # type: ignore
except Exception:
    def warning_on_low_memory(_threshold_mb: float) -> None:
        return


def list_input_files(normal_dir: str, exts: List[str]) -> List[str]:
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(normal_dir, f"*.{ext}")))
    files = sorted(set(files))
    return files


def ensure_normals(pcd: "o3d.geometry.PointCloud") -> "o3d.geometry.PointCloud":
    # If normals are missing, estimate them as a fallback.
    # Your normal/ folder is expected to already contain normals, so this should rarely trigger.
    normals = np.asarray(pcd.normals)
    if normals.size == 0:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
        )
        pcd.normalize_normals()
    return pcd


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def mesh_to_open3d(mesh) -> "o3d.geometry.TriangleMesh":
    v = to_numpy(mesh.v)
    f = to_numpy(mesh.f)
    if v.dtype != np.float64:
        v = v.astype(np.float64)
    if f.dtype != np.int32:
        f = f.astype(np.int32)

    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(v)
    m.triangles = o3d.utility.Vector3iVector(f)
    m.compute_vertex_normals()
    return m


def reconstruct_one(
    reconstructor: "nksr.Reconstructor",
    device: torch.device,
    pcd_path: str,
    mode: str,
    voxel_size: float,
    detail_level: float,
    mise_iter: int,
) -> "o3d.geometry.TriangleMesh":
    pcd = o3d.io.read_point_cloud(pcd_path)
    if pcd.is_empty():
        raise RuntimeError(f"Empty point cloud: {pcd_path}")

    pcd = ensure_normals(pcd)
    pts = np.asarray(pcd.points)
    nrm = np.asarray(pcd.normals)

    if pts.size == 0:
        raise RuntimeError(f"No points: {pcd_path}")
    if nrm.size == 0:
        raise RuntimeError(f"No normals (and estimation failed): {pcd_path}")
    if pts.shape != nrm.shape:
        raise RuntimeError(f"Points/normals shape mismatch: {pts.shape} vs {nrm.shape} in {pcd_path}")

    input_xyz = torch.from_numpy(pts).float().to(device)
    input_normal = torch.from_numpy(nrm).float().to(device)

    if mode == "voxel":
        field = reconstructor.reconstruct(input_xyz, input_normal, voxel_size=voxel_size)
    elif mode == "detail":
        field = reconstructor.reconstruct(input_xyz, input_normal, detail_level=detail_level)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    mesh = field.extract_dual_mesh(mise_iter=mise_iter)
    return mesh_to_open3d(mesh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/result/offline_tracking/Big_Cylinder/003_ICP",
        help="Root folder that contains the 'normal' subfolder.",
    )
    parser.add_argument("--normal_subdir", type=str, default="normal", help="Subfolder name under root.")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory. Default: <root>/nksr_mesh_<mode>")

    parser.add_argument("--mode", type=str, choices=["voxel", "detail"], default="detail",
                        help="Choose exactly one: voxel or detail.")
    parser.add_argument("--voxel_size", type=float, default=0.0050, help="Used only when --mode voxel.")
    parser.add_argument("--detail_level", type=float, default=0.4, help="Used only when --mode detail.")
    parser.add_argument("--mise_iter", type=int, default=1, help="Mise iteration for dual mesh extraction.")

    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 or cpu")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing meshes.")
    parser.add_argument(
        "--exts",
        type=str,
        default="ply,pcd,xyz,xyzn,pts,obj",
        help="Comma-separated extensions to search in normal/ directory.",
    )
    args = parser.parse_args()

    warning_on_low_memory(1024.0)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    reconstructor = nksr.Reconstructor(device)

    normal_dir = os.path.join(args.root, args.normal_subdir)
    if not os.path.isdir(normal_dir):
        raise FileNotFoundError(f"normal dir not found: {normal_dir}")

    exts = [e.strip().lstrip(".") for e in args.exts.split(",") if e.strip()]
    inputs = list_input_files(normal_dir, exts)
    if len(inputs) == 0:
        raise FileNotFoundError(f"No input files found under: {normal_dir} with exts={exts}")

    out_dir = os.path.join(args.root, "mesh_nksr")
    os.makedirs(out_dir, exist_ok=True)

    tag = f"vox{args.voxel_size:g}" if args.mode == "voxel" else f"detail{args.detail_level:g}"

    ok, bad = 0, 0
    failures: List[Tuple[str, str]] = []

    print(f"[INFO] device      : {device}")
    print(f"[INFO] mode        : {args.mode} ({tag})")
    print(f"[INFO] normal_dir   : {normal_dir}")
    print(f"[INFO] out_dir      : {out_dir}")
    print(f"[INFO] num_inputs   : {len(inputs)}")

    for i, p in enumerate(inputs):
        base = os.path.splitext(os.path.basename(p))[0]
        out_path = os.path.join(out_dir, f"{base}_nksr_{tag}.ply")

        if (not args.overwrite) and os.path.exists(out_path):
            print(f"[SKIP] {i+1:04d}/{len(inputs)} exists: {out_path}")
            continue

        try:
            mesh_o3d = reconstruct_one(
                reconstructor=reconstructor,
                device=device,
                pcd_path=p,
                mode=args.mode,
                voxel_size=args.voxel_size,
                detail_level=args.detail_level,
                mise_iter=args.mise_iter,
            )
            ok_write = o3d.io.write_triangle_mesh(out_path, mesh_o3d, write_ascii=False, compressed=False)
            if not ok_write:
                raise RuntimeError(f"Failed to write mesh: {out_path}")

            ok += 1
            print(f"[OK]   {i+1:04d}/{len(inputs)} -> {out_path}")

        except Exception as e:
            bad += 1
            msg = f"{type(e).__name__}: {e}"
            failures.append((p, msg))
            print(f"[FAIL] {i+1:04d}/{len(inputs)} {p}\n       {msg}")
            # Uncomment if you want full traceback:
            # traceback.print_exc()

    print("\n================ Summary ================")
    print(f"Success: {ok}")
    print(f"Failed : {bad}")
    if failures:
        print("Failed files:")
        for fp, err in failures:
            print(f"  - {fp}\n    {err}")


if __name__ == "__main__":
    main()
