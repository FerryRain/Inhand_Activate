"""
@FileName：mesh_refine.py
@Description：
@Author：Ferry
@Time：2026 1/30/26 9:35 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NKSR reconstruct mesh from (xyz, normal) point cloud and save mesh.

Usage:
  python nksr_recon_save.py \
    --input /path/to/recon_0_000244_xyz_normal.ply \
    --output /path/to/recon_mesh.ply \
    --detail_level 0.4 --mise_iter 1

Voxel-size mode:
  python nksr_recon_save.py \
    --input /path/to/pcd.ply \
    --output /path/to/recon_mesh.ply \
    --voxel_size 0.005 --mise_iter 1

Optional visualization:
  --vis
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import open3d as o3d
import nksr


def ensure_normals(pcd: o3d.geometry.PointCloud,
                   normal_radius: float = 0.0,
                   normal_max_nn: int = 30,
                   orient_k: int = 0) -> o3d.geometry.PointCloud:
    """Ensure pcd has normals; if not, estimate (and optionally orient)."""
    if pcd.has_normals() and len(pcd.normals) == len(pcd.points):
        return pcd

    pts = np.asarray(pcd.points)
    if pts.shape[0] < 10:
        raise RuntimeError("Too few points to estimate normals.")

    # auto radius if not provided: ~1% of AABB diagonal
    if normal_radius <= 0:
        aabb = pcd.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(aabb.get_max_bound() - aabb.get_min_bound())
        normal_radius = max(diag * 0.01, 1e-4)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(normal_radius),
            max_nn=int(normal_max_nn),
        )
    )

    if orient_k and orient_k > 0:
        pcd.orient_normals_consistent_tangent_plane(int(orient_k))

    return pcd


def to_o3d_mesh(v: np.ndarray, f: np.ndarray) -> o3d.geometry.TriangleMesh:
    """Convert (V,F) numpy arrays to Open3D TriangleMesh."""
    v = np.asarray(v, dtype=np.float64)
    f = np.asarray(f, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"Invalid vertices shape: {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"Invalid faces shape: {f.shape}")

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f.astype(np.int32))
    return mesh


def clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.remove_non_manifold_edges()
    return mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="/home/ferry/data/Code2/Research/Inhand_Activate/demo/results/process_pcd/Cross.ply", help="Input point cloud (.ply/.pcd/...) with normals preferred")
    parser.add_argument("--output", "-o", default="/home/ferry/data/Code2/Research/Inhand_Activate/demo/results/process_mesh/Cross.ply", help="Output mesh path (.ply/.obj/.stl/.glb/...)")
    parser.add_argument("--device", default="cuda:0", help="Device, e.g., cuda:0 or cpu")

    # reconstruct params: choose ONE
    parser.add_argument("--detail_level", type=float, default=0.3,
                        help="NKSR detail_level (used if --voxel_size <= 0)")
    parser.add_argument("--voxel_size", type=float, default=0.0,
                        help="NKSR voxel_size (if >0, overrides detail_level)")

    parser.add_argument("--mise_iter", type=int, default=1, help="Mise iterations for extract_dual_mesh")
    parser.add_argument("--vis", action="store_true", help="Visualize result with Open3D")

    # normal estimation fallback
    parser.add_argument("--normal_radius", type=float, default=0.0,
                        help="Normal estimation radius if input has no normals (0=auto)")
    parser.add_argument("--normal_max_nn", type=int, default=30)
    parser.add_argument("--orient_k", type=int, default=0,
                        help="If >0, orient normals consistently (slower)")

    # optional point downsample before NKSR (speed)
    parser.add_argument("--voxel_down", type=float, default=0.0,
                        help="Voxel downsample point cloud before reconstruct (0=no downsample)")

    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise FileNotFoundError(f"Not found: {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- load point cloud ----
    pcd = o3d.io.read_point_cloud(str(in_path))
    if pcd.is_empty():
        raise RuntimeError("Loaded point cloud is empty.")

    if args.voxel_down and args.voxel_down > 0:
        pcd = pcd.voxel_down_sample(float(args.voxel_down))

    # pcd = ensure_normals(
    #     pcd,
    #     normal_radius=float(args.normal_radius),
    #     normal_max_nn=int(args.normal_max_nn),
    #     orient_k=int(args.orient_k),
    # )

    xyz = np.asarray(pcd.points, dtype=np.float32)
    nrm = np.asarray(pcd.normals, dtype=np.float32)
    if xyz.shape[0] != nrm.shape[0]:
        raise RuntimeError("Points and normals count mismatch.")

    # ---- reconstruct ----
    device = torch.device(args.device)
    input_xyz = torch.from_numpy(xyz).to(device)
    input_normal = torch.from_numpy(nrm).to(device)

    reconstructor = nksr.Reconstructor(device)

    if args.voxel_size and args.voxel_size > 0:
        print(f"[NKSR] reconstruct with voxel_size={args.voxel_size}")
        field = reconstructor.reconstruct(input_xyz, input_normal, voxel_size=float(args.voxel_size))
    else:
        print(f"[NKSR] reconstruct with detail_level={args.detail_level}")
        field = reconstructor.reconstruct(input_xyz, input_normal, detail_level=float(args.detail_level))

    mesh_nksr = field.extract_dual_mesh(mise_iter=int(args.mise_iter))

    # NKSR mesh typically has attributes v, f (vertices, faces)
    v = np.asarray(mesh_nksr.v.cpu())
    f = np.asarray(mesh_nksr.f.cpu())

    mesh = to_o3d_mesh(v, f)
    mesh = clean_mesh(mesh)
    mesh.compute_vertex_normals()

    ok = o3d.io.write_triangle_mesh(str(out_path), mesh, write_ascii=False, compressed=False, print_progress=False)
    print(f"[OK] Saved mesh: {out_path} (success={ok})")
    print(f"[INFO] Mesh: |V|={len(mesh.vertices)}, |F|={len(mesh.triangles)}")

    if args.vis:
        # show mesh + original pcd
        o3d.visualization.draw_geometries([mesh, pcd], window_name="NKSR Mesh + PCD")


if __name__ == "__main__":
    main()
