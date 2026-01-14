"""
@FileName：mesh_reconstruction.py
@Description：
@Author：Ferry
@Time：2025 11/29/25 12:11 AM
@Copyright：©2024-2025 ShanghaiTech University-RIMLAB
"""

import argparse
import os
import numpy as np
import open3d as o3d


def load_point_cloud(path, estimate_normal_if_missing=False, voxel_size=None):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] < 6:
            raise ValueError(
                f"Expect npy shape (N, 6) [xyz, nx, ny, nz], got {arr.shape}"
            )

        xyz = arr[:, :3].astype(np.float32)
        nrm = arr[:, 3:6].astype(np.float32)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.normals = o3d.utility.Vector3dVector(nrm)

    else:
        pcd = o3d.io.read_point_cloud(path)
        if pcd.is_empty():
            raise RuntimeError(f"Failed to load point cloud from {path}")

        if not pcd.has_normals() and estimate_normal_if_missing:
            print("[Info] No normals in file, estimating normals...")
            if voxel_size is not None and voxel_size > 0:
                pcd = pcd.voxel_down_sample(voxel_size)
                print(f"[Info] Downsampled to {len(pcd.points)} points.")

            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=voxel_size * 2 if voxel_size else 0.05,
                    max_nn=30,
                )
            )
            pcd.orient_normals_consistent_tangent_plane(30)
        elif not pcd.has_normals():
            raise RuntimeError(
                "Point cloud has no normals. "
                "Use --estimate_normal_if_missing 或自己先算好 normal。"
            )

    return pcd


def poisson_reconstruction(pcd, depth=9, density_thresh=0.01):
    print("[Info] Start Poisson reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    densities = np.asarray(densities)
    print(
        f"[Info] Poisson mesh vertices: {len(mesh.vertices)}, "
        f"density range: [{densities.min():.4f}, {densities.max():.4f}]"
    )

    if density_thresh is not None:
        thr = np.quantile(densities, density_thresh)
        print(f"[Info] Density threshold (quantile={density_thresh}): {thr:.4f}")
        keep_idx = np.where(densities > thr)[0]
        mesh = mesh.select_by_index(keep_idx)
        mesh.remove_unreferenced_vertices()
        print(f"[Info] After density crop: {len(mesh.vertices)} vertices")

    mesh.compute_vertex_normals()
    return mesh


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct mesh from normal-bearing point cloud (Poisson)."
    )
    parser.add_argument("--input", type=str, default="/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/result/green_cube_02_000/normal/recon_0_000122_xyz_ascii.ply", help="Input point cloud (.ply/.pcd/.npy ...)")
    parser.add_argument(
        "output",
        type=str,
        nargs="?",
        help="Output mesh file (.ply / .obj). Default: same name with _mesh.ply",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=6,
        help="Poisson tree depth (default: 9, range ~[6,12])",
    )
    parser.add_argument(
        "--density_quantile",
        type=float,
        default=0.01,
        help="Low-density vertices below this quantile will be removed (default: 0.01). "
             "Set to 0 or None to disable.",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.0,
        help="Optional voxel downsample size before reconstruction (0 = no downsample)",
    )
    parser.add_argument(
        "--estimate_normal_if_missing",
        action="store_true",
        help="If input has no normals, estimate normals (for non-npy point clouds).",
    )
    args = parser.parse_args()

    in_path = args.input
    out_path = args.output
    if out_path is None:
        base, _ = os.path.splitext(in_path)
        out_path = base + "_mesh.ply"

    print(f"[Info] Loading point cloud from {in_path}")
    voxel_size = args.voxel_size if args.voxel_size > 0 else None
    pcd = load_point_cloud(
        in_path,
        estimate_normal_if_missing=args.estimate_normal_if_missing,
        voxel_size=voxel_size,
    )

    if voxel_size is not None:
        print(f"[Info] Using voxel_size={voxel_size}, points={len(pcd.points)}")

    if not pcd.has_normals():
        raise RuntimeError("Point cloud still has no normals, abort.")

    mesh = poisson_reconstruction(
        pcd,
        depth=args.depth,
        density_thresh=args.density_quantile if args.density_quantile > 0 else None,
    )

    print(f"[Info] Saving mesh to {out_path}")
    ok = o3d.io.write_triangle_mesh(out_path, mesh)
    if not ok:
        raise RuntimeError(f"Failed to save mesh to {out_path}")
    print("[Done] Mesh reconstruction finished.")


if __name__ == "__main__":
    main()
