"""
@FileName：batch_normal_and_mesh_threshold.py
@Description：
    Batch pipeline for cumulative reconstructions:

    For each recon_0_*_xyz_ascii.ply:
      - If N_points > min_points_face:
          1) Estimate normals using external 'face' executable
          2) Poisson mesh reconstruction
      - Else:
          1) Estimate normals using Open3D
          2) Poisson mesh reconstruction

    Outputs:
      result_cum_5s/normal/recon_0_xxxxx_xyz_normal.ply
      result_cum_5s/mesh/recon_0_xxxxx_mesh.ply

@Author：Ferry (refactor by ChatGPT)
@Time：2026-01-13
"""

import os
import re
import glob
import argparse
import subprocess
import numpy as np
import open3d as o3d


# ---------------- Utilities ----------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def parse_end_frame_id_from_name(path: str) -> int:
    """
    Expect filename like: recon_0_000030_xyz_ascii.ply
    Return 30. If not matched, return -1.
    """
    name = os.path.basename(path)
    m = re.search(r"recon_0_(\d+)_xyz_ascii\.ply$", name)
    return int(m.group(1)) if m else -1


def run_cmd(cmd, dry_run=False):
    print("[CMD]", " ".join(cmd))
    if dry_run:
        return 0, ""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def safe_read_pcd(path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    return pcd


# ---------------- Normal estimation (Open3D) ----------------
def estimate_normals_o3d(
    pcd: o3d.geometry.PointCloud,
    radius: float,
    max_nn: int,
    orient_k: int,
) -> o3d.geometry.PointCloud:
    if pcd.is_empty():
        return pcd

    if radius <= 0:
        # fallback heuristic: 2% of bbox diagonal
        bbox = pcd.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bbox.get_extent())
        radius = max(1e-6, 0.02 * float(diag))

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius), max_nn=int(max_nn))
    )

    # orient normals (may fail for very sparse sets)
    if orient_k and orient_k > 0 and len(pcd.points) >= orient_k:
        try:
            pcd.orient_normals_consistent_tangent_plane(int(orient_k))
        except Exception as e:
            print(f"[WARN] orient_normals_consistent_tangent_plane failed: {e}")
    return pcd


# ---------------- Poisson reconstruction ----------------
def poisson_reconstruction(pcd: o3d.geometry.PointCloud, depth=9, density_quantile=0.01):
    print("[Info] Start Poisson reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(depth))

    densities = np.asarray(densities)
    print(
        f"[Info] Poisson mesh vertices: {len(mesh.vertices)}, "
        f"density range: [{densities.min():.4f}, {densities.max():.4f}]"
    )

    if density_quantile is not None and density_quantile > 0:
        thr = np.quantile(densities, float(density_quantile))
        print(f"[Info] Density threshold (quantile={density_quantile}): {thr:.4f}")
        keep_idx = np.where(densities > thr)[0]
        mesh = mesh.select_by_index(keep_idx)
        mesh.remove_unreferenced_vertices()
        print(f"[Info] After density crop: {len(mesh.vertices)} vertices")

    mesh.compute_vertex_normals()
    return mesh


def mesh_from_normal_ply(
    in_normal_ply: str,
    out_mesh_path: str,
    depth: int,
    density_quantile: float,
    voxel_size: float,
):
    pcd = o3d.io.read_point_cloud(in_normal_ply)
    if pcd.is_empty():
        raise RuntimeError(f"Empty point cloud: {in_normal_ply}")
    if not pcd.has_normals():
        raise RuntimeError(f"Point cloud has no normals: {in_normal_ply}")

    if voxel_size and voxel_size > 0:
        pcd = pcd.voxel_down_sample(float(voxel_size))
        print(f"[Info] Downsample before Poisson: voxel={voxel_size}, points={len(pcd.points)}")

    mesh = poisson_reconstruction(pcd, depth=depth, density_quantile=density_quantile)
    ensure_dir(os.path.dirname(out_mesh_path))
    ok = o3d.io.write_triangle_mesh(out_mesh_path, mesh)
    if not ok:
        raise RuntimeError(f"Failed to save mesh: {out_mesh_path}")


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Batch normal estimation (face or Open3D) + Poisson mesh reconstruction")

    parser.add_argument(
        "--in_dir",
        type=str,
        default="/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/result/cube_obj1/003/",
        help="Directory containing recon_0_*_xyz_ascii.ply",
    )
    parser.add_argument("--pattern", type=str, default="recon_0_*_xyz_ascii.ply")

    # threshold
    parser.add_argument("--min_points_face", type=int, default=1200,
                        help="If point count > this threshold, use external face; else use Open3D normals")

    # face
    parser.add_argument(
        "--face_bin",
        type=str,
        default="/home/ferry/data/Code2/HW/CG2/PaperSharing/face_2/cmake-build-debug/face",
        help="Path to face executable",
    )
    parser.add_argument("--a", type=float, default=0.01, help="face parameter --a")
    parser.add_argument("--use_i", action="store_true", default=True, help="pass --i to face")
    parser.add_argument("--use_h", action="store_true", default=True, help="pass --h to face")

    # open3d normal params (for small point clouds)
    parser.add_argument("--o3d_radius", type=float, default=0.02,
                        help="Open3D normal radius for small point clouds (<= threshold). "
                             "If <=0, use bbox heuristic.")
    parser.add_argument("--o3d_max_nn", type=int, default=30)
    parser.add_argument("--o3d_orient_k", type=int, default=30,
                        help="orient_normals_consistent_tangent_plane(k). Set 0 to disable.")

    # poisson params
    parser.add_argument("--poisson_depth", type=int, default=4)
    parser.add_argument("--density_quantile", type=float, default=0.01)
    parser.add_argument("--mesh_voxel", type=float, default=0.0, help="Optional voxel downsample before Poisson")

    # output dirs
    parser.add_argument("--normal_dir", type=str, default="", help="default: in_dir/normal")
    parser.add_argument("--mesh_dir", type=str, default="", help="default: in_dir/mesh")

    # runtime
    parser.add_argument("--overwrite", default=True, action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    in_dir = os.path.expanduser(args.in_dir)
    normal_dir = os.path.expanduser(args.normal_dir) if args.normal_dir else os.path.join(in_dir, "normal")
    mesh_dir = os.path.expanduser(args.mesh_dir) if args.mesh_dir else os.path.join(in_dir, "mesh")
    ensure_dir(normal_dir)
    ensure_dir(mesh_dir)

    face_bin = os.path.expanduser(args.face_bin)
    if not os.path.exists(face_bin):
        print(f"[WARN] face executable not found: {face_bin}")
        print("[WARN] For N > threshold cases, face step will fail unless path is correct.")

    in_files = sorted(
        glob.glob(os.path.join(os.path.join(in_dir, "pcd"), args.pattern)),
        key=lambda p: parse_end_frame_id_from_name(p),
    )
    if len(in_files) == 0:
        raise RuntimeError(f"No input files matched: {os.path.join(in_dir, args.pattern)}")

    print("====================================================")
    print(f"[INFO] in_dir           : {in_dir}")
    print(f"[INFO] found files      : {len(in_files)}")
    print(f"[INFO] min_points_face  : {args.min_points_face}")
    print(f"[INFO] normal_dir       : {normal_dir}")
    print(f"[INFO] mesh_dir         : {mesh_dir}")
    print(f"[INFO] face_bin         : {face_bin}")
    print(f"[INFO] face --a         : {args.a}")
    print(f"[INFO] o3d_radius       : {args.o3d_radius}")
    print(f"[INFO] o3d_max_nn       : {args.o3d_max_nn}")
    print(f"[INFO] o3d_orient_k     : {args.o3d_orient_k}")
    print(f"[INFO] poisson_depth    : {args.poisson_depth}")
    print(f"[INFO] density_quantile : {args.density_quantile}")
    print(f"[INFO] mesh_voxel       : {args.mesh_voxel}")
    print(f"[INFO] overwrite        : {args.overwrite}")
    print(f"[INFO] dry_run          : {args.dry_run}")
    print("====================================================")

    for idx, in_ply in enumerate(in_files, 1):
        end_id = parse_end_frame_id_from_name(in_ply)
        if end_id < 0:
            print(f"[WARN] skip (cannot parse id): {in_ply}")
            continue

        out_normal = os.path.join(normal_dir, f"recon_0_{end_id:06d}_xyz_normal.ply")
        out_mesh = os.path.join(mesh_dir, f"recon_0_{end_id:06d}_mesh.ply")

        # quick skip
        if (not args.overwrite) and os.path.exists(out_normal) and os.path.exists(out_mesh):
            print(f"\n[{idx}/{len(in_files)}] ID={end_id:06d} -> outputs exist, skip.")
            continue

        # count points from input
        pcd_in = safe_read_pcd(in_ply)
        n_pts = int(len(pcd_in.points))
        print(f"\n[{idx}/{len(in_files)}] ID={end_id:06d}  points={n_pts}")
        print(f"  in  : {in_ply}")
        print(f"  nrm : {out_normal}")
        print(f"  mesh: {out_mesh}")

        if pcd_in.is_empty() or n_pts == 0:
            print("[WARN] empty input point cloud, skip.")
            continue

        # Decide method
        use_face = (n_pts > int(args.min_points_face))

        # 1) normal estimation
        if args.overwrite or (not os.path.exists(out_normal)):
            if use_face:
                if not os.path.exists(face_bin):
                    print("[ERROR] face_bin missing; cannot run face. Skip this file.")
                    continue

                cmd = [face_bin, in_ply, "--o", out_normal]
                if args.use_i:
                    cmd.append("--i")
                cmd += ["--a", f"{args.a}"]
                if args.use_h:
                    cmd.append("--h")

                rc, out = run_cmd(cmd, dry_run=args.dry_run)
                if rc != 0:
                    print("[ERROR] face failed; output:")
                    print(out)
                    continue
            else:
                # Open3D normals for small point clouds
                if args.dry_run:
                    print("[DRY] would estimate normals using Open3D.")
                else:
                    pcd_small = estimate_normals_o3d(
                        pcd_in,
                        radius=float(args.o3d_radius),
                        max_nn=int(args.o3d_max_nn),
                        orient_k=int(args.o3d_orient_k),
                    )
                    ensure_dir(os.path.dirname(out_normal))
                    ok = o3d.io.write_point_cloud(out_normal, pcd_small, write_ascii=False)
                    if not ok:
                        print(f"[ERROR] failed to write normal ply: {out_normal}")
                        continue
        else:
            print("[Info] normal exists, skip normal estimation.")

        # 2) poisson mesh reconstruction
        if (not args.overwrite) and os.path.exists(out_mesh):
            print("[Info] mesh exists, skip mesh reconstruction.")
            continue

        # avoid Poisson on extremely small clouds
        if n_pts < 30:
            print("[WARN] too few points (<30). Skip Poisson mesh (normal file may still be saved).")
            continue

        if args.dry_run:
            print("[DRY] would run Poisson mesh reconstruction.")
        else:
            try:
                mesh_from_normal_ply(
                    in_normal_ply=out_normal,
                    out_mesh_path=out_mesh,
                    depth=int(args.poisson_depth),
                    density_quantile=float(args.density_quantile),
                    voxel_size=float(args.mesh_voxel),
                )
                print("[Done] mesh saved.")
            except Exception as e:
                print(f"[ERROR] mesh reconstruction failed: {e}")
                continue

    print("\n[Done] Batch normal + mesh pipeline finished.")


if __name__ == "__main__":
    main()
