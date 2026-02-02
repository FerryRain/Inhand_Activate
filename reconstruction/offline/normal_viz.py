#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import open3d as o3d


def load_pcd(path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise ValueError(f"Empty point cloud: {path}")
    return pcd


def denoise_pcd(
    pcd: o3d.geometry.PointCloud,
    voxel: float = 0.0,
    sor_nb_neighbors: int = 30,
    sor_std_ratio: float = 2.0,
    use_ror: bool = False,
    ror_radius: float = 0.02,
    ror_min_points: int = 10,
) -> o3d.geometry.PointCloud:
    """Return a cleaned point cloud (kept inliers only)."""

    # 0) optional voxel (usually do it before SOR to stabilize stats & speed)
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)

    # 1) Statistical Outlier Removal
    pcd_sor, ind = pcd.remove_statistical_outlier(
        nb_neighbors=int(sor_nb_neighbors),
        std_ratio=float(sor_std_ratio),
    )
    pcd = pcd_sor

    # 2) optional Radius Outlier Removal
    if use_ror:
        pcd_ror, ind = pcd.remove_radius_outlier(
            nb_points=int(ror_min_points),
            radius=float(ror_radius),
        )
        pcd = pcd_ror

    # 3) clean non-finite
    out = pcd.remove_non_finite_points()
    if isinstance(out, tuple):
        pcd = out[0]

    if len(pcd.points) == 0:
        raise RuntimeError("All points removed by denoising. Relax thresholds.")
    return pcd


def ensure_normals(
    pcd: o3d.geometry.PointCloud,
    estimate: bool,
    k: int,
    radius: float,
    orient: str,
) -> o3d.geometry.PointCloud:
    has_normals = pcd.has_normals() and len(pcd.normals) == len(pcd.points)
    if (not has_normals) or estimate:
        if radius > 0:
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=k))
        else:
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=k))

    n = np.asarray(pcd.normals)
    nn = np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    n = n / nn
    pcd.normals = o3d.utility.Vector3dVector(n.astype(np.float64))

    orient = orient.lower().strip()
    if orient == "camera":
        c = pcd.get_axis_aligned_bounding_box().get_center()
        cam = np.asarray(c) + np.array([0.0, 0.0, 10.0])
        pcd.orient_normals_towards_camera_location(cam)
    elif orient == "consistent":
        pcd.orient_normals_consistent_tangent_plane(k)
    elif orient == "none":
        pass
    else:
        raise ValueError(f"Unknown orient: {orient}")

    return pcd


def colors_from_normals(n: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.lower().strip()

    if mode == "xyz":
        c = (n + 1.0) * 0.5
        return np.clip(c, 0.0, 1.0)

    if mode == "abs":
        return np.clip(np.abs(n), 0.0, 1.0)

    if mode == "z":
        t = (n[:, 2] + 1.0) * 0.5
        t = np.clip(t, 0.0, 1.0)
        return np.stack([t, np.zeros_like(t), 1.0 - t], axis=1)

    if mode == "angle":
        z = np.clip(n[:, 2], -1.0, 1.0)
        ang = np.arccos(z)  # 0..pi
        t = ang / np.pi
        return np.stack([t, np.zeros_like(t), 1.0 - t], axis=1)

    raise ValueError(f"Unknown mode: {mode}")


def make_normal_lines(pts: np.ndarray, n: np.ndarray, colors: np.ndarray, scale: float) -> o3d.geometry.LineSet:
    p0 = pts
    p1 = pts + scale * n
    all_pts = np.vstack([p0, p1]).astype(np.float64)
    lines = np.array([[i, i + len(pts)] for i in range(len(pts))], dtype=np.int32)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(all_pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return ls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=str, default="/home/ferry/data/Code2/Research/Inhand_Activate/demo/results/cube_obj_01/002/recon_normal.ply")

    # denoise
    ap.add_argument("--voxel", type=float, default=0.002, help="voxel downsample before denoise (0 disables)")
    ap.add_argument("--sor_k", type=int, default=40, help="SOR nb_neighbors")
    ap.add_argument("--sor_std", type=float, default=1.5, help="SOR std_ratio (smaller => more aggressive)")
    ap.add_argument("--ror", action="store_true", help="enable radius outlier removal", default=True)
    ap.add_argument("--ror_radius", type=float, default=0.02)
    ap.add_argument("--ror_minpts", type=int, default=12)

    # normals
    ap.add_argument("--mode", type=str, default="xyz", choices=["xyz", "abs", "z", "angle"])
    ap.add_argument("--estimate", action="store_true", help="force re-estimate normals")
    ap.add_argument("--knn", type=int, default=30)
    ap.add_argument("--radius", type=float, default=0.0, help="normal radius (0 => KNN)")
    ap.add_argument("--orient", type=str, default="none", choices=["none", "camera", "consistent"])

    # viz
    ap.add_argument("--show_lines", action="store_true", default=True)
    ap.add_argument("--line_scale", type=float, default=0.01)

    args = ap.parse_args()

    pcd0 = load_pcd(args.in_path)
    print(f"[Load] {len(pcd0.points)} pts")

    pcd = denoise_pcd(
        pcd0,
        voxel=args.voxel,
        sor_nb_neighbors=args.sor_k,
        sor_std_ratio=args.sor_std,
        use_ror=args.ror,
        ror_radius=args.ror_radius,
        ror_min_points=args.ror_minpts,
    )
    print(f"[Denoise] {len(pcd.points)} pts (kept)")

    pcd = ensure_normals(pcd, estimate=args.estimate, k=args.knn, radius=args.radius, orient=args.orient)

    pts = np.asarray(pcd.points)
    n = np.asarray(pcd.normals)
    colors = colors_from_normals(n, args.mode)

    pcd_vis = o3d.geometry.PointCloud(pcd)
    pcd_vis.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    geoms = [pcd_vis, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)]
    if args.show_lines:
        geoms.append(make_normal_lines(pts, n, colors, scale=float(args.line_scale)))

    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"Denoised + Normals colored ({args.mode})",
        width=1280,
        height=800,
        point_show_normal=False,
    )


if __name__ == "__main__":
    main()
