"""
@FileName：normal_video_show.py
@Description：
@Author：Ferry
@Time：2026 1/29/26 9:08 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import cv2


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
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)

    pcd_sor, _ = pcd.remove_statistical_outlier(
        nb_neighbors=int(sor_nb_neighbors),
        std_ratio=float(sor_std_ratio),
    )
    pcd = pcd_sor

    if use_ror:
        pcd_ror, _ = pcd.remove_radius_outlier(
            nb_points=int(ror_min_points),
            radius=float(ror_radius),
        )
        pcd = pcd_ror

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


def set_orbit_camera(ctr: o3d.visualization.ViewControl, lookat: np.ndarray, cam_pos: np.ndarray, up: np.ndarray, zoom: float):
    front = (lookat - cam_pos).astype(np.float64)
    front /= (np.linalg.norm(front) + 1e-12)

    up = up.astype(np.float64)
    if abs(float(np.dot(front, up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    ctr.set_lookat(lookat.astype(np.float64))
    ctr.set_front(front)
    ctr.set_up(up)
    ctr.set_zoom(float(zoom))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=str,
                    default="/home/ferry/data/Code2/Research/Inhand_Activate/demo/results/cube_obj_01/002/recon_normal.ply")

    # denoise
    ap.add_argument("--voxel", type=float, default=0.002)
    ap.add_argument("--sor_k", type=int, default=40)
    ap.add_argument("--sor_std", type=float, default=1.5)
    ap.add_argument("--ror", action="store_true", default=True)  # 你原来就是默认 True
    ap.add_argument("--ror_radius", type=float, default=0.02)
    ap.add_argument("--ror_minpts", type=int, default=12)

    # normals
    ap.add_argument("--mode", type=str, default="xyz", choices=["xyz", "abs", "z", "angle"])
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--knn", type=int, default=30)
    ap.add_argument("--radius", type=float, default=0.0)
    ap.add_argument("--orient", type=str, default="none", choices=["none", "camera", "consistent"])

    # viz
    ap.add_argument("--show_lines", action="store_true", default=True)
    ap.add_argument("--line_scale", type=float, default=0.01)

    # video
    ap.add_argument("--out_path", type=str, default="/home/ferry/data/Code2/Research/Inhand_Activate/demo/results/cube_obj_01/002/normal.mp4", help="must end with .mp4 or .avi")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration", type=float, default=5.0, help="seconds for one full 360 orbit")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--visible", action="store_true", help="show window while recording (default off)")

    # camera orbit style
    ap.add_argument("--elev_deg", type=float, default=25.0, help="camera elevation angle in degrees")
    ap.add_argument("--dist_scale", type=float, default=2.0, help="camera distance scale wrt bbox radius")
    ap.add_argument("--zoom", type=float, default=0.7)

    args = ap.parse_args()

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() not in [".mp4", ".avi"]:
        raise ValueError("--out_path must end with .mp4 or .avi")

    # ---- load + denoise + normals ----
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
    nrm = np.asarray(pcd.normals)
    colors = colors_from_normals(nrm, args.mode)

    pcd_vis = o3d.geometry.PointCloud(pcd)
    pcd_vis.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    geoms = [pcd_vis, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)]
    if args.show_lines:
        geoms.append(make_normal_lines(pts, nrm, colors, scale=float(args.line_scale)))

    # ---- open3d visualizer (render to frames) ----
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Normals Orbit Recorder",
        width=int(args.width),
        height=int(args.height),
        visible=bool(args.visible),
    )

    for g in geoms:
        vis.add_geometry(g)

    # white background
    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    # set initial camera
    aabb = pcd_vis.get_axis_aligned_bounding_box()
    lookat = np.asarray(aabb.get_center(), dtype=np.float64)
    extent = np.asarray(aabb.get_extent(), dtype=np.float64)
    bbox_radius = 0.5 * float(np.linalg.norm(extent)) + 1e-12

    dist = float(args.dist_scale) * bbox_radius
    elev = np.deg2rad(float(args.elev_deg))
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    ctr = vis.get_view_control()
    # warm-up render
    vis.poll_events()
    vis.update_renderer()

    # ---- video writer ----
    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if out_path.suffix.lower() == ".mp4" else "XVID"))
    writer = cv2.VideoWriter(str(out_path), fourcc, float(args.fps), (int(args.width), int(args.height)))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {out_path} (try .avi or check codec)")

    n_frames = int(round(float(args.fps) * float(args.duration)))
    print(f"[Video] writing {n_frames} frames -> {out_path}")

    for i in range(n_frames):
        theta = 2.0 * np.pi * (i / max(n_frames, 1))  # 0..2pi
        cam_pos = lookat + dist * np.array([np.cos(theta) * np.cos(elev),
                                            np.sin(theta) * np.cos(elev),
                                            np.sin(elev)], dtype=np.float64)

        set_orbit_camera(ctr, lookat=lookat, cam_pos=cam_pos, up=up, zoom=float(args.zoom))

        vis.poll_events()
        vis.update_renderer()

        # capture float RGB [0,1]
        img = np.asarray(vis.capture_screen_float_buffer(True), dtype=np.float32)
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)          # RGB uint8
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)                   # to BGR for OpenCV

        # ensure exact size
        if (img.shape[1], img.shape[0]) != (int(args.width), int(args.height)):
            img = cv2.resize(img, (int(args.width), int(args.height)), interpolation=cv2.INTER_AREA)

        writer.write(img)

    writer.release()
    vis.destroy_window()
    print("[Done] saved:", out_path)


if __name__ == "__main__":
    main()
