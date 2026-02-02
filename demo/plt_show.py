"""
@FileName：plt_show.py
@Description：
@Author：Ferry
@Time：2026 1/29/26 3:43 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Render a demo video:
- Video FPS = 20 (default)
- Every 1 second, swap the input point cloud to the next second's ply
- Meanwhile, orbit camera to create continuous rotation
- Works in headless mode using Open3D OffscreenRenderer + OpenCV VideoWriter

Example:
python make_pcd_rotate_update_video.py \
  --pcd_dir /home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/result/demo/cube/001/pcd \
  --out demo_cube.mp4 --fps 20 --t_end 30 --turns 1.0
"""

import os
import re
import glob
import argparse
from pathlib import Path

import numpy as np

import open3d as o3d

try:
    import cv2
except ImportError as e:
    raise ImportError("Please install opencv-python: pip install opencv-python") from e

try:
    import open3d.visualization.rendering as rendering
except Exception as e:
    raise RuntimeError(
        "Open3D rendering module not available. Please use Open3D>=0.15.\n"
        "If you are in a headless server, you may need:\n"
        "  export OPEN3D_CPU_RENDERING=1\n"
        "or install EGL/OSMesa support."
    ) from e


_RECON_RE = re.compile(r"recon_0_(\d+)\.ply$")


def parse_frame_id(p: str) -> int:
    m = _RECON_RE.search(os.path.basename(p))
    if not m:
        raise ValueError(f"Not a recon_0_XXXXXX.ply file: {p}")
    return int(m.group(1))


def list_recon_plys(pcd_dir: str):
    p = str(Path(pcd_dir).resolve())
    files = glob.glob(os.path.join(p, "recon_0_*.ply"))
    items = []
    for f in files:
        try:
            fid = parse_frame_id(f)
            items.append((fid, f))
        except Exception:
            pass
    items.sort(key=lambda x: x[0])
    if not items:
        raise FileNotFoundError(f"No recon_0_*.ply found in: {p}")
    return items


def pick_nearest(items, target_frame: int):
    # items: [(fid, path), ...] sorted
    fids = np.array([x[0] for x in items], dtype=np.int64)
    idx = int(np.argmin(np.abs(fids - target_frame)))
    return items[idx]


def load_pcd(path: str):
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError(f"Empty point cloud: {path}")
    # ensure finite
    pts = np.asarray(pcd.points)
    if not np.isfinite(pts).all():
        mask = np.isfinite(pts).all(axis=1)
        pcd = pcd.select_by_index(np.where(mask)[0].tolist())
    return pcd


def ensure_colors(pcd: o3d.geometry.PointCloud):
    # If no colors, set light gray
    if not pcd.has_colors():
        n = np.asarray(pcd.points).shape[0]
        pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.65, 0.65, 0.65]]), (n, 1)))
    return pcd


def compute_camera_params_from_bbox(aabb: o3d.geometry.AxisAlignedBoundingBox):
    center = aabb.get_center()
    extent = aabb.get_extent()
    diag = float(np.linalg.norm(extent))
    # radius heuristic: keep object comfortably in view
    radius = max(0.15, 1.6 * diag)
    return center, radius


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcd_dir", type=str, default="/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/result/demo/cube/001/pcd", help="Folder containing recon_0_*.ply")
    ap.add_argument("--out", type=str, default="/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/result/demo/cube/001/video_offline.mp4", help="Output video path (.mp4 recommended)")
    ap.add_argument("--fps", type=int, default=20, help="Video FPS (20 frames = 1 second)")
    ap.add_argument("--t_end", type=float, default=None, help="End time (seconds). If None, inferred from max frame / fps")
    ap.add_argument("--turns", type=float, default=3.0, help="How many full 360-degree turns over the whole video")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--point_size", type=float, default=5)
    ap.add_argument("--bg", type=str, default="white", choices=["white", "black"])
    ap.add_argument("--elev_deg", type=float, default=20.0, help="Camera elevation angle in degrees")
    ap.add_argument("--fov_deg", type=float, default=60.0, help="Camera field of view in degrees")
    ap.add_argument("--accumulate", action="store_true",
                    help="If set, accumulate points over time (merge per-second clouds).")
    ap.add_argument("--voxel", type=float, default=0.0,
                    help="If accumulate, optionally voxel-downsample merged cloud (e.g., 0.001).")
    args = ap.parse_args()

    items = list_recon_plys(args.pcd_dir)
    max_frame = items[-1][0]

    if args.t_end is None:
        args.t_end = max_frame / float(args.fps)
        # round up a bit so the last second is included visually
        args.t_end = float(int(np.ceil(args.t_end)))
    t_end = args.t_end

    # Prepare per-second selected clouds
    sec_list = list(range(int(np.floor(t_end)) + 1))  # include 0..t_end (integer seconds)
    per_sec_paths = []
    for s in sec_list:
        target = int(round(s * args.fps))
        fid, path = pick_nearest(items, target)
        per_sec_paths.append((s, fid, path))

    # Load last (or final selected) pcd for stable camera framing
    last_path = per_sec_paths[-1][2]
    pcd_last = ensure_colors(load_pcd(last_path))
    aabb = pcd_last.get_axis_aligned_bounding_box()
    center, radius = compute_camera_params_from_bbox(aabb)

    # Offscreen renderer
    renderer = rendering.OffscreenRenderer(args.width, args.height)
    scene = renderer.scene
    scene.scene.set_sun_light(
        [0.577, -0.577, -0.577],  # direction
        [1.0, 1.0, 1.0],          # color
        75000                      # intensity
    )
    scene.scene.enable_sun_light(True)

    if args.bg == "white":
        scene.set_background([1.0, 1.0, 1.0, 1.0])
    else:
        scene.set_background([0.0, 0.0, 0.0, 1.0])

    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = float(args.point_size)

    total_frames = int(np.round(t_end * args.fps))
    total_frames = max(total_frames, 1)

    # Video writer (mp4)
    out_path = str(Path(args.out).resolve())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, float(args.fps), (args.width, args.height), True)
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {out_path}")

    # Cache per-second point clouds
    per_sec_pcd = []
    for (s, fid, path) in per_sec_paths:
        pcd = ensure_colors(load_pcd(path))
        per_sec_pcd.append((s, fid, path, pcd))

    # Optional accumulate
    merged = o3d.geometry.PointCloud()
    merged_has_init = False

    # Render loop
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    elev = np.deg2rad(args.elev_deg)

    def set_geometry(pcd_to_show: o3d.geometry.PointCloud):
        if scene.has_geometry("pcd"):
            scene.remove_geometry("pcd")
        scene.add_geometry("pcd", pcd_to_show, mat)

    # init geometry
    if args.accumulate:
        merged = o3d.geometry.PointCloud(per_sec_pcd[0][3])
        merged_has_init = True
        if args.voxel > 0:
            merged = merged.voxel_down_sample(args.voxel)
        set_geometry(merged)
    else:
        set_geometry(per_sec_pcd[0][3])

    # camera: use look_at each frame
    cam = scene.camera
    cam.set_projection(args.fov_deg, args.width / args.height, 0.01, 1000.0, rendering.Camera.FovType.Vertical)

    for i in range(total_frames):
        sec = int(i // args.fps)
        sec = min(sec, len(per_sec_pcd) - 1)

        # update point cloud each second (only when sec changes)
        if i % args.fps == 0:
            if args.accumulate:
                if not merged_has_init:
                    merged = o3d.geometry.PointCloud(per_sec_pcd[sec][3])
                    merged_has_init = True
                else:
                    merged += per_sec_pcd[sec][3]
                if args.voxel > 0:
                    merged = merged.voxel_down_sample(args.voxel)
                set_geometry(merged)
            else:
                set_geometry(per_sec_pcd[sec][3])

        # orbit angle
        theta = 2.0 * np.pi * args.turns * (i / max(total_frames - 1, 1))
        # camera position (orbit around center)
        # put some elevation to see 3D structure
        xy_r = radius * np.cos(elev)
        z = center[2] + radius * np.sin(elev)
        eye = np.array([
            center[0] + xy_r * np.cos(theta),
            center[1] + xy_r * np.sin(theta),
            z
        ], dtype=np.float64)

        cam.look_at(center, eye, up)

        img_o3d = renderer.render_to_image()
        img = np.asarray(img_o3d)  # (H, W, 4) uint8 RGBA
        rgb = img[:, :, :3]
        bgr = rgb[:, :, ::-1]
        vw.write(bgr)

        if (i + 1) % (args.fps * 5) == 0:
            s, fid, path, _ = per_sec_pcd[sec]
            print(f"[{i+1}/{total_frames}] t={i/args.fps:.2f}s  sec={s}  using frame={fid}  {os.path.basename(path)}")

    vw.release()
    renderer.release()
    print(f"\nDone. Saved video to: {out_path}")
    print("Tip: If you see black frames in headless env, try: export OPEN3D_CPU_RENDERING=1")


if __name__ == "__main__":
    main()
