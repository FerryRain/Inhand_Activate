"""
@FileName：viz_pcd.py
@Description：
@Author：Ferry
@Time：2026 1/27/26 3:09 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize 4 point clouds (x, y, z, active) with Open3D.

Default mode:
  - Single window, press keys to switch which point cloud is shown:
      1: x, 2: y, 3: z, 4: active
      R: reset view, Q/ESC: quit
Alternative:
  - Sequential mode: show 4 windows one by one.

Usage:
  python viz_4_pcds.py --x x.ply --y y.ply --z z.ply --active active.ply
  python viz_4_pcds.py --x x.ply --y y.ply --z z.ply --active active.ply --mode sequential
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import open3d as o3d
except Exception as e:
    raise RuntimeError("Open3D import failed. Please install open3d in your environment.") from e


def load_pcd(path: str) -> o3d.geometry.PointCloud:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Point cloud not found: {p}")
    pcd = o3d.io.read_point_cloud(str(p))
    if pcd is None or len(pcd.points) == 0:
        raise ValueError(f"Empty/invalid point cloud: {p}")

    # remove non-finite points (Open3D API differs by version)
    out = pcd.remove_non_finite_points()
    if isinstance(out, tuple):
        pcd = out[0]
    if len(pcd.points) == 0:
        raise ValueError(f"All points are non-finite after filtering: {p}")
    return pcd


def preprocess_pcd(pcd: o3d.geometry.PointCloud, voxel: float) -> o3d.geometry.PointCloud:
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    return pcd


def make_axes(size: float = 0.1) -> o3d.geometry.TriangleMesh:
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])


def pcd_extent(pcd: o3d.geometry.PointCloud) -> float:
    aabb = pcd.get_axis_aligned_bounding_box()
    ext = aabb.get_extent()
    return float(np.linalg.norm(np.asarray(ext)))


def show_single_window_switchable(
    clouds: Dict[str, o3d.geometry.PointCloud],
    point_size: float = 3.0,
    show_axes: bool = True,
    axes_scale: float = 0.15,
    start_key: str = "active",
) -> None:
    keys = ["x", "y", "z", "active"]
    if start_key not in clouds:
        start_key = keys[0]

    # precompute a reasonable axis size based on overall scale
    max_diag = 0.0
    for k in keys:
        if k in clouds:
            max_diag = max(max_diag, pcd_extent(clouds[k]))
    axes_size = max(1e-6, axes_scale * max_diag)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="4-PCD Viewer (1:x 2:y 3:z 4:active | R reset | Q/ESC quit)", width=1280, height=800)

    render_opt = vis.get_render_option()
    render_opt.point_size = float(point_size)

    axes = make_axes(size=axes_size) if show_axes else None

    state = {"cur": None, "pcd": None}

    def _set_cloud(name: str):
        if state["pcd"] is not None:
            vis.remove_geometry(state["pcd"], reset_bounding_box=False)
        state["pcd"] = clouds[name]
        vis.add_geometry(state["pcd"], reset_bounding_box=True)
        if axes is not None:
            # ensure axes stays
            pass
        state["cur"] = name
        print(f"[Viewer] showing: {name}")

    # add axes once
    if axes is not None:
        vis.add_geometry(axes, reset_bounding_box=False)

    _set_cloud(start_key)

    def cb_show(name: str):
        def _cb(_vis):
            _set_cloud(name)
            return False
        return _cb

    def cb_reset(_vis):
        _vis.reset_view_point(True)
        return False

    # key bindings
    vis.register_key_callback(ord("1"), cb_show("x"))
    vis.register_key_callback(ord("2"), cb_show("y"))
    vis.register_key_callback(ord("3"), cb_show("z"))
    vis.register_key_callback(ord("4"), cb_show("active"))
    vis.register_key_callback(ord("R"), cb_reset)
    vis.register_key_callback(ord("r"), cb_reset)

    print("[Keys] 1:x  2:y  3:z  4:active   R:reset   Q/ESC:quit")
    vis.run()
    vis.destroy_window()


def show_sequential(
    clouds: Dict[str, o3d.geometry.PointCloud],
    point_size: float = 3.0,
    show_axes: bool = True,
    axes_scale: float = 0.15,
) -> None:
    keys = ["x", "y", "z", "active"]

    max_diag = 0.0
    for k in keys:
        if k in clouds:
            max_diag = max(max_diag, pcd_extent(clouds[k]))
    axes_size = max(1e-6, axes_scale * max_diag)

    for k in keys:
        pcd = clouds[k]
        geoms = [pcd]
        if show_axes:
            geoms.append(make_axes(size=axes_size))
        print(f"[Sequential] Close window to continue: {k}")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"PCD: {k}", width=1280, height=800)
        for g in geoms:
            vis.add_geometry(g)
        opt = vis.get_render_option()
        opt.point_size = float(point_size)
        vis.run()
        vis.destroy_window()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=str, default="/home/ferry/data/Code2/Research/Inhand_Activate/Active/pcd/ablation/x/Cross/recon_0_000580.ply", help="Path to x-axis PCD (ply/pcd/xyz...).")
    ap.add_argument("--y", type=str, default="/home/ferry/data/Code2/Research/Inhand_Activate/Active/pcd/ablation/y/Cross/recon_0_000499.ply", help="Path to y-axis PCD (ply/pcd/xyz...).")
    ap.add_argument("--z", type=str, default="/home/ferry/data/Code2/Research/Inhand_Activate/Active/pcd/ablation/z/Cross/recon_0_000580.ply", help="Path to z-axis PCD (ply/pcd/xyz...).")
    ap.add_argument("--active", type=str, default="/home/ferry/data/Code2/Research/Inhand_Activate/Active/pcd/pcd_active/Cross/recon_0_000600.ply", help="Path to active PCD (ply/pcd/xyz...).")

    ap.add_argument("--voxel", type=float, default=0.0, help="Voxel downsample size. 0 disables.")
    ap.add_argument("--point_size", type=float, default=6.0, help="Open3D point size.")
    ap.add_argument("--no_axes", action="store_true", help="Disable coordinate axes.")
    ap.add_argument("--axes_scale", type=float, default=0.15, help="Axes size = axes_scale * max diag length.")
    ap.add_argument("--mode", choices=["switch", "sequential"], default="switch", help="switch: one window; sequential: 4 windows.")
    ap.add_argument("--start", choices=["x", "y", "z", "active"], default="active", help="Start cloud in switch mode.")

    args = ap.parse_args()

    clouds = {
        "x": preprocess_pcd(load_pcd(args.x), args.voxel),
        "y": preprocess_pcd(load_pcd(args.y), args.voxel),
        "z": preprocess_pcd(load_pcd(args.z), args.voxel),
        "active": preprocess_pcd(load_pcd(args.active), args.voxel),
    }

    show_axes = not args.no_axes

    if args.mode == "sequential":
        show_sequential(clouds, point_size=args.point_size, show_axes=show_axes, axes_scale=args.axes_scale)
    else:
        show_single_window_switchable(
            clouds,
            point_size=args.point_size,
            show_axes=show_axes,
            axes_scale=args.axes_scale,
            start_key=args.start,
        )


if __name__ == "__main__":
    main()
