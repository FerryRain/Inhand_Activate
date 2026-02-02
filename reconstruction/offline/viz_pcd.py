"""
@FileName：viz_pcd.py
@Description：
@Author：Ferry
@Time：2026 1/8/26 3:42 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import numpy as np
import open3d as o3d


def try_load_pointcloud(path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    if pcd is not None and len(pcd.points) > 0:
        return pcd

    # fallback: maybe it's a mesh -> sample points
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh is not None and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_poisson_disk(number_of_points=200000)
        return pcd

    return o3d.geometry.PointCloud()


def estimate_normals_if_needed(pcd: o3d.geometry.PointCloud, radius: float, max_nn: int):
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        pcd.orient_normals_consistent_tangent_plane(k=min(50, len(pcd.points)))


def auto_axis_size(pcd: o3d.geometry.PointCloud) -> float:
    aabb = pcd.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(aabb.get_extent())
    return float(max(0.05, 0.15 * diag))  # heuristic


def main():
    parser = argparse.ArgumentParser(description="Simple point cloud viewer (Open3D).")
    # parser.add_argument("-i", "--input", default="/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/result/offline_tracking/cube_obj_01/003_ICP/pcd/recon_0_000600.ply", help="Path to point cloud (ply/pcd/xyz/xyzn/xyzrgb) or mesh (obj/stl/off).")
    parser.add_argument("-i", "--input",
                        default="/home/ferry/data/Code2/Research/Inhand_Activate/demo/results/process_pcd/Cross.ply",
                        help="Path to point cloud (ply/pcd/xyz/xyzn/xyzrgb) or mesh (obj/stl/off).")

    parser.add_argument("--voxel", type=float, default=0.0, help="Voxel downsample size (e.g. 0.002). 0 = disable.")
    parser.add_argument("--random", type=int, default=0, help="Random sample N points (e.g. 200000). 0 = disable.")

    parser.add_argument("--show-axis", action="store_true", help="Show coordinate frame.")
    parser.add_argument("--axis-size", type=float, default=0.0, help="Coordinate frame size. 0 = auto.")

    parser.add_argument("--point-size", type=float, default=4.0, help="Initial point size.")
    parser.add_argument("--bg", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="Background color RGB in [0,1]. e.g. --bg 0 0 0")

    parser.add_argument("--show-normals", default=False, help="Show normals (if present or estimated).")
    parser.add_argument("--estimate-normals", default=True, help="Estimate normals if not present.")
    parser.add_argument("--normal-radius", type=float, default=0.01, help="Normal estimation search radius (in meters).")
    parser.add_argument("--normal-max-nn", type=int, default=30, help="Normal estimation max nn.")

    args = parser.parse_args()

    in_path = args.input
    if not os.path.exists(in_path):
        print(f"[ERROR] Not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    pcd = try_load_pointcloud(in_path)
    if len(pcd.points) == 0:
        print("[ERROR] Failed to load point cloud/mesh or empty geometry.", file=sys.stderr)
        sys.exit(2)

    # optional sampling
    if args.voxel > 0:
        pcd = pcd.voxel_down_sample(args.voxel)

    if args.random > 0 and len(pcd.points) > args.random:
        idx = np.random.choice(len(pcd.points), size=args.random, replace=False)
        pcd = pcd.select_by_index(idx)

    # normals
    if args.estimate_normals:
        estimate_normals_if_needed(pcd, radius=args.normal_radius, max_nn=args.normal_max_nn)

    # print stats
    aabb = pcd.get_axis_aligned_bounding_box()
    print(f"[INFO] Points: {len(pcd.points)}")
    print(f"[INFO] AABB min: {aabb.get_min_bound()}, max: {aabb.get_max_bound()}")

    # build geometries
    geometries = [pcd]
    axis = None
    if args.show_axis:
        size = args.axis_size if args.axis_size > 0 else auto_axis_size(pcd)
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])
        geometries.append(axis)

    # viewer with key callbacks
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="PointCloud Viewer", width=1280, height=720)
    for g in geometries:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.background_color = np.array(args.bg, dtype=np.float32)
    opt.point_size = float(args.point_size)
    opt.point_show_normal = bool(args.show_normals)

    print("\n[KEYS]")
    print("  +/- : increase/decrease point size")
    print("  N   : toggle normals")
    print("  A   : toggle axis")
    print("  R   : reset view")
    print("  Q/Esc: quit\n")

    def cb_inc_point_size(_):
        opt.point_size = float(opt.point_size + 1.0)
        return False

    def cb_dec_point_size(_):
        opt.point_size = float(max(1.0, opt.point_size - 1.0))
        return False

    def cb_toggle_normals(_):
        opt.point_show_normal = not opt.point_show_normal
        return False

    def cb_toggle_axis(_):
        nonlocal axis
        if axis is None:
            size = args.axis_size if args.axis_size > 0 else auto_axis_size(pcd)
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])
            vis.add_geometry(axis, reset_bounding_box=False)
        else:
            vis.remove_geometry(axis, reset_bounding_box=False)
            axis = None
        return False

    def cb_reset(_):
        vis.reset_view_point(True)
        return False

    vis.register_key_callback(ord('+'), cb_inc_point_size)
    vis.register_key_callback(ord('='), cb_inc_point_size)
    vis.register_key_callback(ord('-'), cb_dec_point_size)
    vis.register_key_callback(ord('N'), cb_toggle_normals)
    vis.register_key_callback(ord('A'), cb_toggle_axis)
    vis.register_key_callback(ord('R'), cb_reset)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
