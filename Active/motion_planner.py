"""
@FileName：motion_planner.py
@Description：Visualize W(world=object@t0), current O(t), fused point cloud, NBV, suggested WORLD axis, and camera position in W.
@Author：Ferry (refactor by ChatGPT)
@Time：2026 1/13/26
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""

from Active.NBV_gpis import GPISNBVv2

import argparse
import copy
import numpy as np
import open3d as o3d


# ------------------------------ IO ------------------------------
def load_T_txt(path: str) -> np.ndarray:
    T = np.loadtxt(path).astype(np.float64)
    if T.shape == (3, 4):
        T = np.vstack([T, [0, 0, 0, 1]])
    if T.shape != (4, 4):
        raise ValueError(f"{path}: expected 4x4 (or 3x4), got {T.shape}")
    return T


# ------------------------------ SO(3) utils ------------------------------
def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v / (n + eps)


def rotation_matrix_from_a_to_b(a, b):
    a = _normalize(a.astype(np.float64))
    b = _normalize(b.astype(np.float64))
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return np.eye(3, dtype=np.float64) if c > 0 else -np.eye(3, dtype=np.float64)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1 - c) / (s * s))


def so3_log(R):
    tr = np.trace(R)
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-9:
        return np.zeros(3, dtype=np.float64)
    w = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ], dtype=np.float64) / (2.0 * np.sin(theta) + 1e-12)
    return w * theta


# ------------------------------ Planner core ------------------------------
def pick_world_axis_and_rvec(T_CO: np.ndarray, T_CO0: np.ndarray, best_dir_O: np.ndarray):
    """
    World W := object frame at t=0.

    Returns:
      axis_idx, rvec_W, aW, vW, T_WO, T_WC
    """
    R_CO,  t_CO  = T_CO[:3, :3],  T_CO[:3, 3]
    R_CO0, t_CO0 = T_CO0[:3, :3], T_CO0[:3, 3]

    # T_WC = inv(T_CO0)
    R_WC = R_CO0.T
    t_WC = -R_CO0.T @ t_CO0

    T_WC = np.eye(4, dtype=np.float64)
    T_WC[:3, :3] = R_WC
    T_WC[:3, 3] = t_WC

    # T_WO = T_WC @ T_CO = inv(T_CO0) @ T_CO
    R_WO = R_WC @ R_CO
    t_WO = R_WC @ t_CO + t_WC
    T_WO = np.eye(4, dtype=np.float64)
    T_WO[:3, :3] = R_WO
    T_WO[:3, 3] = t_WO

    dO = _normalize(np.asarray(best_dir_O, dtype=np.float64))
    aW = _normalize(R_WO @ dO)      # desired direction in world
    vW = _normalize(t_WC - t_WO)    # current direction (object origin -> camera) in world

    R_err = rotation_matrix_from_a_to_b(aW, vW)
    rvec_W = so3_log(R_err)

    axis_idx = int(np.argmax(np.abs(rvec_W)))
    return axis_idx, rvec_W, aW, vW, T_WO, T_WC


# ------------------------------ Open3D helpers ------------------------------
def make_arrow(origin, direction, length):
    direction = _normalize(np.asarray(direction, dtype=np.float64))
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.02 * length,
        cone_radius=0.04 * length,
        cylinder_height=0.75 * length,
        cone_height=0.25 * length,
        resolution=20,
        cylinder_split=4,
        cone_split=1,
    )
    z = np.array([0, 0, 1], dtype=np.float64)
    R = rotation_matrix_from_a_to_b(z, direction)
    arrow.rotate(R, center=np.zeros(3, dtype=np.float64))
    arrow.translate(np.asarray(origin, dtype=np.float64))
    return arrow


def maybe_voxel_downsample_for_viz(pcd: o3d.geometry.PointCloud, voxel: float) -> o3d.geometry.PointCloud:
    if voxel is None or voxel <= 0:
        return pcd
    if pcd.is_empty():
        return pcd
    return pcd.voxel_down_sample(float(voxel))


# ------------------------------ Main ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T_CO", default="Tracking/BundleTrack/results/inhand_y/keyframes/poses/00058.txt",
                    help="path to current tracker.T (4x4 txt), object-in-camera")
    ap.add_argument("--T_CO0", default="Tracking/BundleTrack/results/inhand_y/keyframes/poses/00000.txt",
                    help="path to tracker.init_pose (4x4 txt), init object-in-camera")

    ap.add_argument("--pcd", default="/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/inhand_y.ply",
                    help="fused object point cloud (in object frame) for NBV + visualization")

    ap.add_argument("--axis_size", type=float, default=0.12, help="coordinate frame size")
    ap.add_argument("--arrow_len", type=float, default=0.25, help="arrow length (world axis / direction arrows)")
    ap.add_argument("--viz_voxel", type=float, default=0.0, help="optional voxel downsample for visualization only")

    ap.add_argument("--show_score_sphere", action="store_true", help="show NBV score sphere points (pcd_dirs)")
    ap.add_argument("--show_uncertainty", action="store_true", help="show NBV uncertainty band cloud (pcd_unc)")
    ap.add_argument("--nbv_seed", type=int, default=0, help="seed for NBV")

    ap.add_argument("--cam_marker_scale", type=float, default=0.9,
                    help="camera marker size scale relative to axis_size")
    args = ap.parse_args()

    # 1) Load poses
    T_CO = load_T_txt(args.T_CO)
    T_CO0 = load_T_txt(args.T_CO0)

    # 2) Run NBV on fused point cloud
    est = GPISNBVv2()
    nbv = est.estimate(pcd=args.pcd, seed=int(args.nbv_seed), verbose=True)

    best_dir_O = np.array(nbv["best_dir"], dtype=np.float64).reshape(3,)
    print(f"\n[NBV] best_dir_O = {best_dir_O.tolist()}")

    # 3) Pick world axis + get T_WO/T_WC
    axis_idx, rvec_W, aW, vW, T_WO, T_WC = pick_world_axis_and_rvec(T_CO, T_CO0, best_dir_O)
    axis_name = ["x", "y", "z"][axis_idx]

    cam_pos_W = T_WC[:3, 3].copy()

    print("===============================================")
    print(f"[Axis Pick] rotate about WORLD axis: {axis_name}")
    print(f"[rvec_W] = {rvec_W.tolist()}  (rad)")
    print(f"[|rvec_W|] = {float(np.linalg.norm(rvec_W)):.6f} rad")
    print(f"[camera position in W] = {cam_pos_W.tolist()}")
    print(f"[aW (desired dir in world)] = {aW.tolist()}")
    print(f"[vW (current dir in world)] = {vW.tolist()}")
    print("===============================================")

    # 4) Build visualization geometries
    geoms = []

    # World frame (W) at origin
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(args.axis_size))
    geoms.append(world_frame)

    # Current object frame O(t) in world
    # obj_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(args.axis_size) * 0.9)
    # obj_frame.transform(T_WO)
    # geoms.append(obj_frame)

    # Camera frame in world (at cam_pos_W with orientation R_WC)
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(args.axis_size) * float(args.cam_marker_scale))
    cam_frame.transform(T_WC)
    geoms.append(cam_frame)

    # Camera position sphere in world
    cam_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.06 * float(args.axis_size), resolution=20)
    cam_sphere.translate(cam_pos_W.astype(np.float64))
    cam_sphere.paint_uniform_color([1.0, 0.2, 1.0])  # magenta
    geoms.append(cam_sphere)

    # Object point cloud (NBV uses object frame). Transform to world using T_WO.
    if est.pcd is None:
        pcd = o3d.io.read_point_cloud(args.pcd)
        if pcd.is_empty():
            raise RuntimeError(f"Point cloud is empty: {args.pcd}")
        pcd_vis = pcd
    else:
        pcd_vis = copy.deepcopy(est.pcd)

    pcd_vis = maybe_voxel_downsample_for_viz(pcd_vis, args.viz_voxel)
    pcd_vis.transform(T_WO)
    geoms.append(pcd_vis)

    # Suggested WORLD axis arrow (yellow) at origin
    axis_dirs_W = [np.array([1, 0, 0], dtype=np.float64),
                   np.array([0, 1, 0], dtype=np.float64),
                   np.array([0, 0, 1], dtype=np.float64)]
    world_axis_arrow = make_arrow(origin=np.zeros(3), direction=axis_dirs_W[axis_idx], length=float(args.arrow_len))
    world_axis_arrow.paint_uniform_color([1.0, 1.0, 0.0])  # yellow
    geoms.append(world_axis_arrow)

    # Direction arrows in WORLD (from origin)
    a_arrow = make_arrow(origin=np.zeros(3), direction=aW, length=float(args.arrow_len) * 0.9)
    a_arrow.paint_uniform_color([1.0, 0.5, 0.0])  # orange
    geoms.append(a_arrow)

    # v_arrow = make_arrow(origin=np.zeros(3), direction=vW, length=float(args.arrow_len) * 0.9)
    # v_arrow.paint_uniform_color([0.2, 1.0, 1.0])  # cyan
    # geoms.append(v_arrow)

    # NBV arrow + NBV camera sphere (in object frame) -> transform to world
    # if est.arrow is not None:
    #     nbv_arrow = copy.deepcopy(est.arrow)
    #     nbv_arrow.transform(T_WO)
    #     geoms.append(nbv_arrow)

    if est.cam_sphere is not None:
        nbv_cam_sphere = copy.deepcopy(est.cam_sphere)
        nbv_cam_sphere.transform(T_WO)
        geoms.append(nbv_cam_sphere)

    # Optional: score sphere points / uncertainty points
    if args.show_score_sphere and (est.pcd_dirs is not None):
        pcd_dirs = copy.deepcopy(est.pcd_dirs)
        pcd_dirs = maybe_voxel_downsample_for_viz(pcd_dirs, args.viz_voxel)
        pcd_dirs.transform(T_WO)
        geoms.append(pcd_dirs)

    if args.show_uncertainty and (est.pcd_unc is not None):
        pcd_unc = copy.deepcopy(est.pcd_unc)
        pcd_unc = maybe_voxel_downsample_for_viz(pcd_unc, args.viz_voxel)
        pcd_unc.transform(T_WO)
        geoms.append(pcd_unc)

    # 5) Show
    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"W(world=object@t0) + O(t) + Camera(W) + PCD + NBV + rotate WORLD {axis_name}",
        width=1280,
        height=800
    )


if __name__ == "__main__":
    main()
