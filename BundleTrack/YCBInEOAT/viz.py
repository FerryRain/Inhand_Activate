#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import numpy as np
import cv2
import open3d as o3d
from collections import deque


# ---------------- IO helpers ----------------
def load_K_txt(path: str) -> np.ndarray:
    K = np.loadtxt(path).astype(np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {K.shape} from {path}")
    return K


def load_T_txt(path: str) -> np.ndarray:
    T = np.loadtxt(path).astype(np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Pose must be 4x4, got {T.shape} from {path}")
    return T


def read_color_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def read_mask_u8_255(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Cannot read mask: {path}")
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
    if m.max() <= 1:
        m = (m * 255).astype(np.uint8)
    return m


def read_depth_any(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"]:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"Cannot read depth: {path}")
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        return depth
    elif ext == ".exr":
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"Cannot read EXR depth: {path} (OpenCV may not support EXR)")
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        return depth
    else:
        raise ValueError(f"Unsupported depth format: {path}")


def find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


# ---------------- Preprocess helpers ----------------
def erode_mask(mask_u8_255: np.ndarray, k: int, iters: int) -> np.ndarray:
    if k <= 1 or iters <= 0:
        return mask_u8_255
    kernel = np.ones((k, k), np.uint8)
    return cv2.erode(mask_u8_255, kernel, iterations=iters)


def clean_mask(mask_u8_255: np.ndarray,
               open_k: int,
               close_k: int,
               min_area: int,
               keep_largest: bool) -> np.ndarray:
    """
    mask 去噪：形态学开/闭 + 去除小连通域（可选仅保留最大连通域）
    """
    m = (mask_u8_255 > 0).astype(np.uint8)

    if open_k > 1:
        ker = np.ones((open_k, open_k), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker, iterations=1)

    if close_k > 1:
        ker = np.ones((close_k, close_k), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker, iterations=1)

    if min_area > 0 or keep_largest:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if num <= 1:
            return (m * 255).astype(np.uint8)

        areas = stats[1:, cv2.CC_STAT_AREA]  # exclude background
        keep = np.zeros(num, dtype=bool)
        keep[0] = False

        if keep_largest:
            largest_id = 1 + int(np.argmax(areas))
            keep[largest_id] = True
        else:
            # keep those with area >= min_area
            for cid in range(1, num):
                if stats[cid, cv2.CC_STAT_AREA] >= int(min_area):
                    keep[cid] = True

        m2 = keep[labels].astype(np.uint8)
        return (m2 * 255).astype(np.uint8)

    return (m * 255).astype(np.uint8)


def depth_range_filter(depth: np.ndarray, depth_scale: float, zmin_m: float, zmax_m: float) -> np.ndarray:
    """
    Keep depth only in [zmin_m, zmax_m] (meters after scaling).
    """
    if zmin_m <= 0 and zmax_m <= 0:
        return depth  # disabled

    out = depth.copy()

    if depth.dtype == np.uint16:
        zmin_raw = int(max(0, zmin_m * depth_scale)) if zmin_m > 0 else 0
        zmax_raw = int(zmax_m * depth_scale) if zmax_m > 0 else np.iinfo(np.uint16).max
        invalid = (out < zmin_raw) | (out > zmax_raw)
        out[invalid] = 0
        return out
    else:
        zmin = zmin_m if zmin_m > 0 else -np.inf
        zmax = zmax_m if zmax_m > 0 else np.inf
        invalid = (out < zmin) | (out > zmax)
        out[invalid] = 0.0
        return out


def denoise_depth(depth: np.ndarray,
                  depth_scale: float,
                  median_k: int,
                  bilateral_d: int,
                  bilateral_sigma_m: float,
                  bilateral_sigma_space: float) -> np.ndarray:
    """
    depth 去噪：
    - medianBlur: 去掉椒盐/飞点
    - bilateralFilter: 边缘保持平滑（对 uint16 先转 meters(float)）
    bilateral_sigma_m: 以“米”为单位的深度 sigma（更直观）
    """
    out = depth.copy()

    if median_k and median_k > 1:
        # OpenCV medianBlur supports 8U/16U/32F
        out = cv2.medianBlur(out, int(median_k))

    if bilateral_d and bilateral_d > 0 and bilateral_sigma_m and bilateral_sigma_m > 0:
        # bilateralFilter supports 8U/32F; so convert to float meters
        if out.dtype == np.uint16:
            d_m = out.astype(np.float32) / float(depth_scale)
            valid = d_m > 0
            d_m_f = cv2.bilateralFilter(
                d_m, d=int(bilateral_d),
                sigmaColor=float(bilateral_sigma_m),
                sigmaSpace=float(bilateral_sigma_space)
            )
            d_m_f[~valid] = 0.0
            out = np.clip(d_m_f * float(depth_scale), 0, 65535).astype(np.uint16)
        else:
            d_m = out.astype(np.float32)
            valid = d_m > 0
            d_m_f = cv2.bilateralFilter(
                d_m, d=int(bilateral_d),
                sigmaColor=float(bilateral_sigma_m),
                sigmaSpace=float(bilateral_sigma_space)
            )
            d_m_f[~valid] = 0.0
            out = d_m_f.astype(out.dtype)

    return out


# ---------------- Point cloud helpers ----------------
def build_intrinsic_from_K(rgb_path: str, K: np.ndarray):
    img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {rgb_path}")
    H, W = img.shape[:2]
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    return intrinsic, (W, H), (fx, fy, cx, cy)


def make_masked_rgbd(rgb: np.ndarray,
                     depth: np.ndarray,
                     mask_u8_255: np.ndarray,
                     depth_scale: float,
                     depth_trunc: float):
    if rgb.shape[:2] != depth.shape[:2] or rgb.shape[:2] != mask_u8_255.shape[:2]:
        raise ValueError(f"Shape mismatch: rgb={rgb.shape}, depth={depth.shape}, mask={mask_u8_255.shape}")

    depth_masked = depth.copy()
    depth_masked[mask_u8_255 == 0] = 0

    o3d_color = o3d.geometry.Image(rgb.astype(np.uint8))

    if depth_masked.dtype == np.uint16:
        o3d_depth = o3d.geometry.Image(depth_masked)
    else:
        if depth_masked.dtype != np.float32:
            depth_masked = depth_masked.astype(np.float32)
        o3d_depth = o3d.geometry.Image(depth_masked)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color,
        o3d_depth,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False
    )
    return rgbd


def trajectory_lineset(points_xyz: np.ndarray):
    if len(points_xyz) < 2:
        return None
    lines = [[i, i + 1] for i in range(len(points_xyz) - 1)]
    colors = [[1.0, 0.0, 0.0] for _ in lines]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.float64))
    return ls


def copy_pcd(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    return o3d.geometry.PointCloud(pcd)


def safe_remove_non_finite(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    if pcd.is_empty():
        return pcd
    ret = pcd.remove_non_finite_points()
    # open3d 不同版本：可能返回 (pcd, ind) 或 None(原地修改)
    if isinstance(ret, tuple) and len(ret) >= 1:
        return ret[0]
    return pcd


def denoise_pcd(pcd: o3d.geometry.PointCloud,
                sor_nb: int,
                sor_std: float,
                ror_radius: float,
                ror_min_points: int) -> o3d.geometry.PointCloud:
    if pcd.is_empty():
        return pcd
    out = pcd
    if sor_nb and sor_nb > 0:
        _, ind = out.remove_statistical_outlier(nb_neighbors=int(sor_nb),
                                                std_ratio=float(sor_std))
        out = out.select_by_index(ind)
    if ror_radius and ror_radius > 0:
        _, ind = out.remove_radius_outlier(nb_points=int(ror_min_points),
                                           radius=float(ror_radius))
        out = out.select_by_index(ind)
    return out


def keep_largest_cluster_dbscan(pcd: o3d.geometry.PointCloud, eps: float, min_points: int) -> o3d.geometry.PointCloud:
    if pcd.is_empty():
        return pcd
    labels = np.array(pcd.cluster_dbscan(eps=float(eps), min_points=int(min_points), print_progress=False))
    if labels.size == 0:
        return pcd
    valid = labels[labels >= 0]
    if valid.size == 0:
        return pcd
    counts = np.bincount(valid)
    largest = int(np.argmax(counts))
    ind = np.where(labels == largest)[0]
    return pcd.select_by_index(ind)


# ---------------- Pose smoothing (optional) ----------------
def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Return quaternion [w,x,y,z] from 3x3 rotation matrix."""
    R = R.astype(np.float64)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= (np.linalg.norm(q) + 1e-12)
    return q


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q.astype(np.float64)
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    R = np.array([
        [ww + xx - yy - zz, 2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),       ww - xx + yy - zz, 2*(yz - wx)],
        [2*(xz - wy),       2*(yz + wx),       ww - xx - yy + zz],
    ], dtype=np.float64)
    return R


def average_quaternions_wxyz(quats: np.ndarray) -> np.ndarray:
    """
    Quaternion average via eigenvector of sum(q q^T).
    quats: (N,4) [w,x,y,z]
    """
    if len(quats) == 1:
        return quats[0]
    # hemisphere align
    ref = quats[0]
    qs = quats.copy()
    for i in range(len(qs)):
        if np.dot(qs[i], ref) < 0:
            qs[i] = -qs[i]
    A = np.zeros((4, 4), dtype=np.float64)
    for q in qs:
        A += np.outer(q, q)
    A /= float(len(qs))
    eigvals, eigvecs = np.linalg.eigh(A)
    q_avg = eigvecs[:, np.argmax(eigvals)]
    if q_avg[0] < 0:  # keep w positive (optional)
        q_avg = -q_avg
    q_avg /= (np.linalg.norm(q_avg) + 1e-12)
    return q_avg


class PoseSmoother:
    def __init__(self, window: int):
        self.window = int(window)
        self.buf_R = deque(maxlen=self.window)
        self.buf_t = deque(maxlen=self.window)

    def push_and_get(self, T_ob_in_cam: np.ndarray) -> np.ndarray:
        if self.window <= 1:
            return T_ob_in_cam
        R = T_ob_in_cam[:3, :3]
        t = T_ob_in_cam[:3, 3]
        self.buf_R.append(R)
        self.buf_t.append(t)

        # translation mean
        t_avg = np.mean(np.stack(self.buf_t, axis=0), axis=0)

        # rotation average
        quats = np.stack([rotmat_to_quat_wxyz(r) for r in self.buf_R], axis=0)
        q_avg = average_quaternions_wxyz(quats)
        R_avg = quat_wxyz_to_rotmat(q_avg)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_avg
        T[:3, 3] = t_avg
        return T


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug_dir", type=str,
                    default="/home/ferry/data/Code2/Research/Inhand_Activate/BundleTrack/results/inhand_y",
                    help="BundleTrack debug_dir")
    ap.add_argument("--K_path", type=str,
                    default="/home/ferry/data/Code2/Research/Inhand_Activate/BundleTrack/results/cam_K_A.txt",
                    help="3x3 intrinsic matrix txt")

    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--voxel", type=float, default=0.002)
    ap.add_argument("--depth_scale", type=float, default=-1.0,
                    help="1000 for mm(uint16 png), 1 for meters(float exr). If <0, auto guess.")
    ap.add_argument("--depth_trunc", type=float, default=2.0)

    # mask cleanup
    ap.add_argument("--mask_erode_k", type=int, default=3, help="erode mask kernel size (<=1 disables)")
    ap.add_argument("--mask_erode_iter", type=int, default=1, help="erode iterations")
    ap.add_argument("--mask_open_k", type=int, default=0, help="mask MORPH_OPEN kernel (0 disables)")
    ap.add_argument("--mask_close_k", type=int, default=0, help="mask MORPH_CLOSE kernel (0 disables)")
    ap.add_argument("--mask_min_area", type=int, default=0, help="remove mask components with area < this (0 disables)")
    ap.add_argument("--mask_keep_largest", action="store_true", help="only keep largest connected component in mask")

    # depth range filter (meters after scaling)
    ap.add_argument("--zmin", type=float, default=0.0, help="min depth in meters (0 disables)")
    ap.add_argument("--zmax", type=float, default=0.0, help="max depth in meters (0 disables)")

    # depth denoise
    ap.add_argument("--depth_median_k", type=int, default=5, help="medianBlur kernel for depth (0 disables, odd number)")
    ap.add_argument("--depth_bilateral_d", type=int, default=5, help="bilateralFilter d (0 disables)")
    ap.add_argument("--depth_bilateral_sigma_m", type=float, default=0.0, help="bilateral sigmaColor in meters (e.g., 0.01)")
    ap.add_argument("--depth_bilateral_sigma_space", type=float, default=5.0, help="bilateral sigmaSpace in pixels")

    # per-frame pointcloud denoise (recommended if you have flying depth points)
    ap.add_argument("--frame_sor_nb", type=int, default=0, help="per-frame SOR nb_neighbors (0 disables)")
    ap.add_argument("--frame_sor_std", type=float, default=2.0, help="per-frame SOR std_ratio")
    ap.add_argument("--frame_ror_radius", type=float, default=-1.0, help="per-frame ROR radius; <0 auto=2*voxel, 0 disables")
    ap.add_argument("--frame_ror_min_points", type=int, default=8)

    # final denoise
    ap.add_argument("--sor_nb", type=int, default=30)
    ap.add_argument("--sor_std", type=float, default=2.0)
    ap.add_argument("--ror_radius", type=float, default=-1.0, help="<=0 disables; <0 auto=3*voxel")
    ap.add_argument("--ror_min_points", type=int, default=10)

    # keep largest cluster
    ap.add_argument("--keep_largest_cluster", action="store_true", default=True)
    ap.add_argument("--dbscan_eps", type=float, default=-1.0, help="<0 auto=5*voxel")
    ap.add_argument("--dbscan_min_points", type=int, default=5)

    # pose smoothing (helps pose jitter smear)
    ap.add_argument("--pose_smooth_window", type=int, default=0, help=">1 enables causal smoothing over last N poses")

    ap.add_argument("--show_obj_frames", action="store_true")
    ap.add_argument("--obj_frame_every", type=int, default=30)

    ap.add_argument("--save_ply_obj", type=str, default="./reconstruction_3s.ply")
    ap.add_argument("--save_ply_cam", type=str, default="")
    ap.add_argument("--save_traj_npy", type=str, default="")

    args = ap.parse_args()

    base = os.path.join(args.debug_dir, "keyframes")
    rgb_dir = os.path.join(base, "rgb_full")
    dep_dir = os.path.join(base, "depth")
    msk_dir = os.path.join(base, "mask")
    pos_dir = os.path.join(base, "poses")

    if not os.path.isdir(base):
        raise RuntimeError(f"Not found: {base}")

    rgb_list = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")) +
                      glob.glob(os.path.join(rgb_dir, "*.png")))
    if len(rgb_list) == 0:
        raise RuntimeError(f"No rgb found in {rgb_dir}")

    def fid_from_rgb(p):
        return os.path.splitext(os.path.basename(p))[0]

    fids = [fid_from_rgb(p) for p in rgb_list]
    fids = fids[::max(1, args.stride)]

    K = load_K_txt(args.K_path)
    intrinsic, (W, H), (fx, fy, cx, cy) = build_intrinsic_from_K(rgb_list[0], K)
    print(f"[K] fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}, size=({W}x{H})")

    fused_obj = o3d.geometry.PointCloud()
    accum_cam = o3d.geometry.PointCloud()
    traj = []
    obj_frames = []

    guessed_depth_scale = None
    pose_smoother = PoseSmoother(args.pose_smooth_window) if args.pose_smooth_window and args.pose_smooth_window > 1 else None

    for i, fid in enumerate(fids):
        rgb_path = find_first_existing([
            os.path.join(rgb_dir, fid + ".jpg"),
            os.path.join(rgb_dir, fid + ".png"),
        ])
        dep_path = find_first_existing([
            os.path.join(dep_dir, fid + ".png"),
            os.path.join(dep_dir, fid + ".exr"),
        ])
        msk_path = find_first_existing([
            os.path.join(msk_dir, fid + ".png"),
            os.path.join(msk_dir, fid + ".jpg"),
        ])
        pose_path = os.path.join(pos_dir, fid + ".txt")

        if rgb_path is None or dep_path is None or msk_path is None or (not os.path.exists(pose_path)):
            print(f"[skip] {fid} missing files")
            continue

        T_ob_in_cam = load_T_txt(pose_path)
        if pose_smoother is not None:
            T_ob_in_cam = pose_smoother.push_and_get(T_ob_in_cam)

        T_cam_in_ob = np.linalg.inv(T_ob_in_cam)

        traj.append(T_ob_in_cam[:3, 3].copy())

        rgb = read_color_rgb(rgb_path)
        depth = read_depth_any(dep_path)
        mask = read_mask_u8_255(msk_path)

        # depth_scale guess
        if args.depth_scale < 0:
            if guessed_depth_scale is None:
                guessed_depth_scale = 1000.0 if depth.dtype == np.uint16 else 1.0
            depth_scale = guessed_depth_scale
        else:
            depth_scale = args.depth_scale

        # preprocess: mask erosion + mask cleanup
        mask = erode_mask(mask, args.mask_erode_k, args.mask_erode_iter)
        if (args.mask_open_k and args.mask_open_k > 1) or (args.mask_close_k and args.mask_close_k > 1) or args.mask_min_area > 0 or args.mask_keep_largest:
            mask = clean_mask(mask,
                              open_k=args.mask_open_k,
                              close_k=args.mask_close_k,
                              min_area=args.mask_min_area,
                              keep_largest=args.mask_keep_largest)

        # preprocess: depth range + depth denoise
        depth = depth_range_filter(depth, depth_scale=depth_scale, zmin_m=args.zmin, zmax_m=args.zmax)
        if (args.depth_median_k and args.depth_median_k > 1) or (args.depth_bilateral_d and args.depth_bilateral_d > 0):
            depth = denoise_depth(depth,
                                  depth_scale=depth_scale,
                                  median_k=args.depth_median_k,
                                  bilateral_d=args.depth_bilateral_d,
                                  bilateral_sigma_m=args.depth_bilateral_sigma_m,
                                  bilateral_sigma_space=args.depth_bilateral_sigma_space)

        rgbd = make_masked_rgbd(rgb, depth, mask, depth_scale=depth_scale, depth_trunc=args.depth_trunc)

        pcd_cam = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        pcd_cam = safe_remove_non_finite(pcd_cam)

        if args.voxel > 0:
            pcd_cam = pcd_cam.voxel_down_sample(args.voxel)

        # per-frame pointcloud denoise (before fusion)
        frame_ror_radius = args.frame_ror_radius
        if frame_ror_radius < 0:
            frame_ror_radius = 2.0 * args.voxel if args.voxel > 0 else 0.01
        if args.frame_sor_nb and args.frame_sor_nb > 0 or (frame_ror_radius and frame_ror_radius > 0):
            pcd_cam = denoise_pcd(pcd_cam, args.frame_sor_nb, args.frame_sor_std, frame_ror_radius, args.frame_ror_min_points)

        accum_cam += pcd_cam

        pcd_obj = copy_pcd(pcd_cam)
        pcd_obj.transform(T_cam_in_ob)  # cam -> obj
        fused_obj += pcd_obj

        if args.show_obj_frames and (i % max(1, args.obj_frame_every) == 0):
            fr = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            fr.transform(T_ob_in_cam)
            obj_frames.append(fr)

        if i % 20 == 0:
            print(f"[{i}/{len(fids)}] cam_pts={len(accum_cam.points)}  obj_pts={len(fused_obj.points)}  depth_scale={depth_scale}")

    if len(traj) == 0:
        raise RuntimeError("No valid frames processed.")

    # final downsample
    if args.voxel > 0:
        accum_cam = accum_cam.voxel_down_sample(args.voxel)
        fused_obj = fused_obj.voxel_down_sample(args.voxel)

    # final SOR/ROR
    ror_radius = args.ror_radius
    if ror_radius < 0:
        ror_radius = 3.0 * args.voxel if args.voxel > 0 else 0.01

    before = len(fused_obj.points)
    fused_obj = denoise_pcd(fused_obj, args.sor_nb, args.sor_std, ror_radius, args.ror_min_points)
    after = len(fused_obj.points)
    print(f"[final denoise] points: {before} -> {after} (removed {before-after})")

    # keep largest cluster
    if args.keep_largest_cluster:
        eps = args.dbscan_eps
        if eps < 0:
            eps = 5.0 * args.voxel if args.voxel > 0 else 0.01
        before = len(fused_obj.points)
        fused_obj = keep_largest_cluster_dbscan(fused_obj, eps=eps, min_points=args.dbscan_min_points)
        after = len(fused_obj.points)
        print(f"[largest cluster] eps={eps:.4f}, min_points={args.dbscan_min_points}, points: {before} -> {after}")

    traj_np = np.asarray(traj, dtype=np.float64)
    traj_ls = trajectory_lineset(traj_np)

    if args.save_traj_npy:
        np.save(args.save_traj_npy, traj_np)
        print(f"[saved] trajectory -> {args.save_traj_npy}")

    # View: object frame
    obj_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    geoms_obj = [obj_axis, fused_obj]
    print("\n=== View: OBJECT frame (fused cloud after cleanup) ===")
    o3d.visualization.draw_geometries(geoms_obj, width=1280, height=720)

    if args.save_ply_obj:
        o3d.io.write_point_cloud(args.save_ply_obj, fused_obj)
        print(f"[saved] fused object cloud -> {args.save_ply_obj}")


if __name__ == "__main__":
    main()
