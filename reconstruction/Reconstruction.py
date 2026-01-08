"""
@FileName：Reconstruction.py
@Description：
    Fused reconstruction from BundleTrack keyframes with robust cleanup.

    Defaults are set to:
    --voxel 0.002
    --mask_drop_boundary_px 4
    --post_clean
    --voxel_count_min 8
    --dbscan_min_points 50
@Author：Ferry
@Time：2026 1/7/26 6:14 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""


import os
import glob
import argparse
import numpy as np
import cv2
import open3d as o3d


# ---------------- IO ----------------
def load_K_txt(path: str) -> np.ndarray:
    K = np.loadtxt(path).astype(np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {K.shape} from {path}")
    return K


def load_T_txt(path: str) -> np.ndarray:
    T = np.loadtxt(path).astype(np.float64)
    if T.shape == (3, 4):
        T = np.vstack([T, [0, 0, 0, 1]])
    if T.shape != (4, 4):
        raise ValueError(f"Pose must be 4x4 (or 3x4), got {T.shape} from {path}")
    return T


def read_color_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


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
    if ext in [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".exr"]:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"Cannot read depth: {path}")
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        return depth
    raise ValueError(f"Unsupported depth format: {path}")


def find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


# ---------------- Mask preprocess ----------------
def erode_mask(mask_u8_255: np.ndarray, k: int, iters: int) -> np.ndarray:
    if k <= 1 or iters <= 0:
        return mask_u8_255
    ker = np.ones((k, k), np.uint8)
    return cv2.erode(mask_u8_255, ker, iterations=iters)


def clean_mask(mask_u8_255: np.ndarray,
               open_k: int,
               close_k: int,
               keep_largest: bool) -> np.ndarray:
    m = (mask_u8_255 > 0).astype(np.uint8)

    if open_k > 1:
        ker = np.ones((open_k, open_k), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker, iterations=1)

    if close_k > 1:
        ker = np.ones((close_k, close_k), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker, iterations=1)

    if keep_largest:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if num > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_id = 1 + int(np.argmax(areas))
            m = (labels == largest_id).astype(np.uint8)

    return (m * 255).astype(np.uint8)


def mask_boundary_band(mask_u8_255: np.ndarray, width_px: int):
    w = int(width_px)
    if w <= 0:
        return None
    m = (mask_u8_255 > 0).astype(np.uint8)
    k = 2 * w + 1
    ker = np.ones((k, k), np.uint8)
    dil = cv2.dilate(m, ker, iterations=1)
    ero = cv2.erode(m, ker, iterations=1)
    return (dil > 0) & (ero == 0)


def ablate_mask_boundary(mask_u8_255: np.ndarray, width_px: int) -> np.ndarray:
    band = mask_boundary_band(mask_u8_255, width_px)
    if band is None:
        return mask_u8_255
    out = mask_u8_255.copy()
    out[band] = 0
    return out


# ---------------- Depth preprocess ----------------
def depth_range_filter(depth: np.ndarray, depth_scale: float, zmin_m: float, zmax_m: float) -> np.ndarray:
    if zmin_m <= 0 and zmax_m <= 0:
        return depth

    out = depth.copy()
    if out.dtype == np.uint16:
        zmin_raw = int(max(0, zmin_m * depth_scale)) if zmin_m > 0 else 0
        zmax_raw = int(zmax_m * depth_scale) if zmax_m > 0 else np.iinfo(np.uint16).max
        bad = (out < zmin_raw) | (out > zmax_raw)
        out[bad] = 0
        return out

    zmin = zmin_m if zmin_m > 0 else -np.inf
    zmax = zmax_m if zmax_m > 0 else np.inf
    bad = (out < zmin) | (out > zmax)
    out[bad] = 0.0
    return out


def denoise_depth_median(depth: np.ndarray, median_k: int) -> np.ndarray:
    if median_k <= 1:
        return depth
    mk = int(median_k)
    if mk % 2 == 0:
        mk += 1
    return cv2.medianBlur(depth, mk)


def depth_median_consistency_filter(depth: np.ndarray,
                                   mask_u8_255: np.ndarray,
                                   depth_scale: float,
                                   median_k: int,
                                   jump_mm: float,
                                   jump_ratio: float,
                                   only_on_boundary_band: bool,
                                   boundary_band_width_px: int) -> np.ndarray:
    if median_k <= 1:
        return depth

    mk = int(median_k)
    if mk % 2 == 0:
        mk += 1

    out = depth.copy()
    m = (mask_u8_255 > 0)

    if only_on_boundary_band and boundary_band_width_px > 0:
        band = mask_boundary_band(mask_u8_255, boundary_band_width_px)
        region = (m & band) if band is not None else m
    else:
        region = m

    if not np.any(region):
        return out

    if out.dtype == np.uint16:
        med = cv2.medianBlur(out, mk).astype(np.uint16)
        abs_thr_raw = float(jump_mm) * (float(depth_scale) / 1000.0)
        diff = cv2.absdiff(out, med).astype(np.float32)
        thr = np.maximum(abs_thr_raw, float(jump_ratio) * med.astype(np.float32))
        bad = region & (out > 0) & (med > 0) & (diff > thr)
        out[bad] = 0
        return out

    d = out.astype(np.float32)
    med = cv2.medianBlur(d, mk).astype(np.float32)
    abs_thr_m = float(jump_mm) / 1000.0
    diff = np.abs(d - med)
    thr = np.maximum(abs_thr_m, float(jump_ratio) * med)
    bad = region & (d > 0) & (med > 0) & (diff > thr)
    d[bad] = 0.0
    return d.astype(out.dtype)


# ---------------- Open3D helpers ----------------
def build_intrinsic_from_K(rgb_path: str, K: np.ndarray):
    img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {rgb_path}")
    H, W = img.shape[:2]
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    return o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)


def make_masked_rgbd(rgb: np.ndarray,
                     depth: np.ndarray,
                     mask_u8_255: np.ndarray,
                     depth_scale: float,
                     depth_trunc: float):
    depth_masked = depth.copy()
    depth_masked[mask_u8_255 == 0] = 0

    o3d_color = o3d.geometry.Image(rgb.astype(np.uint8))
    if depth_masked.dtype == np.uint16:
        o3d_depth = o3d.geometry.Image(depth_masked)
    else:
        o3d_depth = o3d.geometry.Image(depth_masked.astype(np.float32))

    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color,
        o3d_depth,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False
    )


def safe_remove_non_finite(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    if pcd.is_empty():
        return pcd
    ret = pcd.remove_non_finite_points()
    if isinstance(ret, tuple):
        return ret[0]
    return pcd


def voxel_count_filter(pcd: o3d.geometry.PointCloud, voxel_size: float, min_count: int) -> o3d.geometry.PointCloud:
    if pcd.is_empty() or voxel_size <= 0 or min_count <= 1:
        return pcd
    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        return pcd
    ijk = np.floor(pts / float(voxel_size)).astype(np.int64)
    _, inv, counts = np.unique(ijk, axis=0, return_inverse=True, return_counts=True)
    keep = counts[inv] >= int(min_count)
    ind = np.where(keep)[0]
    return pcd.select_by_index(ind)


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


def make_xyz_only(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Create a point cloud that contains ONLY xyz (no colors/normals) for ASCII PLY."""
    xyz = o3d.geometry.PointCloud()
    xyz.points = o3d.utility.Vector3dVector(np.asarray(pcd.points).astype(np.float64))
    return xyz


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--debug_dir", type=str,
                    default="/home/ferry/data/Code2/Research/Inhand_Activate/Active",
                    help="debug_dir that contains keyframes/")
    ap.add_argument("--K_path", type=str,
                    default="/home/ferry/data/Code2/Research/Inhand_Activate/BundleTrack/results/cam_K_A.txt",
                    help="3x3 intrinsic matrix txt")

    # core (defaults set to your preferred params)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--voxel", type=float, default=0.002)
    ap.add_argument("--depth_scale", type=float, default=-1.0, help="auto if <0: uint16->1000, float->1")
    ap.add_argument("--depth_trunc", type=float, default=2.0)

    # mask
    ap.add_argument("--mask_erode_k", type=int, default=3)
    ap.add_argument("--mask_erode_iter", type=int, default=1)
    ap.add_argument("--mask_open_k", type=int, default=0)
    ap.add_argument("--mask_close_k", type=int, default=0)
    ap.add_argument("--mask_keep_largest", action="store_true", default=True)
    ap.add_argument("--mask_drop_boundary_px", type=int, default=4)

    # depth
    ap.add_argument("--zmin", type=float, default=0.0)
    ap.add_argument("--zmax", type=float, default=0.0)
    ap.add_argument("--depth_median_k", type=int, default=5)

    ap.add_argument("--depth_consistency_k", type=int, default=5)
    ap.add_argument("--depth_jump_mm", type=float, default=50.0)
    ap.add_argument("--depth_jump_ratio", type=float, default=0.04)
    ap.add_argument("--consistency_only_on_boundary", action="store_true", default=False)

    # post clean (enabled by default)
    ap.add_argument("--post_clean", action="store_true", default=True)
    ap.add_argument("--voxel_count_min", type=int, default=8)
    ap.add_argument("--voxel_count_size", type=float, default=-1.0, help="<0 -> 2*voxel")

    ap.add_argument("--post_ror_radius", type=float, default=-1.0, help="<0 -> 4*voxel; 0 disables")
    ap.add_argument("--post_ror_min_points", type=int, default=20)

    ap.add_argument("--dbscan_eps", type=float, default=-1.0, help="<0 -> 5*voxel")
    ap.add_argument("--dbscan_min_points", type=int, default=50)

    # outputs (NEW)
    ap.add_argument("--save_ply_color", type=str, default="./reconstruction_clean_color_5s.ply",
                    help="colored point cloud output (keeps RGB)")
    ap.add_argument("--save_ply_xyz", type=str, default="./reconstruction_xyz_down_ascii_5s.ply",
                    help="xyz-only ascii ply output (only x y z)")
    ap.add_argument("--xyz_down_voxel", type=float, default=0.004,
                    help="voxel size for xyz-only downsampled cloud")

    ap.add_argument("--no_vis", action="store_true", default=False)

    args = ap.parse_args()

    base = os.path.join(args.debug_dir, "keyframes")
    rgb_dir = os.path.join(base, "rgb_full")
    dep_dir = os.path.join(base, "depth")
    msk_dir = os.path.join(base, "mask")
    pos_dir = os.path.join(base, "poses")

    if not os.path.isdir(base):
        raise RuntimeError(f"Not found: {base}")

    rgb_list = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")) + glob.glob(os.path.join(rgb_dir, "*.png")))
    if len(rgb_list) == 0:
        raise RuntimeError(f"No rgb found in {rgb_dir}")

    def fid_from_rgb(p):
        return os.path.splitext(os.path.basename(p))[0]

    fids = [fid_from_rgb(p) for p in rgb_list][::max(1, args.stride)]

    K = load_K_txt(args.K_path)
    intrinsic = build_intrinsic_from_K(rgb_list[0], K)

    fused_obj = o3d.geometry.PointCloud()
    guessed_depth_scale = None

    for i, fid in enumerate(fids):
        rgb_path = find_first_existing([os.path.join(rgb_dir, fid + ".jpg"),
                                        os.path.join(rgb_dir, fid + ".png")])
        dep_path = find_first_existing([os.path.join(dep_dir, fid + ".png"),
                                        os.path.join(dep_dir, fid + ".exr")])
        msk_path = find_first_existing([os.path.join(msk_dir, fid + ".png"),
                                        os.path.join(msk_dir, fid + ".jpg")])
        pose_path = os.path.join(pos_dir, fid + ".txt")

        if rgb_path is None or dep_path is None or msk_path is None or (not os.path.exists(pose_path)):
            continue

        # BundleTrack pose: object_in_camera (T_ob_in_cam)
        T_ob_in_cam = load_T_txt(pose_path)
        T_cam_in_ob = np.linalg.inv(T_ob_in_cam)

        rgb = read_color_rgb(rgb_path)
        depth = read_depth_any(dep_path)
        mask = read_mask_u8_255(msk_path)

        # depth_scale guess
        if args.depth_scale < 0:
            if guessed_depth_scale is None:
                guessed_depth_scale = 1000.0 if depth.dtype == np.uint16 else 1.0
            depth_scale = guessed_depth_scale
        else:
            depth_scale = float(args.depth_scale)

        # mask preprocess
        mask = erode_mask(mask, args.mask_erode_k, args.mask_erode_iter)
        if args.mask_open_k > 1 or args.mask_close_k > 1 or args.mask_keep_largest:
            mask = clean_mask(mask, args.mask_open_k, args.mask_close_k, args.mask_keep_largest)
        if args.mask_drop_boundary_px > 0:
            mask = ablate_mask_boundary(mask, args.mask_drop_boundary_px)

        # depth preprocess
        depth = depth_range_filter(depth, depth_scale=depth_scale, zmin_m=args.zmin, zmax_m=args.zmax)
        depth = denoise_depth_median(depth, args.depth_median_k)

        if args.depth_consistency_k > 1:
            depth = depth_median_consistency_filter(
                depth=depth,
                mask_u8_255=mask,
                depth_scale=depth_scale,
                median_k=args.depth_consistency_k,
                jump_mm=args.depth_jump_mm,
                jump_ratio=args.depth_jump_ratio,
                only_on_boundary_band=bool(args.consistency_only_on_boundary),
                boundary_band_width_px=args.mask_drop_boundary_px
            )

        rgbd = make_masked_rgbd(rgb, depth, mask, depth_scale=depth_scale, depth_trunc=args.depth_trunc)

        pcd_cam = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        pcd_cam = safe_remove_non_finite(pcd_cam)

        if args.voxel > 0:
            pcd_cam = pcd_cam.voxel_down_sample(args.voxel)

        pcd_obj = o3d.geometry.PointCloud(pcd_cam)
        pcd_obj.transform(T_cam_in_ob)  # cam -> obj
        fused_obj += pcd_obj

        if i % 30 == 0:
            print(f"[{i}/{len(fids)}] fused points = {len(fused_obj.points)}")

    if fused_obj.is_empty():
        raise RuntimeError("No valid frames processed / fused point cloud is empty.")

    # final downsample (for colored main cloud)
    if args.voxel > 0:
        fused_obj = fused_obj.voxel_down_sample(args.voxel)

    # post-clean pipeline (your defaults)
    if args.post_clean:
        vc_size = args.voxel_count_size
        if vc_size < 0:
            vc_size = 2.0 * args.voxel if args.voxel > 0 else 0.01

        before = len(fused_obj.points)
        fused_obj = voxel_count_filter(fused_obj, voxel_size=vc_size, min_count=args.voxel_count_min)
        after = len(fused_obj.points)
        print(f"[post voxel-count] size={vc_size:.4f}, min={args.voxel_count_min}: {before} -> {after}")

        # --- post radius outlier removal (ROR) ---
        pr = args.post_ror_radius
        if pr < 0:
            pr = 4.0 * args.voxel if args.voxel > 0 else 0.02

        if pr > 0:
            before = len(fused_obj.points)
            fused_obj, ind = fused_obj.remove_radius_outlier(
                nb_points=int(args.post_ror_min_points),
                radius=float(pr),
                print_progress=False
            )
            after = len(fused_obj.points)
            print(f"[post ROR] r={pr:.4f}, k={args.post_ror_min_points}: {before} -> {after}")

        eps = args.dbscan_eps
        if eps < 0:
            eps = 5.0 * args.voxel if args.voxel > 0 else 0.02
        before = len(fused_obj.points)
        fused_obj = keep_largest_cluster_dbscan(fused_obj, eps=float(eps), min_points=int(args.dbscan_min_points))
        after = len(fused_obj.points)
        print(f"[post DBSCAN-largest] eps={eps:.4f}, min_pts={args.dbscan_min_points}: {before} -> {after}")

    # ---------------- Save #1: colored cloud ----------------
    o3d.io.write_point_cloud(args.save_ply_color, fused_obj, write_ascii=False)
    print(f"[saved color] {args.save_ply_color}  points={len(fused_obj.points)}  has_colors={fused_obj.has_colors()}")

    # ---------------- Save #2: xyz-only ASCII cloud (downsampled) ----------------
    xyz_src = fused_obj
    if args.xyz_down_voxel and args.xyz_down_voxel > 0:
        xyz_src = xyz_src.voxel_down_sample(float(args.xyz_down_voxel))

    xyz_only = make_xyz_only(xyz_src)  # ensures header only has x y z
    o3d.io.write_point_cloud(args.save_ply_xyz, xyz_only, write_ascii=True)
    print(f"[saved xyz ASCII] {args.save_ply_xyz}  points={len(xyz_only.points)}  (down_voxel={args.xyz_down_voxel})")

    if not args.no_vis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([axis, fused_obj], width=1280, height=720)


if __name__ == "__main__":
    main()
