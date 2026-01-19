"""
@FileName：time_recon_by_seconds_trueframe.py
@Description：
    CUMULATIVE reconstructions from BundleTrack keyframes.
    Windowing is defined by seconds (window_seconds) and fps, but frame selection
    is done by TRUE frame_id parsed from filenames.

    Example (fps=6, window=5s, include_endpoint=True):
      - end_frame_id = round(5*6)=30  => fuse all saved frames with frame_id <= 30
      - end_frame_id = round(10*6)=60 => fuse all saved frames with frame_id <= 60
      ...

    Folder structure:
      debug_dir/keyframes/
        rgb_full/000000.jpg
        depth/000000.png
        mask/000000.png
        poses/000000.txt

    Output:
      recon_0_000030.ply
      recon_0_000030_xyz_ascii.ply

@Author：Ferry (refactor by ChatGPT)
@Time：2026-01-13
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


# ---------------- Frame-id parsing ----------------
def parse_frame_id(fid: str) -> int:
    s = fid.strip()
    if s.isdigit():
        return int(s)
    digits = "".join([c for c in s if c.isdigit()])
    if digits == "":
        raise ValueError(f"Cannot parse frame id from fid='{fid}'")
    return int(digits)


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
        depth_scale=float(depth_scale),
        depth_trunc=float(depth_trunc),
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
    xyz = o3d.geometry.PointCloud()
    xyz.points = o3d.utility.Vector3dVector(np.asarray(pcd.points).astype(np.float64))
    return xyz


# ---------------- Core: reconstruct for a given list of fids ----------------
def reconstruct_prefix(
    fids_prefix,
    rgb_dir, dep_dir, msk_dir, pos_dir,
    intrinsic,
    args,
):
    fused_obj = o3d.geometry.PointCloud()
    guessed_depth_scale = None
    processed = 0
    skipped = 0

    for fid in fids_prefix:
        rgb_path = find_first_existing([os.path.join(rgb_dir, fid + ".jpg"),
                                        os.path.join(rgb_dir, fid + ".png")])
        dep_path = find_first_existing([os.path.join(dep_dir, fid + ".png"),
                                        os.path.join(dep_dir, fid + ".exr")])
        msk_path = find_first_existing([os.path.join(msk_dir, fid + ".png"),
                                        os.path.join(msk_dir, fid + ".jpg")])
        pose_path = os.path.join(pos_dir, fid + ".txt")

        if rgb_path is None or dep_path is None or msk_path is None or (not os.path.exists(pose_path)):
            skipped += 1
            continue

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
            pcd_cam = pcd_cam.voxel_down_sample(float(args.voxel))

        pcd_obj = o3d.geometry.PointCloud(pcd_cam)
        pcd_obj.transform(T_cam_in_ob)  # cam -> obj
        fused_obj += pcd_obj
        processed += 1

    if fused_obj.is_empty():
        return fused_obj, {"processed": processed, "skipped": skipped, "empty": True}

    # final downsample
    if args.voxel > 0:
        fused_obj = fused_obj.voxel_down_sample(float(args.voxel))

    # post clean
    if args.post_clean:
        vc_size = args.voxel_count_size
        if vc_size < 0:
            vc_size = 2.0 * args.voxel if args.voxel > 0 else 0.01

        fused_obj = voxel_count_filter(fused_obj, voxel_size=float(vc_size), min_count=int(args.voxel_count_min))

        pr = args.post_ror_radius
        if pr < 0:
            pr = 4.0 * args.voxel if args.voxel > 0 else 0.02
        if pr > 0:
            fused_obj, _ = fused_obj.remove_radius_outlier(
                nb_points=int(args.post_ror_min_points),
                radius=float(pr),
                print_progress=False
            )

        eps = args.dbscan_eps
        if eps < 0:
            eps = 5.0 * args.voxel if args.voxel > 0 else 0.02
        fused_obj = keep_largest_cluster_dbscan(fused_obj, eps=float(eps), min_points=int(args.dbscan_min_points))

    return fused_obj, {"processed": processed, "skipped": skipped, "empty": False}


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--debug_dir", type=str,
                    default="/home/ferry/data/Code2/Research/Inhand_Activate/Real_deploy/results/10s_active/tetraprism/003",
                    help="debug_dir that contains keyframes/")
    ap.add_argument("--out_dir", type=str, default="/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/result/cube_obj1/003", help="output directory")

    ap.add_argument("--K_path", type=str,
                    default="/home/ferry/data/Code2/Research/Inhand_Activate/Tracking/BundleTrack/results/cam_K_A.txt",
                    help="3x3 intrinsic matrix txt")

    # windows by seconds -> frames by fps
    ap.add_argument("--fps", type=float, default=5, help="recorded fps (frames per second)")
    ap.add_argument("--window_seconds", type=float, default=1, help="save cumulative recon every N seconds")
    ap.add_argument("--include_endpoint", default=True, action=argparse.BooleanOptionalAction,
                    help="If True: end_frame=round(k*window_seconds*fps). "
                         "If False: end_frame=round(k*window_seconds*fps)-1.")

    # optional single-shot
    ap.add_argument("--end_seconds", type=float, default=-1.0,
                    help="if >=0: only reconstruct 0~round(end_seconds*fps) once, then exit.")
    ap.add_argument("--end_frame_id", type=int, default=-1,
                    help="if >=0: only reconstruct 0~end_frame_id once, then exit. "
                         "If both end_seconds and end_frame_id are set, end_frame_id wins.")

    # core
    ap.add_argument("--stride", type=int, default=1,
                    help="use every N-th SAVED frame within [0..end_frame_id] for speed. "
                         "Window definition still uses fps*seconds.")
    ap.add_argument("--voxel", type=float, default=0.002)
    ap.add_argument("--depth_scale", type=float, default=-1.0, help="auto if <0: uint16->1000, float->1")
    ap.add_argument("--depth_trunc", type=float, default=2.0)

    # mask
    ap.add_argument("--mask_erode_k", type=int, default=3)
    ap.add_argument("--mask_erode_iter", type=int, default=1)
    ap.add_argument("--mask_open_k", type=int, default=0)
    ap.add_argument("--mask_close_k", type=int, default=0)
    ap.add_argument("--mask_keep_largest", default=True, action=argparse.BooleanOptionalAction)
    ap.add_argument("--mask_drop_boundary_px", type=int, default=4)

    # depth
    ap.add_argument("--zmin", type=float, default=0.0)
    ap.add_argument("--zmax", type=float, default=0.0)
    ap.add_argument("--depth_median_k", type=int, default=5)

    ap.add_argument("--depth_consistency_k", type=int, default=5)
    ap.add_argument("--depth_jump_mm", type=float, default=50.0)
    ap.add_argument("--depth_jump_ratio", type=float, default=0.04)
    ap.add_argument("--consistency_only_on_boundary", default=False, action=argparse.BooleanOptionalAction)

    # post clean
    ap.add_argument("--post_clean", default=True, action=argparse.BooleanOptionalAction)
    ap.add_argument("--voxel_count_min", type=int, default=8)
    ap.add_argument("--voxel_count_size", type=float, default=-1.0, help="<0 -> 2*voxel")

    ap.add_argument("--post_ror_radius", type=float, default=-1.0, help="<0 -> 4*voxel; 0 disables")
    ap.add_argument("--post_ror_min_points", type=int, default=20)

    ap.add_argument("--dbscan_eps", type=float, default=-1.0, help="<0 -> 5*voxel")
    ap.add_argument("--dbscan_min_points", type=int, default=100)

    # outputs

    ap.add_argument("--xyz_down_voxel", type=float, default=0.001, help="voxel size for xyz-only downsampled cloud")

    # visualization
    ap.add_argument("--no_vis", default=True, action=argparse.BooleanOptionalAction)
    ap.add_argument("--axis_size", type=float, default=0.1)

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

    # build (frame_id, fid) from filenames and sort by true frame_id
    fid_items = []
    for p in rgb_list:
        fid = fid_from_rgb(p)
        try:
            frame_id = parse_frame_id(fid)
        except Exception:
            continue
        fid_items.append((frame_id, fid))

    if len(fid_items) == 0:
        raise RuntimeError("No valid numeric frame ids parsed from rgb filenames.")

    fid_items.sort(key=lambda x: x[0])
    min_frame_id = fid_items[0][0]
    max_frame_id = fid_items[-1][0]
    has_zero = any(fr == 0 for fr, _ in fid_items)
    args.out_dir = os.path.join(args.out_dir, f"pcd")
    os.makedirs(args.out_dir, exist_ok=True)

    K = load_K_txt(args.K_path)
    intrinsic = build_intrinsic_from_K(rgb_list[0], K)

    fps = float(args.fps)
    if fps <= 0:
        raise ValueError("fps must be > 0")
    stride = max(1, int(args.stride))
    window_seconds = float(args.window_seconds)
    if window_seconds <= 0:
        raise ValueError("window_seconds must be > 0")

    print("====================================================")
    print(f"[INFO] saved frames = {len(fid_items)} (min_id={min_frame_id}, max_id={max_frame_id})")
    if not has_zero:
        print("[WARN] frame_id=0 not found in saved frames. '0~xxx' will effectively start from earliest saved frame.")
    print(f"[INFO] fps={fps:.3f}, window_seconds={window_seconds:.3f}")
    print(f"[INFO] include_endpoint={args.include_endpoint}, stride(on saved frames)={stride}")
    print(f"[INFO] out_dir={args.out_dir}")
    print("====================================================")

    def compute_end_frame_id_from_seconds(sec: float) -> int:
        # match your original expectation: 5s@6fps => 30
        end_f = int(round(sec * fps))
        if not args.include_endpoint:
            end_f -= 1
        return max(0, end_f)

    def run_one(end_frame_id: int, tag: str):
        # clamp to available max to avoid endless empty windows
        end_frame_id = min(int(end_frame_id), max_frame_id)

        prefix_all = [fid for (fr, fid) in fid_items if fr <= end_frame_id]
        if len(prefix_all) == 0:
            print(f"[{tag}] 0~{end_frame_id:06d}: NO SAVED FRAMES, skip")
            return

        # stride only affects how many SAVED frames we use, not the window definition
        prefix = prefix_all[::stride]

        first_fr = parse_frame_id(prefix[0])
        last_fr = parse_frame_id(prefix[-1])
        approx_sec = end_frame_id / fps

        print(f"\n[{tag}] recon 0~{end_frame_id:06d} (~{approx_sec:.2f}s)  "
              f"use N={len(prefix)} / avail N={len(prefix_all)}  saved_range={first_fr:06d}~{last_fr:06d}")

        fused_obj, info = reconstruct_prefix(prefix, rgb_dir, dep_dir, msk_dir, pos_dir, intrinsic, args)

        if fused_obj.is_empty():
            print(f"  -> EMPTY (processed={info['processed']}, skipped={info['skipped']})")
            return

        out_color = os.path.join(args.out_dir, f"recon_0_{end_frame_id:06d}.ply")
        o3d.io.write_point_cloud(out_color, fused_obj, write_ascii=False)
        print(f"  [saved color] {out_color}  points={len(fused_obj.points)}  skipped={info['skipped']}")

        xyz_src = fused_obj
        if args.xyz_down_voxel and args.xyz_down_voxel > 0:
            xyz_src = xyz_src.voxel_down_sample(float(args.xyz_down_voxel))
        xyz_only = make_xyz_only(xyz_src)
        out_xyz = os.path.join(args.out_dir, f"recon_0_{end_frame_id:06d}_xyz_ascii.ply")
        o3d.io.write_point_cloud(out_xyz, xyz_only, write_ascii=True)
        print(f"  [saved xyz ASCII] {out_xyz}  points={len(xyz_only.points)}  (down_voxel={args.xyz_down_voxel})")

        if not args.no_vis:
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(args.axis_size))
            o3d.visualization.draw_geometries(
                [axis, fused_obj],
                window_name=f"{tag}: end_frame={end_frame_id:06d}  points={len(fused_obj.points)}",
                width=1280,
                height=720
            )

    # ---------------- Single mode ----------------
    if args.end_frame_id >= 0 or args.end_seconds >= 0:
        if args.end_frame_id >= 0:
            end_frame_id = int(args.end_frame_id)
            if not args.include_endpoint:
                end_frame_id -= 1
            end_frame_id = max(0, end_frame_id)
        else:
            end_frame_id = compute_end_frame_id_from_seconds(float(args.end_seconds))

        run_one(end_frame_id, tag="SINGLE")
        print("\n[Done] Single cumulative reconstruction finished.")
        return

    # ---------------- Series mode: 0~5s, 0~10s, 0~15s ... ----------------
    k = 1
    while True:
        end_sec = k * window_seconds
        end_frame_id = compute_end_frame_id_from_seconds(end_sec)

        final_flag = False
        if end_frame_id >= max_frame_id:
            end_frame_id = max_frame_id
            final_flag = True

        run_one(end_frame_id, tag=f"CUM{k}")

        if final_flag:
            break
        k += 1

    print("\n[Done] Cumulative reconstructions (seconds->true frame id) finished.")


if __name__ == "__main__":
    main()
