"""
@FileName：Reconstructer.py
@Description：BundleTrack keyframes -> fused object point cloud, class wrapper.
@Author：Ferry (refactor by ChatGPT)
@Time：2026 1/9/26 4:46 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""


import os
import glob
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import cv2
import open3d as o3d


# ---------------- Config ----------------
@dataclass
class ReconstructionConfig:
    # core
    stride: int = 1
    voxel: float = 0.002
    depth_scale: float = -1.0          # auto if <0: uint16->1000, float->1
    depth_trunc: float = 2.0

    # mask
    mask_erode_k: int = 3
    mask_erode_iter: int = 1
    mask_open_k: int = 0
    mask_close_k: int = 0
    mask_keep_largest: bool = True
    mask_drop_boundary_px: int = 4

    # depth
    zmin: float = 0.0
    zmax: float = 0.0
    depth_median_k: int = 5

    depth_consistency_k: int = 5
    depth_jump_mm: float = 50.0
    depth_jump_ratio: float = 0.04
    consistency_only_on_boundary: bool = False

    # post clean
    post_clean: bool = True
    voxel_count_min: int = 8
    voxel_count_size: float = -1.0

    post_ror_radius: float = -1.0
    post_ror_min_points: int = 20

    dbscan_eps: float = -1.0
    dbscan_min_points: int = 50

    # outputs (NOTE: default is NOT saving)
    save_ply_color: Optional[str] = None
    save_ply_xyz: Optional[str] = None
    xyz_down_voxel: float = 0.004


class Reconstructor:
    """
    - self.ply holds the last reconstructed colored point cloud.
    """

    def __init__(
        self,
        debug_dir: str = "/home/ferry/data/Code2/Research/Inhand_Activate/BundleTrack/results/AzureKinectDK",
        K_path: str = "/home/ferry/data/Code2/Research/Inhand_Activate/BundleTrack/results/cam_K_A.txt",
        cfg: Optional[ReconstructionConfig] = None,
    ):
        self.debug_dir = debug_dir
        self.K_path = K_path
        self.cfg = cfg if cfg is not None else ReconstructionConfig()

        self.ply: Optional[o3d.geometry.PointCloud] = None
        self.last_info: Dict[str, Any] = {}

        self._guessed_depth_scale: Optional[float] = None

    def reconstruct(
        self,
        debug_dir: Optional[str] = None,
        K_path: Optional[str] = None,
        save_color: Optional[str] = None,
        save_xyz_ascii: Optional[str] = None,
        visualize: bool = False,
        verbose: bool = True,
    ) -> o3d.geometry.PointCloud:
        dbg = debug_dir if debug_dir is not None else self.debug_dir
        kpath = K_path if K_path is not None else self.K_path
        cfg = self.cfg

        base = os.path.join(dbg, "keyframes")
        rgb_dir = os.path.join(base, "rgb_full")
        dep_dir = os.path.join(base, "depth")
        msk_dir = os.path.join(base, "mask")
        pos_dir = os.path.join(base, "poses")

        if not os.path.isdir(base):
            raise RuntimeError(f"Not found: {base}")

        rgb_list = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")) + glob.glob(os.path.join(rgb_dir, "*.png")))
        if len(rgb_list) == 0:
            raise RuntimeError(f"No rgb found in {rgb_dir}")

        def fid_from_rgb(p: str) -> str:
            return os.path.splitext(os.path.basename(p))[0]

        fids = [fid_from_rgb(p) for p in rgb_list][::max(1, int(cfg.stride))]

        K = self._load_K_txt(kpath)
        intrinsic = self._build_intrinsic_from_K(rgb_list[0], K)

        fused_obj = o3d.geometry.PointCloud()

        self._guessed_depth_scale = None
        processed = 0
        skipped = 0

        print("--------------------------------------------------")
        print("---------------Start Reconstruction---------------")

        for i, fid in enumerate(fids):
            rgb_path = self._find_first_existing([os.path.join(rgb_dir, fid + ".jpg"),
                                                 os.path.join(rgb_dir, fid + ".png")])
            dep_path = self._find_first_existing([os.path.join(dep_dir, fid + ".png"),
                                                 os.path.join(dep_dir, fid + ".exr")])
            msk_path = self._find_first_existing([os.path.join(msk_dir, fid + ".png"),
                                                 os.path.join(msk_dir, fid + ".jpg")])
            pose_path = os.path.join(pos_dir, fid + ".txt")

            if rgb_path is None or dep_path is None or msk_path is None or (not os.path.exists(pose_path)):
                skipped += 1
                continue

            T_ob_in_cam = self._load_T_txt(pose_path)
            T_cam_in_ob = np.linalg.inv(T_ob_in_cam)

            rgb = self._read_color_rgb(rgb_path)
            depth = self._read_depth_any(dep_path)
            mask = self._read_mask_u8_255(msk_path)

            depth_scale = self._resolve_depth_scale(depth, cfg.depth_scale)

            mask = self._erode_mask(mask, cfg.mask_erode_k, cfg.mask_erode_iter)
            if cfg.mask_open_k > 1 or cfg.mask_close_k > 1 or cfg.mask_keep_largest:
                mask = self._clean_mask(mask, cfg.mask_open_k, cfg.mask_close_k, cfg.mask_keep_largest)
            if cfg.mask_drop_boundary_px > 0:
                mask = self._ablate_mask_boundary(mask, cfg.mask_drop_boundary_px)

            depth = self._depth_range_filter(depth, depth_scale=depth_scale, zmin_m=cfg.zmin, zmax_m=cfg.zmax)
            depth = self._denoise_depth_median(depth, cfg.depth_median_k)

            if cfg.depth_consistency_k > 1:
                depth = self._depth_median_consistency_filter(
                    depth=depth,
                    mask_u8_255=mask,
                    depth_scale=depth_scale,
                    median_k=cfg.depth_consistency_k,
                    jump_mm=cfg.depth_jump_mm,
                    jump_ratio=cfg.depth_jump_ratio,
                    only_on_boundary_band=bool(cfg.consistency_only_on_boundary),
                    boundary_band_width_px=cfg.mask_drop_boundary_px
                )

            rgbd = self._make_masked_rgbd(rgb, depth, mask, depth_scale=depth_scale, depth_trunc=cfg.depth_trunc)

            pcd_cam = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
            pcd_cam = self._safe_remove_non_finite(pcd_cam)

            if cfg.voxel > 0:
                pcd_cam = pcd_cam.voxel_down_sample(float(cfg.voxel))

            pcd_obj = o3d.geometry.PointCloud(pcd_cam)
            pcd_obj.transform(T_cam_in_ob)  # cam -> obj
            fused_obj += pcd_obj

            processed += 1

        if fused_obj.is_empty():
            raise RuntimeError("No valid frames processed / fused point cloud is empty.")

        # downsample
        if cfg.voxel > 0:
            fused_obj = fused_obj.voxel_down_sample(float(cfg.voxel))

        # post-clean
        post_stats = {}
        if cfg.post_clean:
            fused_obj, post_stats = self._post_clean(fused_obj, cfg, verbose=verbose)

        # store last result
        self.ply = fused_obj
        self.last_info = {
            "debug_dir": dbg,
            "K_path": kpath,
            "processed_frames": processed,
            "skipped_frames": skipped,
            "points": len(fused_obj.points),
            "has_colors": fused_obj.has_colors(),
            "post_stats": post_stats,
        }


        color_out = save_color if save_color is not None else cfg.save_ply_color
        xyz_out = save_xyz_ascii if save_xyz_ascii is not None else cfg.save_ply_xyz

        if color_out:
            self.save_color(color_out, binary=True, verbose=verbose)

        if xyz_out:
            self.save_xyz_ascii(xyz_out, down_voxel=cfg.xyz_down_voxel, verbose=verbose)

        if visualize:
            self.show()


        print("-----------Reconstruction finished!---------------")
        print("--------------------------------------------------")

        return fused_obj

    def show(
        self,
        ply: Optional[o3d.geometry.PointCloud] = None,
        axis_size: float = 0.1,
        width: int = 1280,
        height: int = 720,
        show_axis: bool = True,
    ) -> None:
        pc = ply if ply is not None else self.ply
        if pc is None or pc.is_empty():
            raise RuntimeError("No point cloud available. Run reconstruct() first or pass ply=...")

        geoms = []
        if show_axis:
            geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(axis_size)))
        geoms.append(pc)
        o3d.visualization.draw_geometries(geoms, width=int(width), height=int(height))

    def save_color(self, path: str, ply: Optional[o3d.geometry.PointCloud] = None, binary: bool = True, verbose: bool = True) -> None:
        pc = ply if ply is not None else self.ply
        if pc is None or pc.is_empty():
            raise RuntimeError("No point cloud to save.")
        o3d.io.write_point_cloud(path, pc, write_ascii=(not binary))
        if verbose:
            print(f"[saved color] {path}  points={len(pc.points)}  has_colors={pc.has_colors()}")

    def save_xyz_ascii(
        self,
        path: str,
        ply: Optional[o3d.geometry.PointCloud] = None,
        down_voxel: Optional[float] = None,
        verbose: bool = True,
    ) -> None:
        pc = ply if ply is not None else self.ply
        if pc is None or pc.is_empty():
            raise RuntimeError("No point cloud to save.")

        src = pc
        dv = self.cfg.xyz_down_voxel if down_voxel is None else float(down_voxel)
        if dv and dv > 0:
            src = src.voxel_down_sample(dv)

        xyz_only = self._make_xyz_only(src)
        o3d.io.write_point_cloud(path, xyz_only, write_ascii=True)
        if verbose:
            print(f"[saved xyz ASCII] {path}  points={len(xyz_only.points)}  (down_voxel={dv})")

    def _post_clean(self, pcd: o3d.geometry.PointCloud, cfg: ReconstructionConfig, verbose: bool) -> Tuple[o3d.geometry.PointCloud, Dict[str, Any]]:
        stats: Dict[str, Any] = {}

        vc_size = cfg.voxel_count_size
        if vc_size < 0:
            vc_size = 2.0 * cfg.voxel if cfg.voxel > 0 else 0.01

        before = len(pcd.points)
        pcd = self._voxel_count_filter(pcd, voxel_size=vc_size, min_count=cfg.voxel_count_min)
        after = len(pcd.points)
        stats["voxel_count"] = {"size": float(vc_size), "min": int(cfg.voxel_count_min), "before": int(before), "after": int(after)}


        pr = cfg.post_ror_radius
        if pr < 0:
            pr = 4.0 * cfg.voxel if cfg.voxel > 0 else 0.02

        if pr > 0:
            before = len(pcd.points)
            pcd, _ = pcd.remove_radius_outlier(
                nb_points=int(cfg.post_ror_min_points),
                radius=float(pr),
                print_progress=False
            )
            after = len(pcd.points)
            stats["ror"] = {"radius": float(pr), "min_points": int(cfg.post_ror_min_points), "before": int(before), "after": int(after)}
        else:
            stats["ror"] = {"disabled": True}

        eps = cfg.dbscan_eps
        if eps < 0:
            eps = 5.0 * cfg.voxel if cfg.voxel > 0 else 0.02

        before = len(pcd.points)
        pcd = self._keep_largest_cluster_dbscan(pcd, eps=float(eps), min_points=int(cfg.dbscan_min_points))
        after = len(pcd.points)
        stats["dbscan"] = {"eps": float(eps), "min_points": int(cfg.dbscan_min_points), "before": int(before), "after": int(after)}

        return pcd, stats


    @staticmethod
    def _load_K_txt(path: str) -> np.ndarray:
        K = np.loadtxt(path).astype(np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"K must be 3x3, got {K.shape} from {path}")
        return K

    @staticmethod
    def _load_T_txt(path: str) -> np.ndarray:
        T = np.loadtxt(path).astype(np.float64)
        if T.shape == (3, 4):
            T = np.vstack([T, [0, 0, 0, 1]])
        if T.shape != (4, 4):
            raise ValueError(f"Pose must be 4x4 (or 3x4), got {T.shape} from {path}")
        return T

    @staticmethod
    def _read_color_rgb(path: str) -> np.ndarray:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Cannot read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _read_mask_u8_255(path: str) -> np.ndarray:
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise RuntimeError(f"Cannot read mask: {path}")
        if m.dtype != np.uint8:
            m = m.astype(np.uint8)
        if m.max() <= 1:
            m = (m * 255).astype(np.uint8)
        return m

    @staticmethod
    def _read_depth_any(path: str) -> np.ndarray:
        ext = os.path.splitext(path)[1].lower()
        if ext in [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".exr"]:
            depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise RuntimeError(f"Cannot read depth: {path}")
            if depth.ndim == 3:
                depth = depth[:, :, 0]
            return depth
        raise ValueError(f"Unsupported depth format: {path}")

    @staticmethod
    def _find_first_existing(paths):
        for p in paths:
            if os.path.exists(p):
                return p
        return None

    @staticmethod
    def _build_intrinsic_from_K(rgb_path: str, K: np.ndarray):
        img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot read image: {rgb_path}")
        H, W = img.shape[:2]
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        return o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

    def _resolve_depth_scale(self, depth: np.ndarray, cfg_depth_scale: float) -> float:
        if cfg_depth_scale >= 0:
            return float(cfg_depth_scale)
        if self._guessed_depth_scale is None:
            self._guessed_depth_scale = 1000.0 if depth.dtype == np.uint16 else 1.0
        return float(self._guessed_depth_scale)

    @staticmethod
    def _erode_mask(mask_u8_255: np.ndarray, k: int, iters: int) -> np.ndarray:
        if k <= 1 or iters <= 0:
            return mask_u8_255
        ker = np.ones((int(k), int(k)), np.uint8)
        return cv2.erode(mask_u8_255, ker, iterations=int(iters))

    @staticmethod
    def _clean_mask(mask_u8_255: np.ndarray, open_k: int, close_k: int, keep_largest: bool) -> np.ndarray:
        m = (mask_u8_255 > 0).astype(np.uint8)

        if open_k and open_k > 1:
            ker = np.ones((int(open_k), int(open_k)), np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker, iterations=1)

        if close_k and close_k > 1:
            ker = np.ones((int(close_k), int(close_k)), np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker, iterations=1)

        if keep_largest:
            num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            if num > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                largest_id = 1 + int(np.argmax(areas))
                m = (labels == largest_id).astype(np.uint8)

        return (m * 255).astype(np.uint8)

    @staticmethod
    def _mask_boundary_band(mask_u8_255: np.ndarray, width_px: int):
        w = int(width_px)
        if w <= 0:
            return None
        m = (mask_u8_255 > 0).astype(np.uint8)
        k = 2 * w + 1
        ker = np.ones((k, k), np.uint8)
        dil = cv2.dilate(m, ker, iterations=1)
        ero = cv2.erode(m, ker, iterations=1)
        return (dil > 0) & (ero == 0)

    @classmethod
    def _ablate_mask_boundary(cls, mask_u8_255: np.ndarray, width_px: int) -> np.ndarray:
        band = cls._mask_boundary_band(mask_u8_255, width_px)
        if band is None:
            return mask_u8_255
        out = mask_u8_255.copy()
        out[band] = 0
        return out

    @staticmethod
    def _depth_range_filter(depth: np.ndarray, depth_scale: float, zmin_m: float, zmax_m: float) -> np.ndarray:
        if (zmin_m <= 0) and (zmax_m <= 0):
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

    @staticmethod
    def _denoise_depth_median(depth: np.ndarray, median_k: int) -> np.ndarray:
        if median_k <= 1:
            return depth
        mk = int(median_k)
        if mk % 2 == 0:
            mk += 1
        return cv2.medianBlur(depth, mk)

    @classmethod
    def _depth_median_consistency_filter(
        cls,
        depth: np.ndarray,
        mask_u8_255: np.ndarray,
        depth_scale: float,
        median_k: int,
        jump_mm: float,
        jump_ratio: float,
        only_on_boundary_band: bool,
        boundary_band_width_px: int
    ) -> np.ndarray:
        if median_k <= 1:
            return depth

        mk = int(median_k)
        if mk % 2 == 0:
            mk += 1

        out = depth.copy()
        m = (mask_u8_255 > 0)

        if only_on_boundary_band and boundary_band_width_px > 0:
            band = cls._mask_boundary_band(mask_u8_255, boundary_band_width_px)
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

    @staticmethod
    def _make_masked_rgbd(rgb: np.ndarray, depth: np.ndarray, mask_u8_255: np.ndarray, depth_scale: float, depth_trunc: float):
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

    @staticmethod
    def _safe_remove_non_finite(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        if pcd.is_empty():
            return pcd
        ret = pcd.remove_non_finite_points()
        if isinstance(ret, tuple):
            return ret[0]
        return pcd

    @staticmethod
    def _voxel_count_filter(pcd: o3d.geometry.PointCloud, voxel_size: float, min_count: int) -> o3d.geometry.PointCloud:
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

    @staticmethod
    def _keep_largest_cluster_dbscan(pcd: o3d.geometry.PointCloud, eps: float, min_points: int) -> o3d.geometry.PointCloud:
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

    @staticmethod
    def _make_xyz_only(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        xyz = o3d.geometry.PointCloud()
        xyz.points = o3d.utility.Vector3dVector(np.asarray(pcd.points).astype(np.float64))
        return xyz


# ---------------- Example ----------------
if __name__ == "__main__":
    recon = Reconstructor()
    recon.reconstruct(visualize=False, verbose=True)  # default: no saving
    recon.show()


    # If you want saving:
    # recon.save_color("./reconstruction_clean_color.ply")
    # recon.save_xyz_ascii("./reconstruction_xyz_ascii.ply")
    # or
    # recon.reconstruct(save_color="./color.ply", save_xyz_ascii="./xyz_ascii.ply")
