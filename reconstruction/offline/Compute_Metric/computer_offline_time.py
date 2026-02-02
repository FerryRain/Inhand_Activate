#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@FileName: computer_offline_time.py
@Description:
    Benchmark OFFLINE pipeline total time:
      1) Back-project masked depth for frames [0 .. end_frame] into object frame (+ RGB colors)
      2) Frame-to-frame ICP refinement (each frame aligned to previous) + fusion => pcd_icp.ply (ASCII, colored, downsample=0.01)
      3) FaCE normal estimation => pcd_icp_face.ply
      4) NKSR reconstruction (python API) => mesh_nksr.ply

    Output:
      out_dir/
        pcd_icp.ply            (ASCII, colored, voxel downsample = 0.01)
        pcd_icp_face.ply
        mesh_nksr.ply
        timing.json
"""

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

try:
    import open3d as o3d
except Exception as e:
    raise RuntimeError("Failed to import open3d. Please install it in your current environment.") from e


# ------------------------- Timing -------------------------

@dataclass
class Timer:
    t0: float = 0.0
    acc: float = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.acc += time.perf_counter() - self.t0


# ------------------------- IO utils -------------------------

def read_matrix_4x4(path: Path) -> np.ndarray:
    txt = path.read_text().strip().split()
    if len(txt) == 16:
        vals = [float(x) for x in txt]
        return np.array(vals, dtype=np.float64).reshape(4, 4)
    arr = np.loadtxt(str(path), dtype=np.float64)
    arr = np.array(arr, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"Pose file {path} is not a 4x4 matrix (got shape {arr.shape}).")
    return arr


def read_intrinsics(
    keyframe_root: Path,
    fx: Optional[float],
    fy: Optional[float],
    cx: Optional[float],
    cy: Optional[float],
    K_path: Optional[Path],
) -> np.ndarray:
    if fx is not None and fy is not None and cx is not None and cy is not None:
        return np.array([[fx, 0.0, cx],
                         [0.0, fy, cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    if K_path is None:
        cand = keyframe_root / "intrinsics.txt"
        if cand.exists():
            K_path = cand

    if K_path is None or not K_path.exists():
        raise FileNotFoundError(
            "Camera intrinsics not provided. Use --fx --fy --cx --cy, or provide --K <path>, "
            "or place intrinsics.txt (3x3) under keyframe_root."
        )

    K = np.loadtxt(str(K_path), dtype=np.float64)
    K = np.array(K, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"Intrinsics file {K_path} must be 3x3, got {K.shape}.")
    return K


def list_frames(depth_dir: Path, mask_dir: Path, pose_dir: Path, rgb_dir: Optional[Path]) -> List[int]:
    def stems_png(p: Path) -> set:
        return {int(x.stem) for x in p.glob("*.png")}

    def stems_rgb(p: Path) -> set:
        # allow jpg/png
        s = set()
        for ext in ("*.jpg", "*.png", "*.jpeg"):
            s |= {int(x.stem) for x in p.glob(ext)}
        return s

    dset = stems_png(depth_dir)
    mset = stems_png(mask_dir)
    pset = {int(x.stem) for x in pose_dir.glob("*.txt")}

    common = dset & mset & pset
    if rgb_dir is not None and rgb_dir.exists():
        common = common & stems_rgb(rgb_dir)

    frames = sorted(list(common))
    if not frames:
        raise FileNotFoundError(
            f"No common frames found under:\n  {depth_dir}\n  {mask_dir}\n  {pose_dir}"
            + (f"\n  {rgb_dir}" if rgb_dir is not None else "")
        )
    return frames


def find_rgb_dir(keyframe_root: Path, rgb_dir_arg: Optional[str]) -> Optional[Path]:
    if rgb_dir_arg is not None:
        p = (keyframe_root / rgb_dir_arg)
        return p if p.exists() else None
    # try common names
    for name in ["rgb_full", "rgb", "color", "images", "rgbd_rgb"]:
        p = keyframe_root / name
        if p.exists():
            return p
    return None


def read_rgb(rgb_dir: Path, fid: int) -> np.ndarray:
    # try jpg then png
    for ext in [".jpg", ".png", ".jpeg"]:
        p = rgb_dir / f"{fid:06d}{ext}"
        if p.exists():
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)  # BGR uint8
            if img is None:
                break
            return img
    raise FileNotFoundError(f"RGB image not found for frame {fid} under {rgb_dir}.")


# ------------------------- Geometry -------------------------

def backproject_masked_depth_to_points_and_colors(
    depth: np.ndarray,
    mask: np.ndarray,
    rgb_bgr: Optional[np.ndarray],
    K: np.ndarray,
    depth_scale: float,
    depth_trunc: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      pts_c:   Nx3 points in camera frame
      colors:  Nx3 float in [0,1] (RGB). If rgb_bgr is None, returns zeros.
    """
    assert depth.ndim == 2 and mask.ndim == 2
    m = mask > 0
    if not np.any(m):
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    d = depth.astype(np.float64)
    if np.issubdtype(depth.dtype, np.integer):
        z_all = d / depth_scale
    else:
        z_all = d

    vv, uu = np.where(m)
    z = z_all[vv, uu]

    keep = (z > 0)
    if depth_trunc > 0:
        keep = keep & (z < depth_trunc)
    if not np.any(keep):
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    vv, uu, z = vv[keep], uu[keep], z[keep]

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    x = (uu.astype(np.float64) - cx) * z / fx
    y = (vv.astype(np.float64) - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)

    if rgb_bgr is None:
        colors = np.zeros((pts.shape[0], 3), dtype=np.float64)
    else:
        if rgb_bgr.shape[0] != depth.shape[0] or rgb_bgr.shape[1] != depth.shape[1]:
            raise ValueError(
                f"RGB and depth resolution mismatch: rgb={rgb_bgr.shape[:2]} depth={depth.shape}"
            )
        bgr = rgb_bgr[vv, uu, :].astype(np.float64) / 255.0
        rgb = bgr[:, ::-1]  # BGR -> RGB
        colors = rgb

    return pts, colors


def make_o3d_pcd(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    if points.shape[0] > 0:
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if colors is not None and colors.shape[0] == points.shape[0]:
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.shape[0] == 0:
        return pts
    R = T[:3, :3]
    t = T[:3, 3]
    return (pts @ R.T) + t[None, :]


def preprocess_for_icp(
    pcd: o3d.geometry.PointCloud,
    voxel: float,
    estimate_normals: bool,
    normal_radius: float,
    normal_max_nn: int,
) -> o3d.geometry.PointCloud:
    q = pcd
    if voxel > 0 and len(q.points) > 0:
        q = q.voxel_down_sample(voxel)
    if estimate_normals and len(q.points) > 0:
        q.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius, max_nn=normal_max_nn
            )
        )
        q.normalize_normals()
    return q


def icp_align_to_prev(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    max_corr: float,
    icp_point_to_plane: bool,
    init: np.ndarray,
) -> np.ndarray:
    if icp_point_to_plane:
        est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        est = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    reg = o3d.pipelines.registration.registration_icp(
        source, target, max_corr, init, est
    )
    return reg.transformation


# ------------------------- External: FaCE -------------------------

def run_face(face_bin: Path, in_ply: Path, out_ply: Path, extra_args: List[str]) -> None:
    cmd = [str(face_bin), str(in_ply), "--o", str(out_ply)] + list(extra_args)
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"FaCE failed (code={p.returncode}): {' '.join(cmd)}")


# ------------------------- External: NKSR (python API) -------------------------

def run_nksr(in_ply_with_normals: Path, out_mesh_ply: Path, device_str: str,
            detail_level: float, mise_iter: int) -> None:
    try:
        import torch
        import nksr
    except Exception as e:
        raise RuntimeError(
            "Failed to import torch/nksr. Ensure you run this script in the NKSR environment."
        ) from e

    pcd = o3d.io.read_point_cloud(str(in_ply_with_normals))
    if len(pcd.points) == 0:
        raise RuntimeError(f"Input point cloud is empty: {in_ply_with_normals}")
    if not pcd.has_normals() or len(pcd.normals) != len(pcd.points):
        raise RuntimeError(
            f"Input point cloud must contain per-point normals (FaCE output expected): {in_ply_with_normals}"
        )

    device = torch.device(device_str)
    xyz = torch.from_numpy(np.asarray(pcd.points)).float().to(device)
    nrm = torch.from_numpy(np.asarray(pcd.normals)).float().to(device)

    reconstructor = nksr.Reconstructor(device)
    field = reconstructor.reconstruct(xyz, nrm, detail_level=float(detail_level))
    mesh = field.extract_dual_mesh(mise_iter=int(mise_iter))

    V = mesh.v.detach().cpu().numpy()
    F = mesh.f.detach().cpu().numpy()

    tri = o3d.geometry.TriangleMesh()
    tri.vertices = o3d.utility.Vector3dVector(V.astype(np.float64))
    tri.triangles = o3d.utility.Vector3iVector(F.astype(np.int32))
    tri.remove_duplicated_vertices()
    tri.remove_degenerate_triangles()
    tri.remove_duplicated_triangles()
    tri.remove_non_manifold_edges()
    tri.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out_mesh_ply), tri, write_ascii=False)


# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--keyframe_root",
        type=str,
        default="/home/ferry/data/Code2/Research/Inhand_Activate/Real_deploy/results/offline_tracking/tetraprism/003/keyframes",
        help="Root folder containing depth/, mask/, poses/ (and optionally intrinsics.txt + rgb folder)",
    )
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Default: <keyframe_root>/benchmark_offline_time")
    ap.add_argument("--end_frame", type=int, default=540,
                    help="Use frames with frame_id <= end_frame.")
    ap.add_argument("--max_frames", type=int, default=None,
                    help="Alternatively, take first N frames after sorting.")
    ap.add_argument("--depth_scale", type=float, default=1000.0,
                    help="Depth scale for integer depth (mm->m: 1000).")
    ap.add_argument("--depth_trunc", type=float, default=2.0,
                    help="Depth truncation in meters. Set <=0 to disable.")
    ap.add_argument("--K", type=str, default="/home/ferry/data/Code2/Research/Inhand_Activate/Active/AzureKinectDK/cam_K.txt",
                    help="Path to 3x3 intrinsics matrix text file.")
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)

    # RGB folder
    ap.add_argument("--rgb_dir", type=str, default=None,
                    help="RGB folder name under keyframe_root (e.g., rgb_full). If not set, auto-detect.")

    # ICP params
    ap.add_argument("--icp_voxel", type=float, default=0.0015)
    ap.add_argument("--icp_max_corr", type=float, default=0.01)
    ap.add_argument("--icp_point_to_plane", action="store_true")
    ap.add_argument("--icp_normal_radius", type=float, default=0.01)
    ap.add_argument("--icp_normal_max_nn", type=int, default=30)

    # Fused PCD post-process (USER REQUEST)
    ap.add_argument("--fuse_voxel", type=float, default=0.01,
                    help="Final fused cloud voxel downsample in meters. Default=0.01 (1 cm).")
    ap.add_argument("--outlier_nb", type=int, default=20)
    ap.add_argument("--outlier_std", type=float, default=2.0)

    # FaCE
    ap.add_argument("--face_bin", type=str,
                    default="~/data/Code2/HW/CG2/PaperSharing/face_2/cmake-build-debug/face")
    ap.add_argument("--face_extra", type=str, default="--h",
                    help="Extra args passed to FaCE (space-separated), e.g. '--h' or ''.")

    # NKSR
    ap.add_argument("--nksr_device", type=str, default="cuda:0")
    ap.add_argument("--nksr_detail_level", type=float, default=0.4)
    ap.add_argument("--nksr_mise_iter", type=int, default=1)

    ap.add_argument("--skip_face", action="store_true")
    ap.add_argument("--skip_nksr", action="store_true")

    args = ap.parse_args()

    keyframe_root = Path(args.keyframe_root)
    depth_dir = keyframe_root / "depth"
    mask_dir = keyframe_root / "mask"
    pose_dir = keyframe_root / "poses"

    rgb_dir = find_rgb_dir(keyframe_root, args.rgb_dir)

    out_dir = Path(args.out_dir) if args.out_dir is not None else (keyframe_root / "benchmark_offline_time")
    out_dir.mkdir(parents=True, exist_ok=True)

    K = read_intrinsics(
        keyframe_root=keyframe_root,
        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
        K_path=Path(args.K) if args.K is not None else None,
    )

    frames = list_frames(depth_dir, mask_dir, pose_dir, rgb_dir)
    frames = [f for f in frames if f <= args.end_frame]
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    if not frames:
        raise ValueError("No frames selected after applying end_frame/max_frames filters.")

    if rgb_dir is None:
        print("[WARN] RGB directory not found. Colors will be set to zeros in the fused point cloud.")
    else:
        print(f"[INFO] Using RGB directory: {rgb_dir}")

    timing: Dict[str, float] = {}

    t_total = Timer()
    with t_total:
        # -------- Stage A: Back-project + ICP + fuse --------
        t_proj = Timer()
        t_icp = Timer()
        t_fuse = Timer()

        fused = o3d.geometry.PointCloud()
        prev_pcd_aligned: Optional[o3d.geometry.PointCloud] = None

        for i, fid in enumerate(frames):
            depth_path = depth_dir / f"{fid:06d}.png"
            mask_path = mask_dir / f"{fid:06d}.png"
            pose_path = pose_dir / f"{fid:06d}.txt"

            with t_proj:
                depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                if depth is None:
                    raise FileNotFoundError(depth_path)
                mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if mask is None:
                    raise FileNotFoundError(mask_path)

                rgb = None
                if rgb_dir is not None:
                    rgb = read_rgb(rgb_dir, fid)

                T_CO = read_matrix_4x4(pose_path)
                T_OC = np.linalg.inv(T_CO)

                pts_c, cols = backproject_masked_depth_to_points_and_colors(
                    depth=depth, mask=mask, rgb_bgr=rgb, K=K,
                    depth_scale=args.depth_scale,
                    depth_trunc=args.depth_trunc,
                )
                pts_o = transform_points(T_OC, pts_c)
                pcd_o = make_o3d_pcd(pts_o, cols)

            # ICP align current to previous aligned frame
            if i == 0 or prev_pcd_aligned is None or len(pcd_o.points) == 0:
                pcd_aligned = pcd_o
            else:
                with t_icp:
                    src = preprocess_for_icp(
                        pcd_o,
                        voxel=args.icp_voxel,
                        estimate_normals=args.icp_point_to_plane,
                        normal_radius=args.icp_normal_radius,
                        normal_max_nn=args.icp_normal_max_nn,
                    )
                    tgt = preprocess_for_icp(
                        prev_pcd_aligned,
                        voxel=args.icp_voxel,
                        estimate_normals=args.icp_point_to_plane,
                        normal_radius=args.icp_normal_radius,
                        normal_max_nn=args.icp_normal_max_nn,
                    )
                    if len(src.points) == 0 or len(tgt.points) == 0:
                        T = np.eye(4)
                    else:
                        T = icp_align_to_prev(
                            source=src,
                            target=tgt,
                            max_corr=args.icp_max_corr,
                            icp_point_to_plane=args.icp_point_to_plane,
                            init=np.eye(4),
                        )

                pcd_aligned = pcd_o
                pcd_aligned.transform(T)

            prev_pcd_aligned = pcd_aligned
            fused += pcd_aligned

        with t_fuse:
            if args.fuse_voxel > 0 and len(fused.points) > 0:
                fused = fused.voxel_down_sample(args.fuse_voxel)

            if len(fused.points) > 0:
                fused, _ = fused.remove_statistical_outlier(
                    nb_neighbors=args.outlier_nb, std_ratio=args.outlier_std
                )

        pcd_icp_path = out_dir / "pcd_icp.ply"
        # USER REQUEST: ASCII + color + downsample 0.01 already applied above
        o3d.io.write_point_cloud(str(pcd_icp_path), fused, write_ascii=True, compressed=False)

        timing["stage_proj_backproject_s"] = t_proj.acc
        timing["stage_icp_align_s"] = t_icp.acc
        timing["stage_fuse_filter_s"] = t_fuse.acc

        # -------- Stage B: FaCE normals --------
        face_out_path = out_dir / "pcd_icp_face.ply"
        t_face = Timer()

        if args.skip_face:
            timing["stage_face_s"] = 0.0
            face_result = pcd_icp_path
        else:
            face_bin = Path(args.face_bin).expanduser()
            if not face_bin.exists():
                raise FileNotFoundError(f"FaCE binary not found: {face_bin}")
            extra = args.face_extra.strip()
            extra_args = extra.split() if len(extra) > 0 else []
            with t_face:
                run_face(face_bin, pcd_icp_path, face_out_path, extra_args)
            timing["stage_face_s"] = t_face.acc
            face_result = face_out_path

        # -------- Stage C: NKSR reconstruction --------
        t_nksr = Timer()
        mesh_out = out_dir / "mesh_nksr.ply"

        if args.skip_nksr:
            timing["stage_nksr_s"] = 0.0
        else:
            with t_nksr:
                run_nksr(
                    in_ply_with_normals=face_result,
                    out_mesh_ply=mesh_out,
                    device_str=args.nksr_device,
                    detail_level=args.nksr_detail_level,
                    mise_iter=args.nksr_mise_iter,
                )
            timing["stage_nksr_s"] = t_nksr.acc

    timing["total_s"] = t_total.acc
    timing["num_frames"] = int(len(frames))
    timing["frame_first"] = int(frames[0])
    timing["frame_last"] = int(frames[-1])
    timing["out_dir"] = str(out_dir)
    timing["rgb_dir"] = str(rgb_dir) if rgb_dir is not None else None
    timing["fuse_voxel_m"] = float(args.fuse_voxel)

    (out_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    print(json.dumps(timing, indent=2))
    print(f"[OK] Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
