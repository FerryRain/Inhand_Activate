"""
@FileName：compute_tau_time.py
@Description：
@Author：Ferry
@Time：2026 1/22/26 11:51 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@FileName：compute_metric_nksr_vs_gt.py
@Description：
    Batch evaluation with STRICT reference alignment (same alignment pipeline as your strictref script),
    but ONLY evaluates: NKSR mesh (result_dir/mesh_nksr/*.ply) vs GT mesh (gt_root/mesh/GT_mesh.stl).

    Alignment (STRICT):
      manual scale -> auto-unit(mm<->m) -> auto-voxel -> PCA(24-way) ->
      multiscale ICP (point-to-point coarse, point-to-plane + Tukey loss fine)
      score by Precision@tau (pick_tau_mm) on eval downsampled clouds.

    Alignment source:
      --align_source {pcd, pcd_online} (default: pcd/offline)

    Output:
      out_dir/summary_nksr_vs_gt.csv (default out_dir: result_dir/eval_strictref_nksr_vs_gt)

@Author：Ferry (modified by ChatGPT)
@Time：2026-01-22
"""

import os
import re
import csv
import glob
import time
import argparse
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import open3d as o3d


# ============================================================
# Basic IO
# ============================================================
def read_point_cloud(path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise RuntimeError(f"Empty point cloud: {path}")
    return pcd


def read_triangle_mesh(path: str) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise RuntimeError(f"Mesh has no triangles: {path}")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    return mesh


def remove_non_finite(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    if pcd.is_empty():
        return pcd
    ret = pcd.remove_non_finite_points()
    if isinstance(ret, tuple):
        return ret[0]
    return pcd


def preprocess_pcd(pcd: o3d.geometry.PointCloud, voxel: float) -> o3d.geometry.PointCloud:
    pcd = remove_non_finite(pcd)
    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(float(voxel))
    return pcd


def apply_scale(geom, s: float):
    if s == 1.0:
        return
    if isinstance(geom, (o3d.geometry.PointCloud, o3d.geometry.TriangleMesh)):
        geom.scale(float(s), center=(0, 0, 0))


# ============================================================
# Auto unit (mm<->m) and auto voxel
# ============================================================
def pcd_diag_and_center(pcd: o3d.geometry.PointCloud):
    aabb = pcd.get_axis_aligned_bounding_box()
    ext = np.asarray(aabb.get_extent(), dtype=np.float64)
    diag = float(np.linalg.norm(ext))
    cen = np.asarray(aabb.get_center(), dtype=np.float64)
    return ext, diag, cen


def maybe_auto_unit_to_meters(recon_pcd, gt_pcd, enable=True):
    """
    Returns (s_rec, s_gt) to scale recon/gt into meters, if needed.
    Same heuristic as your strictref script.
    """
    if not enable:
        return 1.0, 1.0
    _, dr, _ = pcd_diag_and_center(recon_pcd)
    _, dg, _ = pcd_diag_and_center(gt_pcd)
    if dr <= 0 or dg <= 0:
        return 1.0, 1.0
    if dr > 5.0 and dg > 5.0:
        return 0.001, 0.001
    big = max(dr, dg)
    small = min(dr, dg)
    ratio = big / (small + 1e-12)
    if ratio > 200.0:
        if dr > dg:
            return 0.001, 1.0
        else:
            return 1.0, 0.001
    return 1.0, 1.0


def median_nn_distance(pcd: o3d.geometry.PointCloud, sample_n=5000) -> float:
    pts = np.asarray(pcd.points)
    if pts.shape[0] < 10:
        return 0.0
    n = min(int(sample_n), pts.shape[0])
    idx = np.random.choice(pts.shape[0], n, replace=False)
    q = pts[idx]
    kdt = o3d.geometry.KDTreeFlann(pcd)
    dists = []
    for i in range(q.shape[0]):
        _, _, ds = kdt.search_knn_vector_3d(q[i], 2)
        if len(ds) >= 2:
            dists.append(np.sqrt(ds[1]))
    if len(dists) == 0:
        return 0.0
    return float(np.median(np.asarray(dists)))


def auto_voxel_from_recon(recon_pcd: o3d.geometry.PointCloud, enable=True, fallback=0.003) -> float:
    if not enable:
        return float(fallback)
    tmp = preprocess_pcd(o3d.geometry.PointCloud(recon_pcd), voxel=0.0)
    mnn = median_nn_distance(tmp, sample_n=5000)
    if mnn <= 0:
        return float(fallback)
    vx = max(5.0 * mnn, 1e-4)
    return float(vx)


# ============================================================
# PCA init (24-way)
# ============================================================
def make_pca_frame(pcd: o3d.geometry.PointCloud):
    pts = np.asarray(pcd.points)
    c = pts.mean(axis=0)
    X = pts - c[None, :]
    C = (X.T @ X) / max(1, X.shape[0])
    w, V = np.linalg.eigh(C)
    V = V[:, np.argsort(w)[::-1]]
    if np.linalg.det(V) < 0:
        V[:, 2] *= -1
    return c, V


def all_right_handed_axis_mats(V):
    mats = []
    perms = [
        (0, 1, 2), (0, 2, 1),
        (1, 0, 2), (1, 2, 0),
        (2, 0, 1), (2, 1, 0),
    ]
    signs = [
        (1, 1, 1),
        (1, -1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
    ]
    for p in perms:
        P = V[:, p]
        for s in signs:
            R = P @ np.diag(s)
            if np.linalg.det(R) > 0:
                mats.append(R)
    return mats  # 24


def nn_mean(src_pcd: o3d.geometry.PointCloud, tgt_pcd: o3d.geometry.PointCloud, sample_n=6000) -> float:
    src_pts = np.asarray(src_pcd.points)
    if src_pts.shape[0] == 0:
        return 1e9
    n = min(int(sample_n), src_pts.shape[0])
    idx = np.random.choice(src_pts.shape[0], n, replace=False)
    q = src_pts[idx]
    kdt = o3d.geometry.KDTreeFlann(tgt_pcd)
    ds = []
    for i in range(q.shape[0]):
        _, ind, d2 = kdt.search_knn_vector_3d(q[i], 1)
        if len(ind) == 1 and len(d2) == 1:
            ds.append(np.sqrt(d2[0]))
    if len(ds) == 0:
        return 1e9
    return float(np.mean(ds))


def pca_init_transform(src: o3d.geometry.PointCloud, tgt: o3d.geometry.PointCloud):
    cs, Vs = make_pca_frame(src)
    ct, Vt = make_pca_frame(tgt)
    candidates = all_right_handed_axis_mats(Vs)
    best_T = np.eye(4)
    best_err = 1e18
    Rt = Vt
    for Rsrc in candidates:
        R = Rt @ Rsrc.T
        t = ct - R @ cs
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        src_tmp = o3d.geometry.PointCloud(src)
        src_tmp.transform(T)
        err = nn_mean(src_tmp, tgt, sample_n=6000)
        if err < best_err:
            best_err = err
            best_T = T
    return best_T, best_err


# ============================================================
# ICP multiscale (point-to-plane + Tukey)
# ============================================================
def estimate_normals_for_icp(pcd: o3d.geometry.PointCloud, radius: float, max_nn=30):
    if pcd.is_empty():
        return pcd
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius), max_nn=int(max_nn)))
    pcd.normalize_normals()
    return pcd


def icp_multiscale(src_full, tgt_full, init_T, voxel_base, iters=(80, 120, 160), verbose=False):
    voxels = [4.0 * voxel_base, 2.0 * voxel_base, 1.0 * voxel_base]
    Ts = init_T.copy()

    iters = tuple(iters)
    levels = min(len(voxels), len(iters))

    for lvl in range(levels):
        vx = voxels[lvl]
        it = iters[lvl]

        src = preprocess_pcd(o3d.geometry.PointCloud(src_full), vx)
        tgt = preprocess_pcd(o3d.geometry.PointCloud(tgt_full), vx)

        max_corr = 6.0 * vx if lvl == 0 else (4.0 * vx if lvl == 1 else 2.5 * vx)

        if lvl == 0:
            est = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        else:
            nr = max(3.0 * vx, 1e-3)
            estimate_normals_for_icp(src, nr)
            estimate_normals_for_icp(tgt, nr)
            try:
                loss = o3d.pipelines.registration.TukeyLoss(k=max_corr)
                est = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
            except Exception:
                est = o3d.pipelines.registration.TransformationEstimationPointToPlane()

        reg = o3d.pipelines.registration.registration_icp(
            src, tgt,
            max_correspondence_distance=float(max_corr),
            init=Ts,
            estimation_method=est,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(it))
        )
        Ts = reg.transformation

        if verbose:
            print(f"[ICP lvl{lvl}] voxel={vx:.6f} max_corr={max_corr:.6f} fitness={reg.fitness:.4f} rmse={reg.inlier_rmse:.6f}")

    return Ts


# ============================================================
# Mesh distance helpers
# ============================================================
def make_scene(mesh_legacy: o3d.geometry.TriangleMesh):
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_t)
    return scene


def point_to_mesh_distance(scene: o3d.t.geometry.RaycastingScene, points_xyz: np.ndarray) -> np.ndarray:
    pts = o3d.core.Tensor(points_xyz.astype(np.float32))
    d = scene.compute_distance(pts)
    return d.numpy().astype(np.float64)


def sample_mesh(mesh: o3d.geometry.TriangleMesh, n: int, method: str = "uniform") -> o3d.geometry.PointCloud:
    if method == "poisson":
        return mesh.sample_points_poisson_disk(number_of_points=int(n))
    return mesh.sample_points_uniformly(number_of_points=int(n))


# ============================================================
# Metrics (only F-score needed)
# ============================================================
def fscore_from_dists(d_src_to_dst: np.ndarray, d_dst_to_src: np.ndarray, tau: float):
    P = float(np.mean(d_src_to_dst < tau)) if len(d_src_to_dst) > 0 else 0.0
    R = float(np.mean(d_dst_to_src < tau)) if len(d_dst_to_src) > 0 else 0.0
    F = 0.0 if (P + R) == 0 else float(2.0 * P * R / (P + R))
    return P, R, F


def eval_mesh_vs_gt_fscore_only(
    rec_mesh: o3d.geometry.TriangleMesh,
    scene_gt: o3d.t.geometry.RaycastingScene,
    gt_samples_xyz: np.ndarray,
    mesh_samples: int,
    sample_method: str,
    taus_m: List[float],
) -> Dict[str, Any]:
    # rec -> gt
    p_rec = sample_mesh(rec_mesh, mesh_samples, sample_method)
    rec_xyz = np.asarray(p_rec.points, dtype=np.float64)
    d_rec_to_gt = point_to_mesh_distance(scene_gt, rec_xyz)

    # gt -> rec
    scene_rec = make_scene(rec_mesh)
    d_gt_to_rec = point_to_mesh_distance(scene_rec, gt_samples_xyz)

    out = {"fscore": {}}
    for tau in taus_m:
        _, _, F = fscore_from_dists(d_rec_to_gt, d_gt_to_rec, tau)
        out["fscore"][tau] = F
    return out


def precision_at_tau(rec_pcd: o3d.geometry.PointCloud, gt_pcd: o3d.geometry.PointCloud, tau_m: float) -> float:
    d = np.asarray(rec_pcd.compute_point_cloud_distance(gt_pcd), dtype=np.float64)
    if d.size == 0:
        return 0.0
    return float(np.mean(d < tau_m))


# ============================================================
# Indexing recon files
# ============================================================
def parse_id_from_name(path: str) -> Optional[int]:
    name = os.path.basename(path)
    m = re.search(r"recon_0_(\d+)", name)
    if not m:
        return None
    return int(m.group(1))


def build_recon_index(
    result_dir: str,
    pcd_online_subdir: str,
    pcd_offline_subdir: str,
    mesh_nksr_subdir: str,
) -> Dict[int, Dict[str, str]]:
    pcd_online_dir = os.path.join(result_dir, pcd_online_subdir)
    pcd_offline_dir = os.path.join(result_dir, pcd_offline_subdir)
    mesh_nksr_dir = os.path.join(result_dir, mesh_nksr_subdir)

    by_id: Dict[int, Dict[str, str]] = {}

    if os.path.isdir(pcd_online_dir):
        for p in sorted(glob.glob(os.path.join(pcd_online_dir, "*.ply"))):
            fid = parse_id_from_name(p)
            if fid is None:
                continue
            by_id.setdefault(fid, {})["pcd_online"] = p

    if os.path.isdir(pcd_offline_dir):
        for p in sorted(glob.glob(os.path.join(pcd_offline_dir, "*.ply"))):
            fid = parse_id_from_name(p)
            if fid is None:
                continue
            by_id.setdefault(fid, {})["pcd"] = p

    if os.path.isdir(mesh_nksr_dir):
        for p in sorted(glob.glob(os.path.join(mesh_nksr_dir, "*.ply"))):
            fid = parse_id_from_name(p)
            if fid is None:
                continue
            by_id.setdefault(fid, {})["mesh_nksr"] = p

    return by_id


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def parse_id_list(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    out: List[int] = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


# ============================================================
# CSV flatten
# ============================================================
def flatten_nksr_gt_fscore(report: Optional[Dict[str, Any]], taus_m: List[float], prefix: str = "nksr_vs_gt") -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    if report is None:
        for tau in taus_m:
            row[f"{prefix}_fscore@{tau*1000.0:.0f}mm"] = np.nan
        return row
    for tau in taus_m:
        row[f"{prefix}_fscore@{tau*1000.0:.0f}mm"] = float(report["fscore"][tau])
    return row


def write_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        raise RuntimeError("No rows to write.")
    keys_set = set()
    for r in rows:
        keys_set.update(r.keys())
    keys = sorted(keys_set)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ============================================================
# STRICT reference alignment on ONE candidate
# ============================================================
def strict_align_one_candidate(
    rec_pcd_path: str,
    gt_pcd_path: str,
    gt_mesh_path: str,
    rec_scale: float,
    gt_scale: float,
    auto_unit: bool,
    auto_voxel: bool,
    voxel_fallback: float,
    icp_iters: Tuple[int, int, int],
    eval_voxel: float,
    pick_tau_m: float,
    verbose_icp: bool = False,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "T": 4x4 (maps recon->gt),
        "s_rec": float,
        "s_gt": float,
        "score": float (Precision@pick_tau),
        "voxel": float,
        "time_align_s": float,
      }
    """
    gt_pcd = read_point_cloud(gt_pcd_path)
    gt_mesh = read_triangle_mesh(gt_mesh_path)  # only for consistent scaling
    rec_pcd = read_point_cloud(rec_pcd_path)

    # manual scales first
    apply_scale(gt_pcd, gt_scale)
    apply_scale(gt_mesh, gt_scale)
    apply_scale(rec_pcd, rec_scale)

    # strict auto unit
    s_rec, s_gt = maybe_auto_unit_to_meters(rec_pcd, gt_pcd, enable=auto_unit)
    if s_rec != 1.0:
        apply_scale(rec_pcd, s_rec)
    if s_gt != 1.0:
        apply_scale(gt_pcd, s_gt)
        apply_scale(gt_mesh, s_gt)

    # strict auto voxel
    voxel = auto_voxel_from_recon(rec_pcd, enable=True, fallback=voxel_fallback) if auto_voxel else float(voxel_fallback)

    # alignment uses downsampled PCDs
    src_ds = preprocess_pcd(o3d.geometry.PointCloud(rec_pcd), voxel)
    tgt_ds = preprocess_pcd(o3d.geometry.PointCloud(gt_pcd), voxel)

    t0 = time.time()
    T0, _ = pca_init_transform(src_ds, tgt_ds)
    T = icp_multiscale(rec_pcd, gt_pcd, T0, voxel_base=voxel, iters=icp_iters, verbose=verbose_icp)
    t_align = time.time() - t0

    rec_aligned = o3d.geometry.PointCloud(rec_pcd)
    rec_aligned.transform(T)

    # score by Precision@pick_tau on eval downsampled clouds
    if eval_voxel and eval_voxel > 0:
        rec_eval = preprocess_pcd(rec_aligned, eval_voxel)
        gt_eval = preprocess_pcd(o3d.geometry.PointCloud(gt_pcd), eval_voxel)
    else:
        rec_eval = rec_aligned
        gt_eval = gt_pcd

    score = precision_at_tau(rec_eval, gt_eval, pick_tau_m)

    return {"T": T, "s_rec": s_rec, "s_gt": s_gt, "score": score, "voxel": voxel, "time_align_s": t_align}


# ============================================================
# GT cache (for per_frame mode)
# ============================================================
def build_gt_cache_entry(
    gt_mesh_base: o3d.geometry.TriangleMesh,
    s_gt: float,
    mesh_metric_samples: int,
    mesh_sample_method: str,
) -> Dict[str, Any]:
    gt_mesh = o3d.geometry.TriangleMesh(gt_mesh_base)
    apply_scale(gt_mesh, s_gt)

    scene_gt = make_scene(gt_mesh)
    gt_samples = sample_mesh(gt_mesh, mesh_metric_samples, mesh_sample_method)
    gt_samples_xyz = np.asarray(gt_samples.points, dtype=np.float64)

    return {
        "gt_mesh": gt_mesh,
        "scene_gt": scene_gt,
        "gt_samples_xyz": gt_samples_xyz,
    }


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--result_dir", default="/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/result/offline_tracking/cube_obj_01/003_ICP",
                    help="dir containing pcd_online/, pcd/, mesh_nksr/")
    ap.add_argument("--gt_root", default="/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/GT_data/cube_obj_01/GT",
                    help="GT folder containing mesh/GT_mesh.stl and ply/GT_normal.ply")
    ap.add_argument("--fps", type=float, default=20, help="time_s = frame_id / fps")

    ap.add_argument("--pcd_online_subdir", type=str, default="pcd_online")
    ap.add_argument("--pcd_offline_subdir", type=str, default="pcd")
    ap.add_argument("--mesh_nksr_subdir", type=str, default="mesh_nksr")

    ap.add_argument("--align_source", choices=["pcd", "pcd_online"], default="pcd",
                    help="Which PCD folder to use for STRICT alignment (T_ref). Default: offline pcd/")

    # default: 0..10mm
    ap.add_argument("--thresholds_mm", type=str, default="0,1,2,3,4,5,6,7,8,9,10",
                    help="F-score thresholds (mm), e.g. '1,2,5,10'")

    ap.add_argument("--pick_tau_mm", type=float, default=5.0,
                    help="ref selection tau (mm) using Precision@tau")

    ap.add_argument("--rec_scale", type=float, default=1.0)
    ap.add_argument("--gt_scale", type=float, default=1.0)

    ap.add_argument("--no_auto_unit", action="store_true")
    ap.add_argument("--no_auto_voxel", default=True)
    ap.add_argument("--voxel", type=float, default=0.003, help="fallback voxel if auto_voxel disabled")

    # eval_voxel: used ONLY for Precision@tau scoring when picking T_ref
    ap.add_argument("--eval_voxel", type=float, default=0.005,
                    help="downsample voxel for Precision@tau scoring (0 disables)")

    # STRICT ICP iters: 3-level by default
    ap.add_argument("--icp_iter0", type=int, default=300)
    ap.add_argument("--icp_iter1", type=int, default=300)
    ap.add_argument("--icp_iter2", type=int, default=300)
    ap.add_argument("--verbose_icp", action="store_true")

    ap.add_argument("--align_mode", choices=["fixed_id", "search_best", "per_frame"], default="search_best",
                    help="fixed_id/search_best = strict ref then fixed T for all; per_frame = strict per-frame (slow)")
    ap.add_argument("--align_id", type=int, default=-1, help="fixed_id reference frame id; -1 means last")
    ap.add_argument("--search_ids", type=str, default="100,200,300,400,600,800,1000,-1",
                    help="search_best explicit ids, e.g. '100,200,-1' (-1 means last). If empty, uses --search_k uniform.")
    ap.add_argument("--search_k", type=int, default=7, help="search_best fallback candidate count (uniform + last)")

    ap.add_argument("--mesh_metric_samples", type=int, default=20000)
    ap.add_argument("--mesh_sample_method", choices=["uniform", "poisson"], default="uniform")

    ap.add_argument("--out_dir", type=str, default="./",
                    help="default: result_dir/eval_strictref_nksr_vs_gt")

    args = ap.parse_args()

    result_dir = os.path.expanduser(args.result_dir)
    out_dir = os.path.expanduser(args.out_dir) if args.out_dir else os.path.join(result_dir, "eval_strictref_nksr_vs_gt")
    ensure_dir(out_dir)

    # thresholds
    taus_m = [float(x.strip()) / 1000.0 for x in args.thresholds_mm.split(",") if x.strip()]
    if not taus_m:
        raise RuntimeError("Empty thresholds_mm.")
    pick_tau_m = float(args.pick_tau_mm) / 1000.0

    # GT paths
    gt_pcd_path = os.path.join(os.path.expanduser(args.gt_root), "ply", "GT_normal.ply")
    gt_mesh_path = os.path.join(os.path.expanduser(args.gt_root), "mesh", "GT_mesh.stl")
    if not os.path.isfile(gt_pcd_path):
        raise RuntimeError(f"Missing GT point cloud: {gt_pcd_path}")
    if not os.path.isfile(gt_mesh_path):
        raise RuntimeError(f"Missing GT mesh: {gt_mesh_path}")

    # index frames
    index = build_recon_index(
        result_dir=result_dir,
        pcd_online_subdir=args.pcd_online_subdir,
        pcd_offline_subdir=args.pcd_offline_subdir,
        mesh_nksr_subdir=args.mesh_nksr_subdir,
    )
    ids = sorted(index.keys())
    if not ids:
        raise RuntimeError("No frames found (cannot parse recon_0_XXXXXX from filenames).")

    icp_iters = (int(args.icp_iter0), int(args.icp_iter1), int(args.icp_iter2))
    align_key = "pcd" if args.align_source == "pcd" else "pcd_online"

    # ---------------- Choose candidate ids ----------------
    if args.align_mode == "fixed_id":
        ref_id = ids[-1] if args.align_id < 0 else args.align_id
        cand_ids = [ref_id]
    elif args.align_mode == "search_best":
        explicit = parse_id_list(args.search_ids)
        if explicit:
            cand_ids = []
            for v in explicit:
                cand_ids.append(ids[-1] if v < 0 else v)
            seen = set()
            cand_ids = [x for x in cand_ids if (x not in seen and not seen.add(x))]
        else:
            K = min(max(1, args.search_k), len(ids))
            idxs = np.linspace(0, len(ids) - 1, K, dtype=int).tolist()
            cand_ids = [ids[i] for i in idxs]
            if ids[-1] not in cand_ids:
                cand_ids.append(ids[-1])
    else:
        cand_ids = []

    # ---------------- Find strict ref (T_ref, best_s_rec, best_s_gt) ----------------
    T_ref = None
    best_s_rec = 1.0
    best_s_gt = 1.0

    if args.align_mode in ["fixed_id", "search_best"]:
        best_score = -1.0
        best_id = None

        for cid in cand_ids:
            if cid not in index:
                print(f"[REF_CAND][SKIP] id={cid} not found.")
                continue

            rec_align_path = index[cid].get(align_key, "")
            if not rec_align_path or (not os.path.isfile(rec_align_path)):
                print(f"[REF_CAND][SKIP] id={cid:06d} missing align_source={align_key}.")
                continue

            info = strict_align_one_candidate(
                rec_pcd_path=rec_align_path,
                gt_pcd_path=gt_pcd_path,
                gt_mesh_path=gt_mesh_path,
                rec_scale=args.rec_scale,
                gt_scale=args.gt_scale,
                auto_unit=(not args.no_auto_unit),
                auto_voxel=(not args.no_auto_voxel),
                voxel_fallback=float(args.voxel),
                icp_iters=icp_iters,
                eval_voxel=float(args.eval_voxel),
                pick_tau_m=pick_tau_m,
                verbose_icp=args.verbose_icp,
            )

            print(
                f"[REF_CAND] id={cid:06d}  src={align_key}  P@{args.pick_tau_mm:.2f}mm={info['score']:.4f}  "
                f"voxel={info['voxel']:.6f}  s_rec={info['s_rec']} s_gt={info['s_gt']}  "
                f"align_time={info['time_align_s']:.2f}s"
            )

            if info["score"] > best_score:
                best_score = info["score"]
                best_id = cid
                T_ref = info["T"]
                best_s_rec = info["s_rec"]
                best_s_gt = info["s_gt"]

        if T_ref is None:
            raise RuntimeError("Failed to compute T_ref (no valid candidates with align_source).")
        print(f"[ALIGN_REF] mode={args.align_mode}  align_source={align_key}  ref_id={best_id:06d}  best_P@{args.pick_tau_mm:.2f}mm={best_score:.4f}")

    # ---------------- Load GT mesh base once (only manual gt_scale) ----------------
    gt_mesh_base = read_triangle_mesh(gt_mesh_path)
    apply_scale(gt_mesh_base, args.gt_scale)

    # Fixed-mode precompute (best_s_gt)
    if args.align_mode in ["fixed_id", "search_best"]:
        gt_fixed = build_gt_cache_entry(
            gt_mesh_base=gt_mesh_base,
            s_gt=best_s_gt,
            mesh_metric_samples=args.mesh_metric_samples,
            mesh_sample_method=args.mesh_sample_method,
        )
        scene_gt_fixed = gt_fixed["scene_gt"]
        gt_samples_xyz_fixed = gt_fixed["gt_samples_xyz"]
    else:
        scene_gt_fixed = None
        gt_samples_xyz_fixed = None

    # per_frame cache by s_gt
    gt_cache: Dict[float, Dict[str, Any]] = {}

    # ---------------- Evaluate all frames ----------------
    rows: List[Dict[str, Any]] = []
    missing_nksr = 0
    ok_nksr = 0
    missing_align_source = 0

    for i, fid in enumerate(ids):
        time_s = float(fid) / float(args.fps)
        row: Dict[str, Any] = {"frame_id": fid, "time_s": time_s}

        # ---- get T and scales ----
        if args.align_mode == "per_frame":
            rec_align_path = index[fid].get(align_key, "")
            if not rec_align_path or (not os.path.isfile(rec_align_path)):
                T = None
                s_rec_frame = np.nan
                s_gt_frame = np.nan
                scene_gt_use = None
                gt_samples_xyz_use = None
                missing_align_source += 1
            else:
                info = strict_align_one_candidate(
                    rec_pcd_path=rec_align_path,
                    gt_pcd_path=gt_pcd_path,
                    gt_mesh_path=gt_mesh_path,
                    rec_scale=args.rec_scale,
                    gt_scale=args.gt_scale,
                    auto_unit=(not args.no_auto_unit),
                    auto_voxel=(not args.no_auto_voxel),
                    voxel_fallback=float(args.voxel),
                    icp_iters=icp_iters,
                    eval_voxel=float(args.eval_voxel),
                    pick_tau_m=pick_tau_m,
                    verbose_icp=False,
                )
                T = info["T"]
                s_rec_frame = float(info["s_rec"])
                s_gt_frame = float(info["s_gt"])

                if s_gt_frame not in gt_cache:
                    gt_cache[s_gt_frame] = build_gt_cache_entry(
                        gt_mesh_base=gt_mesh_base,
                        s_gt=s_gt_frame,
                        mesh_metric_samples=args.mesh_metric_samples,
                        mesh_sample_method=args.mesh_sample_method,
                    )
                gt_entry = gt_cache[s_gt_frame]
                scene_gt_use = gt_entry["scene_gt"]
                gt_samples_xyz_use = gt_entry["gt_samples_xyz"]
        else:
            T = T_ref
            s_rec_frame = float(best_s_rec)
            scene_gt_use = scene_gt_fixed
            gt_samples_xyz_use = gt_samples_xyz_fixed

        # ---- if no valid alignment ----
        if T is None or scene_gt_use is None or gt_samples_xyz_use is None or (not np.isfinite(s_rec_frame)):
            row.update(flatten_nksr_gt_fscore(None, taus_m, prefix="nksr_vs_gt"))
            rows.append(row)
            continue

        # ---- NKSR vs GT ----
        nksr_path = index[fid].get("mesh_nksr", "")
        rep = None
        if nksr_path and os.path.isfile(nksr_path):
            mesh_nksr = read_triangle_mesh(nksr_path)
            apply_scale(mesh_nksr, args.rec_scale)
            apply_scale(mesh_nksr, s_rec_frame)
            mesh_nksr.transform(T)

            rep = eval_mesh_vs_gt_fscore_only(
                rec_mesh=mesh_nksr,
                scene_gt=scene_gt_use,
                gt_samples_xyz=gt_samples_xyz_use,
                mesh_samples=args.mesh_metric_samples,
                sample_method=args.mesh_sample_method,
                taus_m=taus_m,
            )
            ok_nksr += 1
        else:
            missing_nksr += 1

        row["nksr_path"] = nksr_path if nksr_path else ""
        row.update(flatten_nksr_gt_fscore(rep, taus_m, prefix="nksr_vs_gt"))
        rows.append(row)

        # periodic logging
        if (i % 10) == 0 or (i == len(ids) - 1):
            f5 = row.get("nksr_vs_gt_fscore@5mm", np.nan)
            f10 = row.get("nksr_vs_gt_fscore@10mm", np.nan)
            print(f"[{i+1:04d}/{len(ids):04d}] id={fid:06d} t={time_s:.2f}s  NKSR-vs-GT F@5mm={f5:.4f}  F@10mm={f10:.4f}")

    csv_path = os.path.join(out_dir, "summary_nksr_vs_gt.csv")
    write_csv(csv_path, rows)

    print(f"\n[SAVED] {csv_path}")
    print(f"[INFO] NKSR mesh available frames: {ok_nksr} / {len(ids)} (missing {missing_nksr})")
    if args.align_mode == "per_frame":
        print(f"[INFO] Missing align_source frames: {missing_align_source}")
    print("[Done]")


if __name__ == "__main__":
    main()
