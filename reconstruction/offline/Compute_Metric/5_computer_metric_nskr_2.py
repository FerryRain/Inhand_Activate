#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@FileName：computer_metric_nskr.py
@Description：
    Batch evaluation with STRICT reference alignment (same alignment as your old strictref script).

    Behavior (single CSV):
    A) Online PCD-PCD:   recon pcd (result_dir/pcd_online/*.ply) vs GT pcd (gt_root/ply/GT_normal.ply)
    B) Offline PCD-PCD:  recon pcd (result_dir/pcd/*.ply)        vs GT pcd (gt_root/ply/GT_normal.ply)
    C) Mesh-GT:          mesh (result_dir/mesh/*.ply)            vs GT mesh (gt_root/mesh/GT_mesh.stl)
    D) NKSR-Mesh:        nksr mesh (result_dir/mesh_nksr/*.ply)  vs original mesh (result_dir/mesh/*.ply)

    Alignment (STRICT):
      manual scale -> auto-unit(mm<->m) -> auto-voxel -> PCA(24-way) ->
      multiscale ICP (point-to-point coarse, point-to-plane + Tukey loss fine)
      score by Precision@tau (pick_tau_mm) on eval downsampled clouds.

    Alignment source:
      --align_source {pcd, pcd_online} (default: pcd/offline)

    Output:
      out_dir/summary.csv (default: result_dir/eval_strictref)

@Author：Ferry
@Time：2026-01-21
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
    Same strict heuristic as your old script.
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


def sample_mesh(mesh: o3d.geometry.TriangleMesh, n: int, method: str = "poisson") -> o3d.geometry.PointCloud:
    if method == "poisson":
        return mesh.sample_points_poisson_disk(number_of_points=int(n))
    return mesh.sample_points_uniformly(number_of_points=int(n))


# ============================================================
# Metrics
# ============================================================
def fscore_from_dists(d_src_to_dst: np.ndarray, d_dst_to_src: np.ndarray, tau: float):
    P = float(np.mean(d_src_to_dst < tau)) if len(d_src_to_dst) > 0 else 0.0
    R = float(np.mean(d_dst_to_src < tau)) if len(d_dst_to_src) > 0 else 0.0
    F = 0.0 if (P + R) == 0 else float(2.0 * P * R / (P + R))
    return P, R, F


def normal_consistency_simple(rec_pcd: o3d.geometry.PointCloud, gt_pcd: o3d.geometry.PointCloud) -> Dict[str, float]:
    if (not rec_pcd.has_normals()) or (not gt_pcd.has_normals()):
        return {"mean_abs_cos": np.nan, "mean_angle_deg": np.nan}

    rec_xyz = np.asarray(rec_pcd.points, dtype=np.float64)
    rec_n = np.asarray(rec_pcd.normals, dtype=np.float64)
    gt_n = np.asarray(gt_pcd.normals, dtype=np.float64)

    kdt = o3d.geometry.KDTreeFlann(gt_pcd)
    idx = np.zeros((rec_xyz.shape[0],), dtype=np.int64)
    for i in range(rec_xyz.shape[0]):
        _, ind, _ = kdt.search_knn_vector_3d(rec_xyz[i], 1)
        idx[i] = int(ind[0]) if len(ind) == 1 else 0

    n1 = gt_n[idx]
    dot = np.sum(rec_n * n1, axis=1)
    abs_cos = np.clip(np.abs(dot), 0.0, 1.0)
    ang = np.degrees(np.arccos(abs_cos))
    return {
        "mean_abs_cos": float(np.mean(abs_cos)) if abs_cos.size else np.nan,
        "mean_angle_deg": float(np.mean(ang)) if ang.size else np.nan,
    }


def eval_pcd_pcd(rec_pcd: o3d.geometry.PointCloud, gt_pcd: o3d.geometry.PointCloud, taus_m: List[float]) -> Dict[str, Any]:
    d_rg = np.asarray(rec_pcd.compute_point_cloud_distance(gt_pcd), dtype=np.float64)
    d_gr = np.asarray(gt_pcd.compute_point_cloud_distance(rec_pcd), dtype=np.float64)

    acc = float(d_rg.mean()) if d_rg.size else 0.0
    comp = float(d_gr.mean()) if d_gr.size else 0.0
    cd_l1 = acc + comp
    cd_l2 = float((d_rg**2).mean() + (d_gr**2).mean()) if (d_rg.size and d_gr.size) else 0.0

    out = {
        "accuracy_mean": acc,
        "completeness_mean": comp,
        "chamfer_l1": cd_l1,
        "chamfer_l2": cd_l2,
        "fscore": {},
        "normal": normal_consistency_simple(rec_pcd, gt_pcd),
    }
    for tau in taus_m:
        P, R, F = fscore_from_dists(d_rg, d_gr, tau)
        out["fscore"][tau] = {"precision": P, "recall": R, "fscore": F}
    return out


def eval_mesh_mesh_against_fixed_ref(
    rec_mesh: o3d.geometry.TriangleMesh,
    scene_ref: o3d.t.geometry.RaycastingScene,
    ref_samples_xyz: np.ndarray,
    mesh_samples: int,
    sample_method: str,
    taus_m: List[float],
) -> Dict[str, Any]:
    # rec -> ref
    p_rec = sample_mesh(rec_mesh, mesh_samples, sample_method)
    rec_xyz = np.asarray(p_rec.points, dtype=np.float64)
    d_rec_to_ref = point_to_mesh_distance(scene_ref, rec_xyz)

    # ref -> rec
    scene_rec = make_scene(rec_mesh)
    d_ref_to_rec = point_to_mesh_distance(scene_rec, ref_samples_xyz)

    acc = float(np.mean(d_rec_to_ref)) if d_rec_to_ref.size else 0.0
    comp = float(np.mean(d_ref_to_rec)) if d_ref_to_rec.size else 0.0
    cd_l1 = acc + comp
    cd_l2 = float(np.mean(d_rec_to_ref ** 2) + np.mean(d_ref_to_rec ** 2)) if (d_rec_to_ref.size and d_ref_to_rec.size) else 0.0

    out = {
        "accuracy_mean": acc,
        "completeness_mean": comp,
        "chamfer_l1": cd_l1,
        "chamfer_l2": cd_l2,
        "fscore": {},
    }
    for tau in taus_m:
        P, R, F = fscore_from_dists(d_rec_to_ref, d_ref_to_rec, tau)
        out["fscore"][tau] = {"precision": P, "recall": R, "fscore": F}
    return out


def eval_mesh_mesh_pair(
    mesh_a: o3d.geometry.TriangleMesh,
    mesh_b: o3d.geometry.TriangleMesh,
    mesh_samples: int,
    sample_method: str,
    taus_m: List[float],
) -> Dict[str, Any]:
    # a -> b
    p_a = sample_mesh(mesh_a, mesh_samples, sample_method)
    a_xyz = np.asarray(p_a.points, dtype=np.float64)
    scene_b = make_scene(mesh_b)
    d_a_to_b = point_to_mesh_distance(scene_b, a_xyz)

    # b -> a
    p_b = sample_mesh(mesh_b, mesh_samples, sample_method)
    b_xyz = np.asarray(p_b.points, dtype=np.float64)
    scene_a = make_scene(mesh_a)
    d_b_to_a = point_to_mesh_distance(scene_a, b_xyz)

    acc = float(np.mean(d_a_to_b)) if d_a_to_b.size else 0.0
    comp = float(np.mean(d_b_to_a)) if d_b_to_a.size else 0.0
    cd_l1 = acc + comp
    cd_l2 = float(np.mean(d_a_to_b ** 2) + np.mean(d_b_to_a ** 2)) if (d_a_to_b.size and d_b_to_a.size) else 0.0

    out = {
        "accuracy_mean": acc,
        "completeness_mean": comp,
        "chamfer_l1": cd_l1,
        "chamfer_l2": cd_l2,
        "fscore": {},
    }
    for tau in taus_m:
        P, R, F = fscore_from_dists(d_a_to_b, d_b_to_a, tau)
        out["fscore"][tau] = {"precision": P, "recall": R, "fscore": F}
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
    mesh_subdir: str,
    mesh_nksr_subdir: str,
) -> Dict[int, Dict[str, str]]:
    pcd_online_dir = os.path.join(result_dir, pcd_online_subdir)
    pcd_offline_dir = os.path.join(result_dir, pcd_offline_subdir)
    mesh_dir = os.path.join(result_dir, mesh_subdir)
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

    if os.path.isdir(mesh_dir):
        for p in sorted(glob.glob(os.path.join(mesh_dir, "*.ply"))):
            fid = parse_id_from_name(p)
            if fid is None:
                continue
            by_id.setdefault(fid, {})["mesh"] = p

    if os.path.isdir(mesh_nksr_dir):
        for p in sorted(glob.glob(os.path.join(mesh_nksr_dir, "*.ply"))):
            fid = parse_id_from_name(p)
            if fid is None:
                continue
            by_id.setdefault(fid, {})["mesh_nksr"] = p

    # 不强行过滤：允许某些 frame 缺在线/离线 pcd 或 nksr mesh，输出 NaN
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
def flatten_pcd_report_optional(report: Optional[Dict[str, Any]], taus_m: List[float], prefix: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    if report is None:
        row.update({
            f"{prefix}_accuracy_mean": np.nan,
            f"{prefix}_completeness_mean": np.nan,
            f"{prefix}_chamfer_l1": np.nan,
            f"{prefix}_chamfer_l2": np.nan,
            f"{prefix}_normal_mean_abs_cos": np.nan,
            f"{prefix}_normal_mean_angle_deg": np.nan,
        })
        for tau in taus_m:
            row[f"{prefix}_fscore@{tau:.6f}"] = np.nan
            row[f"{prefix}_precision@{tau:.6f}"] = np.nan
            row[f"{prefix}_recall@{tau:.6f}"] = np.nan
        return row

    row.update({
        f"{prefix}_accuracy_mean": report["accuracy_mean"],
        f"{prefix}_completeness_mean": report["completeness_mean"],
        f"{prefix}_chamfer_l1": report["chamfer_l1"],
        f"{prefix}_chamfer_l2": report["chamfer_l2"],
        f"{prefix}_normal_mean_abs_cos": report["normal"]["mean_abs_cos"],
        f"{prefix}_normal_mean_angle_deg": report["normal"]["mean_angle_deg"],
    })
    for tau in taus_m:
        row[f"{prefix}_fscore@{tau:.6f}"] = report["fscore"][tau]["fscore"]
        row[f"{prefix}_precision@{tau:.6f}"] = report["fscore"][tau]["precision"]
        row[f"{prefix}_recall@{tau:.6f}"] = report["fscore"][tau]["recall"]
    return row


def flatten_mesh_report(report: Optional[Dict[str, Any]], taus_m: List[float], prefix: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    if report is None:
        row.update({
            f"{prefix}_accuracy_mean": np.nan,
            f"{prefix}_completeness_mean": np.nan,
            f"{prefix}_chamfer_l1": np.nan,
            f"{prefix}_chamfer_l2": np.nan,
        })
        for tau in taus_m:
            row[f"{prefix}_fscore@{tau:.6f}"] = np.nan
            row[f"{prefix}_precision@{tau:.6f}"] = np.nan
            row[f"{prefix}_recall@{tau:.6f}"] = np.nan
        return row

    row.update({
        f"{prefix}_accuracy_mean": report["accuracy_mean"],
        f"{prefix}_completeness_mean": report["completeness_mean"],
        f"{prefix}_chamfer_l1": report["chamfer_l1"],
        f"{prefix}_chamfer_l2": report["chamfer_l2"],
    })
    for tau in taus_m:
        row[f"{prefix}_fscore@{tau:.6f}"] = report["fscore"][tau]["fscore"]
        row[f"{prefix}_precision@{tau:.6f}"] = report["fscore"][tau]["precision"]
        row[f"{prefix}_recall@{tau:.6f}"] = report["fscore"][tau]["recall"]
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
    gt_pcd_base: o3d.geometry.PointCloud,
    gt_mesh_base: o3d.geometry.TriangleMesh,
    s_gt: float,
    mesh_metric_samples: int,
    mesh_sample_method: str,
) -> Dict[str, Any]:
    gt_pcd = o3d.geometry.PointCloud(gt_pcd_base)
    gt_mesh = o3d.geometry.TriangleMesh(gt_mesh_base)
    apply_scale(gt_pcd, s_gt)
    apply_scale(gt_mesh, s_gt)

    scene_gt = make_scene(gt_mesh)
    gt_samples = sample_mesh(gt_mesh, mesh_metric_samples, mesh_sample_method)
    gt_samples_xyz = np.asarray(gt_samples.points, dtype=np.float64)

    return {
        "gt_pcd": gt_pcd,
        "gt_mesh": gt_mesh,
        "scene_gt": scene_gt,
        "gt_samples_xyz": gt_samples_xyz,
    }


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--result_dir", default="/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/result/ablation/offline_tracking/cube_purple/xyz",
                    help="dir containing pcd_online/, pcd/, mesh/, mesh_nksr/")
    ap.add_argument("--gt_root", default="/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/GT_data/cube_purple/GT",
                    help="GT folder containing mesh/GT_mesh.stl and ply/GT_normal.ply")
    ap.add_argument("--fps", type=float, default=20, help="time_s = frame_id / fps")

    ap.add_argument("--pcd_online_subdir", type=str, default="pcd_online")
    ap.add_argument("--pcd_offline_subdir", type=str, default="pcd")
    ap.add_argument("--mesh_subdir", type=str, default="mesh")
    ap.add_argument("--mesh_nksr_subdir", type=str, default="mesh_nksr")

    ap.add_argument("--align_source", choices=["pcd", "pcd_online"], default="pcd",
                    help="Which PCD folder to use for STRICT alignment (T_ref). Default: offline pcd/")

    ap.add_argument("--thresholds_mm", type=str, default="0.5, 1, 5", help="fscore thresholds (mm)")
    ap.add_argument("--pick_tau_mm", type=float, default=5.0, help="ref selection tau (mm) using Precision@tau")

    ap.add_argument("--rec_scale", type=float, default=1.0)
    ap.add_argument("--gt_scale", type=float, default=1.0)
    ap.add_argument("--no_auto_unit", action="store_true")
    ap.add_argument("--no_auto_voxel", default=True)  # <-- FIXED (was mistakenly default=True)
    ap.add_argument("--voxel", type=float, default=0.003, help="fallback voxel if auto_voxel disabled")

    # eval_voxel: used ONLY for Precision@tau scoring when picking T_ref
    ap.add_argument("--eval_voxel", type=float, default=0.005, help="downsample voxel for Precision@tau scoring (0 disables)")

    # pcd eval voxels
    ap.add_argument("--pcd_online_eval_voxel", type=float, default=0.005, help="downsample voxel for ONLINE pcd metrics (0 disables)")
    ap.add_argument("--pcd_offline_eval_voxel", type=float, default=0.005, help="downsample voxel for OFFLINE pcd metrics (0 disables)")

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

    ap.add_argument("--out_dir", type=str, default="", help="default: result_dir/eval_strictref")

    args = ap.parse_args()

    result_dir = os.path.expanduser(args.result_dir)
    out_dir = os.path.expanduser(args.out_dir) if args.out_dir else os.path.join(result_dir, "eval_strictref")
    ensure_dir(out_dir)

    taus_m = [float(x.strip()) / 1000.0 for x in args.thresholds_mm.split(",") if x.strip()]
    if not taus_m:
        raise RuntimeError("Empty thresholds_mm.")
    pick_tau_m = float(args.pick_tau_mm) / 1000.0

    gt_pcd_path = os.path.join(os.path.expanduser(args.gt_root), "ply", "GT_normal.ply")
    gt_mesh_path = os.path.join(os.path.expanduser(args.gt_root), "mesh", "GT_mesh.stl")
    if not os.path.isfile(gt_pcd_path):
        raise RuntimeError(f"Missing GT point cloud: {gt_pcd_path}")
    if not os.path.isfile(gt_mesh_path):
        raise RuntimeError(f"Missing GT mesh: {gt_mesh_path}")

    index = build_recon_index(
        result_dir=result_dir,
        pcd_online_subdir=args.pcd_online_subdir,
        pcd_offline_subdir=args.pcd_offline_subdir,
        mesh_subdir=args.mesh_subdir,
        mesh_nksr_subdir=args.mesh_nksr_subdir,
    )
    ids = sorted(index.keys())
    if not ids:
        raise RuntimeError("No frames found (cannot parse recon_0_XXXXXX from filenames).")

    icp_iters = (int(args.icp_iter0), int(args.icp_iter1), int(args.icp_iter2))

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

    align_key = "pcd" if args.align_source == "pcd" else "pcd_online"

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

    # ---------------- Load GT base once (only manual gt_scale) ----------------
    gt_pcd_base = read_point_cloud(gt_pcd_path)
    gt_mesh_base = read_triangle_mesh(gt_mesh_path)
    apply_scale(gt_pcd_base, args.gt_scale)
    apply_scale(gt_mesh_base, args.gt_scale)

    # Fixed-mode precompute (best_s_gt)
    if args.align_mode in ["fixed_id", "search_best"]:
        gt_fixed = build_gt_cache_entry(
            gt_pcd_base=gt_pcd_base,
            gt_mesh_base=gt_mesh_base,
            s_gt=best_s_gt,
            mesh_metric_samples=args.mesh_metric_samples,
            mesh_sample_method=args.mesh_sample_method,
        )
        gt_pcd_fixed = gt_fixed["gt_pcd"]
        scene_gt_fixed = gt_fixed["scene_gt"]
        gt_samples_xyz_fixed = gt_fixed["gt_samples_xyz"]
    else:
        gt_pcd_fixed = None
        scene_gt_fixed = None
        gt_samples_xyz_fixed = None

    # per_frame cache by s_gt
    gt_cache: Dict[float, Dict[str, Any]] = {}

    # ---------------- Evaluate all frames ----------------
    rows: List[Dict[str, Any]] = []
    missing_nksr = 0
    have_nksr = 0

    missing_mesh = 0
    missing_online = 0
    missing_offline = 0

    for i, fid in enumerate(ids):
        time_s = float(fid) / float(args.fps)
        row: Dict[str, Any] = {"frame_id": fid, "time_s": time_s}

        # ---- get T and (possibly per-frame) scales ----
        if args.align_mode == "per_frame":
            rec_align_path = index[fid].get(align_key, "")
            if not rec_align_path or (not os.path.isfile(rec_align_path)):
                # per-frame 模式下，如果这一帧缺少对齐源，直接输出 NaN
                T = None
                s_rec_frame = np.nan
                s_gt_frame = np.nan
                gt_pcd_use = None
                scene_gt_use = None
                gt_samples_xyz_use = None
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
                        gt_pcd_base=gt_pcd_base,
                        gt_mesh_base=gt_mesh_base,
                        s_gt=s_gt_frame,
                        mesh_metric_samples=args.mesh_metric_samples,
                        mesh_sample_method=args.mesh_sample_method,
                    )
                gt_entry = gt_cache[s_gt_frame]
                gt_pcd_use = gt_entry["gt_pcd"]
                scene_gt_use = gt_entry["scene_gt"]
                gt_samples_xyz_use = gt_entry["gt_samples_xyz"]
        else:
            T = T_ref
            s_rec_frame = float(best_s_rec)
            gt_pcd_use = gt_pcd_fixed
            scene_gt_use = scene_gt_fixed
            gt_samples_xyz_use = gt_samples_xyz_fixed

        # 如果这一帧没有有效 T（per_frame 缺对齐源），全部输出 NaN
        if T is None or gt_pcd_use is None or scene_gt_use is None or gt_samples_xyz_use is None or (not np.isfinite(s_rec_frame)):
            row.update(flatten_pcd_report_optional(None, taus_m, prefix="online_pcd_vs_gt"))
            row.update(flatten_pcd_report_optional(None, taus_m, prefix="offline_pcd_vs_gt"))
            row.update(flatten_mesh_report(None, taus_m, prefix="mesh_vs_gt"))
            row.update(flatten_mesh_report(None, taus_m, prefix="nksr_vs_mesh"))
            rows.append(row)
            continue

        # ============================================================
        # Online PCD vs GT
        # ============================================================
        online_path = index[fid].get("pcd_online", "")
        online_report = None
        if online_path and os.path.isfile(online_path):
            rec_online = read_point_cloud(online_path)
            apply_scale(rec_online, args.rec_scale)
            apply_scale(rec_online, s_rec_frame)
            rec_online.transform(T)

            if args.pcd_online_eval_voxel and args.pcd_online_eval_voxel > 0:
                rec_online_eval = preprocess_pcd(o3d.geometry.PointCloud(rec_online), float(args.pcd_online_eval_voxel))
                gt_pcd_eval = preprocess_pcd(o3d.geometry.PointCloud(gt_pcd_use), float(args.pcd_online_eval_voxel))
            else:
                rec_online_eval = rec_online
                gt_pcd_eval = gt_pcd_use

            online_report = eval_pcd_pcd(rec_online_eval, gt_pcd_eval, taus_m=taus_m)
        else:
            missing_online += 1

        row.update(flatten_pcd_report_optional(online_report, taus_m, prefix="online_pcd_vs_gt"))

        # ============================================================
        # Offline PCD vs GT
        # ============================================================
        offline_path = index[fid].get("pcd", "")
        offline_report = None
        if offline_path and os.path.isfile(offline_path):
            rec_offline = read_point_cloud(offline_path)
            apply_scale(rec_offline, args.rec_scale)
            apply_scale(rec_offline, s_rec_frame)
            rec_offline.transform(T)

            if args.pcd_offline_eval_voxel and args.pcd_offline_eval_voxel > 0:
                rec_offline_eval = preprocess_pcd(o3d.geometry.PointCloud(rec_offline), float(args.pcd_offline_eval_voxel))
                gt_pcd_eval2 = preprocess_pcd(o3d.geometry.PointCloud(gt_pcd_use), float(args.pcd_offline_eval_voxel))
            else:
                rec_offline_eval = rec_offline
                gt_pcd_eval2 = gt_pcd_use

            offline_report = eval_pcd_pcd(rec_offline_eval, gt_pcd_eval2, taus_m=taus_m)
        else:
            missing_offline += 1

        row.update(flatten_pcd_report_optional(offline_report, taus_m, prefix="offline_pcd_vs_gt"))

        # ============================================================
        # Mesh vs GT
        # ============================================================
        mesh_path = index[fid].get("mesh", "")
        rep_mesh_gt = None
        mesh_orig = None
        if mesh_path and os.path.isfile(mesh_path):
            mesh_orig = read_triangle_mesh(mesh_path)
            apply_scale(mesh_orig, args.rec_scale)
            apply_scale(mesh_orig, s_rec_frame)
            mesh_orig.transform(T)

            rep_mesh_gt = eval_mesh_mesh_against_fixed_ref(
                rec_mesh=mesh_orig,
                scene_ref=scene_gt_use,
                ref_samples_xyz=gt_samples_xyz_use,
                mesh_samples=args.mesh_metric_samples,
                sample_method=args.mesh_sample_method,
                taus_m=taus_m,
            )
        else:
            missing_mesh += 1

        row.update(flatten_mesh_report(rep_mesh_gt, taus_m, prefix="mesh_vs_gt"))

        # ============================================================
        # NKSR mesh vs original mesh
        # ============================================================
        rep_nksr_mesh: Optional[Dict[str, Any]] = None
        nksr_path = index[fid].get("mesh_nksr", "")
        if mesh_orig is not None and nksr_path and os.path.isfile(nksr_path):
            have_nksr += 1
            mesh_nksr = read_triangle_mesh(nksr_path)
            apply_scale(mesh_nksr, args.rec_scale)
            apply_scale(mesh_nksr, s_rec_frame)
            mesh_nksr.transform(T)

            rep_nksr_mesh = eval_mesh_mesh_pair(
                mesh_a=mesh_nksr,
                mesh_b=mesh_orig,
                mesh_samples=args.mesh_metric_samples,
                sample_method=args.mesh_sample_method,
                taus_m=taus_m,
            )
        else:
            missing_nksr += 1

        row.update(flatten_mesh_report(rep_nksr_mesh, taus_m, prefix="nksr_vs_mesh"))

        rows.append(row)

        # periodic logging
        if (i % 5) == 0 or (i == len(ids) - 1):
            tau0 = taus_m[0]
            online_acc = row.get("online_pcd_vs_gt_accuracy_mean", np.nan)
            offline_acc = row.get("offline_pcd_vs_gt_accuracy_mean", np.nan)
            mesh_f = row.get(f"mesh_vs_gt_fscore@{tau0:.6f}", np.nan)
            nksr_f = row.get(f"nksr_vs_mesh_fscore@{tau0:.6f}", np.nan)
            print(
                f"[{i+1:04d}/{len(ids):04d}] id={fid:06d} t={time_s:.2f}s  "
                f"OnlinePCD acc={online_acc:.6f}  OfflinePCD acc={offline_acc:.6f}  "
                f"Mesh-vs-GT F@{tau0*1000:.1f}mm={mesh_f:.4f}  "
                f"NKSR-vs-Mesh F@{tau0*1000:.1f}mm={nksr_f}"
            )

    csv_path = os.path.join(out_dir, "summary.csv")
    write_csv(csv_path, rows)
    print(f"\n[SAVED] {csv_path}")
    print(f"[INFO] Missing online PCD frames : {missing_online}")
    print(f"[INFO] Missing offline PCD frames: {missing_offline}")
    print(f"[INFO] Missing mesh frames       : {missing_mesh}")
    print(f"[INFO] mesh_nksr available frames: {have_nksr} / {len(ids)} (missing {missing_nksr})")
    print("[Done]")


if __name__ == "__main__":
    main()
