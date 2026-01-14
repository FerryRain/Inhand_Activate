#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch evaluation with STRICT reference alignment (same as your old single-frame script).

Workflow:
1) Build index from:
   result_dir/normal/*.ply (pcd with normals)
   result_dir/mesh/*.ply   (mesh)
2) Pick reference transform T_ref using STRICT alignment on candidate frames:
   manual scale -> auto-unit(mm<->m) -> auto-voxel -> PCA(24-way) -> multiscale ICP(point-to-plane with Tukey)
   score by Precision@tau (default tau=pick_tau_mm) on eval downsampled clouds
3) Apply fixed (s_rec, s_gt, T_ref) to ALL frames (no per-frame ICP), compute:
   - pcd_pcd: accuracy_mean, completeness_mean, chamfer_l1, chamfer_l2, fscore@taus (+ normal consistency mean_abs_cos, mean_angle_deg)
   - mesh_mesh: accuracy_mean, completeness_mean, chamfer_l1, chamfer_l2, fscore@taus
4) Save summary.csv to out_dir (default: result_dir/eval_strictref)

Extra:
- Visualization for alignment inspection:
  --vis_ref_ids  : visualize STRICT alignment for selected reference candidates
  --vis_eval_ids : visualize final aligned results for selected frames

Example:
python compute_metric_batch_strictref.py \
  --result_dir /.../result/green_cube_00_001 \
  --gt_root /.../GT_data/cube/GT \
  --align_mode search_best \
  --search_ids 173,186,-1 \
  --pick_tau_mm 0.5 \
  --vis_ref_ids 173,-1 \
  --vis_eval_ids 100,120,-1
"""

import os
import re
import csv
import glob
import time
import argparse
from typing import Dict, Any, List, Optional

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
# Auto unit (mm<->m) and auto voxel (STRICT = same as your old)
# ============================================================
def pcd_diag_and_center(pcd: o3d.geometry.PointCloud):
    aabb = pcd.get_axis_aligned_bounding_box()
    ext = np.asarray(aabb.get_extent(), dtype=np.float64)
    diag = float(np.linalg.norm(ext))
    cen = np.asarray(aabb.get_center(), dtype=np.float64)
    return ext, diag, cen


def maybe_auto_unit_to_meters(recon_pcd, gt_pcd, enable=True):
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
# PCA init (24-way) (STRICT = same as your old)
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
        _, _, d2 = kdt.search_knn_vector_3d(q[i], 1)
        if len(d2) == 1:
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
# ICP multiscale (point-to-plane) (STRICT = same as your old)
# ============================================================
def estimate_normals_for_icp(pcd: o3d.geometry.PointCloud, radius: float, max_nn=30):
    if pcd.is_empty():
        return pcd
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius), max_nn=int(max_nn)))
    pcd.normalize_normals()
    return pcd


def icp_multiscale(src_full, tgt_full, init_T, voxel_base, iters=(60, 90), verbose=False):
    voxels = [2.0 * voxel_base, 1.0 * voxel_base]
    Ts = init_T.copy()
    for lvl, (vx, it) in enumerate(zip(voxels, iters)):
        src = preprocess_pcd(o3d.geometry.PointCloud(src_full), vx)
        tgt = preprocess_pcd(o3d.geometry.PointCloud(tgt_full), vx)
        nr = max(2.0 * vx, 1e-3)
        estimate_normals_for_icp(src, nr)
        estimate_normals_for_icp(tgt, nr)
        max_corr = 2.5 * vx
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


def eval_mesh_mesh(rec_mesh: o3d.geometry.TriangleMesh,
                   gt_mesh: o3d.geometry.TriangleMesh,
                   scene_gt: o3d.t.geometry.RaycastingScene,
                   gt_samples_xyz: np.ndarray,
                   mesh_samples: int,
                   sample_method: str,
                   taus_m: List[float]) -> Dict[str, Any]:
    p_rec = sample_mesh(rec_mesh, mesh_samples, sample_method)
    rec_xyz = np.asarray(p_rec.points, dtype=np.float64)
    d_rec_to_gt = point_to_mesh_distance(scene_gt, rec_xyz)

    scene_rec = make_scene(rec_mesh)
    d_gt_to_rec = point_to_mesh_distance(scene_rec, gt_samples_xyz)

    acc = float(np.mean(d_rec_to_gt)) if d_rec_to_gt.size else 0.0
    comp = float(np.mean(d_gt_to_rec)) if d_gt_to_rec.size else 0.0
    cd_l1 = acc + comp
    cd_l2 = float(np.mean(d_rec_to_gt**2) + np.mean(d_gt_to_rec**2)) if (d_rec_to_gt.size and d_gt_to_rec.size) else 0.0

    out = {
        "accuracy_mean": acc,
        "completeness_mean": comp,
        "chamfer_l1": cd_l1,
        "chamfer_l2": cd_l2,
        "fscore": {},
    }
    for tau in taus_m:
        P, R, F = fscore_from_dists(d_rec_to_gt, d_gt_to_rec, tau)
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


def build_recon_index(result_dir: str) -> Dict[int, Dict[str, str]]:
    normal_dir = os.path.join(result_dir, "normal")
    mesh_dir = os.path.join(result_dir, "mesh")
    if not os.path.isdir(normal_dir):
        raise RuntimeError(f"Missing folder: {normal_dir}")
    if not os.path.isdir(mesh_dir):
        raise RuntimeError(f"Missing folder: {mesh_dir}")

    pcd_files = sorted(glob.glob(os.path.join(normal_dir, "*.ply")))
    mesh_files = sorted(glob.glob(os.path.join(mesh_dir, "*.ply")))

    by_id: Dict[int, Dict[str, str]] = {}
    for p in pcd_files:
        fid = parse_id_from_name(p)
        if fid is None:
            continue
        by_id.setdefault(fid, {})["pcd"] = p

    for p in mesh_files:
        fid = parse_id_from_name(p)
        if fid is None:
            continue
        if "mesh" not in by_id.get(fid, {}):
            by_id.setdefault(fid, {})["mesh"] = p
        else:
            old = by_id[fid]["mesh"]
            if len(os.path.basename(p)) < len(os.path.basename(old)):
                by_id[fid]["mesh"] = p

    return {k: v for k, v in by_id.items() if "pcd" in v}


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
# Visualization helpers
# ============================================================
def resolve_ids_with_last(id_list: List[int], all_ids: List[int]) -> List[int]:
    if not all_ids:
        return []
    last = all_ids[-1]
    out = []
    seen = set()
    for x in id_list:
        v = last if x < 0 else x
        if v in all_ids and v not in seen:
            out.append(v)
            seen.add(v)
    return out


def visualize_alignment(
    gt_pcd: o3d.geometry.PointCloud,
    rec_pcd_aligned: o3d.geometry.PointCloud,
    title: str,
    voxel: float = 0.002,
    gt_mesh: Optional[o3d.geometry.TriangleMesh] = None,
    rec_mesh_aligned: Optional[o3d.geometry.TriangleMesh] = None,
):
    if voxel and voxel > 0:
        gt_vis = preprocess_pcd(o3d.geometry.PointCloud(gt_pcd), voxel)
        rec_vis = preprocess_pcd(o3d.geometry.PointCloud(rec_pcd_aligned), voxel)
    else:
        gt_vis = o3d.geometry.PointCloud(gt_pcd)
        rec_vis = o3d.geometry.PointCloud(rec_pcd_aligned)

    gt_vis.paint_uniform_color([0.2, 0.8, 0.2])   # green
    rec_vis.paint_uniform_color([0.9, 0.2, 0.2])  # red

    geoms = [gt_vis, rec_vis, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)]

    if gt_mesh is not None and rec_mesh_aligned is not None:
        gt_m = o3d.geometry.TriangleMesh(gt_mesh)
        rec_m = o3d.geometry.TriangleMesh(rec_mesh_aligned)
        if not gt_m.has_vertex_normals():
            gt_m.compute_vertex_normals()
        if not rec_m.has_vertex_normals():
            rec_m.compute_vertex_normals()
        gt_m.paint_uniform_color([0.2, 0.8, 0.2])
        rec_m.paint_uniform_color([0.9, 0.2, 0.2])
        geoms.extend([gt_m, rec_m])

    print(f"[VIS] {title}  (close window to continue)")
    try:
        o3d.visualization.draw_geometries(geoms, window_name=title)
    except Exception as e:
        print(f"[VIS][SKIP] Open3D window failed (headless?). Error: {repr(e)}")


# ============================================================
# CSV
# ============================================================
def flatten_row(fid: int, time_s: float,
                pcd_report: Dict[str, Any],
                mesh_report: Optional[Dict[str, Any]],
                taus_m: List[float]) -> Dict[str, Any]:
    row = {
        "frame_id": fid,
        "time_s": time_s,
        "pcd_pcd_accuracy_mean": pcd_report["accuracy_mean"],
        "pcd_pcd_completeness_mean": pcd_report["completeness_mean"],
        "pcd_pcd_chamfer_l1": pcd_report["chamfer_l1"],
        "pcd_pcd_chamfer_l2": pcd_report["chamfer_l2"],
        "pcd_pcd_normal_mean_abs_cos": pcd_report["normal"]["mean_abs_cos"],
        "pcd_pcd_normal_mean_angle_deg": pcd_report["normal"]["mean_angle_deg"],
    }
    for tau in taus_m:
        row[f"pcd_pcd_fscore@{tau:.6f}"] = pcd_report["fscore"][tau]["fscore"]
        row[f"pcd_pcd_precision@{tau:.6f}"] = pcd_report["fscore"][tau]["precision"]
        row[f"pcd_pcd_recall@{tau:.6f}"] = pcd_report["fscore"][tau]["recall"]

    # Always include mesh columns (avoid CSV field mismatch)
    if mesh_report is None:
        row.update({
            "mesh_mesh_accuracy_mean": np.nan,
            "mesh_mesh_completeness_mean": np.nan,
            "mesh_mesh_chamfer_l1": np.nan,
            "mesh_mesh_chamfer_l2": np.nan,
        })
        for tau in taus_m:
            row[f"mesh_mesh_fscore@{tau:.6f}"] = np.nan
            row[f"mesh_mesh_precision@{tau:.6f}"] = np.nan
            row[f"mesh_mesh_recall@{tau:.6f}"] = np.nan
    else:
        row.update({
            "mesh_mesh_accuracy_mean": mesh_report["accuracy_mean"],
            "mesh_mesh_completeness_mean": mesh_report["completeness_mean"],
            "mesh_mesh_chamfer_l1": mesh_report["chamfer_l1"],
            "mesh_mesh_chamfer_l2": mesh_report["chamfer_l2"],
        })
        for tau in taus_m:
            row[f"mesh_mesh_fscore@{tau:.6f}"] = mesh_report["fscore"][tau]["fscore"]
            row[f"mesh_mesh_precision@{tau:.6f}"] = mesh_report["fscore"][tau]["precision"]
            row[f"mesh_mesh_recall@{tau:.6f}"] = mesh_report["fscore"][tau]["recall"]

    return row


def write_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        raise RuntimeError("No rows to write.")

    # Robust fieldnames: union of all keys (stable sorted)
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
# STRICT reference alignment on ONE candidate (exact old behavior)
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
    icp_iter0: int,
    icp_iter1: int,
    eval_voxel: float,
    pick_tau_m: float,
    verbose_icp: bool = False,
) -> Dict[str, Any]:
    """
    Returns dict:
      {
        "T": 4x4 (GT <- Recon),
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

    # strict auto unit (per candidate)
    s_rec, s_gt = maybe_auto_unit_to_meters(rec_pcd, gt_pcd, enable=auto_unit)
    if s_rec != 1.0:
        apply_scale(rec_pcd, s_rec)
    if s_gt != 1.0:
        apply_scale(gt_pcd, s_gt)
        apply_scale(gt_mesh, s_gt)

    # strict auto voxel (per candidate)
    if auto_voxel:
        voxel = auto_voxel_from_recon(rec_pcd, enable=True, fallback=voxel_fallback)
    else:
        voxel = float(voxel_fallback)

    # alignment uses downsampled PCDs (stable)
    src_ds = preprocess_pcd(o3d.geometry.PointCloud(rec_pcd), voxel)
    tgt_ds = preprocess_pcd(o3d.geometry.PointCloud(gt_pcd), voxel)

    t0 = time.time()
    T0, _ = pca_init_transform(src_ds, tgt_ds)
    T = icp_multiscale(rec_pcd, gt_pcd, T0, voxel_base=voxel, iters=(icp_iter0, icp_iter1), verbose=verbose_icp)
    t_align = time.time() - t0

    rec_aligned = o3d.geometry.PointCloud(rec_pcd)
    rec_aligned.transform(T)

    if eval_voxel and eval_voxel > 0:
        rec_eval = preprocess_pcd(rec_aligned, eval_voxel)
        gt_eval = preprocess_pcd(o3d.geometry.PointCloud(gt_pcd), eval_voxel)
    else:
        rec_eval = rec_aligned
        gt_eval = gt_pcd

    score = precision_at_tau(rec_eval, gt_eval, pick_tau_m)

    return {
        "T": T,
        "s_rec": s_rec,
        "s_gt": s_gt,
        "score": score,
        "voxel": voxel,
        "time_align_s": t_align,
    }


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--result_dir", default="/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/result/yellow_cylinder_002",
                    help="dir containing normal/ and mesh/")
    ap.add_argument("--gt_root", default="/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/GT_data/yellow_cylinder_01/GT",
                    help="GT folder containing mesh/GT_mesh.stl and ply/GT_normal.ply")
    ap.add_argument("--fps", type=float, default=4.0, help="time_s = frame_id / fps")

    ap.add_argument("--thresholds_mm", type=str, default="2,5,10", help="fscore thresholds (mm)")
    ap.add_argument("--pick_tau_mm", type=float, default=5.0, help="ref selection tau (mm) using Precision@tau")

    ap.add_argument("--rec_scale", type=float, default=1.0)
    ap.add_argument("--gt_scale", type=float, default=1.0)
    ap.add_argument("--no_auto_unit", action="store_true")
    ap.add_argument("--no_auto_voxel", action="store_true")
    ap.add_argument("--voxel", type=float, default=0.003, help="fallback voxel if auto_voxel disabled")
    ap.add_argument("--eval_voxel", type=float, default=0.002, help="downsample voxel for pcd metrics (0 disables)")
    ap.add_argument("--icp_iter0", type=int, default=60)
    ap.add_argument("--icp_iter1", type=int, default=90)
    ap.add_argument("--verbose_icp", action="store_true")

    ap.add_argument("--align_mode", choices=["fixed_id", "search_best", "per_frame"], default="search_best",
                    help="fixed_id/search_best = strict ref then fixed T for all; per_frame = strict per-frame (slow)")
    ap.add_argument("--align_id", type=int, default=-1, help="fixed_id reference frame id; -1 means last")
    ap.add_argument("--search_ids", type=str, default="100,200,-1",
                    help="search_best explicit ids, e.g. '100,200,-1' (-1 means last). If empty, uses --search_k uniform.")
    ap.add_argument("--search_k", type=int, default=7, help="search_best fallback candidate count (uniform + last)")

    ap.add_argument("--skip_mesh", action="store_true", help="skip mesh metrics")
    ap.add_argument("--mesh_metric_samples", type=int, default=20000)
    ap.add_argument("--mesh_sample_method", choices=["uniform", "poisson"], default="uniform")

    ap.add_argument("--out_dir", type=str, default="", help="default: result_dir/eval_strictref")
    ap.add_argument("--save_T", type=str, default="", help="save chosen T_ref to txt")
    ap.add_argument("--save_aligned", action="store_true", help="save aligned pcd/mesh for all frames (large)")

    # ---- NEW: visualization options ----
    ap.add_argument("--vis_eval_ids", type=str, default="",
                    help="visualize final aligned eval results for ids, e.g. '100,120,-1' (-1 means last)")
    ap.add_argument("--vis_ref_ids", type=str, default="100, 200, -1",
                    help="visualize strict alignment for ref candidates ids, e.g. '109,144,-1'")
    ap.add_argument("--vis_voxel", type=float, default=0.002,
                    help="voxel downsample for visualization (0 disables)")
    ap.add_argument("--vis_mesh", action="store_true",default=True,
                    help="also show meshes in visualization (if available)")

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

    index = build_recon_index(result_dir)
    ids = sorted(index.keys())
    if not ids:
        raise RuntimeError("No recon point clouds found in result_dir/normal.")

    # Parse visualization ids
    vis_ref_ids = resolve_ids_with_last(parse_id_list(args.vis_ref_ids), ids) if args.vis_ref_ids.strip() else []
    vis_eval_ids = resolve_ids_with_last(parse_id_list(args.vis_eval_ids), ids) if args.vis_eval_ids.strip() else []

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

    # ---------------- Find strict ref (T_ref, s_rec, s_gt) ----------------
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

            rec_pcd_path = index[cid]["pcd"]

            info = strict_align_one_candidate(
                rec_pcd_path=rec_pcd_path,
                gt_pcd_path=gt_pcd_path,
                gt_mesh_path=gt_mesh_path,
                rec_scale=args.rec_scale,
                gt_scale=args.gt_scale,
                auto_unit=(not args.no_auto_unit),
                auto_voxel=(not args.no_auto_voxel),
                voxel_fallback=float(args.voxel),
                icp_iter0=args.icp_iter0,
                icp_iter1=args.icp_iter1,
                eval_voxel=float(args.eval_voxel),
                pick_tau_m=pick_tau_m,
                verbose_icp=args.verbose_icp,
            )

            print(f"[REF_CAND] id={cid:06d}  P@{args.pick_tau_mm:.2f}mm={info['score']:.4f}  "
                  f"voxel={info['voxel']:.6f}  s_rec={info['s_rec']} s_gt={info['s_gt']}  "
                  f"align_time={info['time_align_s']:.2f}s")

            # ---- Visualize STRICT alignment for selected candidates ----
            if cid in vis_ref_ids:
                gt_tmp = read_point_cloud(gt_pcd_path)
                rec_tmp = read_point_cloud(rec_pcd_path)

                apply_scale(gt_tmp, args.gt_scale)
                apply_scale(rec_tmp, args.rec_scale)

                apply_scale(gt_tmp, info["s_gt"])
                apply_scale(rec_tmp, info["s_rec"])

                rec_tmp.transform(info["T"])

                if args.vis_mesh:
                    # optional meshes for visualization (if exist)
                    gt_m = read_triangle_mesh(gt_mesh_path)
                    apply_scale(gt_m, args.gt_scale)
                    apply_scale(gt_m, info["s_gt"])

                    rec_m = None
                    rec_mesh_path = index[cid].get("mesh", "")
                    if rec_mesh_path and os.path.isfile(rec_mesh_path):
                        rec_m = read_triangle_mesh(rec_mesh_path)
                        apply_scale(rec_m, args.rec_scale)
                        apply_scale(rec_m, info["s_rec"])
                        rec_m.transform(info["T"])
                    visualize_alignment(
                        gt_pcd=gt_tmp,
                        rec_pcd_aligned=rec_tmp,
                        title=f"REF_ALIGN id={cid:06d} P@{args.pick_tau_mm:.2f}mm={info['score']:.4f}",
                        voxel=float(args.vis_voxel),
                        gt_mesh=gt_m,
                        rec_mesh_aligned=rec_m,
                    )
                else:
                    visualize_alignment(
                        gt_pcd=gt_tmp,
                        rec_pcd_aligned=rec_tmp,
                        title=f"REF_ALIGN id={cid:06d} P@{args.pick_tau_mm:.2f}mm={info['score']:.4f}",
                        voxel=float(args.vis_voxel),
                    )

            if info["score"] > best_score:
                best_score = info["score"]
                best_id = cid
                T_ref = info["T"]
                best_s_rec = info["s_rec"]
                best_s_gt = info["s_gt"]

        if T_ref is None:
            raise RuntimeError("Failed to compute T_ref.")

        print(f"[ALIGN_REF] mode={args.align_mode}  ref_id={best_id:06d}  best_P@{args.pick_tau_mm:.2f}mm={best_score:.4f}")
        if args.save_T:
            np.savetxt(args.save_T, T_ref, fmt="%.10f")
            print(f"[SAVED] T_ref -> {args.save_T}")

    # ---------------- Load GT ONCE with chosen scales ----------------
    gt_pcd = read_point_cloud(gt_pcd_path)
    gt_mesh = read_triangle_mesh(gt_mesh_path)

    apply_scale(gt_pcd, args.gt_scale)
    apply_scale(gt_mesh, args.gt_scale)
    apply_scale(gt_pcd, best_s_gt)
    apply_scale(gt_mesh, best_s_gt)

    # Precompute GT scene and GT samples for mesh metrics
    if not args.skip_mesh:
        scene_gt = make_scene(gt_mesh)
        gt_samples = sample_mesh(gt_mesh, args.mesh_metric_samples, args.mesh_sample_method)
        gt_samples_xyz = np.asarray(gt_samples.points, dtype=np.float64)
    else:
        scene_gt = None
        gt_samples_xyz = None

    # ---------------- Evaluate all frames ----------------
    rows: List[Dict[str, Any]] = []
    aligned_dir = os.path.join(out_dir, "aligned")
    if args.save_aligned:
        ensure_dir(os.path.join(aligned_dir, "pcd"))
        ensure_dir(os.path.join(aligned_dir, "mesh"))

    for i, fid in enumerate(ids):
        rec_pcd = read_point_cloud(index[fid]["pcd"])
        apply_scale(rec_pcd, args.rec_scale)
        apply_scale(rec_pcd, best_s_rec)

        rec_mesh = None
        rec_mesh_path = index[fid].get("mesh", "")
        if (not args.skip_mesh) and rec_mesh_path and os.path.isfile(rec_mesh_path):
            rec_mesh = read_triangle_mesh(rec_mesh_path)
            apply_scale(rec_mesh, args.rec_scale)
            apply_scale(rec_mesh, best_s_rec)

        # Align
        if args.align_mode == "per_frame":
            info = strict_align_one_candidate(
                rec_pcd_path=index[fid]["pcd"],
                gt_pcd_path=gt_pcd_path,
                gt_mesh_path=gt_mesh_path,
                rec_scale=args.rec_scale,
                gt_scale=args.gt_scale,
                auto_unit=(not args.no_auto_unit),
                auto_voxel=(not args.no_auto_voxel),
                voxel_fallback=float(args.voxel),
                icp_iter0=args.icp_iter0,
                icp_iter1=args.icp_iter1,
                eval_voxel=float(args.eval_voxel),
                pick_tau_m=pick_tau_m,
                verbose_icp=False,
            )
            T = info["T"]
        else:
            T = T_ref

        rec_pcd_aligned = o3d.geometry.PointCloud(rec_pcd)
        rec_pcd_aligned.transform(T)

        rec_mesh_aligned = None
        if rec_mesh is not None:
            rec_mesh_aligned = o3d.geometry.TriangleMesh(rec_mesh)
            rec_mesh_aligned.transform(T)

        # ---- Visualize selected eval frames ----
        if fid in vis_eval_ids:
            if args.vis_mesh:
                visualize_alignment(
                    gt_pcd=gt_pcd,
                    rec_pcd_aligned=rec_pcd_aligned,
                    title=f"EVAL_ALIGN id={fid:06d} t={float(fid)/float(args.fps):.2f}s",
                    voxel=float(args.vis_voxel),
                    gt_mesh=gt_mesh,
                    rec_mesh_aligned=rec_mesh_aligned,
                )
            else:
                visualize_alignment(
                    gt_pcd=gt_pcd,
                    rec_pcd_aligned=rec_pcd_aligned,
                    title=f"EVAL_ALIGN id={fid:06d} t={float(fid)/float(args.fps):.2f}s",
                    voxel=float(args.vis_voxel),
                )

        # PCD eval downsample
        if args.eval_voxel and args.eval_voxel > 0:
            rec_eval = preprocess_pcd(rec_pcd_aligned, args.eval_voxel)
            gt_eval = preprocess_pcd(o3d.geometry.PointCloud(gt_pcd), args.eval_voxel)
        else:
            rec_eval = rec_pcd_aligned
            gt_eval = gt_pcd

        pcd_report = eval_pcd_pcd(rec_eval, gt_eval, taus_m=taus_m)

        if args.skip_mesh or rec_mesh_aligned is None:
            mesh_report = None
        else:
            mesh_report = eval_mesh_mesh(
                rec_mesh_aligned, gt_mesh,
                scene_gt=scene_gt,
                gt_samples_xyz=gt_samples_xyz,
                mesh_samples=args.mesh_metric_samples,
                sample_method=args.mesh_sample_method,
                taus_m=taus_m
            )

        time_s = float(fid) / float(args.fps)
        rows.append(flatten_row(fid, time_s, pcd_report, mesh_report, taus_m))

        if args.save_aligned:
            o3d.io.write_point_cloud(
                os.path.join(aligned_dir, "pcd", f"aligned_{fid:06d}.ply"),
                rec_pcd_aligned, write_ascii=False
            )
            if rec_mesh_aligned is not None:
                o3d.io.write_triangle_mesh(
                    os.path.join(aligned_dir, "mesh", f"aligned_{fid:06d}.ply"),
                    rec_mesh_aligned, write_ascii=False
                )

        if (i % 5) == 0 or (i == len(ids) - 1):
            f0 = pcd_report["fscore"][taus_m[0]]["fscore"]
            print(f"[{i+1:04d}/{len(ids):04d}] id={fid:06d}  t={time_s:.2f}s  "
                  f"pcd_acc={pcd_report['accuracy_mean']:.6f}  pcd_comp={pcd_report['completeness_mean']:.6f}  "
                  f"F@{taus_m[0]*1000:.1f}mm={f0:.4f}")

    csv_path = os.path.join(out_dir, "summary.csv")
    write_csv(csv_path, rows)
    print(f"\n[SAVED] {csv_path}")
    print("[Done]")


if __name__ == "__main__":
    main()
