"""
@FileName：compare_all_time.py
@Description：
@Author：Ferry
@Time：2026 1/13/26 11:25 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""

import os
import re
import json
import csv
import argparse
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
        raise RuntimeError(
            f"Mesh has no triangles (maybe it's a point cloud PLY): {path}\n"
            f"Please pass this file as --*_pcd instead of --*_mesh."
        )
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


def sample_mesh(mesh: o3d.geometry.TriangleMesh, n: int, method: str = "poisson") -> o3d.geometry.PointCloud:
    if method == "poisson":
        return mesh.sample_points_poisson_disk(number_of_points=int(n))
    return mesh.sample_points_uniformly(number_of_points=int(n))


def pcd_diag_and_center(pcd: o3d.geometry.PointCloud):
    aabb = pcd.get_axis_aligned_bounding_box()
    ext = np.asarray(aabb.get_extent(), dtype=np.float64)
    diag = float(np.linalg.norm(ext))
    cen = np.asarray(aabb.get_center(), dtype=np.float64)
    return ext, diag, cen


def print_pcd_stats(name: str, pcd: o3d.geometry.PointCloud):
    ext, diag, cen = pcd_diag_and_center(pcd)
    print(f"[{name}] N={len(pcd.points)}  extent={ext}  diag={diag:.6f}  center={cen}")
    return diag


def apply_scale(geom, s: float):
    if s == 1.0:
        return
    if isinstance(geom, (o3d.geometry.PointCloud, o3d.geometry.TriangleMesh)):
        geom.scale(float(s), center=(0, 0, 0))


def apply_transform(geom, T: np.ndarray):
    if geom is None:
        return
    if isinstance(geom, (o3d.geometry.PointCloud, o3d.geometry.TriangleMesh)):
        geom.transform(T)


# ============================================================
# Auto unit (mm<->m) and auto voxel
# ============================================================
def maybe_auto_unit_to_meters(recon_pcd, gt_pcd, enable=True):
    """
    Heuristic: if diag ratio ~ 1000, assume one is in mm and the other in meters.
    Convert the larger-diag geometry by *0.001 to bring to meters.
    Also if BOTH diags are huge, convert both by 0.001.
    """
    if not enable:
        return 1.0, 1.0  # (recon_extra_scale, gt_extra_scale)

    _, dr, _ = pcd_diag_and_center(recon_pcd)
    _, dg, _ = pcd_diag_and_center(gt_pcd)

    if dr <= 0 or dg <= 0:
        return 1.0, 1.0

    # if both look like mm-scale
    if dr > 5.0 and dg > 5.0:
        return 0.001, 0.001

    big = max(dr, dg)
    small = min(dr, dg)
    ratio = big / (small + 1e-12)

    # typical mm vs m mismatch ratio ~ 1000
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
    w, V = np.linalg.eigh(C)  # ascending
    V = V[:, np.argsort(w)[::-1]]  # descending
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
# ICP multiscale (point-to-plane)
# ============================================================
def estimate_normals_for_icp(pcd: o3d.geometry.PointCloud, radius: float, max_nn=30):
    if pcd.is_empty():
        return pcd
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius), max_nn=int(max_nn)))
    pcd.normalize_normals()
    return pcd


def icp_multiscale(src_full, tgt_full, init_T, voxel_base, iters=(60, 90)):
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
        print(f"[ICP lvl{lvl}] voxel={vx:.6f} max_corr={max_corr:.6f} fitness={reg.fitness:.4f} rmse={reg.inlier_rmse:.6f}")

    return Ts


# ============================================================
# Fast NN (prefer Open3D core NNS; fallback)
# ============================================================
def nn_1_core(src_xyz: np.ndarray, dst_xyz: np.ndarray):
    src = o3d.core.Tensor(src_xyz.astype(np.float32))
    dst = o3d.core.Tensor(dst_xyz.astype(np.float32))
    nns = o3d.core.nns.NearestNeighborSearch(dst)
    nns.knn_index()
    idx, dist2 = nns.knn_search(src, 1)
    idx = idx.numpy().reshape(-1).astype(np.int64)
    dist = np.sqrt(dist2.numpy().reshape(-1).astype(np.float64))
    return idx, dist


def nn_1_kdtree(src_xyz: np.ndarray, dst_pcd: o3d.geometry.PointCloud):
    kdt = o3d.geometry.KDTreeFlann(dst_pcd)
    idxs = np.zeros((src_xyz.shape[0],), dtype=np.int64)
    dists = np.zeros((src_xyz.shape[0],), dtype=np.float64)
    for i in range(src_xyz.shape[0]):
        _, ind, d2 = kdt.search_knn_vector_3d(src_xyz[i], 1)
        if len(ind) == 1:
            idxs[i] = int(ind[0])
            dists[i] = float(np.sqrt(d2[0]))
        else:
            idxs[i] = 0
            dists[i] = 1e9
    return idxs, dists


def nn_1(src_xyz: np.ndarray, dst_xyz: np.ndarray, dst_pcd_for_fallback: o3d.geometry.PointCloud):
    try:
        return nn_1_core(src_xyz, dst_xyz)
    except Exception:
        return nn_1_kdtree(src_xyz, dst_pcd_for_fallback)


# ============================================================
# Metrics: PCD↔PCD
# ============================================================
def fscore_from_dists(d_src_to_dst: np.ndarray, d_dst_to_src: np.ndarray, tau: float):
    P = float(np.mean(d_src_to_dst < tau)) if len(d_src_to_dst) > 0 else 0.0
    R = float(np.mean(d_dst_to_src < tau)) if len(d_dst_to_src) > 0 else 0.0
    F = 0.0 if (P + R) == 0 else float(2.0 * P * R / (P + R))
    return P, R, F


def normal_consistency(rec_xyz, rec_n, gt_xyz, gt_n, gt_pcd_for_fallback, tau=None):
    if rec_n is None or gt_n is None:
        return None

    idx, d = nn_1(rec_xyz, gt_xyz, gt_pcd_for_fallback)
    n0 = rec_n
    n1 = gt_n[idx]

    if tau is not None:
        m = d < tau
        if not np.any(m):
            return {"mean_abs_cos": 0.0, "mean_angle_deg": 90.0, "pct_lt_15": 0.0, "pct_lt_30": 0.0, "pairs": 0}
        n0 = n0[m]
        n1 = n1[m]

    dot = np.sum(n0 * n1, axis=1)
    abs_cos = np.clip(np.abs(dot), 0.0, 1.0)
    ang = np.degrees(np.arccos(abs_cos))
    return {
        "mean_abs_cos": float(np.mean(abs_cos)),
        "mean_angle_deg": float(np.mean(ang)),
        "pct_lt_15": float(np.mean(ang < 15.0)),
        "pct_lt_30": float(np.mean(ang < 30.0)),
        "pairs": int(len(ang)),
    }


def eval_pcd_pcd(rec_pcd, gt_pcd, taus_m, require_normals=True):
    rec_xyz = np.asarray(rec_pcd.points, dtype=np.float64)
    gt_xyz = np.asarray(gt_pcd.points, dtype=np.float64)

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
        "normal_consistency": None,
        "normal_consistency@tau": {},
    }

    for tau in taus_m:
        P, R, F = fscore_from_dists(d_rg, d_gr, tau)
        out["fscore"][tau] = {"precision": P, "recall": R, "fscore": F}

    rec_n = np.asarray(rec_pcd.normals, dtype=np.float64) if rec_pcd.has_normals() else None
    gt_n = np.asarray(gt_pcd.normals, dtype=np.float64) if gt_pcd.has_normals() else None

    if require_normals and (rec_n is None or gt_n is None):
        raise RuntimeError(
            "Normals are required for Normal Consistency, but missing.\n"
            f"rec_has_normals={rec_pcd.has_normals()}, gt_has_normals={gt_pcd.has_normals()}"
        )

    out["normal_consistency"] = normal_consistency(rec_xyz, rec_n, gt_xyz, gt_n, gt_pcd, tau=None)
    for tau in taus_m:
        out["normal_consistency@tau"][tau] = normal_consistency(rec_xyz, rec_n, gt_xyz, gt_n, gt_pcd, tau=tau)

    return out


# ============================================================
# RaycastingScene: point-to-mesh & occupancy IoU
# ============================================================
def make_scene(mesh_legacy: o3d.geometry.TriangleMesh):
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_t)
    return scene


def point_to_mesh_distance(scene: o3d.t.geometry.RaycastingScene, points_xyz: np.ndarray) -> np.ndarray:
    pts = o3d.core.Tensor(points_xyz.astype(np.float32))
    d = scene.compute_distance(pts)  # unsigned
    return d.numpy().astype(np.float64)


def occupancy_from_mesh(scene: o3d.t.geometry.RaycastingScene, points_xyz: np.ndarray) -> np.ndarray:
    pts = o3d.core.Tensor(points_xyz.astype(np.float32))
    try:
        occ = scene.compute_occupancy(pts)
        return occ.numpy().astype(bool)
    except Exception:
        sd = scene.compute_signed_distance(pts)
        return (sd.numpy().astype(np.float64) < 0.0)


# ============================================================
# Metrics: Mesh↔Mesh / PCD→Mesh / IoU
# ============================================================
def eval_mesh_mesh(rec_mesh, gt_mesh, mesh_samples, sample_method, taus_m):
    scene_gt = make_scene(gt_mesh)
    scene_rec = make_scene(rec_mesh)

    p_rec = sample_mesh(rec_mesh, mesh_samples, sample_method)
    p_gt = sample_mesh(gt_mesh, mesh_samples, sample_method)

    rec_xyz = np.asarray(p_rec.points, dtype=np.float64)
    gt_xyz = np.asarray(p_gt.points, dtype=np.float64)

    d_rec_to_gt = point_to_mesh_distance(scene_gt, rec_xyz)
    d_gt_to_rec = point_to_mesh_distance(scene_rec, gt_xyz)

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


def eval_pcd_mesh(pcd_src, mesh_dst, taus_m):
    scene = make_scene(mesh_dst)
    src_xyz = np.asarray(pcd_src.points, dtype=np.float64)
    d = point_to_mesh_distance(scene, src_xyz)

    out = {
        "mean_distance": float(np.mean(d)) if d.size else 0.0,
        "p50": float(np.median(d)) if d.size else 0.0,
        "p95": float(np.percentile(d, 95.0)) if d.size else 0.0,
        "inlier_ratio": {},
    }
    for tau in taus_m:
        out["inlier_ratio"][tau] = float(np.mean(d < tau)) if d.size else 0.0
    return out


def eval_mesh_iou(rec_mesh, gt_mesh, voxel_size, margin, max_points=5_000_000):
    scene_gt = make_scene(gt_mesh)
    scene_rec = make_scene(rec_mesh)

    bb_gt = gt_mesh.get_axis_aligned_bounding_box()
    bb_rec = rec_mesh.get_axis_aligned_bounding_box()
    mn = np.minimum(bb_gt.min_bound, bb_rec.min_bound) - margin
    mx = np.maximum(bb_gt.max_bound, bb_rec.max_bound) + margin

    extent = mx - mn
    dims = np.ceil(extent / voxel_size).astype(np.int64) + 1
    total = int(dims[0] * dims[1] * dims[2])

    if total > max_points:
        raise RuntimeError(
            f"IoU grid too large: {total} points (dims={dims}). "
            f"Increase --iou_voxel or reduce --iou_margin."
        )

    xs = mn[0] + np.arange(dims[0]) * voxel_size
    ys = mn[1] + np.arange(dims[1]) * voxel_size
    zs = mn[2] + np.arange(dims[2]) * voxel_size

    occ_gt = np.zeros(total, dtype=bool)
    occ_rec = np.zeros(total, dtype=bool)

    idx = 0
    chunk = 200_000
    for ix in range(dims[0]):
        X = xs[ix]
        Y, Z = np.meshgrid(ys, zs, indexing="ij")
        slab = np.stack([np.full_like(Y, X), Y, Z], axis=-1).reshape(-1, 3).astype(np.float64)

        s0 = 0
        while s0 < slab.shape[0]:
            s1 = min(s0 + chunk, slab.shape[0])
            pts = slab[s0:s1]

            gt_o = occupancy_from_mesh(scene_gt, pts)
            rec_o = occupancy_from_mesh(scene_rec, pts)

            n = s1 - s0
            occ_gt[idx:idx + n] = gt_o
            occ_rec[idx:idx + n] = rec_o
            idx += n
            s0 = s1

    inter = int(np.sum(occ_gt & occ_rec))
    union = int(np.sum(occ_gt | occ_rec))
    iou = 0.0 if union == 0 else float(inter / union)

    return {
        "voxel_size": float(voxel_size),
        "margin": float(margin),
        "dims": dims.tolist(),
        "total_points": total,
        "intersection": inter,
        "union": union,
        "iou": iou,
    }


# ============================================================
# Pretty print helpers
# ============================================================
def fmt_m_mm(x):
    return f"{x:.6f} m ({x*1000.0:.3f} mm)"


# ============================================================
# Batch utils
# ============================================================
def parse_id_from_recon_normal_name(path: str) -> int:
    # recon_0_000030_xyz_normal.ply
    name = os.path.basename(path)
    m = re.match(r"recon_0_(\d+)_xyz_normal\.ply$", name)
    return int(m.group(1)) if m else -1


def default_paths_from_folders(result_dir: str, gt_dir: str):
    gt_pcd = os.path.join(gt_dir, "ply", "GT_normal.ply")
    gt_mesh = os.path.join(gt_dir, "mesh", "GT_mesh.stl")
    normal_dir = os.path.join(result_dir, "normal")
    mesh_dir = os.path.join(result_dir, "mesh")
    return gt_pcd, gt_mesh, normal_dir, mesh_dir


# ============================================================
# Single evaluation (core), returns dict
# ============================================================
def evaluate_one(
    gt_pcd_path: str,
    gt_mesh_path: str,
    rec_pcd_path: str,
    rec_mesh_path: str,
    taus_m,
    args,
):
    np.random.seed(int(args.seed))

    # load
    gt_pcd = read_point_cloud(gt_pcd_path)
    rec_pcd = read_point_cloud(rec_pcd_path)
    gt_mesh = read_triangle_mesh(gt_mesh_path)
    rec_mesh = read_triangle_mesh(rec_mesh_path)

    # manual scales first
    apply_scale(gt_pcd, args.gt_scale)
    apply_scale(gt_mesh, args.gt_scale)
    apply_scale(rec_pcd, args.rec_scale)
    apply_scale(rec_mesh, args.rec_scale)

    print("============== Before autosize ==============")
    d_rec = print_pcd_stats("RECON_PCD", rec_pcd)
    d_gt = print_pcd_stats("GT_PCD", gt_pcd)

    # autosize unit to meters
    s_rec, s_gt = maybe_auto_unit_to_meters(rec_pcd, gt_pcd, enable=(not args.no_auto_unit))
    if s_rec != 1.0 or s_gt != 1.0:
        print(f"[AUTO_UNIT] apply rec *= {s_rec}, gt *= {s_gt}")
        apply_scale(rec_pcd, s_rec)
        apply_scale(rec_mesh, s_rec)
        apply_scale(gt_pcd, s_gt)
        apply_scale(gt_mesh, s_gt)

    print("============== After autosize ==============")
    d_rec2 = print_pcd_stats("RECON_PCD", rec_pcd)
    d_gt2 = print_pcd_stats("GT_PCD", gt_pcd)
    ratio = (d_gt2 / (d_rec2 + 1e-12)) if (d_gt2 > 0 and d_rec2 > 0) else np.inf
    print(f"[SCALE] gt_diag/recon_diag = {ratio:.6f}")
    print("===========================================\n")

    # auto voxel from recon
    if args.no_auto_voxel:
        voxel = float(args.voxel)
        print(f"[VOXEL] auto_voxel disabled -> voxel={voxel:.6f}")
    else:
        voxel = auto_voxel_from_recon(rec_pcd, enable=True, fallback=float(args.voxel))
        print(f"[AUTO_VOXEL] voxel={voxel:.6f} (from recon median NN)\n")

    # alignment uses downsampled PCDs
    src_ds = preprocess_pcd(o3d.geometry.PointCloud(rec_pcd), voxel)
    tgt_ds = preprocess_pcd(o3d.geometry.PointCloud(gt_pcd), voxel)

    print(f"[INFO] recon points: {len(rec_pcd.points)} (down: {len(src_ds.points)})")
    print(f"[INFO] gt    points: {len(gt_pcd.points)} (down: {len(tgt_ds.points)})")

    # PCA init
    print("[STEP] PCA 24-way init ...")
    T0, err0 = pca_init_transform(src_ds, tgt_ds)
    print(f"[PCA init] mean NN (coarse) = {err0:.6f} m")

    # ICP refine
    print("[STEP] Multi-scale ICP refine ...")
    T = icp_multiscale(rec_pcd, gt_pcd, T0, voxel_base=voxel, iters=(args.icp_iter0, args.icp_iter1))
    print("\nFinal T (GT <- Recon):\n", T)

    # apply T to recon geometries
    rec_pcd_aligned = o3d.geometry.PointCloud(rec_pcd)
    rec_pcd_aligned.transform(T)

    rec_mesh_aligned = o3d.geometry.TriangleMesh(rec_mesh)
    rec_mesh_aligned.transform(T)

    # evaluation downsample
    if args.eval_voxel and args.eval_voxel > 0:
        rec_eval = preprocess_pcd(o3d.geometry.PointCloud(rec_pcd_aligned), args.eval_voxel)
        gt_eval = preprocess_pcd(o3d.geometry.PointCloud(gt_pcd), args.eval_voxel)
    else:
        rec_eval = preprocess_pcd(o3d.geometry.PointCloud(rec_pcd_aligned), 0.0)
        gt_eval = preprocess_pcd(o3d.geometry.PointCloud(gt_pcd), 0.0)

    report = {
        "paths": {
            "gt_pcd": gt_pcd_path,
            "gt_mesh": gt_mesh_path,
            "rec_pcd": rec_pcd_path,
            "rec_mesh": rec_mesh_path,
        },
        "scales": {
            "rec_scale_manual": float(args.rec_scale),
            "gt_scale_manual": float(args.gt_scale),
            "rec_scale_auto": float(s_rec),
            "gt_scale_auto": float(s_gt),
        },
        "alignment": {
            "voxel_base": float(voxel),
            "T_gt_from_rec": T.tolist(),
            "pca_init_nn_mean": float(err0),
        },
        "metrics": {}
    }

    # A) PCD↔PCD
    pcd_report = eval_pcd_pcd(rec_eval, gt_eval, taus_m=taus_m, require_normals=True)
    report["metrics"]["pcd_pcd"] = pcd_report

    # B) Mesh↔Mesh
    if not args.skip_mesh_metrics:
        mesh_report = eval_mesh_mesh(
            rec_mesh_aligned, gt_mesh,
            mesh_samples=args.mesh_metric_samples,
            sample_method=args.mesh_sample_method,
            taus_m=taus_m
        )
        report["metrics"]["mesh_mesh"] = mesh_report

    # C) Point-to-mesh
    if not args.skip_pcd_mesh:
        recpcd_to_gtmesh = eval_pcd_mesh(rec_eval, gt_mesh, taus_m=taus_m)
        gtpcd_to_recmesh = eval_pcd_mesh(gt_eval, rec_mesh_aligned, taus_m=taus_m)
        report["metrics"]["pcd_to_mesh"] = {
            "recon_pcd_to_gt_mesh": recpcd_to_gtmesh,
            "gt_pcd_to_recon_mesh": gtpcd_to_recmesh,
        }

    # D) IoU
    if not args.skip_iou:
        iou_report = eval_mesh_iou(rec_mesh_aligned, gt_mesh, voxel_size=args.iou_voxel, margin=args.iou_margin)
        report["metrics"]["iou"] = iou_report

    # optional save aligned
    if args.save_aligned:
        out_aligned_pcd = args.save_aligned.replace("{id}", args.cur_id).replace("{type}", "pcd")
        out_aligned_mesh = args.save_aligned.replace("{id}", args.cur_id).replace("{type}", "mesh")
        os.makedirs(os.path.dirname(out_aligned_pcd), exist_ok=True)
        o3d.io.write_point_cloud(out_aligned_pcd, rec_pcd_aligned, write_ascii=False)
        o3d.io.write_triangle_mesh(out_aligned_mesh, rec_mesh_aligned, write_ascii=False)
        report["outputs"] = {
            "aligned_rec_pcd": out_aligned_pcd,
            "aligned_rec_mesh": out_aligned_mesh,
        }

    # optional vis (single case only recommended)
    if args.vis:
        gt_vis = o3d.geometry.PointCloud(gt_pcd)
        rec_vis = o3d.geometry.PointCloud(rec_pcd_aligned)
        if not gt_vis.has_colors():
            gt_vis.paint_uniform_color([0.2, 0.8, 0.2])
        if not rec_vis.has_colors():
            rec_vis.paint_uniform_color([0.9, 0.2, 0.2])
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([axis, gt_vis, rec_vis], width=1400, height=900)

    return report


# ============================================================
# Main (batch)
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--result_dir",
        default="/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/result/green_cube_00_001",
        help="Folder containing normal/ and mesh/ subfolders",
    )
    ap.add_argument(
        "--gt_dir",
        default="/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/GT_data/cube/GT",
        help="GT folder (contains mesh/GT_mesh.stl and ply/GT_normal.ply)",
    )

    # evaluate one id only (optional)
    ap.add_argument("--id", type=int, default=-1, help="Only evaluate recon_0_{id:06d} if set >=0")

    # sampling for mesh metrics
    ap.add_argument("--mesh_metric_samples", type=int, default=200000)
    ap.add_argument("--mesh_sample_method", choices=["uniform", "poisson"], default="poisson")

    # autosize defaults ON
    ap.add_argument("--no_auto_unit", action="store_true")
    ap.add_argument("--no_auto_voxel", action="store_true")

    # manual extra scales
    ap.add_argument("--rec_scale", type=float, default=1.0)
    ap.add_argument("--gt_scale", type=float, default=1.0)

    # alignment base voxel (used if auto_voxel disabled)
    ap.add_argument("--voxel", type=float, default=0.003, help="base voxel for alignment (m) when auto_voxel disabled")

    # ICP iters
    ap.add_argument("--icp_iter0", type=int, default=60)
    ap.add_argument("--icp_iter1", type=int, default=90)

    # evaluation downsample voxel
    ap.add_argument("--eval_voxel", type=float, default=0.002, help="downsample voxel for PCD metrics (m), 0 disables")

    # thresholds
    ap.add_argument("--thresholds_mm", type=str, default="2,5,10")

    # IoU
    ap.add_argument("--iou_voxel", type=float, default=0.005)
    ap.add_argument("--iou_margin", type=float, default=0.01)

    # batch speed switches
    ap.add_argument("--skip_iou", action="store_true", help="Skip IoU (very slow in batch)")
    ap.add_argument("--skip_mesh_metrics", action="store_true", help="Skip mesh↔mesh surface metrics (slow)")
    ap.add_argument("--skip_pcd_mesh", action="store_true", help="Skip point-to-mesh metrics")

    # outputs
    ap.add_argument("--out_dir", type=str, default="", help="Default: result_dir/eval")
    ap.add_argument("--save_aligned", type=str, default="",
                    help="If set, save aligned recon. Use template with {id} and {type}. "
                         "Example: result_dir/eval/aligned/recon_0_{id}_{type}.ply")
    ap.add_argument("--vis", action="store_true", help="Visualize (only recommended with --id)")

    # reproducibility
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    taus_m = [float(x.strip()) / 1000.0 for x in args.thresholds_mm.split(",") if x.strip()]

    result_dir = os.path.expanduser(args.result_dir)
    gt_dir = os.path.expanduser(args.gt_dir)
    gt_pcd_path, gt_mesh_path, normal_dir, mesh_dir = default_paths_from_folders(result_dir, gt_dir)

    if not os.path.exists(gt_pcd_path):
        raise RuntimeError(f"GT pcd not found: {gt_pcd_path}")
    if not os.path.exists(gt_mesh_path):
        raise RuntimeError(f"GT mesh not found: {gt_mesh_path}")
    if not os.path.isdir(normal_dir):
        raise RuntimeError(f"normal dir not found: {normal_dir}")
    if not os.path.isdir(mesh_dir):
        raise RuntimeError(f"mesh dir not found: {mesh_dir}")

    out_dir = os.path.expanduser(args.out_dir) if args.out_dir else os.path.join(result_dir, "eval")
    metrics_dir = os.path.join(out_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # gather recon normal files
    normal_files = sorted(glob.glob(os.path.join(normal_dir, "recon_0_*_xyz_normal.ply")),
                          key=parse_id_from_recon_normal_name)

    if args.id >= 0:
        target = os.path.join(normal_dir, f"recon_0_{args.id:06d}_xyz_normal.ply")
        if target not in normal_files:
            if not os.path.exists(target):
                raise RuntimeError(f"Requested id not found: {target}")
            normal_files = [target]
        else:
            normal_files = [target]

    if len(normal_files) == 0:
        raise RuntimeError(f"No recon normal ply found in: {normal_dir}")

    # summary csv
    summary_csv = os.path.join(out_dir, "summary.csv")
    summary_fields = [
        "id",
        "rec_points",
        "accuracy_mm",
        "completeness_mm",
        "chamfer_l1_mm",
        "f@2mm",
        "f@5mm",
        "f@10mm",
        "normal_mean_angle_deg",
        "mesh_chamfer_l1_mm",
        "iou",
        "metrics_json",
    ]

    rows = []

    print("====================================================")
    print(f"[INFO] result_dir : {result_dir}")
    print(f"[INFO] gt_dir     : {gt_dir}")
    print(f"[INFO] GT_pcd     : {gt_pcd_path}")
    print(f"[INFO] GT_mesh    : {gt_mesh_path}")
    print(f"[INFO] normal_dir : {normal_dir}")
    print(f"[INFO] mesh_dir   : {mesh_dir}")
    print(f"[INFO] out_dir    : {out_dir}")
    print(f"[INFO] cases      : {len(normal_files)}")
    print(f"[INFO] skip_iou={args.skip_iou}, skip_mesh_metrics={args.skip_mesh_metrics}, skip_pcd_mesh={args.skip_pcd_mesh}")
    print("====================================================\n")

    for i, rec_pcd_path in enumerate(normal_files, 1):
        rid = parse_id_from_recon_normal_name(rec_pcd_path)
        if rid < 0:
            print(f"[WARN] cannot parse id, skip: {rec_pcd_path}")
            continue

        rec_mesh_path = os.path.join(mesh_dir, f"recon_0_{rid:06d}_mesh.ply")
        if not os.path.exists(rec_mesh_path):
            print(f"[WARN] mesh missing for id={rid:06d}, skip this case:\n  need: {rec_mesh_path}")
            continue

        args.cur_id = f"{rid:06d}"

        print(f"\n================= [{i}/{len(normal_files)}] ID={rid:06d} =================")
        print(f"REC_PCD : {rec_pcd_path}")
        print(f"REC_MESH: {rec_mesh_path}")

        try:
            report = evaluate_one(
                gt_pcd_path=gt_pcd_path,
                gt_mesh_path=gt_mesh_path,
                rec_pcd_path=rec_pcd_path,
                rec_mesh_path=rec_mesh_path,
                taus_m=taus_m,
                args=args,
            )
        except Exception as e:
            print(f"[ERROR] evaluation failed for id={rid:06d}: {e}")
            continue

        # save json
        json_path = os.path.join(metrics_dir, f"recon_0_{rid:06d}_metrics.json")
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)

        # build summary row
        rec_points = int(len(o3d.io.read_point_cloud(rec_pcd_path).points))
        pcdm = report["metrics"]["pcd_pcd"]
        f2 = pcdm["fscore"].get(0.002, {}).get("fscore", 0.0)
        f5 = pcdm["fscore"].get(0.005, {}).get("fscore", 0.0)
        f10 = pcdm["fscore"].get(0.010, {}).get("fscore", 0.0)
        nca = pcdm["normal_consistency"]["mean_angle_deg"] if pcdm.get("normal_consistency") else 90.0

        mesh_cd_l1 = ""
        if (not args.skip_mesh_metrics) and ("mesh_mesh" in report["metrics"]):
            mesh_cd_l1 = report["metrics"]["mesh_mesh"]["chamfer_l1"] * 1000.0  # mm

        iou_val = ""
        if (not args.skip_iou) and ("iou" in report["metrics"]):
            iou_val = report["metrics"]["iou"]["iou"]

        row = {
            "id": f"{rid:06d}",
            "rec_points": rec_points,
            "accuracy_mm": pcdm["accuracy_mean"] * 1000.0,
            "completeness_mm": pcdm["completeness_mean"] * 1000.0,
            "chamfer_l1_mm": pcdm["chamfer_l1"] * 1000.0,
            "f@2mm": f2,
            "f@5mm": f5,
            "f@10mm": f10,
            "normal_mean_angle_deg": nca,
            "mesh_chamfer_l1_mm": mesh_cd_l1,
            "iou": iou_val,
            "metrics_json": json_path,
        }
        rows.append(row)

        print(f"[SAVED] {json_path}")

    # write summary csv
    os.makedirs(out_dir, exist_ok=True)
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n[Done] Batch evaluation finished.")
    print(f"[SUMMARY] {summary_csv}")
    print(f"[METRICS]  {metrics_dir}")


if __name__ == "__main__":
    import glob
    main()
