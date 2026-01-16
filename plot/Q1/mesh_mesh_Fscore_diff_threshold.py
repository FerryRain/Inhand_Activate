#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Align two meshes (recon vs GT) and compute Mesh--Mesh F-score curve over tau in [0, 10] mm.

Pipeline:
1) Load meshes, optional auto unit (mm<->m) + optional manual scales
2) Sample point clouds from meshes (for alignment)
3) PCA 24-way init + multiscale ICP (point-to-point coarse, point-to-plane fine)
4) Apply T to recon mesh
5) Mesh--Mesh distances via Open3D RaycastingScene
6) Sweep tau_mm in [tau_mm_min, tau_mm_max] with step tau_mm_step -> Precision/Recall/F-score
7) Save CSV + figure

Example:
python mesh_mesh_fscore_vs_tau_align.py \
  --rec_mesh /path/to/recon.ply \
  --gt_mesh  /path/to/GT_mesh.stl \
  --out_csv  best_vs_tau.csv \
  --out_fig  fig_q1_right.png \
  --save_aligned_mesh aligned_recon.ply
"""

import argparse
import csv
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


# ----------------------------
# IO + basic utils
# ----------------------------
def read_triangle_mesh(path: str) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise RuntimeError(f"Mesh has no triangles: {path}")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    return mesh


def apply_scale(geom, s: float):
    if s == 1.0:
        return
    geom.scale(float(s), center=(0, 0, 0))


def mesh_diag(mesh: o3d.geometry.TriangleMesh) -> float:
    aabb = mesh.get_axis_aligned_bounding_box()
    ext = np.asarray(aabb.get_extent(), dtype=np.float64)
    return float(np.linalg.norm(ext))


def maybe_auto_unit_to_meters_from_diag(rec_mesh, gt_mesh, enable=True):
    """
    Heuristic unit fix:
    - If one is ~1000x larger, scale that one by 0.001.
    - If both huge (>5m diag), scale both by 0.001.
    """
    if not enable:
        return 1.0, 1.0
    dr = mesh_diag(rec_mesh)
    dg = mesh_diag(gt_mesh)
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


def sample_mesh(mesh: o3d.geometry.TriangleMesh, n: int, method: str = "uniform") -> o3d.geometry.PointCloud:
    if method == "poisson":
        return mesh.sample_points_poisson_disk(number_of_points=int(n))
    return mesh.sample_points_uniformly(number_of_points=int(n))


def preprocess_pcd(pcd: o3d.geometry.PointCloud, voxel: float) -> o3d.geometry.PointCloud:
    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(float(voxel))
    return pcd


# ----------------------------
# PCA init (24-way)
# ----------------------------
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


# ----------------------------
# Multiscale ICP
# ----------------------------
def estimate_normals_for_icp(pcd: o3d.geometry.PointCloud, radius: float, max_nn=30):
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius), max_nn=int(max_nn)))
    pcd.normalize_normals()
    return pcd


def icp_multiscale(src_full, tgt_full, init_T, voxel_base, iters=(60, 90, 120), verbose=False):
    voxels = [4.0 * voxel_base, 2.0 * voxel_base, 1.0 * voxel_base]
    Ts = init_T.copy()

    for lvl, (vx, it) in enumerate(zip(voxels, iters)):
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


# ----------------------------
# Mesh--Mesh distances (Raycasting)
# ----------------------------
def make_scene(mesh_legacy: o3d.geometry.TriangleMesh):
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_t)
    return scene


def point_to_mesh_distance(scene: o3d.t.geometry.RaycastingScene, points_xyz: np.ndarray) -> np.ndarray:
    pts = o3d.core.Tensor(points_xyz.astype(np.float32))
    d = scene.compute_distance(pts)
    return d.numpy().astype(np.float64)


def fscore_from_dists(d_src_to_dst: np.ndarray, d_dst_to_src: np.ndarray, tau: float):
    P = float(np.mean(d_src_to_dst < tau)) if d_src_to_dst.size else 0.0
    R = float(np.mean(d_dst_to_src < tau)) if d_dst_to_src.size else 0.0
    F = 0.0 if (P + R) == 0 else float(2.0 * P * R / (P + R))
    return P, R, F


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec_mesh", type=str, default="/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/result/green_cube_00_001/mesh/recon_0_000150_mesh.ply", help="Reconstructed mesh path (ply/obj/stl...)")
    ap.add_argument("--gt_mesh", type=str, default="/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/GT_data/cube_01/GT/mesh/GT_mesh.stl", help="Ground-truth mesh path (ply/obj/stl...)")

    ap.add_argument("--rec_scale", type=float, default=1.0)
    ap.add_argument("--gt_scale", type=float, default=1.0)
    ap.add_argument("--no_auto_unit", action="store_true", help="Disable auto mm<->m heuristic")

    ap.add_argument("--align_samples", type=int, default=30000, help="Points sampled from each mesh for alignment")
    ap.add_argument("--align_sample_method", choices=["uniform", "poisson"], default="uniform")

    ap.add_argument("--icp_voxel", type=float, default=0.003, help="Base voxel (meters) for ICP multiscale")
    ap.add_argument("--icp_iter0", type=int, default=60)
    ap.add_argument("--icp_iter1", type=int, default=90)
    ap.add_argument("--icp_iter2", type=int, default=120)
    ap.add_argument("--verbose_icp", action="store_true")

    ap.add_argument("--metric_samples", type=int, default=20000, help="Points for mesh-mesh distance eval")
    ap.add_argument("--metric_sample_method", choices=["uniform", "poisson"], default="uniform")

    ap.add_argument("--tau_mm_min", type=float, default=0.0)
    ap.add_argument("--tau_mm_max", type=float, default=20.0)
    ap.add_argument("--tau_mm_step", type=float, default=0.5)

    ap.add_argument("--out_csv", type=str, default="mesh_mesh_fscore_vs_tau.csv")
    ap.add_argument("--out_fig", type=str, default="fig_q1_right.png")
    ap.add_argument("--save_aligned_mesh", type=str, default="", help="Optional: save aligned recon mesh")

    # plot style
    ap.add_argument("--label_fs", type=int, default=10)
    ap.add_argument("--tick_fs", type=int, default=9)
    ap.add_argument("--legend_fs", type=int, default=8)

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    np.random.seed(args.seed)

    # ---- load meshes
    rec_mesh = read_triangle_mesh(args.rec_mesh)
    gt_mesh = read_triangle_mesh(args.gt_mesh)

    # ---- manual scales
    apply_scale(rec_mesh, args.rec_scale)
    apply_scale(gt_mesh, args.gt_scale)

    # ---- auto unit (mm<->m)
    s_rec, s_gt = maybe_auto_unit_to_meters_from_diag(rec_mesh, gt_mesh, enable=(not args.no_auto_unit))
    apply_scale(rec_mesh, s_rec)
    apply_scale(gt_mesh, s_gt)

    print(f"[UNIT] rec_scale_total={args.rec_scale * s_rec:.6g}, gt_scale_total={args.gt_scale * s_gt:.6g}")

    # ---- sample point clouds for alignment
    rec_pcd_full = sample_mesh(rec_mesh, args.align_samples, args.align_sample_method)
    gt_pcd_full = sample_mesh(gt_mesh, args.align_samples, args.align_sample_method)

    # ---- PCA init
    T0, err0 = pca_init_transform(rec_pcd_full, gt_pcd_full)
    print(f"[PCA_INIT] nn_mean={err0:.6f}")

    # ---- ICP multiscale refinement
    iters = (args.icp_iter0, args.icp_iter1, args.icp_iter2)
    T = icp_multiscale(
        rec_pcd_full, gt_pcd_full, T0,
        voxel_base=float(args.icp_voxel),
        iters=iters,
        verbose=args.verbose_icp
    )
    print("[ALIGN] Done. Applying transform to recon mesh.")

    rec_mesh_aligned = o3d.geometry.TriangleMesh(rec_mesh)
    rec_mesh_aligned.transform(T)

    if args.save_aligned_mesh:
        o3d.io.write_triangle_mesh(args.save_aligned_mesh, rec_mesh_aligned, write_ascii=False)
        print(f"[SAVED] aligned mesh -> {args.save_aligned_mesh}")

    # ---- prepare raycasting scenes
    scene_gt = make_scene(gt_mesh)
    scene_rec = make_scene(rec_mesh_aligned)

    # ---- sample points for metric
    rec_eval_pcd = sample_mesh(rec_mesh_aligned, args.metric_samples, args.metric_sample_method)
    gt_eval_pcd = sample_mesh(gt_mesh, args.metric_samples, args.metric_sample_method)
    rec_xyz = np.asarray(rec_eval_pcd.points, dtype=np.float64)
    gt_xyz = np.asarray(gt_eval_pcd.points, dtype=np.float64)

    d_rec_to_gt = point_to_mesh_distance(scene_gt, rec_xyz)
    d_gt_to_rec = point_to_mesh_distance(scene_rec, gt_xyz)

    # ---- sweep tau
    if args.tau_mm_step <= 0:
        raise ValueError("--tau_mm_step must be > 0")
    taus_mm = np.arange(args.tau_mm_min, args.tau_mm_max + 1e-12, args.tau_mm_step, dtype=float)
    taus_m = taus_mm / 1000.0

    rows = []
    for tau_mm, tau_m in zip(taus_mm, taus_m):
        P, R, F = fscore_from_dists(d_rec_to_gt, d_gt_to_rec, float(tau_m))
        rows.append((float(tau_mm), float(F), float(P), float(R)))

    # ---- save CSV
    out_csv = args.out_csv
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau_mm", "fscore", "precision", "recall"])
        for tau_mm, F, P, R in rows:
            w.writerow([f"{tau_mm:.3f}", f"{F:.6f}", f"{P:.6f}", f"{R:.6f}"])
    print(f"[SAVED] {out_csv}")

    # ---- plot (right figure style)
    plt.figure(figsize=(3.25, 3.0), dpi=160)
    ax = plt.gca()

    F_arr = np.array([r[1] for r in rows], dtype=float)
    ax.plot(taus_mm, F_arr, color="black", lw=1.8, linestyle="-", label="Ours")

    ax.set_xlim(args.tau_mm_min, args.tau_mm_max)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Threshold $\tau$ (mm)", fontsize=args.label_fs)
    ax.set_ylabel("Shape F-Score", fontsize=args.label_fs)
    ax.tick_params(axis="both", which="major", labelsize=args.tick_fs)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc="lower right", frameon=True, fancybox=True, framealpha=1.0, fontsize=args.legend_fs)

    plt.tight_layout()
    out_fig = args.out_fig
    plt.savefig(out_fig, bbox_inches="tight")
    print(f"[SAVED] {out_fig}")


if __name__ == "__main__":
    main()
