#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import open3d as o3d


def to_cpu(geom):
    if hasattr(geom, "cpu"):
        try:
            return geom.cpu()
        except Exception:
            return geom
    return geom


def read_pcd_or_mesh(path: str):
    ext = os.path.splitext(path)[1].lower()

    # mesh-like extensions
    if ext in [".obj", ".stl", ".off", ".gltf", ".glb"]:
        mesh = to_cpu(o3d.io.read_triangle_mesh(path))
        if mesh.is_empty():
            raise RuntimeError(f"Failed to read mesh: {path}")
        return None, mesh

    # .ply could be mesh or pcd; try mesh first but accept only if triangles exist
    if ext == ".ply":
        mesh = to_cpu(o3d.io.read_triangle_mesh(path))
        if (mesh is not None) and (not mesh.is_empty()) and (len(mesh.triangles) > 0):
            return None, mesh

    # fallback pcd
    pcd = to_cpu(o3d.io.read_point_cloud(path))
    if pcd.is_empty():
        raise RuntimeError(f"Failed to read point cloud (or empty): {path}")
    return pcd, None


def sample_mesh(mesh: o3d.geometry.TriangleMesh, n: int, method: str):
    mesh = to_cpu(mesh)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    if method == "poisson":
        return to_cpu(mesh.sample_points_poisson_disk(number_of_points=int(n)))
    return to_cpu(mesh.sample_points_uniformly(number_of_points=int(n)))


def remove_non_finite(pcd):
    if pcd.is_empty():
        return pcd
    ret = pcd.remove_non_finite_points()
    if isinstance(ret, tuple):
        return ret[0]
    return pcd


def estimate_normals(pcd, radius, max_nn=30):
    if pcd.is_empty():
        return pcd
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius), max_nn=int(max_nn)))
    pcd.normalize_normals()
    return pcd


def preprocess(pcd, voxel):
    pcd = to_cpu(pcd)
    pcd = remove_non_finite(pcd)
    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(float(voxel))
    return pcd


def median_nn_distance(pcd, sample_n=5000):
    pts = np.asarray(pcd.points)
    if pts.shape[0] < 10:
        return 0.0
    n = min(int(sample_n), pts.shape[0])
    idx = np.random.choice(pts.shape[0], n, replace=False)
    q = pts[idx]

    kdt = o3d.geometry.KDTreeFlann(pcd)
    dists = []
    for i in range(q.shape[0]):
        _, inds, ds = kdt.search_knn_vector_3d(q[i], 2)
        if len(ds) >= 2:
            dists.append(np.sqrt(ds[1]))
    if len(dists) == 0:
        return 0.0
    return float(np.median(np.asarray(dists)))


def pcd_stats(name, pcd):
    aabb = pcd.get_axis_aligned_bounding_box()
    ext = aabb.get_extent()
    diag = float(np.linalg.norm(ext))
    cen = aabb.get_center()
    print(f"[{name}] N={len(pcd.points)}  AABB extent={ext}  diag={diag:.6f}  center={cen}")
    return np.array(ext, dtype=np.float64), diag, np.array(cen, dtype=np.float64)


def make_pca_frame(pcd):
    pts = np.asarray(pcd.points)
    c = pts.mean(axis=0)
    X = pts - c[None, :]
    # covariance
    C = (X.T @ X) / max(1, X.shape[0])
    w, V = np.linalg.eigh(C)  # ascending
    V = V[:, np.argsort(w)[::-1]]  # descending eigenvalues
    # ensure right-handed (det=+1)
    if np.linalg.det(V) < 0:
        V[:, 2] *= -1
    return c, V


def all_right_handed_axis_mats(V):
    """
    Generate 24 right-handed rotation matrices from PCA axes:
    permutations (6) * sign flips (4) with det=+1.
    """
    mats = []
    perms = [
        (0,1,2), (0,2,1),
        (1,0,2), (1,2,0),
        (2,0,1), (2,1,0),
    ]
    signs = [
        ( 1, 1, 1),
        ( 1,-1,-1),
        (-1, 1,-1),
        (-1,-1, 1),
    ]
    for p in perms:
        P = V[:, p]
        for s in signs:
            R = P @ np.diag(s)
            if np.linalg.det(R) < 0:
                continue
            mats.append(R)
    return mats  # should be 24


def nn_mean(src_pcd, tgt_pcd, sample_n=6000):
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


def pca_init_transform(src, tgt):
    cs, Vs = make_pca_frame(src)
    ct, Vt = make_pca_frame(tgt)

    candidates = all_right_handed_axis_mats(Vs)
    best_T = np.eye(4)
    best_err = 1e18

    # target basis fixed
    Rt = Vt

    for Rsrc in candidates:
        # map src -> tgt: x_t = Rt * Rsrc^T * (x_s - cs) + ct
        R = Rt @ Rsrc.T
        t = ct - R @ cs
        T = np.eye(4)
        T[:3,:3] = R
        T[:3, 3] = t

        src_tmp = o3d.geometry.PointCloud(src)
        src_tmp.transform(T)
        err = nn_mean(src_tmp, tgt, sample_n=6000)
        if err < best_err:
            best_err = err
            best_T = T

    return best_T, best_err


def icp_multiscale(src_full, tgt_full, init_T, voxel_base, iters=(60, 80)):
    # build pyramids (coarse -> fine)
    voxels = [2.0 * voxel_base, 1.0 * voxel_base]
    Ts = init_T.copy()

    for lvl, (vx, it) in enumerate(zip(voxels, iters)):
        src = preprocess(src_full, vx)
        tgt = preprocess(tgt_full, vx)

        # normals for point-to-plane
        nr = max(2.0 * vx, 1e-3)
        src = estimate_normals(src, nr)
        tgt = estimate_normals(tgt, nr)

        max_corr = 2.5 * vx

        # robust kernel if available
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


def compute_metrics(recon, gt, thresholds_m):
    d_rg = np.asarray(recon.compute_point_cloud_distance(gt), dtype=np.float64)
    d_gr = np.asarray(gt.compute_point_cloud_distance(recon), dtype=np.float64)

    chamfer_l1 = float(d_rg.mean() + d_gr.mean())
    chamfer_l2 = float((d_rg**2).mean() + (d_gr**2).mean())
    rms_sym = float(np.sqrt(0.5 * ((d_rg**2).mean() + (d_gr**2).mean())))
    mean_sym = float(0.5 * (d_rg.mean() + d_gr.mean()))
    hausdorff = float(max(d_rg.max(initial=0.0), d_gr.max(initial=0.0)))

    out = {
        "chamfer_L1_m": chamfer_l1,
        "chamfer_L2_m2": chamfer_l2,
        "mean_sym_m": mean_sym,
        "rms_sym_m": rms_sym,
        "hausdorff_m": hausdorff,
        "F": {}
    }

    for th in thresholds_m:
        th = float(th)
        P = float((d_rg < th).mean()) if d_rg.size else 0.0
        R = float((d_gr < th).mean()) if d_gr.size else 0.0
        F = 0.0 if (P + R) < 1e-12 else float(2.0 * P * R / (P + R))
        out["F"][th] = (P, R, F)

    return out


def print_report(m):
    def mm(x): return x * 1000.0
    print("\n================ Evaluation Report ================")
    print(f"Chamfer L1  : {m['chamfer_L1_m']:.6f} m   ({mm(m['chamfer_L1_m']):.3f} mm)")
    print(f"Chamfer L2  : {m['chamfer_L2_m2']:.10f} m^2")
    print(f"Mean Sym    : {m['mean_sym_m']:.6f} m   ({mm(m['mean_sym_m']):.3f} mm)")
    print(f"RMS  Sym    : {m['rms_sym_m']:.6f} m   ({mm(m['rms_sym_m']):.3f} mm)")
    print(f"Hausdorff   : {m['hausdorff_m']:.6f} m   ({mm(m['hausdorff_m']):.3f} mm)")
    for th, (P, R, F) in m["F"].items():
        print(f"Fscore@{int(th*1000)}mm:  P={P:.4f}  R={R:.4f}  F={F:.4f}")
    print("===================================================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True)
    ap.add_argument("--gt", required=True)

    ap.add_argument("--gt_is_mesh_sample_n", type=int, default=200000)
    ap.add_argument("--gt_sample_method", choices=["uniform", "poisson"], default="poisson")

    # scale handling
    ap.add_argument("--recon_scale", type=float, default=1.0, help="multiply recon coordinates by this factor")
    ap.add_argument("--gt_scale", type=float, default=1.0, help="multiply gt coordinates by this factor")
    ap.add_argument("--auto_scale", action="store_true",
                    help="auto scale GT to match recon diag if mismatch is large (m vs mm)")

    # voxel
    ap.add_argument("--voxel", type=float, default=0.003, help="base voxel for alignment (m)")
    ap.add_argument("--auto_voxel", action="store_true",
                    help="set voxel = 5 * median_nn(recon) after scaling (recommended)")

    # evaluation
    ap.add_argument("--eval_voxel", type=float, default=0.002, help="downsample voxel for metric eval (m), 0 disables")
    ap.add_argument("--thresholds_mm", type=str, default="2,5,10")

    ap.add_argument("--save_aligned", type=str, default="")
    ap.add_argument("--save_T", type=str, default="")
    ap.add_argument("--vis", action="store_true")

    args = ap.parse_args()

    thresholds_m = [float(x)/1000.0 for x in args.thresholds_mm.split(",") if x.strip()]

    # load recon
    recon_pcd, recon_mesh = read_pcd_or_mesh(args.recon)
    if recon_mesh is not None:
        recon_pcd = sample_mesh(recon_mesh, args.gt_is_mesh_sample_n, args.gt_sample_method)
    recon_pcd = to_cpu(recon_pcd)

    # load gt
    gt_pcd, gt_mesh = read_pcd_or_mesh(args.gt)
    if gt_mesh is not None:
        gt_pcd = sample_mesh(gt_mesh, args.gt_is_mesh_sample_n, args.gt_sample_method)
    gt_pcd = to_cpu(gt_pcd)

    # apply manual scales
    if args.recon_scale != 1.0:
        recon_pcd.scale(float(args.recon_scale), center=(0,0,0))
    if args.gt_scale != 1.0:
        gt_pcd.scale(float(args.gt_scale), center=(0,0,0))

    # print stats + detect scale mismatch
    _, recon_diag, _ = pcd_stats("RECON", recon_pcd)
    _, gt_diag, _ = pcd_stats("GT", gt_pcd)

    if recon_diag > 0 and gt_diag > 0:
        ratio = gt_diag / recon_diag
        print(f"[SCALE] gt_diag/recon_diag = {ratio:.6f}")
        if args.auto_scale and (ratio > 50.0 or ratio < 0.02):
            # scale GT to match recon
            s = recon_diag / gt_diag
            gt_pcd.scale(float(s), center=(0,0,0))
            print(f"[AUTO_SCALE] Applied gt_scale *= {s:.12f} (to match recon diag)")
            _, recon_diag, _ = pcd_stats("RECON", recon_pcd)
            _, gt_diag, _ = pcd_stats("GT (scaled)", gt_pcd)

    # auto voxel by recon median NN
    voxel = float(args.voxel)
    if args.auto_voxel:
        tmp = preprocess(recon_pcd, 0.0)
        mnn = median_nn_distance(tmp, sample_n=5000)
        if mnn > 0:
            voxel = max(5.0 * mnn, 1e-4)
            print(f"[AUTO_VOXEL] median_nn={mnn:.6f} -> voxel={voxel:.6f}")

    # downsample a bit for init search stability
    src_ds = preprocess(recon_pcd, voxel)
    tgt_ds = preprocess(gt_pcd, voxel)

    print(f"[INFO] recon points: {len(recon_pcd.points)} (down: {len(src_ds.points)})")
    print(f"[INFO] gt    points: {len(gt_pcd.points)} (down: {len(tgt_ds.points)})")

    # PCA init (robust for large rotations + sparse features)
    print("[STEP] PCA 24-way init ...")
    T0, err0 = pca_init_transform(src_ds, tgt_ds)
    print(f"[PCA init] mean NN (coarse) = {err0:.6f} m")

    # ICP refine (multi-scale)
    print("[STEP] Multi-scale ICP refine ...")
    T = icp_multiscale(recon_pcd, gt_pcd, T0, voxel_base=voxel, iters=(60, 90))
    print("\nFinal T (GT <- Recon):\n", T)

    recon_aligned = o3d.geometry.PointCloud(recon_pcd)
    recon_aligned.transform(T)

    # evaluation clouds
    if args.eval_voxel and args.eval_voxel > 0:
        recon_eval = preprocess(recon_aligned, float(args.eval_voxel))
        gt_eval = preprocess(gt_pcd, float(args.eval_voxel))
    else:
        recon_eval = preprocess(recon_aligned, 0.0)
        gt_eval = preprocess(gt_pcd, 0.0)

    # metrics
    metrics = compute_metrics(recon_eval, gt_eval, thresholds_m)
    print_report(metrics)


    if args.vis:
        gt_vis = o3d.geometry.PointCloud(gt_pcd)
        rec_vis = o3d.geometry.PointCloud(recon_aligned)
        if not gt_vis.has_colors():
            gt_vis.paint_uniform_color([0.2, 0.8, 0.2])
        if not rec_vis.has_colors():
            rec_vis.paint_uniform_color([0.9, 0.2, 0.2])
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([axis, gt_vis, rec_vis], width=1400, height=900)


if __name__ == "__main__":
    main()
