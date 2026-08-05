#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


DEFAULT_PRED_ROOT = "/home/ferry/data/Code2/Research/TRELLIS/TRELLIS.2/output_3d"
DEFAULT_GT_ROOT = "/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/GT_data"
DEFAULT_OUT_DIR = "/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/Compute_Metric/trellis_mesh_eval"

DEFAULT_MAPPING = {
    "cube_obj_01": "cube",
    "cube_obj_02": "corner",
    "tetraprism": "cross",
}


def read_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise RuntimeError(f"Mesh has no triangles: {path}")
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    return mesh


def resolve_pred_mesh(pred_root: Path, pred_name: str) -> Path:
    candidates = [
        pred_root / f"{pred_name}.ply",
        pred_root / f"mesh_{pred_name}.ply",
        pred_root / pred_name / f"{pred_name}.ply",
        pred_root / pred_name / f"{pred_name}_output.ply",
        pred_root / pred_name / "mesh.ply",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find TRELLIS mesh for {pred_name}; tried: {candidates}")


def resolve_gt_mesh(gt_root: Path, gt_name: str) -> Path:
    candidates = [
        gt_root / gt_name / "GT" / "mesh" / "GT_mesh.stl",
        gt_root / gt_name / "GT" / "ply" / "GT.ply",
        gt_root / gt_name / "GT" / "ply" / "GT_cube.ply",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find GT mesh for {gt_name}; tried: {candidates}")


def mesh_diag(mesh: o3d.geometry.TriangleMesh) -> float:
    extent = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent(), dtype=np.float64)
    return float(np.linalg.norm(extent))


def unit_scale_from_mode(mesh: o3d.geometry.TriangleMesh, mode: str) -> float:
    if mode == "m":
        return 1.0
    if mode == "mm":
        return 0.001
    diag = mesh_diag(mesh)
    return 0.001 if diag > 5.0 else 1.0


def sample_mesh(mesh: o3d.geometry.TriangleMesh, n: int, method: str) -> o3d.geometry.PointCloud:
    if method == "poisson":
        return mesh.sample_points_poisson_disk(number_of_points=int(n))
    return mesh.sample_points_uniformly(number_of_points=int(n))


def preprocess_pcd(pcd: o3d.geometry.PointCloud, voxel: float) -> o3d.geometry.PointCloud:
    if voxel > 0:
        pcd = pcd.voxel_down_sample(float(voxel))
    ret = pcd.remove_non_finite_points()
    if isinstance(ret, tuple):
        pcd = ret[0]
    return pcd


def median_nn_distance(pcd: o3d.geometry.PointCloud, sample_n: int = 5000) -> float:
    pts = np.asarray(pcd.points)
    if pts.shape[0] < 10:
        return 0.0
    n = min(int(sample_n), pts.shape[0])
    idx = np.random.choice(pts.shape[0], n, replace=False)
    query = pts[idx]
    tree = o3d.geometry.KDTreeFlann(pcd)
    dists = []
    for point in query:
        _, _, dist2 = tree.search_knn_vector_3d(point, 2)
        if len(dist2) >= 2:
            dists.append(np.sqrt(dist2[1]))
    return float(np.median(np.asarray(dists))) if dists else 0.0


def auto_voxel_from_recon(recon_pcd: o3d.geometry.PointCloud, fallback: float = 0.003) -> float:
    tmp = preprocess_pcd(o3d.geometry.PointCloud(recon_pcd), voxel=0.0)
    mnn = median_nn_distance(tmp, sample_n=5000)
    if mnn <= 0:
        return float(fallback)
    return float(max(5.0 * mnn, 1e-4))


def make_pca_frame(pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(pcd.points, dtype=np.float64)
    center = pts.mean(axis=0)
    x = pts - center[None, :]
    cov = (x.T @ x) / max(1, x.shape[0])
    eigvals, eigvecs = np.linalg.eigh(cov)
    frame = eigvecs[:, np.argsort(eigvals)[::-1]]
    if np.linalg.det(frame) < 0:
        frame[:, 2] *= -1
    return center, frame


def right_handed_axis_candidates(frame: np.ndarray) -> List[np.ndarray]:
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    signs = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    mats = []
    for perm in perms:
        p = frame[:, perm]
        for sign in signs:
            r = p @ np.diag(sign)
            if np.linalg.det(r) > 0:
                mats.append(r)
    return mats


def nn_mean(src: o3d.geometry.PointCloud, dst: o3d.geometry.PointCloud, sample_n: int = 6000) -> float:
    src_pts = np.asarray(src.points, dtype=np.float64)
    if src_pts.size == 0:
        return float("inf")
    n = min(sample_n, src_pts.shape[0])
    idx = np.random.choice(src_pts.shape[0], n, replace=False)
    tree = o3d.geometry.KDTreeFlann(dst)
    dists = []
    for point in src_pts[idx]:
        _, _, dist2 = tree.search_knn_vector_3d(point, 1)
        if dist2:
            dists.append(np.sqrt(dist2[0]))
    return float(np.mean(dists)) if dists else float("inf")


def pca_init_transform(src: o3d.geometry.PointCloud, dst: o3d.geometry.PointCloud) -> Tuple[np.ndarray, float]:
    src_center, src_frame = make_pca_frame(src)
    dst_center, dst_frame = make_pca_frame(dst)
    best_t = np.eye(4)
    best_err = float("inf")
    for src_candidate in right_handed_axis_candidates(src_frame):
        rot = dst_frame @ src_candidate.T
        trans = dst_center - rot @ src_center
        t = np.eye(4)
        t[:3, :3] = rot
        t[:3, 3] = trans
        tmp = o3d.geometry.PointCloud(src)
        tmp.transform(t)
        err = nn_mean(tmp, dst)
        if err < best_err:
            best_err = err
            best_t = t
    return best_t, best_err


def estimate_normals(pcd: o3d.geometry.PointCloud, radius: float) -> None:
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius), max_nn=40))
    pcd.normalize_normals()


def refine_icp(
    src_full: o3d.geometry.PointCloud,
    dst_full: o3d.geometry.PointCloud,
    init_t: np.ndarray,
    voxel_base: float,
    iters: Tuple[int, int, int] = (300, 300, 300),
) -> Tuple[np.ndarray, float, float]:
    transform = init_t.copy()
    last_fitness = 0.0
    last_rmse = float("nan")
    levels = [(4.0 * voxel_base, iters[0]), (2.0 * voxel_base, iters[1]), (1.0 * voxel_base, iters[2])]
    for level, (voxel, max_iter) in enumerate(levels):
        src = preprocess_pcd(o3d.geometry.PointCloud(src_full), voxel)
        dst = preprocess_pcd(o3d.geometry.PointCloud(dst_full), voxel)

        if level == 0:
            max_corr = max(6.0 * voxel, 1e-5)
            method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        else:
            max_corr = max((4.0 if level == 1 else 2.5) * voxel, 1e-5)
            radius = max(3.0 * voxel, 1e-3)
            estimate_normals(src, radius)
            estimate_normals(dst, radius)
            try:
                loss = o3d.pipelines.registration.TukeyLoss(k=max_corr)
                method = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
            except Exception:
                method = o3d.pipelines.registration.TransformationEstimationPointToPlane()

        reg = o3d.pipelines.registration.registration_icp(
            src,
            dst,
            max_correspondence_distance=float(max_corr),
            init=transform,
            estimation_method=method,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iter)),
        )
        transform = reg.transformation
        last_fitness = float(reg.fitness)
        last_rmse = float(reg.inlier_rmse)
    return transform, last_fitness, last_rmse


def save_alignment_overlay(
    pred_mesh: o3d.geometry.TriangleMesh,
    gt_mesh: o3d.geometry.TriangleMesh,
    out_path: Path,
    sample_n: int,
    sample_method: str,
) -> None:
    n = min(int(sample_n), 12000)
    pred_pts = np.asarray(sample_mesh(pred_mesh, n, sample_method).points, dtype=np.float64)
    gt_pts = np.asarray(sample_mesh(gt_mesh, n, sample_method).points, dtype=np.float64)
    all_pts = np.vstack([pred_pts, gt_pts])
    mn = all_pts.min(axis=0)
    mx = all_pts.max(axis=0)
    pad = max(float(np.max(mx - mn)) * 0.04, 1e-6)
    mn -= pad
    mx += pad

    views = [
        ("XY", 0, 1),
        ("XZ", 0, 2),
        ("YZ", 1, 2),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=180)
    for ax, (title, i, j) in zip(axes, views):
        ax.scatter(gt_pts[:, i], gt_pts[:, j], s=0.25, c="#2ca02c", alpha=0.55, label="GT")
        ax.scatter(pred_pts[:, i], pred_pts[:, j], s=0.25, c="#d62728", alpha=0.55, label="TRELLIS aligned")
        ax.set_title(title)
        ax.set_xlim(mn[i], mx[i])
        ax.set_ylim(mn[j], mx[j])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.3, alpha=0.25)
    axes[0].legend(loc="upper right", markerscale=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_scene(mesh: o3d.geometry.TriangleMesh) -> o3d.t.geometry.RaycastingScene:
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)
    return scene


def point_to_mesh_distance(scene: o3d.t.geometry.RaycastingScene, points: np.ndarray) -> np.ndarray:
    tensor = o3d.core.Tensor(points.astype(np.float32))
    return scene.compute_distance(tensor).numpy().astype(np.float64)


def fscore_from_distances(d_pred_gt: np.ndarray, d_gt_pred: np.ndarray, tau: float) -> Tuple[float, float, float]:
    precision = float(np.mean(d_pred_gt < tau)) if d_pred_gt.size else 0.0
    recall = float(np.mean(d_gt_pred < tau)) if d_gt_pred.size else 0.0
    fscore = 0.0 if precision + recall == 0 else float(2.0 * precision * recall / (precision + recall))
    return precision, recall, fscore


def eval_mesh_mesh(
    pred_mesh: o3d.geometry.TriangleMesh,
    gt_mesh: o3d.geometry.TriangleMesh,
    sample_n: int,
    sample_method: str,
    taus_m: Iterable[float],
) -> Dict[str, float]:
    pred_scene = make_scene(pred_mesh)
    gt_scene = make_scene(gt_mesh)
    pred_pts = np.asarray(sample_mesh(pred_mesh, sample_n, sample_method).points, dtype=np.float64)
    gt_pts = np.asarray(sample_mesh(gt_mesh, sample_n, sample_method).points, dtype=np.float64)
    d_pred_gt = point_to_mesh_distance(gt_scene, pred_pts)
    d_gt_pred = point_to_mesh_distance(pred_scene, gt_pts)
    out = {
        "accuracy_m": float(np.mean(d_pred_gt)),
        "completeness_m": float(np.mean(d_gt_pred)),
        "chamfer_l1_m": float(np.mean(d_pred_gt) + np.mean(d_gt_pred)),
        "chamfer_l2_m2": float(np.mean(d_pred_gt**2) + np.mean(d_gt_pred**2)),
        "pred_to_gt_p50_m": float(np.percentile(d_pred_gt, 50.0)),
        "pred_to_gt_p95_m": float(np.percentile(d_pred_gt, 95.0)),
        "gt_to_pred_p50_m": float(np.percentile(d_gt_pred, 50.0)),
        "gt_to_pred_p95_m": float(np.percentile(d_gt_pred, 95.0)),
    }
    for tau in taus_m:
        precision, recall, fscore = fscore_from_distances(d_pred_gt, d_gt_pred, tau)
        tau_mm = tau * 1000.0
        out[f"precision@{tau_mm:g}mm"] = precision
        out[f"recall@{tau_mm:g}mm"] = recall
        out[f"fscore@{tau_mm:g}mm"] = fscore
    return out


def occupancy(mesh_scene: o3d.t.geometry.RaycastingScene, points: np.ndarray) -> np.ndarray:
    tensor = o3d.core.Tensor(points.astype(np.float32))
    try:
        return mesh_scene.compute_occupancy(tensor).numpy().astype(bool)
    except Exception:
        return (mesh_scene.compute_signed_distance(tensor).numpy() < 0.0)


def eval_iou(
    pred_mesh: o3d.geometry.TriangleMesh,
    gt_mesh: o3d.geometry.TriangleMesh,
    voxel: float,
    margin: float,
    max_points: int,
) -> Dict[str, float]:
    pred_scene = make_scene(pred_mesh)
    gt_scene = make_scene(gt_mesh)
    pred_bb = pred_mesh.get_axis_aligned_bounding_box()
    gt_bb = gt_mesh.get_axis_aligned_bounding_box()
    mn = np.minimum(pred_bb.min_bound, gt_bb.min_bound) - margin
    mx = np.maximum(pred_bb.max_bound, gt_bb.max_bound) + margin
    dims = np.ceil((mx - mn) / voxel).astype(np.int64) + 1
    total = int(dims[0] * dims[1] * dims[2])
    if total > max_points:
        return {
            "iou": float("nan"),
            "iou_grid_points": total,
            "iou_intersection": 0,
            "iou_union": 0,
        }

    xs = mn[0] + np.arange(dims[0]) * voxel
    ys = mn[1] + np.arange(dims[1]) * voxel
    zs = mn[2] + np.arange(dims[2]) * voxel
    intersection = 0
    union = 0
    for x in xs:
        yy, zz = np.meshgrid(ys, zs, indexing="ij")
        slab = np.stack([np.full_like(yy, x), yy, zz], axis=-1).reshape(-1, 3)
        pred_occ = occupancy(pred_scene, slab)
        gt_occ = occupancy(gt_scene, slab)
        intersection += int(np.sum(pred_occ & gt_occ))
        union += int(np.sum(pred_occ | gt_occ))
    return {
        "iou": 0.0 if union == 0 else float(intersection / union),
        "iou_grid_points": total,
        "iou_intersection": intersection,
        "iou_union": union,
    }


def parse_mapping(items: List[str]) -> Dict[str, str]:
    if not items:
        return dict(DEFAULT_MAPPING)
    mapping = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Mapping must be GT=PRED, got: {item}")
        gt_name, pred_name = item.split("=", 1)
        mapping[gt_name.strip()] = pred_name.strip()
    return mapping


def parse_float_list(text: str) -> List[float]:
    values = []
    for item in (text or "").split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("Empty float list.")
    return values


def evaluate_one_scale(
    args,
    gt_name: str,
    pred_name: str,
    pred_path: Path,
    gt_path: Path,
    pred_mesh_unit: o3d.geometry.TriangleMesh,
    gt_mesh: o3d.geometry.TriangleMesh,
    pred_diag_before: float,
    gt_diag: float,
    pred_unit_scale: float,
    gt_unit_scale: float,
    base_scale: float,
    scale_factor: float,
    taus_m: List[float],
) -> Tuple[Dict[str, float], o3d.geometry.TriangleMesh, np.ndarray]:
    pred_mesh = o3d.geometry.TriangleMesh(pred_mesh_unit)
    scale = base_scale * float(scale_factor)
    pred_mesh.scale(scale, center=(0, 0, 0))

    align_samples = sample_mesh(pred_mesh, args.align_samples, args.sample_method)
    gt_align_samples = sample_mesh(gt_mesh, args.align_samples, args.sample_method)
    voxel = args.voxel if args.voxel > 0 else auto_voxel_from_recon(align_samples, fallback=0.003)
    pred_ds = preprocess_pcd(o3d.geometry.PointCloud(align_samples), voxel)
    gt_ds = preprocess_pcd(o3d.geometry.PointCloud(gt_align_samples), voxel)

    init_t, pca_nn = pca_init_transform(pred_ds, gt_ds)
    transform, icp_fitness, icp_rmse = refine_icp(
        align_samples,
        gt_align_samples,
        init_t,
        voxel,
        iters=(args.icp_iter0, args.icp_iter1, args.icp_iter2),
    )
    pred_aligned = o3d.geometry.TriangleMesh(pred_mesh)
    pred_aligned.transform(transform)

    metrics = eval_mesh_mesh(pred_aligned, gt_mesh, args.metric_samples, args.sample_method, taus_m)
    if args.iou_voxel > 0:
        metrics.update(eval_iou(pred_aligned, gt_mesh, args.iou_voxel, args.iou_margin, args.iou_max_points))

    row = {
        "gt_name": gt_name,
        "pred_name": pred_name,
        "gt_mesh": str(gt_path),
        "pred_mesh": str(pred_path),
        "pred_diag_before_m": pred_diag_before,
        "gt_diag_m": gt_diag,
        "pred_unit_scale": pred_unit_scale,
        "gt_unit_scale": gt_unit_scale,
        "base_scale_pred_to_gt": base_scale,
        "scale_factor": float(scale_factor),
        "scale_pred_to_gt": scale,
        "align_voxel_m": voxel,
        "pca_mean_nn_m": pca_nn,
        "icp_fitness": icp_fitness,
        "icp_inlier_rmse_m": icp_rmse,
        **metrics,
    }
    return row, pred_aligned, transform


def evaluate_one(args, gt_name: str, pred_name: str, taus_m: List[float]) -> Dict[str, float]:
    pred_path = resolve_pred_mesh(Path(args.pred_root), pred_name)
    gt_path = resolve_gt_mesh(Path(args.gt_root), gt_name)

    pred_mesh = read_mesh(pred_path)
    gt_mesh = read_mesh(gt_path)

    pred_unit_scale = unit_scale_from_mode(pred_mesh, args.pred_unit)
    gt_unit_scale = unit_scale_from_mode(gt_mesh, args.gt_unit)
    if pred_unit_scale != 1.0:
        pred_mesh.scale(pred_unit_scale, center=(0, 0, 0))
    if gt_unit_scale != 1.0:
        gt_mesh.scale(gt_unit_scale, center=(0, 0, 0))

    pred_diag_before = mesh_diag(pred_mesh)
    gt_diag = mesh_diag(gt_mesh)

    base_scale = 1.0
    if args.scale_mode == "bbox_diag":
        base_scale = gt_diag / pred_diag_before

    scale_factors = parse_float_list(args.scale_factors)
    pick_key = f"fscore@{args.pick_tau_mm:g}mm"
    sweep_rows = []
    best_row = None
    best_mesh = None
    best_transform = None
    best_score = -1.0

    for scale_factor in scale_factors:
        row, pred_aligned, transform = evaluate_one_scale(
            args=args,
            gt_name=gt_name,
            pred_name=pred_name,
            pred_path=pred_path,
            gt_path=gt_path,
            pred_mesh_unit=pred_mesh,
            gt_mesh=gt_mesh,
            pred_diag_before=pred_diag_before,
            gt_diag=gt_diag,
            pred_unit_scale=pred_unit_scale,
            gt_unit_scale=gt_unit_scale,
            base_scale=base_scale,
            scale_factor=scale_factor,
            taus_m=taus_m,
        )
        sweep_rows.append(row)
        score = float(row.get(pick_key, -1.0))
        if score > best_score:
            best_score = score
            best_row = row
            best_mesh = pred_aligned
            best_transform = transform

    if best_row is None or best_mesh is None or best_transform is None:
        raise RuntimeError(f"No valid scale candidates for {gt_name} vs {pred_name}")

    object_dir = Path(args.out_dir) / f"{gt_name}_vs_{pred_name}"
    object_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = object_dir / "pred_aligned.ply"
    transform_path = object_dir / "transform_gt_from_pred.txt"
    overlay_path = object_dir / "alignment_overlay.png"
    scale_sweep_path = object_dir / "scale_sweep.csv"
    o3d.io.write_triangle_mesh(str(aligned_path), best_mesh, write_ascii=False)
    np.savetxt(transform_path, best_transform, fmt="%.10f")
    write_summary(scale_sweep_path, sweep_rows)
    if not args.no_vis:
        save_alignment_overlay(best_mesh, gt_mesh, overlay_path, args.vis_samples, args.sample_method)

    return {
        **best_row,
        "gt_mesh": str(gt_path),
        "pred_mesh": str(pred_path),
        "aligned_pred_mesh": str(aligned_path),
        "transform_gt_from_pred": str(transform_path),
        "alignment_overlay": str(overlay_path) if not args.no_vis else "",
        "scale_sweep_csv": str(scale_sweep_path),
        "pick_scale_metric": pick_key,
    }


def write_summary(path: Path, rows: List[Dict[str, float]]) -> None:
    keys = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: List[Dict[str, float]], taus_m: List[float]) -> None:
    tau_key = f"fscore@{taus_m[0] * 1000.0:g}mm" if taus_m else ""
    header = f"{'GT':<14} {'TRELLIS':<10} {'ScaleFac':>8} {'ChamferL1(mm)':>14} {'Acc(mm)':>10} {'Comp(mm)':>10}"
    if tau_key:
        header += f" {tau_key:>12}"
    header += f" {'IoU':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        line = (
            f"{row['gt_name']:<14} {row['pred_name']:<10} "
            f"{row.get('scale_factor', float('nan')):8.3f} "
            f"{row['chamfer_l1_m'] * 1000.0:14.3f} "
            f"{row['accuracy_m'] * 1000.0:10.3f} "
            f"{row['completeness_m'] * 1000.0:10.3f}"
        )
        if tau_key:
            line += f" {row[tau_key]:12.4f}"
        line += f" {row.get('iou', float('nan')):8.4f}"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_root", default=DEFAULT_PRED_ROOT)
    parser.add_argument("--gt_root", default=DEFAULT_GT_ROOT)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--mapping", nargs="*", default=[], help="Mappings in GT_NAME=PRED_NAME format")
    parser.add_argument("--thresholds_mm", default="2,5,10")
    parser.add_argument("--pick_tau_mm", type=float, default=5.0, help="Metric threshold used to select best scale")
    parser.add_argument("--scale_factors", default="1.0", help="Comma-separated multipliers relative to --scale_mode base scale")
    parser.add_argument("--align_samples", type=int, default=30000)
    parser.add_argument("--metric_samples", type=int, default=50000)
    parser.add_argument("--sample_method", choices=["uniform", "poisson"], default="uniform")
    parser.add_argument("--scale_mode", choices=["bbox_diag", "none"], default="bbox_diag")
    parser.add_argument("--pred_unit", choices=["auto", "m", "mm"], default="m")
    parser.add_argument("--gt_unit", choices=["auto", "m", "mm"], default="auto")
    parser.add_argument("--voxel", type=float, default=0.0, help="ICP voxel in mesh units; 0 uses 5x recon median NN like 5_computer_metric_nskr_2.py")
    parser.add_argument("--icp_iter0", type=int, default=300)
    parser.add_argument("--icp_iter1", type=int, default=300)
    parser.add_argument("--icp_iter2", type=int, default=300)
    parser.add_argument("--vis_samples", type=int, default=12000)
    parser.add_argument("--no_vis", action="store_true")
    parser.add_argument("--iou_voxel", type=float, default=0.003)
    parser.add_argument("--iou_margin", type=float, default=0.006)
    parser.add_argument("--iou_max_points", type=int, default=5000000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    mapping = parse_mapping(args.mapping)
    taus_m = [float(item.strip()) / 1000.0 for item in args.thresholds_mm.split(",") if item.strip()]

    rows = []
    for gt_name, pred_name in mapping.items():
        print(f"[RUN] {gt_name} vs {pred_name}")
        rows.append(evaluate_one(args, gt_name, pred_name, taus_m))

    summary_path = Path(args.out_dir) / "summary.csv"
    write_summary(summary_path, rows)
    print_table(rows, taus_m)
    print(f"[SAVED] {summary_path}")


if __name__ == "__main__":
    main()
