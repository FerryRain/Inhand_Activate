from __future__ import annotations

from typing import Dict

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


class ReconstructionEvaluator:
    def __init__(self, mesh: o3d.geometry.TriangleMesh, cfg: Dict, seed: int = 0):
        count = int(cfg["evaluation"].get("gt_sample_count", 30000))
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        triangles = np.asarray(mesh.triangles, dtype=np.int32)
        triangle_vertices = vertices[triangles]
        areas = 0.5 * np.linalg.norm(
            np.cross(
                triangle_vertices[:, 1] - triangle_vertices[:, 0],
                triangle_vertices[:, 2] - triangle_vertices[:, 0],
            ),
            axis=1,
        )
        valid = areas > 1e-15
        triangle_vertices = triangle_vertices[valid]
        areas = areas[valid]
        probabilities = areas / areas.sum()
        rng = np.random.RandomState(int(seed))
        selected = triangle_vertices[rng.choice(len(triangle_vertices), size=count, p=probabilities)]
        u = np.sqrt(rng.rand(count, 1))
        v = rng.rand(count, 1)
        self.gt_points = (
            (1.0 - u) * selected[:, 0]
            + u * (1.0 - v) * selected[:, 1]
            + u * v * selected[:, 2]
        ).astype(np.float32)
        self.thresholds = [float(value) for value in cfg["evaluation"]["thresholds_m"]]
        self.gt_tree = cKDTree(self.gt_points)

    def evaluate(self, fused_points: np.ndarray) -> Dict[str, float]:
        points = np.asarray(fused_points, dtype=np.float32)
        if len(points) == 0:
            metrics = {
                "chamfer_m": float("inf"),
                "point_count": 0,
                "surface_thickness_m": float("inf"),
                "outlier_ratio": 1.0,
            }
            for threshold in self.thresholds:
                label = int(round(threshold * 1000.0))
                metrics.update({
                    "precision@%d" % label: 0.0,
                    "recall@%d" % label: 0.0,
                    "f@%d" % label: 0.0,
                })
            metrics["surface_coverage"] = 0.0
            return metrics
        fused_tree = cKDTree(points)
        dist_fused_to_gt = self.gt_tree.query(points, k=1, workers=-1)[0]
        dist_gt_to_fused = fused_tree.query(self.gt_points, k=1, workers=-1)[0]
        metrics = {
            "chamfer_m": float(dist_fused_to_gt.mean() + dist_gt_to_fused.mean()),
            "point_count": int(len(points)),
            "surface_thickness_m": float(np.percentile(dist_fused_to_gt, 90)),
            "outlier_ratio": float(np.mean(dist_fused_to_gt > 0.010)),
        }
        for threshold in self.thresholds:
            label = int(round(threshold * 1000.0))
            precision = float(np.mean(dist_fused_to_gt <= threshold))
            recall = float(np.mean(dist_gt_to_fused <= threshold))
            f_score = 2.0 * precision * recall / max(precision + recall, 1e-12)
            metrics.update({
                "precision@%d" % label: precision,
                "recall@%d" % label: recall,
                "f@%d" % label: float(f_score),
            })
        coverage_threshold = int(round(float(self.thresholds[1]) * 1000.0)) if len(self.thresholds) > 1 else int(round(self.thresholds[0] * 1000.0))
        metrics["surface_coverage"] = metrics["recall@%d" % coverage_threshold]
        return metrics


def trapezoid_auc(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float(values[0]) if len(values) else float("nan")
    return float(np.mean(0.5 * (values[:-1] + values[1:])))
