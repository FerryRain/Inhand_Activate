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
        self.backend = str(cfg["evaluation"].get("backend", "scipy_cpu"))
        self._torch = None
        self._knn_points = None
        self._gt_tensor = None
        if self.backend == "pytorch3d_cuda":
            import torch
            from pytorch3d.ops import knn_points

            if not torch.cuda.is_available():
                raise RuntimeError("evaluation.backend=pytorch3d_cuda requires CUDA")
            self._torch = torch
            self._knn_points = knn_points
            self._gt_tensor = torch.as_tensor(
                self.gt_points, dtype=torch.float32, device="cuda"
            )[None]

    def _nearest_distances(self, points: np.ndarray):
        points = np.asarray(points, dtype=np.float32)
        if self.backend != "pytorch3d_cuda":
            fused_tree = cKDTree(points)
            return (
                self.gt_tree.query(points, k=1, workers=-1)[0],
                fused_tree.query(self.gt_points, k=1, workers=-1)[0],
            )
        torch = self._torch
        with torch.no_grad():
            fused = torch.as_tensor(points, dtype=torch.float32, device="cuda")[None]
            fused_to_gt = self._knn_points(fused, self._gt_tensor, K=1).dists
            gt_to_fused = self._knn_points(self._gt_tensor, fused, K=1).dists
            fused_to_gt = torch.sqrt(torch.clamp_min(fused_to_gt[0, :, 0], 0.0))
            gt_to_fused = torch.sqrt(torch.clamp_min(gt_to_fused[0, :, 0], 0.0))
            return (
                fused_to_gt.detach().cpu().numpy(),
                gt_to_fused.detach().cpu().numpy(),
            )

    def distances_to_gt(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        if not len(points):
            return np.zeros((0,), dtype=np.float32)
        if self.backend != "pytorch3d_cuda":
            return self.gt_tree.query(points, k=1, workers=-1)[0]
        torch = self._torch
        with torch.no_grad():
            values = self._knn_points(
                torch.as_tensor(points, dtype=torch.float32, device="cuda")[None],
                self._gt_tensor,
                K=1,
            ).dists[0, :, 0]
            return torch.sqrt(torch.clamp_min(values, 0.0)).detach().cpu().numpy()

    def distances_from_gt(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        if not len(points):
            return np.full((len(self.gt_points),), np.inf, dtype=np.float32)
        if self.backend != "pytorch3d_cuda":
            return cKDTree(points).query(self.gt_points, k=1, workers=-1)[0]
        torch = self._torch
        with torch.no_grad():
            values = self._knn_points(
                self._gt_tensor,
                torch.as_tensor(points, dtype=torch.float32, device="cuda")[None],
                K=1,
            ).dists[0, :, 0]
            return torch.sqrt(torch.clamp_min(values, 0.0)).detach().cpu().numpy()

    def recall_subset(
        self,
        fused_points: np.ndarray,
        target_points: np.ndarray,
        threshold: float = 0.005,
    ) -> float:
        points = np.asarray(fused_points, dtype=np.float32)
        targets = np.asarray(target_points, dtype=np.float32)
        if not len(targets):
            return float("nan")
        if not len(points):
            return 0.0
        if self.backend != "pytorch3d_cuda":
            distances = cKDTree(points).query(targets, k=1, workers=-1)[0]
        else:
            torch = self._torch
            with torch.no_grad():
                distances_sq = self._knn_points(
                    torch.as_tensor(targets, dtype=torch.float32, device="cuda")[None],
                    torch.as_tensor(points, dtype=torch.float32, device="cuda")[None],
                    K=1,
                ).dists[0, :, 0]
                distances = torch.sqrt(torch.clamp_min(distances_sq, 0.0)).detach().cpu().numpy()
        return float(np.mean(distances <= float(threshold)))

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
        dist_fused_to_gt, dist_gt_to_fused = self._nearest_distances(points)
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
