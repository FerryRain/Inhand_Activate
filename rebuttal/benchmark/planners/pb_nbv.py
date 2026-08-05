from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import binary_dilation
from sklearn.mixture import GaussianMixture

from ..fusion import backproject_object
from ..geometry import current_view_direction_object, look_at_basis
from .base import Planner


class PBNBVPlanner(Planner):
    """Scale-normalized adaptation of the PB-NBV representation and score core."""

    name = "pb_nbv"

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        pcfg = cfg["planners"]["pb_nbv"]
        self.voxel_divisor = float(pcfg["voxel_divisor"])
        self.max_ellipsoids = int(pcfg["max_ellipsoids"])
        self.partition_count = int(pcfg["partitions"])
        self.use_partition = bool(pcfg["use_partition"])
        self.occupied_penalty = float(pcfg["occupied_penalty"])
        self.ray_stride = int(pcfg.get("ray_stride", 8))
        self.camera_distance = float(cfg["camera"]["distance"])
        self.fx = float(cfg["camera"]["intrinsics"][0][0])
        self.fy = float(cfg["camera"]["intrinsics"][1][1])
        self.occupied_ellipsoids: List[Tuple[np.ndarray, np.ndarray]] = []
        self.frontier_ellipsoids: List[Tuple[np.ndarray, np.ndarray]] = []
        self.visited_partitions = set()
        self.voxel_size = float("nan")
        self.voxel_counts = {}

    def reset(self, initial_observation, fusion):
        self.visited_partitions = set()
        for observation in fusion.observations:
            self.visited_partitions.add(self._partition(current_view_direction_object(observation.executed_pose)))
        self._update_representation(fusion)

    def update(self, observation, fusion):
        self.visited_partitions.add(self._partition(current_view_direction_object(observation.executed_pose)))
        self._update_representation(fusion)

    def _partition(self, direction: np.ndarray) -> int:
        longitude = math.atan2(float(direction[2]), float(direction[0]))
        return int(math.floor((longitude + math.pi) / (2.0 * math.pi) * self.partition_count)) % self.partition_count

    @staticmethod
    def _indices(points: np.ndarray, lower: np.ndarray, voxel: float, shape: np.ndarray) -> np.ndarray:
        idx = np.floor((points - lower[None]) / voxel).astype(np.int32)
        valid = np.all((idx >= 0) & (idx < shape[None]), axis=1)
        return idx[valid]

    def _update_representation(self, fusion):
        start = time.perf_counter()
        points = fusion.points
        if len(points) < 20:
            self.occupied_ellipsoids = []
            self.frontier_ellipsoids = []
            self.last_update_time = time.perf_counter() - start
            return
        data_min = points.min(axis=0)
        data_max = points.max(axis=0)
        diagonal = float(np.linalg.norm(data_max - data_min))
        self.voxel_size = max(diagonal / self.voxel_divisor, 4e-4)
        margin = max(0.30 * diagonal, 4.0 * self.voxel_size)
        lower = data_min - margin
        upper = data_max + margin
        shape = np.ceil((upper - lower) / self.voxel_size).astype(np.int32) + 1
        shape = np.minimum(shape, 96)
        upper = lower + shape * self.voxel_size

        occupied = np.zeros(tuple(shape.tolist()), dtype=bool)
        occupied_idx = self._indices(points, lower, self.voxel_size, shape)
        if len(occupied_idx):
            occupied[tuple(occupied_idx.T)] = True
        empty = np.zeros_like(occupied)

        for observation, frame_points in zip(fusion.observations, fusion.frame_points):
            if not len(frame_points):
                continue
            rotation = observation.executed_pose[:3, :3]
            translation = observation.executed_pose[:3, 3]
            camera_origin = rotation.T @ (-translation)
            for hit in frame_points[:: self.ray_stride]:
                vector = hit - camera_origin
                distance = float(np.linalg.norm(vector))
                if distance <= self.voxel_size:
                    continue
                sample_count = max(2, int(distance / self.voxel_size))
                fractions = np.linspace(0.0, max(0.0, 1.0 - 1.5 * self.voxel_size / distance), sample_count)
                samples = camera_origin[None] + fractions[:, None] * vector[None]
                ray_idx = self._indices(samples, lower, self.voxel_size, shape)
                if len(ray_idx):
                    empty[tuple(ray_idx.T)] = True
        empty[occupied] = False
        neighborhood = np.ones((3, 3, 3), dtype=bool)
        unknown = ~(occupied | empty)
        frontier = unknown & binary_dilation(occupied, structure=neighborhood) & binary_dilation(empty, structure=neighborhood)

        occupied_centers = lower[None] + (np.argwhere(occupied) + 0.5) * self.voxel_size
        frontier_centers = lower[None] + (np.argwhere(frontier) + 0.5) * self.voxel_size
        self.occupied_ellipsoids = self._fit_ellipsoids(occupied_centers, self.voxel_size, seed=0)
        self.frontier_ellipsoids = self._fit_ellipsoids(frontier_centers, self.voxel_size, seed=1)
        self.voxel_counts = {
            "occupied": int(occupied.sum()),
            "empty": int(empty.sum()),
            "unknown": int(unknown.sum()),
            "frontier": int(frontier.sum()),
        }
        self.last_update_time = time.perf_counter() - start

    def _fit_ellipsoids(self, points: np.ndarray, voxel: float, seed: int):
        if len(points) == 0:
            return []
        if len(points) < 8:
            return [(points.mean(axis=0), np.eye(3) * (1.5 * voxel) ** 2)]
        rng = np.random.RandomState(seed)
        fit_points = points
        if len(fit_points) > 5000:
            fit_points = fit_points[rng.choice(len(fit_points), 5000, replace=False)]
        max_components = min(self.max_ellipsoids, max(1, len(fit_points) // 20))
        best = None
        best_bic = float("inf")
        for components in range(1, max_components + 1):
            model = GaussianMixture(
                n_components=components,
                covariance_type="full",
                reg_covar=max(voxel * voxel * 0.1, 1e-9),
                n_init=1,
                random_state=seed,
                max_iter=100,
            )
            model.fit(fit_points)
            bic = float(model.bic(fit_points))
            if bic < best_bic:
                best_bic = bic
                best = model
        ellipsoids = []
        for mean, covariance in zip(best.means_, best.covariances_):
            # 95% Gaussian equiprobability ellipsoid, used as an MVEE approximation.
            ellipsoids.append((mean.astype(np.float64), covariance.astype(np.float64) * 7.815))
        return ellipsoids

    def _projected_areas(self, direction: np.ndarray):
        ellipsoids = [(0, item) for item in self.occupied_ellipsoids]
        ellipsoids += [(1, item) for item in self.frontier_ellipsoids]
        if not ellipsoids:
            return 0.0
        camera_position = np.asarray(direction, dtype=np.float64)
        camera_position /= np.linalg.norm(camera_position) + 1e-12
        camera_position *= self.camera_distance
        basis = look_at_basis(camera_position)
        projected = []
        for kind, (center, covariance) in ellipsoids:
            center_camera = (center - camera_position) @ basis
            x, y, z = center_camera
            if z <= 1e-5:
                continue
            covariance_camera = basis.T @ covariance @ basis
            jacobian = np.array(
                [[self.fx / z, 0.0, -self.fx * x / (z * z)],
                 [0.0, self.fy / z, -self.fy * y / (z * z)]],
                dtype=np.float64,
            )
            covariance_image = jacobian @ covariance_camera @ jacobian.T
            determinant = max(float(np.linalg.det(covariance_image)), 0.0)
            area = math.pi * math.sqrt(determinant)
            projected.append((float(z), kind, area))
        projected.sort(key=lambda item: item[0])
        score = 0.0
        for rank, (_, kind, area) in enumerate(projected):
            weighted = (0.5 ** rank) * area
            score += weighted if kind == 1 else -self.occupied_penalty * weighted
        return float(score)

    def score_views(self, candidate_views: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        scores = np.array([self._projected_areas(direction) for direction in candidate_views], dtype=np.float32)
        if self.use_partition and self.visited_partitions:
            unvisited = set(range(self.partition_count)) - self.visited_partitions
            if unvisited:
                neighbors = set()
                for visited in self.visited_partitions:
                    for delta in (-1, 1):
                        candidate = (visited + delta) % self.partition_count
                        if candidate in unvisited:
                            neighbors.add(candidate)
                allowed = neighbors if neighbors else unvisited
                mask = np.array([self._partition(direction) in allowed for direction in candidate_views])
                if mask.any():
                    scores[~mask] = -np.inf
        self.last_score_time = time.perf_counter() - start
        self.last_scores = scores
        return scores

    def diagnostics(self):
        out = super().diagnostics()
        out.update({
            "voxel_size_m": float(self.voxel_size),
            "occupied_ellipsoids": len(self.occupied_ellipsoids),
            "frontier_ellipsoids": len(self.frontier_ellipsoids),
            "voxel_counts": self.voxel_counts,
        })
        return out
