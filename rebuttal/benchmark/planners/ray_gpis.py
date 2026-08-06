from __future__ import annotations

import time
import sys
from pathlib import Path
from typing import Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Active.NBV_gpis_Field import GPISNBVv3

from .base import Planner


class RayGPISPlanner(Planner):
    name = "ray_gpis"

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        pcfg = cfg["planners"]["ray_gpis"]
        self.variant = str(pcfg.get("variant", "full"))
        self.estimator = GPISNBVv3(
            voxel=float(pcfg["voxel_size"]),
            max_train=int(pcfg["max_train"]),
            train_iters=int(pcfg["train_iters"]),
            device="cuda" if __import__("torch").cuda.is_available() else "cpu",
            grid_res=int(pcfg.get("grid_res", 20)),
            n_dirs=int(cfg["candidates"]["count"]),
            hit_eps=float(pcfg.get("hit_eps_m", 0.0)),
            novelty_beta=float(pcfg["novelty_beta"]),
            band_halfwidth=float(pcfg.get("band_halfwidth", 0.06)),
            rf_k=int(pcfg["receptive_field_k"]),
            rf_min_cos=float(pcfg["receptive_field_min_cos"]),
            band_samples=int(pcfg["band_samples"]),
            gpis_add_outer_samples=bool(pcfg.get("add_outer_samples", False)),
        )
        self._scores = None
        self._hit_mask = None

    def reset(self, initial_observation, fusion):
        self._estimate(fusion)

    def update(self, observation, fusion):
        self._estimate(fusion)

    def _estimate(self, fusion):
        torch = __import__("torch")
        if self.estimator.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        self.estimator.estimate(fusion.cloud, seed=int(self.cfg["seed"]), verbose=False)
        if self.variant == "novelty_only":
            scores = self.estimator._nov_rf.copy()
        elif self.variant == "uncertainty_only":
            scores = self.estimator._unc_rf.copy()
        elif self.variant == "pointwise":
            scores = self.estimator._unc_base * self.estimator._nov_base
        else:
            scores = self.estimator.scores.copy()
        if self.variant == "hit_only":
            self._hit_mask = np.asarray(self.estimator._hit_mask, dtype=bool).copy()
            scores[~self._hit_mask] = -np.inf
        self._scores = np.asarray(scores, dtype=np.float32)
        if self.estimator.device == "cuda":
            torch.cuda.synchronize()
        self.last_update_time = time.perf_counter() - start

    def score_views(self, candidate_views: np.ndarray) -> np.ndarray:
        torch = __import__("torch")
        if self.estimator.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        estimator_directions = np.asarray(self.estimator.dirs)
        candidates = np.asarray(candidate_views)
        if len(candidates) == len(estimator_directions) and np.allclose(candidates, estimator_directions, atol=1e-6):
            scores = self._scores.copy()
        else:
            nearest = np.argmax(candidates @ estimator_directions.T, axis=1)
            scores = self._scores[nearest]
        if self.estimator.device == "cuda":
            torch.cuda.synchronize()
        self.last_score_time = time.perf_counter() - start
        self.last_scores = scores
        return scores

    def diagnostics(self):
        out = super().diagnostics()
        selected_index = (
            int(np.nanargmax(self.last_scores))
            if self.last_scores is not None and np.any(np.isfinite(self.last_scores))
            else int(self.estimator.best_idx)
        )
        selected_hit = bool(self.estimator._hit_mask[selected_index])
        out.update({
            "variant": self.variant,
            "device": self.estimator.device,
            "hit_ratio": float(self.estimator.get_last_nbv()["hit_ratio"]),
            "train_points": int(min(len(self.estimator.pcd.points), self.estimator.max_train)),
            "selected_direction_hit": selected_hit,
        })
        return out
