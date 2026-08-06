from __future__ import annotations

import time
from typing import Dict

import numpy as np

from ..geometry import (
    ACTIONS,
    candidate_action_assignments,
    current_view_direction_object,
    next_view_direction,
)
from .base import Planner


class PoseNoveltyPlanner(Planner):
    """Pose-only viewpoint-coverage planner.

    The planner never reads the fused cloud.  It predicts the viewing direction
    after each executable primitive and maximizes the minimum geodesic distance
    to all previously acquired viewing directions.
    """

    name = "pose_novelty"

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.angles_deg = cfg["actions"]["angles_deg"]
        self.history = []
        self.current_pose = None
        self.last_action_scores = {}

    def _record(self, observation):
        start = time.perf_counter()
        self.current_pose = np.asarray(observation.executed_pose, dtype=np.float64).copy()
        direction = current_view_direction_object(self.current_pose)
        self.history.append(np.asarray(direction, dtype=np.float64))
        self.last_update_time = time.perf_counter() - start

    def reset(self, initial_observation, fusion):
        self.history = []
        self.last_action_scores = {}
        self._record(initial_observation)

    def update(self, observation, fusion):
        self._record(observation)

    def score_views(self, candidate_views: np.ndarray) -> np.ndarray:
        if self.current_pose is None or not self.history:
            raise RuntimeError("PoseNoveltyPlanner must be reset before scoring")
        start = time.perf_counter()
        history = np.stack(self.history, axis=0)
        predicted = np.stack(
            [
                next_view_direction(self.current_pose, action, self.angles_deg)
                for action in ACTIONS
            ],
            axis=0,
        )
        cosines = np.clip(predicted @ history.T, -1.0, 1.0)
        action_scores = np.min(np.arccos(cosines), axis=1)
        assignments = candidate_action_assignments(
            candidate_views, self.current_pose, self.angles_deg
        )
        scores = action_scores[assignments].astype(np.float32)
        self.last_action_scores = {
            action: float(action_scores[index]) for index, action in enumerate(ACTIONS)
        }
        self.last_score_time = time.perf_counter() - start
        self.last_scores = scores
        return scores

    def diagnostics(self):
        out = super().diagnostics()
        out.update({
            "device": "cpu",
            "history_views": int(len(self.history)),
            "action_scores_rad": dict(self.last_action_scores),
            "uses_reconstruction": False,
        })
        return out
