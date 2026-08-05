from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np


class Planner(ABC):
    name = "planner"

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.last_scores: Optional[np.ndarray] = None
        self.last_update_time = 0.0
        self.last_score_time = 0.0

    @abstractmethod
    def reset(self, initial_observation, fusion):
        pass

    @abstractmethod
    def update(self, observation, fusion):
        pass

    @abstractmethod
    def score_views(self, candidate_views: np.ndarray) -> np.ndarray:
        pass

    def select_nbv(self, candidate_views: np.ndarray) -> np.ndarray:
        scores = self.score_views(candidate_views)
        self.last_scores = np.asarray(scores, dtype=np.float32)
        return np.asarray(candidate_views[int(np.nanargmax(scores))], dtype=np.float32)

    def diagnostics(self) -> Dict:
        return {
            "representation_update_s": float(self.last_update_time),
            "candidate_scoring_s": float(self.last_score_time),
            "planning_s": float(self.last_update_time + self.last_score_time),
        }
