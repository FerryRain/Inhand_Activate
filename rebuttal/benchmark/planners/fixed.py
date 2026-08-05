from __future__ import annotations

import numpy as np

from .base import Planner


class FixedSchedulePlanner(Planner):
    name = "fixed"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.schedule = list(cfg["planners"]["fixed"]["schedule"])
        self.index = 0

    def reset(self, initial_observation, fusion):
        self.index = 0
        self.last_scores = None

    def update(self, observation, fusion):
        return None

    def score_views(self, candidate_views):
        return np.full((len(candidate_views),), np.nan, dtype=np.float32)

    def next_action(self):
        action = self.schedule[self.index % len(self.schedule)]
        self.index += 1
        return action
