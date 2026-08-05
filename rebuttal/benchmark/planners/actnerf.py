from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..geometry import virtual_camera_rays
from .base import Planner


def positional_encoding(values: torch.Tensor, levels: int) -> torch.Tensor:
    encoded = [values]
    for level in range(levels):
        frequency = (2.0 ** level) * math.pi
        encoded.extend([torch.sin(frequency * values), torch.cos(frequency * values)])
    return torch.cat(encoded, dim=-1)


class TinyRadianceField(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        position_dim = 3 * (1 + 2 * 4)
        direction_dim = 3 * (1 + 2 * 2)
        self.trunk = nn.Sequential(
            nn.Linear(position_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.sigma = nn.Linear(hidden_dim, 1)
        self.color = nn.Sequential(
            nn.Linear(hidden_dim + direction_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

    def forward(self, positions: torch.Tensor, directions: torch.Tensor):
        features = self.trunk(positional_encoding(positions, 4))
        sigma = F.softplus(self.sigma(features) - 1.0)
        color = self.color(torch.cat([features, positional_encoding(directions, 2)], dim=-1))
        return sigma, color


class ActNeRFPlanner(Planner):
    """Adapted ActNeRF: warm-started NeRF ensemble RGB rendering variance."""

    name = "actnerf"

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        pcfg = cfg["planners"]["actnerf"]
        self.ensemble_size = int(pcfg["ensemble_size"])
        self.hidden_dim = int(pcfg["hidden_dim"])
        self.train_iterations = int(pcfg["train_iterations"])
        self.rays_per_batch = int(pcfg["rays_per_batch"])
        self.samples_per_ray = int(pcfg["samples_per_ray"])
        self.score_resolution = int(pcfg["score_resolution"])
        self.learning_rate = float(pcfg["learning_rate"])
        self.opacity_threshold = float(pcfg["opacity_threshold"])
        self.near = float(pcfg["near"])
        self.far = float(pcfg["far"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.camera_distance = float(cfg["camera"]["distance"])
        width = float(cfg["camera"]["width"])
        fx = float(cfg["camera"]["intrinsics"][0][0])
        self.fov_deg = math.degrees(2.0 * math.atan(width / (2.0 * fx)))
        self.models: List[TinyRadianceField] = []
        self.optimizers = []
        self.observations = []
        self.planner_seed = int(cfg.get("runtime", {}).get("planner_seed", 0))
        self.rng = np.random.RandomState(int(cfg["seed"]) + self.planner_seed)
        self._camera_directions = self._make_camera_directions()
        self.roi_fraction = 0.0

    def _make_camera_directions(self):
        camera = self.cfg["camera"]
        height, width = int(camera["height"]), int(camera["width"])
        K = np.asarray(camera["intrinsics"], dtype=np.float64)
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        directions = np.stack(
            [(xx + 0.5 - K[0, 2]) / K[0, 0],
             (yy + 0.5 - K[1, 2]) / K[1, 1],
             np.ones_like(xx)], axis=-1
        )
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        return directions.astype(np.float32)

    def _initialize_ensemble(self):
        self.models = []
        self.optimizers = []
        for index in range(self.ensemble_size):
            torch.manual_seed(int(self.cfg["seed"]) + self.planner_seed + 1009 * index)
            model = TinyRadianceField(self.hidden_dim).to(self.device)
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
            self.models.append(model)
            self.optimizers.append(torch.optim.Adam(model.parameters(), lr=self.learning_rate))

    def reset(self, initial_observation, fusion):
        self._initialize_ensemble()
        self.observations = list(fusion.observations)
        self._train_update()

    def update(self, observation, fusion):
        self.observations = list(fusion.observations)
        self._train_update()

    def _sample_rays(self, batch_size: int):
        ray_origins = []
        ray_directions = []
        targets = []
        masks = []
        per_observation = max(1, batch_size // max(len(self.observations), 1))
        for observation in self.observations:
            foreground = np.flatnonzero(observation.mask.reshape(-1))
            all_pixels = np.arange(observation.mask.size)
            fg_count = min(len(foreground), per_observation // 2)
            selected_fg = self.rng.choice(foreground, fg_count, replace=len(foreground) < fg_count) if fg_count else np.zeros(0, dtype=int)
            rest_count = per_observation - len(selected_fg)
            selected_all = self.rng.choice(all_pixels, rest_count, replace=len(all_pixels) < rest_count)
            selected = np.concatenate([selected_fg, selected_all])
            directions_camera = self._camera_directions.reshape(-1, 3)[selected]
            rotation = observation.executed_pose[:3, :3]
            translation = observation.executed_pose[:3, 3]
            origins_object = rotation.T @ (-translation)
            directions_object = directions_camera @ rotation
            ray_origins.append(np.broadcast_to(origins_object, directions_object.shape))
            ray_directions.append(directions_object)
            target_rgb = observation.rgb.reshape(-1, 3)[selected].astype(np.float32) / 255.0
            target_mask = observation.mask.reshape(-1)[selected].astype(np.float32)
            target_rgb *= target_mask[:, None]
            targets.append(target_rgb)
            masks.append(target_mask)
        return tuple(
            torch.from_numpy(np.concatenate(items, axis=0)).to(self.device, dtype=torch.float32)
            for items in (ray_origins, ray_directions, targets, masks)
        )

    def _render(self, model, origins, directions, training=False):
        ray_count = len(origins)
        z_values = torch.linspace(self.near, self.far, self.samples_per_ray, device=self.device)
        if training:
            interval = (self.far - self.near) / self.samples_per_ray
            z_values = z_values[None] + (torch.rand((ray_count, self.samples_per_ray), device=self.device) - 0.5) * interval
        else:
            z_values = z_values[None].expand(ray_count, -1)
        positions = origins[:, None, :] + directions[:, None, :] * z_values[..., None]
        expanded_directions = directions[:, None, :].expand_as(positions)
        sigma, colors = model(positions.reshape(-1, 3), expanded_directions.reshape(-1, 3))
        sigma = sigma.reshape(ray_count, self.samples_per_ray)
        colors = colors.reshape(ray_count, self.samples_per_ray, 3)
        deltas = z_values[:, 1:] - z_values[:, :-1]
        deltas = torch.cat([deltas, torch.full_like(deltas[:, :1], (self.far - self.near) / self.samples_per_ray)], dim=1)
        alpha = 1.0 - torch.exp(-sigma * deltas)
        transmittance = torch.cumprod(
            torch.cat([torch.ones((ray_count, 1), device=self.device), 1.0 - alpha + 1e-8], dim=1), dim=1
        )[:, :-1]
        weights = alpha * transmittance
        rgb = torch.sum(weights[..., None] * colors, dim=1)
        opacity = torch.sum(weights, dim=1)
        return rgb, opacity

    def _train_update(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        if not self.observations:
            return
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            for _ in range(self.train_iterations):
                origins, directions, targets, masks = self._sample_rays(self.rays_per_batch)
                prediction, opacity = self._render(model, origins, directions, training=True)
                rgb_loss = F.mse_loss(prediction, targets)
                opacity_loss = F.binary_cross_entropy(torch.clamp(opacity, 1e-5, 1.0 - 1e-5), masks)
                loss = rgb_loss + 0.1 * opacity_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            model.eval()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_update_time = time.perf_counter() - start

    @torch.no_grad()
    def _render_in_chunks(self, model, origins: torch.Tensor, directions: torch.Tensor, chunk: int = 2048):
        rgbs, opacities = [], []
        for start in range(0, len(origins), chunk):
            rgb, opacity = self._render(model, origins[start:start + chunk], directions[start:start + chunk])
            rgbs.append(rgb.cpu())
            opacities.append(opacity.cpu())
        return torch.cat(rgbs), torch.cat(opacities)

    def score_views(self, candidate_views: np.ndarray) -> np.ndarray:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        origins = []
        directions = []
        for candidate in candidate_views:
            ray_origins, ray_directions = virtual_camera_rays(
                candidate,
                self.camera_distance,
                self.score_resolution,
                self.score_resolution,
                self.fov_deg,
            )
            origins.append(ray_origins.reshape(-1, 3))
            directions.append(ray_directions.reshape(-1, 3))
        origins_t = torch.from_numpy(np.concatenate(origins, axis=0)).to(self.device)
        directions_t = torch.from_numpy(np.concatenate(directions, axis=0)).to(self.device)
        ensemble_rgb = []
        ensemble_opacity = []
        for model in self.models:
            rgb, opacity = self._render_in_chunks(model, origins_t, directions_t)
            ensemble_rgb.append(rgb)
            ensemble_opacity.append(opacity)
        colors = torch.stack(ensemble_rgb, dim=0)
        opacities = torch.stack(ensemble_opacity, dim=0)
        variance = torch.mean(torch.sum((colors - colors.mean(dim=0, keepdim=True)) ** 2, dim=-1), dim=0)
        roi = opacities.mean(dim=0) > self.opacity_threshold
        self.roi_fraction = float(roi.float().mean().item())
        rays_per_view = self.score_resolution ** 2
        variance = variance.reshape(len(candidate_views), rays_per_view)
        roi = roi.reshape(len(candidate_views), rays_per_view)
        scores = torch.sum(variance * roi, dim=1).numpy().astype(np.float32)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_score_time = time.perf_counter() - start
        self.last_scores = scores
        return scores

    def diagnostics(self):
        out = super().diagnostics()
        gpu_memory = 0.0
        if self.device.type == "cuda":
            gpu_memory = float(torch.cuda.max_memory_allocated() / (1024.0 ** 2))
        out.update({
            "device": str(self.device),
            "ensemble_size": self.ensemble_size,
            "training_views": len(self.observations),
            "valid_surface": bool(self.roi_fraction > 1e-4),
            "roi_fraction": float(self.roi_fraction),
            "gpu_memory_mb": gpu_memory,
        })
        return out
