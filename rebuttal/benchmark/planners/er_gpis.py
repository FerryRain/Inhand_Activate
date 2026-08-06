from __future__ import annotations

import time
from typing import Dict, Tuple

import gpytorch
import numpy as np
import open3d as o3d
import torch

from .base import Planner


class InverseMultiquadricKernel(gpytorch.kernels.Kernel):
    """Inverse-multiquadric kernel used by the official ER-GPIS code."""

    has_lengthscale = True

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            dist2 = ((x1 - x2) ** 2).sum(dim=-1)
        else:
            diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)
            dist2 = (diff ** 2).sum(dim=-1)
        lengthscale2 = self.lengthscale.squeeze() ** 2
        return torch.rsqrt(dist2 + lengthscale2)


class ExplorationGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(InverseMultiquadricKernel())

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class ERGPISPlanner(Planner):
    """Visual NBV adaptation of the ER-GPIS exploration model.

    ER-GPIS was proposed for contact exploration.  This adapter retains its
    exploration-GP principle: surface observations and normal-offset dummy
    samples train a separate E-GPIS, and candidate directions are ranked only
    by E-GPIS posterior uncertainty.  The original tactile local sliding and
    contact-recovery controllers are intentionally excluded because every
    method in this benchmark must execute the same three reorientation actions.
    """

    name = "er_gpis"

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        pcfg = cfg["planners"]["er_gpis"]
        self.voxel_size = float(pcfg.get("voxel_size", 0.002))
        self.max_train = int(pcfg.get("max_train", 900))
        self.train_iters = int(pcfg.get("train_iters", 40))
        self.learning_rate = float(pcfg.get("learning_rate", 0.1))
        self.noise = float(pcfg.get("noise", 1e-4))
        self.dummy_offset = float(pcfg.get("dummy_offset_scaled", 0.04))
        self.radial_samples = int(pcfg.get("radial_samples", 24))
        self.radial_min = float(pcfg.get("radial_min_scaled", 0.25))
        self.radial_max = float(pcfg.get("radial_max_scaled", 1.20))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.likelihood = None
        self.center = None
        self.radius = None
        self.train_points = 0
        self.update_index = 0

    def reset(self, initial_observation, fusion):
        self.update_index = 0
        self._estimate(fusion)

    def update(self, observation, fusion):
        self.update_index += 1
        self._estimate(fusion)

    def _training_set(self, fusion) -> Tuple[np.ndarray, np.ndarray]:
        cloud = o3d.geometry.PointCloud(fusion.cloud)
        if self.voxel_size > 0:
            cloud = cloud.voxel_down_sample(self.voxel_size)
        if len(cloud.points) < 20:
            raise ValueError("ER-GPIS point cloud is too small after preprocessing")

        points = np.asarray(cloud.points, dtype=np.float32)
        center = np.asarray(cloud.get_oriented_bounding_box().center, dtype=np.float32)
        radius = max(float(np.max(np.linalg.norm(points - center[None, :], axis=1))), 1e-6)

        # Each retained surface sample generates two normal-offset dummy
        # samples, so cap the surface set to keep the total exact-GP size fixed.
        surface_cap = max(8, self.max_train // 3)
        if len(points) > surface_cap:
            seed = int(self.cfg["seed"]) + 7919 * int(self.update_index)
            indices = np.random.RandomState(seed).choice(
                len(points), size=surface_cap, replace=False
            )
            sampled = o3d.geometry.PointCloud()
            sampled.points = o3d.utility.Vector3dVector(points[indices].astype(np.float64))
        else:
            sampled = cloud

        search_radius = max(4.0 * self.voxel_size, 0.01)
        sampled.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=search_radius, max_nn=30
            )
        )
        surface = np.asarray(sampled.points, dtype=np.float32)
        normals = np.asarray(sampled.normals, dtype=np.float32)
        radial = surface - center[None, :]
        radial /= np.linalg.norm(radial, axis=1, keepdims=True) + 1e-12
        invalid = ~np.all(np.isfinite(normals), axis=1)
        normals[invalid] = radial[invalid]
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
        flip = np.sum(normals * radial, axis=1) < 0.0
        normals[flip] *= -1.0

        surface_n = (surface - center[None, :]) / radius
        inside_n = surface_n - self.dummy_offset * normals
        outside_n = surface_n + self.dummy_offset * normals
        train_x = np.concatenate([surface_n, inside_n, outside_n], axis=0)
        train_y = np.concatenate(
            [
                np.zeros(len(surface_n), dtype=np.float32),
                np.ones(len(surface_n), dtype=np.float32),
                -np.ones(len(surface_n), dtype=np.float32),
            ],
            axis=0,
        )
        self.center = center
        self.radius = radius
        return train_x.astype(np.float32), train_y

    def _estimate(self, fusion):
        if self.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        train_x, train_y = self._training_set(fusion)
        tx = torch.from_numpy(train_x).to(self.device)
        ty = torch.from_numpy(train_y).to(self.device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        likelihood.noise = torch.tensor(self.noise, device=self.device)
        model = ExplorationGP(tx, ty, likelihood).to(self.device)
        with torch.no_grad():
            model.covar_module.base_kernel.lengthscale = torch.tensor(
                0.15, device=self.device
            )
            model.covar_module.outputscale = torch.tensor(1.0, device=self.device)
            model.mean_module.constant = torch.tensor(0.0, device=self.device)
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        # GPyTorch's default threshold switches a 900-point exact GP to CG.
        # The IMQ covariance can be poorly conditioned for dense RGB-D surface
        # samples, so retain the exact Cholesky solver used by this baseline.
        with gpytorch.settings.max_cholesky_size(self.max_train + 1), \
                gpytorch.settings.cholesky_jitter(1e-4):
            for _ in range(self.train_iters):
                optimizer.zero_grad(set_to_none=True)
                loss = -mll(model(tx), ty)
                loss.backward()
                optimizer.step()
        model.eval()
        likelihood.eval()
        self.model = model
        self.likelihood = likelihood
        self.train_points = int(len(train_x))
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.last_update_time = time.perf_counter() - start

    def score_views(self, candidate_views: np.ndarray) -> np.ndarray:
        if self.model is None or self.likelihood is None:
            raise RuntimeError("ERGPISPlanner must be reset before scoring")
        if self.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        # Keep the benchmark-owned candidate set byte-identical across
        # planners; normalization must not mutate the caller's float32 array.
        directions = np.asarray(candidate_views, dtype=np.float32).copy()
        directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
        radial = torch.linspace(
            self.radial_min,
            self.radial_max,
            self.radial_samples,
            device=self.device,
        )
        dirs_t = torch.from_numpy(directions).to(self.device)
        query = (dirs_t[:, None, :] * radial[None, :, None]).reshape(-1, 3)
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_train + 1), \
                gpytorch.settings.cholesky_jitter(1e-4), gpytorch.settings.fast_pred_var():
            posterior = self.likelihood(self.model(query))
            variances = posterior.variance.reshape(len(directions), self.radial_samples)
            scores = torch.max(variances, dim=1).values
        result = scores.detach().cpu().numpy().astype(np.float32)
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.last_score_time = time.perf_counter() - start
        self.last_scores = result
        return result

    def diagnostics(self):
        out = super().diagnostics()
        out.update({
            "device": self.device,
            "kernel": "inverse_multiquadric",
            "train_points": int(self.train_points),
            "radial_samples": int(self.radial_samples),
            "adaptation": "E-GPIS global surface-uncertainty core",
        })
        return out
