"""
@FileName：NBV_gpis_Field.py
@Description：
@Author：Ferry
@Time：2026 1/24/26 1:42 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""
"""
@FileName：Uncertainty_compute_and_plot.py
@Description：
@Author：Ferry
@Time：2026 1/24/26 2:02 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""
import argparse


import math
import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Any

import numpy as np
import open3d as o3d
import torch
import gpytorch

import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import re
import glob
from pathlib import Path
from typing import Dict

_RECON_RE = re.compile(r"recon_0_(\d+)\.ply$", re.IGNORECASE)

def _parse_recon_frame_id(p: str) -> int:
    m = _RECON_RE.search(os.path.basename(p))
    if m is None:
        return -1
    try:
        return int(m.group(1))
    except Exception:
        return -1

def find_last_recon_in_dir(seq_dir: str) -> str:
    """Return path of the last recon_0_*.ply in seq_dir; raise if none."""
    paths = glob.glob(os.path.join(seq_dir, "recon_0_*.ply"))
    if len(paths) == 0:
        raise FileNotFoundError(f"No recon_0_*.ply found in: {seq_dir}")
    paths = sorted(paths, key=_parse_recon_frame_id)
    return paths[-1]


# ------------------------------ Repro ------------------------------
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------ IO ------------------------------
def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    ext = path.lower().split(".")[-1]
    if ext in ["ply", "pcd", "xyz", "xyzn", "xyzrgb"]:
        pcd = o3d.io.read_point_cloud(path)
    elif ext in ["txt"]:
        pts = np.loadtxt(path).astype(np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError("TXT must contain at least 3 columns: x y z")
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[:, :3]))
    else:
        raise ValueError(f"Unsupported file extension: .{ext}")

    if len(pcd.points) == 0:
        raise ValueError("Loaded point cloud is empty.")
    return pcd


def remove_non_finite(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    # Open3D versions differ: may return (pcd, indices) or modify in-place
    out = pcd.remove_non_finite_points()
    if isinstance(out, tuple):
        return out[0]
    return pcd


# ------------------------------ Geometry utils ------------------------------
def compute_median_1nn(pcd: o3d.geometry.PointCloud, sample_n: int = 2000) -> float:
    pts = np.asarray(pcd.points)
    n = pts.shape[0]
    if n < 5:
        return float(np.linalg.norm(pts.std(axis=0)) + 1e-6)

    idx = np.random.choice(n, size=min(sample_n, n), replace=False)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    dists = []
    for i in idx:
        _, _, nn_dist2 = kdtree.search_knn_vector_3d(pcd.points[i], 2)
        if len(nn_dist2) >= 2:
            dists.append(math.sqrt(nn_dist2[1]))
    if len(dists) == 0:
        return float(np.linalg.norm(pts.std(axis=0)) + 1e-6)
    return float(np.median(dists))


def fibonacci_sphere(n: int) -> np.ndarray:
    pts = np.zeros((n, 3), dtype=np.float32)
    phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
    if n == 1:
        pts[0] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return pts

    for i in range(n):
        y = 1.0 - (2.0 * i) / (n - 1)
        r = math.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        x = math.cos(theta) * r
        z = math.sin(theta) * r
        pts[i] = np.array([x, y, z], dtype=np.float32)
    return pts


def planar_polar_dirs(n: int) -> np.ndarray:
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    dirs = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], axis=1).astype(np.float32)
    return dirs


def rotation_matrix_from_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return np.eye(3, dtype=np.float64) if c > 0 else -np.eye(3, dtype=np.float64)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1 - c) / (s * s))
    return R


def normalize01(x: np.ndarray) -> np.ndarray:
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


# ------------------------------ GPIS (Exact GP surrogate) ------------------------------
class GPISModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


@dataclass
class GPISConfig:
    max_train: int = 2000
    train_iters: int = 60
    lr: float = 0.15
    add_outer_samples: bool = True
    outer_offset_scale: float = 1.0   # multiplied by median_1nn (world units)
    noise: float = 1e-4


# ------------------------------ NBVConfigV3 ------------------------------
@dataclass
class NBVConfigV3:
    n_dirs: int = 2000
    planar: bool = False

    # ray marching for first-hit detection (scaled parameter t)
    t_max_scale: float = 2.2
    ray_step_scale: float = 0.02
    hit_eps_world: float = 0.01

    # miss depth inference (spherical neighbor interpolation)
    interp_k: int = 8
    interp_min_cos: float = 0.6
    interp_tau: float = 0.08

    # scoring near t0 (per-direction band query)
    band_halfwidth_scale: float = 0.06
    band_samples: int = 24

    # novelty base
    novelty_beta: float = 4.0

    # receptive field on direction sphere
    rf_k: int = 32
    rf_min_cos: float = 0.96     # ~16 deg; set None to disable cutoff

    # reducers
    unc_rf_reduce: str = "max"   # "max" or "mean"
    nov_rf_reduce: str = "mean"  # "mean" or "max"

    # uncertainty visualization
    delta_mu: float = 0.15
    grid_res: int = 35


def build_training_set(pcd: o3d.geometry.PointCloud,
                       center: np.ndarray,
                       radius: float,
                       median_1nn: float,
                       cfg: GPISConfig) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(pcd.points).astype(np.float32)
    n = pts.shape[0]
    if n > cfg.max_train:
        idx = np.random.choice(n, size=cfg.max_train, replace=False)
        pts_surf = pts[idx]
    else:
        pts_surf = pts

    X_list = [pts_surf]
    y_list = [np.zeros((pts_surf.shape[0],), dtype=np.float32)]

    if cfg.add_outer_samples:
        dirs = pts_surf - center[None, :]
        norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
        dirs = dirs / norms
        offset = cfg.outer_offset_scale * median_1nn
        pts_out = pts_surf + offset * dirs
        X_list.append(pts_out)
        y_list.append(np.ones((pts_out.shape[0],), dtype=np.float32))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    # normalize to scaled coordinates around center
    Xn = (X - center[None, :]) / max(radius, 1e-6)
    return Xn.astype(np.float32), y.astype(np.float32)


def train_gpis(train_x: np.ndarray, train_y: np.ndarray, device: str, cfg: GPISConfig):
    tx = torch.from_numpy(train_x).to(device)
    ty = torch.from_numpy(train_y).to(device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = torch.tensor(cfg.noise, device=device)

    model = GPISModel(tx, ty, likelihood).to(device)

    # mild initial values (safe defaults)
    with torch.no_grad():
        model.covar_module.base_kernel.lengthscale = torch.tensor(0.08, device=device)
        model.covar_module.outputscale = torch.tensor(1.0, device=device)
        model.mean_module.constant = torch.tensor(0.5, device=device)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(cfg.train_iters):
        optimizer.zero_grad(set_to_none=True)
        output = model(tx)
        loss = -mll(output, ty)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()
    return model, likelihood


# =========================
# Kernel state helpers
# =========================
def extract_kernel_state(model, likelihood) -> Dict[str, Any]:
    with torch.no_grad():
        ls = model.covar_module.base_kernel.lengthscale.detach().cpu().view(-1).numpy().copy()
        os_ = float(model.covar_module.outputscale.detach().cpu().item())
        m = float(model.mean_module.constant.detach().cpu().item())
        nz = float(likelihood.noise.detach().cpu().item())
    return {"lengthscale": ls, "outputscale": os_, "mean": m, "noise": nz}



def build_gpis_from_state(
    train_x: np.ndarray,
    train_y: np.ndarray,
    device: str,
    state: Dict[str, Any],
) -> Tuple[gpytorch.models.ExactGP, gpytorch.likelihoods.GaussianLikelihood]:
    """Build ExactGP with fixed hyperparameters; NO optimization."""
    tx = torch.from_numpy(train_x).to(device)
    ty = torch.from_numpy(train_y).to(device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = GPISModel(tx, ty, likelihood).to(device)

    # apply fixed hypers (match exact parameter shapes)
    with torch.no_grad():
        # lengthscale: match shape exactly (could be [1,3] or [1,1,3] depending on gpytorch)
        ls_param = model.covar_module.base_kernel.lengthscale
        ls_val = torch.as_tensor(state["lengthscale"], device=device, dtype=ls_param.dtype).view(-1)
        if ls_val.numel() != ls_param.numel():
            raise RuntimeError(
                f"lengthscale numel mismatch: state={ls_val.numel()} vs param={ls_param.numel()} "
                f"(state shape {tuple(ls_val.shape)}, param shape {tuple(ls_param.shape)})"
            )
        ls_param.copy_(ls_val.view_as(ls_param))

        # outputscale
        os_param = model.covar_module.outputscale
        os_val = torch.as_tensor(float(state["outputscale"]), device=device, dtype=os_param.dtype).view_as(os_param)
        os_param.copy_(os_val)

        # mean constant
        m_param = model.mean_module.constant
        m_val = torch.as_tensor(float(state["mean"]), device=device, dtype=m_param.dtype).view_as(m_param)
        m_param.copy_(m_val)

        # noise
        n_param = likelihood.noise
        n_val = torch.as_tensor(float(state["noise"]), device=device, dtype=n_param.dtype).view_as(n_param)
        n_param.copy_(n_val)

    model.eval()
    likelihood.eval()
    return model, likelihood



@torch.no_grad()
def gpis_predict(model, likelihood, Xq: torch.Tensor, batch: int = 200000):
    means = []
    vars_ = []
    with gpytorch.settings.fast_pred_var():
        for i in range(0, Xq.shape[0], batch):
            xb = Xq[i:i + batch]
            pred = likelihood(model(xb))
            means.append(pred.mean.detach())
            vars_.append(pred.variance.detach())
    return torch.cat(means, dim=0), torch.cat(vars_, dim=0)


# ------------------------------ Hit depth (KDTree + ray marching) ------------------------------
def estimate_hit_depth_scaled(pcd: o3d.geometry.PointCloud,
                              kdtree: o3d.geometry.KDTreeFlann,
                              center_world: np.ndarray,
                              radius: float,
                              dir_world: np.ndarray,
                              t_max_scaled: float,
                              step_scaled: float,
                              hit_eps_world: float) -> Optional[float]:
    """
    Ray: x(t) = center + (t * radius) * dir, with t in scaled units.
    Hit if NN distance < hit_eps_world.
    Return first t_hit (scaled) or None.
    """
    t = 0.0
    u = dir_world / (np.linalg.norm(dir_world) + 1e-12)

    while t <= t_max_scaled:
        xw = center_world + (t * radius) * u
        _, _, dist2 = kdtree.search_knn_vector_3d(xw, 1)
        if len(dist2) > 0 and dist2[0] < (hit_eps_world * hit_eps_world):
            return float(t)
        t += step_scaled
    return None


# ------------------------------ Miss depth inference on sphere ------------------------------
def fill_missing_depths(dirs: np.ndarray,
                        t_hit: np.ndarray,
                        k: int = 8,
                        min_cos: float = 0.6,
                        tau: float = 0.08) -> np.ndarray:
    """
    dirs: (N,3) unit directions
    t_hit: (N,) scaled depth; miss => np.nan
    return t0: (N,) scaled depth (hit keeps its own), miss interpolated from nearby hit dirs
    """
    N = dirs.shape[0]
    hit_mask = np.isfinite(t_hit)
    hit_dirs = dirs[hit_mask]
    hit_t = t_hit[hit_mask]
    if hit_dirs.shape[0] < 3:
        return np.full((N,), np.nan, dtype=np.float32)

    t0 = t_hit.copy().astype(np.float32)

    for i in range(N):
        if np.isfinite(t0[i]):
            continue
        u = dirs[i]
        cos = hit_dirs @ u  # (Nh,)
        j_sorted = np.argsort(-cos)
        j_top = j_sorted[:k]
        if cos[j_top[0]] < min_cos:
            t0[i] = np.nan
            continue

        c = cos[j_top]
        w = np.exp((c - c.max()) / max(tau, 1e-6))
        w = w / (w.sum() + 1e-12)
        t0[i] = float(np.sum(w * hit_t[j_top]))
    return t0


# ------------------------------ RF neighbors on direction sphere ------------------------------
def build_dir_neighbors(dirs: np.ndarray, k: int, min_cos: Optional[float]) -> np.ndarray:
    """
    Build receptive-field neighbor indices on the direction sphere.

    dirs: (N,3) unit
    return: nbrs (N,k) int32, invalid entries are -1
    """
    dirs = dirs.astype(np.float32)
    N = dirs.shape[0]
    if N == 0:
        return np.zeros((0, 0), dtype=np.int32)
    k = int(min(max(k, 1), N))

    # cosine similarity matrix (N,N)
    cos = (dirs @ dirs.T).astype(np.float32)

    # top-k indices per row (unordered), then sort by cosine descending
    idx = np.argpartition(-cos, kth=k - 1, axis=1)[:, :k]  # (N,k)
    row = np.arange(N)[:, None]
    top_cos = cos[row, idx]
    order = np.argsort(-top_cos, axis=1)
    nbrs = idx[row, order].astype(np.int32)  # (N,k) sorted

    # apply cutoff
    if min_cos is not None:
        mask = cos[row, nbrs] >= float(min_cos)
        nbrs[~mask] = -1

    return nbrs


@torch.no_grad()
def compute_band_max_var_all_dirs(
    model, likelihood,
    dirs: np.ndarray,
    t0: np.ndarray,
    t_max_scaled: float,
    band_halfwidth_scaled: float,
    n_band_samples: int,
    device: str,
) -> np.ndarray:
    """
    For each direction u_i, compute max posterior variance in band [t0-dt, t0+dt].
    This is done in one batched GP query: total points = N * n_band_samples.
    Return: max_var (N,) float32
    """
    dirs_t = torch.from_numpy(dirs.astype(np.float32)).to(device)  # (N,3)
    t0_t = torch.from_numpy(t0.astype(np.float32)).to(device)      # (N,)

    N = dirs_t.shape[0]
    dt = float(band_halfwidth_scaled)

    S = int(max(2, n_band_samples))
    offs = torch.linspace(-dt, dt, S, device=device)               # (S,)
    ts = t0_t[:, None] + offs[None, :]                             # (N,S)
    ts = torch.clamp(ts, 0.0, float(t_max_scaled))

    # invalid t0 -> set all samples to 0 (will be masked later)
    valid = torch.isfinite(t0_t)
    ts[~valid, :] = 0.0

    Xq = (ts[..., None] * dirs_t[:, None, :]).reshape(-1, 3).contiguous()  # (N*S,3)
    _, var = gpis_predict(model, likelihood, Xq, batch=200000)
    var = var.view(N, -1)  # (N,S)

    max_var = torch.max(var, dim=1).values
    max_var[~valid] = 0.0
    return max_var.detach().cpu().numpy().astype(np.float32)


def compute_anchor_novelty_all_dirs(
    kdtree: o3d.geometry.KDTreeFlann,
    center_world: np.ndarray,
    radius: float,
    dirs: np.ndarray,
    t0: np.ndarray,
    median_1nn_world: float,
    novelty_beta: float,
) -> np.ndarray:
    """
    novelty_i = 1 - exp(-(d_nn(x0_i)/d0)^2), where x0_i = center + (t0_i*radius)*u_i.
    Return: novelty (N,) float32, invalid t0 -> 0.
    """
    N = dirs.shape[0]
    novelty = np.zeros((N,), dtype=np.float32)

    d0 = max(float(novelty_beta) * float(median_1nn_world), 1e-6)

    for i in range(N):
        if not np.isfinite(t0[i]):
            novelty[i] = 0.0
            continue
        u = dirs[i]
        x0_world = center_world + (float(t0[i]) * float(radius)) * u
        _, _, dist2 = kdtree.search_knn_vector_3d(x0_world, 1)
        d_nn = math.sqrt(dist2[0]) if len(dist2) > 0 else 1e6
        novelty[i] = float(1.0 - math.exp(- (d_nn / d0) ** 2))
    return novelty


# ------------------------------ Uncertainty viz near surface band ------------------------------
def build_uncertainty_cloud(model, likelihood,
                            center_world: np.ndarray,
                            radius: float,
                            aabb_min_world: np.ndarray,
                            aabb_max_world: np.ndarray,
                            grid_res: int,
                            delta_mu: float,
                            device: str):
    xs = np.linspace(aabb_min_world[0], aabb_max_world[0], grid_res, dtype=np.float32)
    ys = np.linspace(aabb_min_world[1], aabb_max_world[1], grid_res, dtype=np.float32)
    zs = np.linspace(aabb_min_world[2], aabb_max_world[2], grid_res, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=1)

    pts_scaled = (pts - center_world[None, :]) / max(radius, 1e-6)
    Xq = torch.from_numpy(pts_scaled).to(device)

    mu, var = gpis_predict(model, likelihood, Xq, batch=150000)
    mu_np = mu.detach().cpu().numpy()
    var_np = var.detach().cpu().numpy()

    mask = np.abs(mu_np) < float(delta_mu)
    pts_band = pts[mask]
    var_band = var_np[mask]
    if pts_band.shape[0] == 0:
        return None

    v01 = normalize01(var_band)
    # red (high) -> green (low)
    colors = np.stack([v01, 1.0 - v01, np.zeros_like(v01)], axis=1).astype(np.float64)

    pcd_u = o3d.geometry.PointCloud()
    pcd_u.points = o3d.utility.Vector3dVector(pts_band.astype(np.float64))
    pcd_u.colors = o3d.utility.Vector3dVector(colors)
    return pcd_u


# ------------------------------ Visualization helpers ------------------------------
def make_arrow(origin: np.ndarray, direction: np.ndarray, length: float) -> o3d.geometry.TriangleMesh:
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.01 * length,
        cone_radius=0.02 * length,
        cylinder_height=0.75 * length,
        cone_height=0.25 * length,
        resolution=20,
        cylinder_split=4,
        cone_split=1,
    )
    R = rotation_matrix_from_a_to_b(np.array([0, 0, 1], dtype=np.float64), direction.astype(np.float64))
    arrow.rotate(R, center=np.array([0, 0, 0], dtype=np.float64))
    arrow.translate(origin.astype(np.float64))
    return arrow


def set_camera_look_from(vis: o3d.visualization.Visualizer, cam_pos: np.ndarray, lookat: np.ndarray):
    ctr = vis.get_view_control()
    front = (lookat - cam_pos).astype(np.float64)
    front /= (np.linalg.norm(front) + 1e-12)

    up = np.array([0, 0, 1], dtype=np.float64)
    if abs(float(np.dot(front, up))) > 0.95:
        up = np.array([0, 1, 0], dtype=np.float64)

    ctr.set_lookat(lookat.astype(np.float64))
    ctr.set_front(front)
    ctr.set_up(up)
    ctr.set_zoom(0.7)


# ============================================================
# Class wrapper
# ============================================================
class GPISNBVv3:
    """
    GPIS + NBV-v3 (RF uncertainty * RF novelty) wrapper.

    Visualization:
      geoms = [pcd_vis(raw object), arrow(blue), cam_sphere(red), pcd_dirs(score sphere)] + optional pcd_unc
      window: 1280x800
      camera: set_camera_look_from(cam_pos, center)
    """

    def __init__(
            self,
            voxel: float = 0.01,
            max_train: int = 2000,
            train_iters: int = 60,
            device: Optional[str] = None,

            grid_res: int = 35,
            n_dirs: int = 2000,
            planar: bool = False,
            hit_eps: float = 0.01,

            novelty_beta: float = 4.0,
            band_halfwidth: float = 0.06,
            delta_mu: float = 0.15,

            # RF params
            rf_k: int = 32,
            rf_min_cos: float = 0.96,
            unc_rf_reduce: str = "max",
            nov_rf_reduce: str = "mean",

            gpis_lr: float = 0.15,
            gpis_add_outer_samples: bool = False,
            gpis_outer_offset_scale: float = 1.0,
            gpis_noise: float = 1e-4,

            t_max_scale: float = 2.2,
            ray_step_scale: float = 0.02,

            interp_k: int = 8,
            interp_min_cos: float = 0.6,
            interp_tau: float = 0.08,

            band_samples: int = 24,
    ):
        self.voxel = float(voxel)
        self.max_train = int(max_train)
        self.train_iters = int(train_iters)
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        self.grid_res = int(grid_res)
        self.n_dirs = int(n_dirs)
        self.planar = bool(planar)
        self.hit_eps = float(hit_eps)

        self.novelty_beta = float(novelty_beta)
        self.band_halfwidth = float(band_halfwidth)
        self.delta_mu = float(delta_mu)

        self.rf_k = int(rf_k)
        self.rf_min_cos = float(rf_min_cos)
        self.unc_rf_reduce = str(unc_rf_reduce)
        self.nov_rf_reduce = str(nov_rf_reduce)

        self.gpis_lr = float(gpis_lr)
        self.gpis_add_outer_samples = bool(gpis_add_outer_samples)
        self.gpis_outer_offset_scale = float(gpis_outer_offset_scale)
        self.gpis_noise = float(gpis_noise)

        self.t_max_scale = float(t_max_scale)
        self.ray_step_scale = float(ray_step_scale)

        self.interp_k = int(interp_k)
        self.interp_min_cos = float(interp_min_cos)
        self.interp_tau = float(interp_tau)

        self.band_samples = int(band_samples)

        # internal states
        self.center: Optional[np.ndarray] = None
        self.radius: Optional[float] = None
        self.median_1nn: Optional[float] = None
        self.hit_eps_world: Optional[float] = None

        self.model = None
        self.likelihood = None

        # keep ORIGINAL raw point cloud for visualization
        self.pcd_raw: Optional[o3d.geometry.PointCloud] = None

        # viz objects
        self.pcd: Optional[o3d.geometry.PointCloud] = None          # pcd_vis (raw object cloud)
        self.pcd_dirs: Optional[o3d.geometry.PointCloud] = None     # direction-score sphere points
        self.pcd_unc: Optional[o3d.geometry.PointCloud] = None      # variance near |mu|<delta_mu
        self.arrow: Optional[o3d.geometry.TriangleMesh] = None
        self.cam_sphere: Optional[o3d.geometry.TriangleMesh] = None
        self.cam_pos: Optional[np.ndarray] = None

        # NBV results
        self.dirs: Optional[np.ndarray] = None
        self.scores: Optional[np.ndarray] = None
        self.best_idx: Optional[int] = None
        self.best_dir: Optional[np.ndarray] = None
        self.best_score: Optional[float] = None
        self.best_t0_scaled: Optional[float] = None
        self._last_nbv: Optional[Dict[str, Any]] = None

        # Optional debug buffers
        self._unc_rf: Optional[np.ndarray] = None
        self._nov_rf: Optional[np.ndarray] = None
        self._unc_base: Optional[np.ndarray] = None
        self._nov_base: Optional[np.ndarray] = None

        self._ref_geom_cache: Dict[str, Dict[str, Any]] = {}

    def fit_kernel_once(self, pcd: Union[str, o3d.geometry.PointCloud], seed: int = 0, verbose: bool = True) -> Dict[
        str, Any]:
        """
        Fit GP hyperparameters ONCE for one object (using one reference frame).
        Returns kernel_state dict. Subsequent estimate() calls can reuse it.
        """
        set_seed(seed)

        # ---- load ----
        if isinstance(pcd, o3d.geometry.PointCloud):
            if len(pcd.points) == 0:
                raise ValueError("Input point cloud is empty.")
            pcd_in = o3d.geometry.PointCloud(pcd)
        elif isinstance(pcd, str):
            pcd_in = load_point_cloud(pcd)
        else:
            raise TypeError("pcd must be a path (str) or open3d.geometry.PointCloud")

        pcd_in = remove_non_finite(pcd_in)

        # ---- preprocess ----
        if self.voxel > 0:
            pcd_proc = pcd_in.voxel_down_sample(self.voxel)
        else:
            pcd_proc = pcd_in

        if len(pcd_proc.points) < 20:
            raise ValueError("Point cloud too small after preprocessing.")

        pts = np.asarray(pcd_proc.points).astype(np.float32)
        obb = pcd_proc.get_oriented_bounding_box()
        center = np.asarray(obb.center, dtype=np.float32)

        radius = float(np.max(np.linalg.norm(pts - center[None, :], axis=1)))
        radius = max(radius, 1e-6)

        median_1nn = compute_median_1nn(pcd_proc, sample_n=2000)

        gpcfg = GPISConfig(
            max_train=self.max_train,
            train_iters=self.train_iters,  # only here we optimize
            lr=self.gpis_lr,
            add_outer_samples=self.gpis_add_outer_samples,
            outer_offset_scale=self.gpis_outer_offset_scale,
            noise=self.gpis_noise,
        )

        train_x, train_y = build_training_set(pcd_proc, center, radius, median_1nn, gpcfg)
        model, likelihood = train_gpis(train_x, train_y, self.device, gpcfg)

        state = extract_kernel_state(model, likelihood)
        if verbose:
            print("[KernelFit] lengthscale:", state["lengthscale"])
            print("[KernelFit] outputscale:", state["outputscale"])
            print("[KernelFit] mean:", state["mean"])
            print("[KernelFit] noise:", state["noise"])

        return state

    def _get_ref_geom_from_last_frame(self, pcd_path: str, verbose: bool = False) -> Optional[Dict[str, Any]]:
        """
        If pcd_path looks like .../recon_0_XXXXXX.ply, then we treat its directory as a sequence,
        and compute reference geometry (center/radius/median_1nn/hit_eps) from the LAST frame once.
        Cache per directory.
        """
        if not isinstance(pcd_path, str):
            return None

        base = os.path.basename(pcd_path)
        if _RECON_RE.search(base) is None:
            return None  # not a recon sequence file

        seq_dir = os.path.dirname(os.path.abspath(pcd_path))
        if seq_dir in self._ref_geom_cache:
            return self._ref_geom_cache[seq_dir]

        # locate last frame
        last_path = find_last_recon_in_dir(seq_dir)

        # load & preprocess last frame
        pcd_ref = load_point_cloud(last_path)
        pcd_ref = remove_non_finite(pcd_ref)

        if self.voxel > 0:
            pcd_ref = pcd_ref.voxel_down_sample(self.voxel)

        if len(pcd_ref.points) < 20:
            raise ValueError(f"[RefGeom] Last frame too small after preprocessing: {last_path}")

        pts = np.asarray(pcd_ref.points).astype(np.float32)
        obb = pcd_ref.get_oriented_bounding_box()
        center = np.asarray(obb.center, dtype=np.float32)

        radius = float(np.max(np.linalg.norm(pts - center[None, :], axis=1)))
        radius = max(radius, 1e-6)

        median_1nn = compute_median_1nn(pcd_ref, sample_n=2000)

        # keep hit_eps consistent with your original logic
        hit_eps_world = self.hit_eps if self.hit_eps > 0 else max(2.0 * median_1nn, 1e-4)

        ref = {
            "center": center,
            "radius": radius,
            "median_1nn": float(median_1nn),
            "hit_eps_world": float(hit_eps_world),
            "ref_path": last_path,
        }
        self._ref_geom_cache[seq_dir] = ref

        if verbose:
            print(f"[RefGeom] seq_dir={seq_dir}")
            print(f"[RefGeom] using LAST frame: {os.path.basename(last_path)}")
            print(f"[RefGeom] center={center.tolist()}")
            print(f"[RefGeom] radius={radius:.6f}")
            print(f"[RefGeom] median_1nn={median_1nn:.6f}")
            print(f"[RefGeom] hit_eps_world={hit_eps_world:.6f}")

        return ref


    def estimate(
            self,
            pcd: Union[str, o3d.geometry.PointCloud],
            seed: int = 0,
            verbose: bool = True,
            kernel_state: Optional[Dict[str, Any]] = None,  # NEW
    ) -> Dict[str, Any]:

        set_seed(seed)

        # ---- load ----
        if isinstance(pcd, o3d.geometry.PointCloud):
            if len(pcd.points) == 0:
                raise ValueError("Input point cloud is empty.")
            pcd_in = o3d.geometry.PointCloud(pcd)
        elif isinstance(pcd, str):
            pcd_in = load_point_cloud(pcd)
        else:
            raise TypeError("pcd must be a path (str) or open3d.geometry.PointCloud")

        pcd_in = remove_non_finite(pcd_in)

        # cache original raw point cloud for visualization (no downsample, keep original colors)
        self.pcd_raw = o3d.geometry.PointCloud(pcd_in)

        # ---- preprocess: voxel downsample for GPIS/NBV compute ----
        if self.voxel > 0:
            pcd_proc = pcd_in.voxel_down_sample(self.voxel)
        else:
            pcd_proc = pcd_in

        if len(pcd_proc.points) < 20:
            raise ValueError("Point cloud too small after preprocessing.")

        pts = np.asarray(pcd_proc.points).astype(np.float32)

        # ---------------------------------------------------------
        # NEW: use LAST-frame reference geometry for center & radius
        # ---------------------------------------------------------
        ref = None
        if isinstance(pcd, str):
            ref = self._get_ref_geom_from_last_frame(pcd, verbose=verbose)

        if ref is not None:
            center = ref["center"].astype(np.float32)
            radius = float(ref["radius"])
            median_1nn = float(ref["median_1nn"])
            hit_eps_world = float(ref["hit_eps_world"])
        else:
            # fallback to per-frame geometry (old behavior)
            obb = pcd_proc.get_oriented_bounding_box()
            center = np.asarray(obb.center, dtype=np.float32)

            radius = float(np.max(np.linalg.norm(pts - center[None, :], axis=1)))
            radius = max(radius, 1e-6)

            median_1nn = compute_median_1nn(pcd_proc, sample_n=2000)
            hit_eps_world = self.hit_eps if self.hit_eps > 0 else max(2.0 * median_1nn, 1e-4)

        # optional guard: avoid pathological small radius due to bad ref (rare)
        radius = max(radius, 1e-6)

        if verbose:
            print(f"[Info] points={len(pcd_proc.points)}")
            print(f"[Info] center={center.tolist()}")
            print(f"[Info] radius={radius:.6f} (world units)")
            print(f"[Info] median_1nn={median_1nn:.6f} (world units)")
            print(f"[Info] hit_eps={hit_eps_world:.6f} (world units)")
            print(f"[Info] device={self.device}")

        # ---- configs ----
        gpcfg = GPISConfig(
            max_train=self.max_train,
            train_iters=self.train_iters,
            lr=self.gpis_lr,
            add_outer_samples=self.gpis_add_outer_samples,
            outer_offset_scale=self.gpis_outer_offset_scale,
            noise=self.gpis_noise,
        )
        nbv = NBVConfigV3(
            n_dirs=self.n_dirs,
            planar=self.planar,
            t_max_scale=self.t_max_scale,
            ray_step_scale=self.ray_step_scale,
            hit_eps_world=hit_eps_world,
            interp_k=self.interp_k,
            interp_min_cos=self.interp_min_cos,
            interp_tau=self.interp_tau,
            band_halfwidth_scale=self.band_halfwidth,
            band_samples=self.band_samples,
            novelty_beta=self.novelty_beta,
            rf_k=self.rf_k,
            rf_min_cos=self.rf_min_cos,
            unc_rf_reduce=self.unc_rf_reduce,
            nov_rf_reduce=self.nov_rf_reduce,
            delta_mu=self.delta_mu,
            grid_res=self.grid_res,
        )

        # ---- train GPIS surrogate ----
        # ---- train/build GPIS surrogate ----
        train_x, train_y = build_training_set(pcd_proc, center, radius, median_1nn, gpcfg)

        if kernel_state is None:
            # default behavior: optimize hypers (old behavior)
            model, likelihood = train_gpis(train_x, train_y, self.device, gpcfg)
        else:
            # NEW: fixed hypers, no optimization
            model, likelihood = build_gpis_from_state(train_x, train_y, self.device, kernel_state)

        # ---- uncertainty visualization cloud (near |mu|<delta_mu) ----
        aabb = pcd_proc.get_axis_aligned_bounding_box()
        minb = np.asarray(aabb.get_min_bound(), dtype=np.float32)
        maxb = np.asarray(aabb.get_max_bound(), dtype=np.float32)
        margin = 0.15 * (maxb - minb)
        minb2 = minb - margin
        maxb2 = maxb + margin

        pcd_unc = build_uncertainty_cloud(
            model, likelihood,
            center_world=center,
            radius=radius,
            aabb_min_world=minb2,
            aabb_max_world=maxb2,
            grid_res=nbv.grid_res,
            delta_mu=nbv.delta_mu,
            device=self.device
        )

        # ---- directions ----
        if nbv.planar:
            dirs = planar_polar_dirs(nbv.n_dirs)
        else:
            dirs = fibonacci_sphere(nbv.n_dirs)
        dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)

        # ---- KDTree for hit detection + novelty ----
        kdtree = o3d.geometry.KDTreeFlann(pcd_proc)

        # 1) per-direction first hit depth
        t_hit = np.full((dirs.shape[0],), np.nan, dtype=np.float32)
        for i in range(dirs.shape[0]):
            th = estimate_hit_depth_scaled(
                pcd_proc, kdtree,
                center_world=center,
                radius=radius,
                dir_world=dirs[i],
                t_max_scaled=nbv.t_max_scale,
                step_scaled=nbv.ray_step_scale,
                hit_eps_world=nbv.hit_eps_world
            )
            if th is not None:
                t_hit[i] = float(th)

        hit_ratio = float(np.isfinite(t_hit).mean())
        if verbose:
            print(f"[NBV-v3] hit_ratio={hit_ratio:.3f} (fraction of directions that hit point cloud)")

        # 2) infer missing depths for miss directions
        t0 = fill_missing_depths(
            dirs=dirs,
            t_hit=t_hit,
            k=nbv.interp_k,
            min_cos=nbv.interp_min_cos,
            tau=nbv.interp_tau
        )

        # 3) score directions with receptive field:
        #    Score(u_i) = UncRF(u_i) * NovRF(u_i)
        #    where UncRF aggregates per-dir band max var over angular neighbors,
        #          NovRF aggregates per-dir anchor novelty over angular neighbors.

        # 3.1 per-direction base uncertainty: max var in band around t0
        max_var = compute_band_max_var_all_dirs(
            model, likelihood,
            dirs=dirs,
            t0=t0,
            t_max_scaled=nbv.t_max_scale,
            band_halfwidth_scaled=nbv.band_halfwidth_scale,
            n_band_samples=nbv.band_samples,
            device=self.device,
        )  # (N,)

        # 3.2 per-direction base novelty at anchor x0
        novelty_base = compute_anchor_novelty_all_dirs(
            kdtree=kdtree,
            center_world=center,
            radius=radius,
            dirs=dirs,
            t0=t0,
            median_1nn_world=median_1nn,
            novelty_beta=nbv.novelty_beta,
        )  # (N,)

        # 3.3 build angular RF neighbors
        nbrs = build_dir_neighbors(dirs, k=nbv.rf_k, min_cos=nbv.rf_min_cos)  # (N,k), -1 invalid
        nbrs_clip = np.clip(nbrs, 0, dirs.shape[0] - 1)

        # gather (N,k)
        unc_g = max_var[nbrs_clip]
        nov_g = novelty_base[nbrs_clip]

        valid = (nbrs >= 0)
        # invalidate neighbors whose t0 is nan (no anchor)
        valid = valid & np.isfinite(t0[nbrs_clip])

        # mask invalid
        unc_g = np.where(valid, unc_g, 0.0).astype(np.float32)
        nov_g = np.where(valid, nov_g, 0.0).astype(np.float32)

        # reduce
        unc_reduce = nbv.unc_rf_reduce.lower().strip()
        nov_reduce = nbv.nov_rf_reduce.lower().strip()

        if unc_reduce == "mean":
            cnt = np.maximum(valid.sum(axis=1).astype(np.float32), 1.0)
            unc_rf = unc_g.sum(axis=1) / cnt
        else:  # default "max"
            unc_rf = unc_g.max(axis=1)

        if nov_reduce == "max":
            nov_rf = nov_g.max(axis=1)
        else:  # default "mean"
            cnt = np.maximum(valid.sum(axis=1).astype(np.float32), 1.0)
            nov_rf = nov_g.sum(axis=1) / cnt

        scores = (unc_rf * nov_rf).astype(np.float32)

        if verbose:
            print(f"[NBV-v3+RF] rf_k={nbv.rf_k}, rf_min_cos={nbv.rf_min_cos}")
            print(f"[NBV-v3+RF] unc_rf_reduce={nbv.unc_rf_reduce}, nov_rf_reduce={nbv.nov_rf_reduce}")

        best_idx = int(np.argmax(scores))
        best_dir = dirs[best_idx]
        best_score = float(scores[best_idx])

        if verbose:
            print(f"[NBV-v3] best_idx={best_idx}")
            print(f"[NBV-v3] best_score={best_score:.6e}")
            print(f"[NBV-v3] best_dir={best_dir.tolist()}")
            print(f"[NBV-v3] best_t0_scaled={float(t0[best_idx])}")
            print(f"[NBV-v3] best_unc_rf={float(unc_rf[best_idx]):.6e}, best_nov_rf={float(nov_rf[best_idx]):.6e}")
            print(f"[NBV-v3] best_unc_base={float(max_var[best_idx]):.6e}, best_nov_base={float(novelty_base[best_idx]):.6e}")

        # =========================================================
        # Visualization objects
        # =========================================================
        pcd_vis = o3d.geometry.PointCloud(self.pcd_raw)  # raw object cloud
        if not pcd_vis.has_colors():
            pcd_vis.paint_uniform_color([0.65, 0.65, 0.65])

        origin = center
        arrow_len = 1.2 * radius
        arrow = make_arrow(origin, best_dir, arrow_len)
        arrow.paint_uniform_color([0.1, 0.1, 0.95])  # blue

        cam_dist = 2.0 * radius
        cam_pos = center + cam_dist * best_dir
        cam_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03 * radius, resolution=20)
        cam_sphere.translate(cam_pos.astype(np.float64))
        cam_sphere.paint_uniform_color([0.95, 0.1, 0.1])  # red

        # Direction-score sphere points (debug): red(high) -> blue(low)
        score01 = normalize01(scores)
        sphere_pts = center[None, :] + (1.5 * radius) * dirs
        colors = np.stack([score01, np.zeros_like(score01), 1.0 - score01], axis=1).astype(np.float64)

        pcd_dirs = o3d.geometry.PointCloud()
        pcd_dirs.points = o3d.utility.Vector3dVector(sphere_pts.astype(np.float64))
        pcd_dirs.colors = o3d.utility.Vector3dVector(colors)

        # ---- store state ----
        self.center = center
        self.radius = radius
        self.median_1nn = median_1nn
        self.hit_eps_world = hit_eps_world

        self.model = model
        self.likelihood = likelihood

        self.pcd_unc = pcd_unc

        self.pcd = pcd_vis
        self.arrow = arrow
        self.cam_sphere = cam_sphere
        self.cam_pos = cam_pos
        self.pcd_dirs = pcd_dirs

        self.dirs = dirs
        self.scores = scores
        self.best_idx = best_idx
        self.best_dir = best_dir
        self.best_score = best_score
        self.best_t0_scaled = float(t0[best_idx])

        self._unc_rf = unc_rf
        self._nov_rf = nov_rf
        self._unc_base = max_var
        self._nov_base = novelty_base

        nbv_out = {
            "best_idx": best_idx,
            "best_score": best_score,
            "best_dir": best_dir.astype(np.float32).tolist(),
            "best_t0_scaled": float(t0[best_idx]),
            "center": center.astype(np.float32).tolist(),
            "radius": float(radius),
            "cam_pos": cam_pos.astype(np.float32).tolist(),
            "hit_ratio": hit_ratio,
            "device": self.device,

            # RF decomposition (useful for paper plots / debug)
            "best_unc_rf": float(unc_rf[best_idx]),
            "best_nov_rf": float(nov_rf[best_idx]),
            "best_unc_base": float(max_var[best_idx]),
            "best_nov_base": float(novelty_base[best_idx]),
            "rf_k": int(nbv.rf_k),
            "rf_min_cos": float(nbv.rf_min_cos),
            "unc_rf_reduce": str(nbv.unc_rf_reduce),
            "nov_rf_reduce": str(nbv.nov_rf_reduce),
        }
        self._last_nbv = nbv_out
        return nbv_out

    def viz(self):
        """
        Visualization:
          geoms = [raw object cloud, arrow, cam_sphere, score sphere points] + optional pcd_unc
        """
        if self.pcd is None or self.arrow is None or self.cam_sphere is None or self.pcd_dirs is None:
            raise RuntimeError("No result to visualize. Call estimate() first.")
        if self.center is None or self.cam_pos is None:
            raise RuntimeError("Missing camera/center state. Call estimate() first.")

        geoms = [self.pcd, self.arrow, self.cam_sphere, self.pcd_dirs]
        if self.pcd_unc is not None:
            geoms.append(self.pcd_unc)

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="GPIS Uncertainty + NBV (v3: RF(uncertainty) * RF(novelty))",
            width=1280,
            height=800
        )
        for g in geoms:
            vis.add_geometry(g)

        set_camera_look_from(vis, cam_pos=np.asarray(self.cam_pos), lookat=np.asarray(self.center))

        print("[UI] Gray: input point cloud")
        print("[UI] Colored volume points: GP variance near |mu|<delta_mu (red=high, green=low)")
        print("[UI] Blue arrow: best view direction; Red sphere: camera position")
        print("[UI] Outer colored sphere points: direction scores (red=high)")
        print("[UI] Close the window to exit.")
        vis.run()
        vis.destroy_window()

    def get_last_nbv(self) -> Dict[str, Any]:
        if self._last_nbv is None:
            raise RuntimeError("No NBV available. Call estimate() first.")
        return self._last_nbv

    # -------------------------- GUI viz (your original) --------------------------
    def _get_camera_extrinsic(self, cam_pos, target_pos):
        """
        计算从 cam_pos 看向 target_pos 的相机外参矩阵 (World-to-Camera)
        """
        cam_pos = np.asarray(cam_pos)
        target_pos = np.asarray(target_pos)

        # 1. 计算前向量 (Z axis)
        z_axis = target_pos - cam_pos
        z_axis /= np.linalg.norm(z_axis)

        # 2. 确定上向量 (Up)
        up = np.array([0, 0, 1.0])
        if abs(np.dot(up, z_axis)) > 0.99:  # 防止平行
            up = np.array([0, 1.0, 0])

        # 3. 计算右向量 (X axis) 和 修正后的上向量 (Y axis)
        x_axis = np.cross(up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        # 4. 构建 Camera-to-World 旋转平移矩阵 (R | t)
        rot = np.eye(4)
        rot[:3, 0] = x_axis
        rot[:3, 1] = y_axis
        rot[:3, 2] = z_axis
        rot[:3, 3] = cam_pos

        # 5. 返回外参 (World-to-Camera)
        return np.linalg.inv(rot)

    def viz_2(self):
        """
        美化版可视化：
        - 物体：不透明灰色
        - 评分球：半透明彩色
        - NBV：绿色相机视锥体
        """
        if self.pcd is None or self.cam_pos is None or self.center is None:
            raise RuntimeError("Missing data to visualize. Call estimate() first.")

        # --- 1. 初始化 GUI 应用 ---
        app = gui.Application.instance
        app.initialize()

        win = app.create_window("NBV Modern Visualization", 1280, 800)
        scene_widget = gui.SceneWidget()
        scene_widget.scene = rendering.Open3DScene(win.renderer)
        win.add_child(scene_widget)

        # --- 2. 创建材质 ---
        mat_obj = rendering.MaterialRecord()
        mat_obj.shader = "defaultUnlit"
        mat_obj.base_color = [0.8, 0.8, 0.8, 1.0]
        mat_obj.point_size = 5.0

        mat_scores = rendering.MaterialRecord()
        mat_scores.shader = "defaultUnlit"
        mat_scores.base_color = [1.0, 1.0, 1.0, 0.3]
        mat_scores.has_alpha = True
        mat_scores.point_size = 5.0

        mat_unc = rendering.MaterialRecord()
        mat_unc.shader = "defaultUnlit"
        mat_unc.base_color = [1.0, 1.0, 1.0, 0.4]
        mat_unc.has_alpha = True
        mat_unc.point_size = 3.0

        # --- 3. 创建相机视锥体 (Frustum) ---
        extrinsic = self._get_camera_extrinsic(self.cam_pos, self.center)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            1280, 800, 1000, 1000, 640, 400
        )
        frustum = o3d.geometry.LineSet.create_camera_visualization(
            view_width_px=1280, view_height_px=800,
            intrinsic=intrinsic.intrinsic_matrix,
            extrinsic=extrinsic,
            scale=0.02
        )
        frustum.paint_uniform_color([0.0, 1.0, 0.2])  # green

        # --- 4. 添加到场景 ---
        scene_widget.scene.add_geometry("object_pcd", self.pcd, mat_obj)

        if self.pcd_dirs is not None:
            scene_widget.scene.add_geometry("direction_scores", self.pcd_dirs, mat_scores)

        if self.pcd_unc is not None:
            scene_widget.scene.add_geometry("uncertainty_vol", self.pcd_unc, mat_unc)

        mat_line = rendering.MaterialRecord()
        mat_line.shader = "unlitLine"
        mat_line.line_width = 2.0
        scene_widget.scene.add_geometry("camera_frustum", frustum, mat_line)

        # --- 5. 设置观察视角 ---
        bounds = self.pcd.get_axis_aligned_bounding_box()
        scene_widget.setup_camera(60, bounds, self.center)

        scale = 2
        extent = bounds.get_extent()
        radius = 0.5 * float(np.linalg.norm(extent))
        cam = scene_widget.scene.camera

        front = np.array([0.0, 0.0, -1.0])
        eye = np.asarray(self.center) - front * (radius / np.tan(np.deg2rad(60.0) * 0.5)) * scale
        up = np.array([0.0, -1.0, 0.0])

        cam.look_at(self.center, eye, up)
        scene_widget.scene.set_background([0.1, 0.1, 0.1, 1.0])

        print("[Visualizer] Green: NBV Camera Frustum")
        print("[Visualizer] Transparent Sphere: Score distribution (Red=High)")
        print("[Visualizer] Grey: Input Point Cloud")

        app.run()

    def _get_lookat_matrix(self, cam_pos, target_pos):
        """辅助函数：根据相机位置和目标位置计算位姿矩阵"""
        cam_pos = np.asarray(cam_pos)
        target_pos = np.asarray(target_pos)

        z = target_pos - cam_pos
        z /= np.linalg.norm(z)

        up = np.array([0, 0, 1])
        if abs(np.dot(up, z)) > 0.99:
            up = np.array([0, 1, 0])

        x = np.cross(up, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)

        rot = np.eye(4)
        rot[:3, 0] = x
        rot[:3, 1] = y
        rot[:3, 2] = z
        rot[:3, 3] = cam_pos

        return np.linalg.inv(rot)


# ------------------------------ minimal demo ------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_path",
        type=str,
        default="/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/result/offline_tracking/cube_obj_02/002_ICP/pcd_online/recon_0_000300.ply"
    )
    ap.add_argument("--no_viz", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    # RF params (optional)
    ap.add_argument("--rf_k", type=int, default=32)
    ap.add_argument("--rf_min_cos", type=float, default=0.96)
    ap.add_argument("--unc_rf_reduce", type=str, default="max", choices=["max", "mean"])
    ap.add_argument("--nov_rf_reduce", type=str, default="mean", choices=["mean", "max"])

    args = ap.parse_args()

    est = GPISNBVv3(
        rf_k=args.rf_k,
        rf_min_cos=args.rf_min_cos,
        unc_rf_reduce=args.unc_rf_reduce,
        nov_rf_reduce=args.nov_rf_reduce,
    )
    nbv = est.estimate(args.in_path, seed=args.seed, verbose=True)

    print(f"\nNBV best_dir: {nbv['best_dir']}")
    if not args.no_viz:
        est.viz()
