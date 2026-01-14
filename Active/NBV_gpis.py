"""
@FileName：NBV.py
@Description：
@Author：Ferry
@Time：2026 1/9/26 5:24 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""

import math
import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Any

import numpy as np
import open3d as o3d
import torch
import gpytorch


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


@dataclass
class NBVConfigV2:
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

    # scoring near t0
    band_halfwidth_scale: float = 0.06
    band_samples: int = 24

    # novelty
    novelty_beta: float = 4.0

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


# ------------------------------ NBV scoring v2: uncertainty * novelty ------------------------------
def score_direction_v2(model, likelihood,
                       kdtree: o3d.geometry.KDTreeFlann,
                       center_world: np.ndarray,
                       radius: float,
                       dir_world: np.ndarray,
                       t0_scaled: float,
                       t_max_scaled: float,
                       band_halfwidth_scaled: float,
                       n_band_samples: int,
                       median_1nn_world: float,
                       novelty_beta: float,
                       device: str) -> float:
    """
    Score(u) = max_var_near_t0(u) * novelty(x0(u))
      - max_var_near_t0: max posterior variance in [t0-dt, t0+dt]
      - novelty: monotonic with distance from x0 to nearest observed point
    """
    if not np.isfinite(t0_scaled):
        return 0.0

    t1 = max(0.0, t0_scaled - band_halfwidth_scaled)
    t2 = min(t_max_scaled, t0_scaled + band_halfwidth_scaled)
    if t2 <= t1 + 1e-9:
        return 0.0

    u = dir_world / (np.linalg.norm(dir_world) + 1e-12)

    ts = torch.linspace(t1, t2, n_band_samples, device=device)
    u_t = torch.tensor(u, device=device, dtype=torch.float32)
    Xq = (ts[:, None] * u_t[None, :]).contiguous()

    _, var = gpis_predict(model, likelihood, Xq, batch=50000)
    max_var = float(var.max().item())

    x0_world = center_world + (t0_scaled * radius) * u
    _, _, dist2 = kdtree.search_knn_vector_3d(x0_world, 1)
    d_nn = math.sqrt(dist2[0]) if len(dist2) > 0 else 1e6

    d0 = max(novelty_beta * median_1nn_world, 1e-6)
    novelty = 1.0 - math.exp(- (d_nn / d0) ** 2)  # [0, 1)

    return max_var * novelty


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
# Class wrapper (script-like visualization, but object uses RAW point cloud)
# ============================================================
class GPISNBVv2:
    """
    GPIS + NBV-v2 wrapper.

    Visualization is script-like (new_gpis_test_2.py) except:
      - The *inside object* point cloud is the ORIGINAL raw point cloud (after remove_non_finite),
        i.e., NOT the voxel-downsampled cloud used for training/scoring.

    Everything else stays aligned with the script:
      geoms = [pcd_vis(raw object), arrow(blue), cam_sphere(red), pcd_dirs(score sphere)] + optional pcd_unc
      window: 1280x800
      camera: set_camera_look_from(cam_pos, center)
      prints: same text
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

            gpis_lr: float = 0.15,
            gpis_add_outer_samples: bool = True,
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

        # viz objects (script-like)
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

    def estimate(self, pcd: Union[str, o3d.geometry.PointCloud], seed: int = 0, verbose: bool = True) -> Dict[str, Any]:
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

        obb = pcd_proc.get_oriented_bounding_box()
        center = np.asarray(obb.center, dtype=np.float32)

        radius = float(np.max(np.linalg.norm(pts - center[None, :], axis=1)))
        radius = max(radius, 1e-6)

        median_1nn = compute_median_1nn(pcd_proc, sample_n=2000)
        hit_eps_world = self.hit_eps if self.hit_eps > 0 else max(2.0 * median_1nn, 1e-4)

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
        nbv = NBVConfigV2(
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
            delta_mu=self.delta_mu,
            grid_res=self.grid_res,
        )

        # ---- train GPIS surrogate ----
        train_x, train_y = build_training_set(pcd_proc, center, radius, median_1nn, gpcfg)
        model, likelihood = train_gpis(train_x, train_y, self.device, gpcfg)

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
            print(f"[NBV-v2] hit_ratio={hit_ratio:.3f} (fraction of directions that hit point cloud)")

        # 2) infer missing depths for miss directions
        t0 = fill_missing_depths(
            dirs=dirs,
            t_hit=t_hit,
            k=nbv.interp_k,
            min_cos=nbv.interp_min_cos,
            tau=nbv.interp_tau
        )

        # 3) score directions: uncertainty * novelty (toward unobserved)
        scores = np.zeros((dirs.shape[0],), dtype=np.float32)
        for i in range(dirs.shape[0]):
            scores[i] = score_direction_v2(
                model, likelihood,
                kdtree=kdtree,
                center_world=center,
                radius=radius,
                dir_world=dirs[i],
                t0_scaled=float(t0[i]),
                t_max_scaled=nbv.t_max_scale,
                band_halfwidth_scaled=nbv.band_halfwidth_scale,
                n_band_samples=nbv.band_samples,
                median_1nn_world=median_1nn,
                novelty_beta=nbv.novelty_beta,
                device=self.device
            )

        best_idx = int(np.argmax(scores))
        best_dir = dirs[best_idx]
        best_score = float(scores[best_idx])

        if verbose:
            print(f"[NBV-v2] best_idx={best_idx}")
            print(f"[NBV-v2] best_score={best_score:.6e}")
            print(f"[NBV-v2] best_dir={best_dir.tolist()}")
            print(f"[NBV-v2] best_t0_scaled={float(t0[best_idx])}")

        # =========================================================
        # Visualization objects (script-like), but pcd_vis uses RAW cloud
        # =========================================================
        pcd_vis = o3d.geometry.PointCloud(self.pcd_raw)  # raw object cloud
        # If raw has no colors, follow script and paint gray; if it has colors, keep them.
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
        }
        self._last_nbv = nbv_out
        return nbv_out

    def viz(self):
        """
        Script-like visualization:
          geoms = [raw object cloud, arrow, cam_sphere, score sphere points] + optional pcd_unc
          window: 1280x800
          camera: set_camera_look_from(cam_pos, center)
          prints: same text
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
            window_name="GPIS Uncertainty + NBV (v2: uncertainty * novelty)",
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


# ------------------------------ minimal demo ------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=str,
                    default="/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/inhand_y.ply")
    ap.add_argument("--no_viz", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    est = GPISNBVv2()
    nbv = est.estimate(args.in_path, seed=args.seed, verbose=True)

    print(f"\nNBV:{nbv['best_dir']}")
    # for k, v in nbv.items():
    #     print(f"  {k}: {v}")

    if not args.no_viz:
        est.viz()
