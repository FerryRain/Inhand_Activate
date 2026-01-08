"""
@FileName：new_gpis_test.py
@Description：
@Author：Ferry
@Time：2026 1/8/26 2:45 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import random
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import open3d as o3d
import torch
import gpytorch


# ------------------------------ Utils ------------------------------
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def compute_median_1nn(pcd: o3d.geometry.PointCloud, sample_n: int = 2000) -> float:
    pts = np.asarray(pcd.points)
    n = pts.shape[0]
    if n < 5:
        return float(np.linalg.norm(pts.std(axis=0)) + 1e-6)

    idx = np.random.choice(n, size=min(sample_n, n), replace=False)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    dists = []
    for i in idx:
        _, nn_idx, nn_dist2 = kdtree.search_knn_vector_3d(pcd.points[i], 2)
        if len(nn_dist2) >= 2:
            dists.append(math.sqrt(nn_dist2[1]))
    if len(dists) == 0:
        return float(np.linalg.norm(pts.std(axis=0)) + 1e-6)
    return float(np.median(dists))


def fibonacci_sphere(n: int) -> np.ndarray:
    # uniform-ish directions on S^2
    # returns (n,3)
    pts = np.zeros((n, 3), dtype=np.float32)
    phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
    for i in range(n):
        y = 1.0 - (2.0 * i) / (n - 1) if n > 1 else 0.0
        r = math.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        x = math.cos(theta) * r
        z = math.sin(theta) * r
        pts[i] = np.array([x, y, z], dtype=np.float32)
    return pts


def planar_polar_dirs(n: int) -> np.ndarray:
    # XY plane 360 degrees
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    dirs = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], axis=1).astype(np.float32)
    return dirs


def rotation_matrix_from_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Rodrigues: rotate vector a to b
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


# ------------------------------ GPIS (Exact GP) ------------------------------
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
    outer_offset_scale: float = 1.0  # multiplied by median 1NN (in world units)
    noise: float = 1e-4


@dataclass
class NBVConfig:
    n_dirs: int = 1000
    n_ray_samples: int = 64
    ray_step: float = 0.02           # in scaled units
    hit_eps: float = 0.03            # in world units (auto tuned from 1NN if not provided)
    visibility_alpha: float = 0.5    # hit => score *= alpha
    delta_mu: float = 0.15           # surface-band width in GP mean space
    band_extend: float = 0.05        # extend beyond hit depth (scaled)
    t_max_scale: float = 2.2         # ray max length in scaled units


def build_training_set(pcd: o3d.geometry.PointCloud,
                       center: np.ndarray,
                       radius: float,
                       median_1nn: float,
                       cfg: GPISConfig) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(pcd.points).astype(np.float32)
    # Subsample surface points
    n = pts.shape[0]
    if n > cfg.max_train:
        idx = np.random.choice(n, size=cfg.max_train, replace=False)
        pts_surf = pts[idx]
    else:
        pts_surf = pts

    # Labels: surface = 0
    X = [pts_surf]
    y = [np.zeros((pts_surf.shape[0],), dtype=np.float32)]

    if cfg.add_outer_samples:
        # Outer samples: move outward from center along radial direction, label = 1
        dirs = pts_surf - center[None, :]
        norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
        dirs = dirs / norms
        offset = cfg.outer_offset_scale * median_1nn
        pts_out = pts_surf + offset * dirs
        X.append(pts_out)
        y.append(np.ones((pts_out.shape[0],), dtype=np.float32))

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    # Normalize coordinates (scaled space): x' = (x - c) / r
    Xn = (X - center[None, :]) / max(radius, 1e-6)
    return Xn.astype(np.float32), y.astype(np.float32)


def train_gpis(train_x: np.ndarray, train_y: np.ndarray, device: str, cfg: GPISConfig):
    tx = torch.from_numpy(train_x).to(device)
    ty = torch.from_numpy(train_y).to(device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = torch.tensor(cfg.noise, device=device)

    model = GPISModel(tx, ty, likelihood).to(device)

    # Initialize lengthscale roughly
    # In scaled space, median 1NN ~ median_1nn / radius; we can set ls ~ 2*that
    with torch.no_grad():
        model.covar_module.base_kernel.lengthscale = torch.tensor(0.08, device=device)  # safe default
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
    # returns mean, var (both 1D tensors)
    means = []
    vars_ = []
    with gpytorch.settings.fast_pred_var():
        for i in range(0, Xq.shape[0], batch):
            xb = Xq[i:i+batch]
            pred = likelihood(model(xb))
            means.append(pred.mean.detach())
            vars_.append(pred.variance.detach())
    return torch.cat(means, dim=0), torch.cat(vars_, dim=0)


# ------------------------------ NBV scoring ------------------------------
def estimate_hit_depth_scaled(pcd: o3d.geometry.PointCloud,
                              kdtree: o3d.geometry.KDTreeFlann,
                              center_world: np.ndarray,
                              radius: float,
                              dir_world: np.ndarray,
                              t_max_scaled: float,
                              step_scaled: float,
                              hit_eps_world: float) -> Optional[float]:
    # Step along ray in world, but parameterized by scaled t.
    # x(t) = c + (t_scaled * radius) * u
    t = 0.0
    while t <= t_max_scaled:
        xw = center_world + (t * radius) * dir_world
        _, idx, dist2 = kdtree.search_knn_vector_3d(xw, 1)
        if len(dist2) > 0 and dist2[0] < (hit_eps_world * hit_eps_world):
            return t
        t += step_scaled
    return None


def score_direction(model, likelihood,
                    center_world: np.ndarray,
                    radius: float,
                    dir_world: np.ndarray,
                    t_hit_scaled: Optional[float],
                    nbv: NBVConfig,
                    device: str) -> float:
    # Sample along ray up to hit (+band) or tmax
    t_end = nbv.t_max_scale if t_hit_scaled is None else min(nbv.t_max_scale, t_hit_scaled + nbv.band_extend)
    ts = torch.linspace(0.0, t_end, nbv.n_ray_samples, device=device)

    # Query points in scaled space: x' = (x - c)/r = t_scaled * u  (since origin at c)
    u = torch.tensor(dir_world, device=device, dtype=torch.float32)
    u = u / (torch.norm(u) + 1e-12)
    Xq = (ts[:, None] * u[None, :]).contiguous()

    mu, var = gpis_predict(model, likelihood, Xq, batch=50000)

    # Surface-band weight using GP mean
    delta = nbv.delta_mu
    w = torch.exp(-(mu * mu) / (2.0 * delta * delta))

    # If miss and w is tiny everywhere => this direction likely doesn't intersect surface band
    if t_hit_scaled is None:
        if float(w.max().item()) < 1e-3:
            return 0.0

    U = torch.sum(w * var) * (t_end / max(1, nbv.n_ray_samples - 1))
    score = float(U.item())

    # visibility prior: hit => downweight
    if t_hit_scaled is not None:
        score *= nbv.visibility_alpha
    return score


# ------------------------------ Visualization ------------------------------
def build_uncertainty_cloud(model, likelihood,
                            center_world: np.ndarray,
                            radius: float,
                            aabb_min_world: np.ndarray,
                            aabb_max_world: np.ndarray,
                            grid_res: int,
                            delta_mu: float,
                            device: str):
    # build grid in world, map to scaled and query GP
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
    # Keep surface band (otherwise too many points)
    mask = np.abs(mu_np) < float(delta_mu)
    pts_band = pts[mask]
    var_band = var_np[mask]

    if pts_band.shape[0] == 0:
        return None, None

    v01 = normalize01(var_band)
    colors = np.stack([v01, 1.0 - v01, np.zeros_like(v01)], axis=1).astype(np.float64)  # red->green-ish

    pcd_u = o3d.geometry.PointCloud()
    pcd_u.points = o3d.utility.Vector3dVector(pts_band.astype(np.float64))
    pcd_u.colors = o3d.utility.Vector3dVector(colors)
    return pcd_u, (pts_band, var_band)


def make_arrow(origin: np.ndarray, direction: np.ndarray, length: float) -> o3d.geometry.TriangleMesh:
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    # Arrow is along +Z by default
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
    # Open3D camera params: front points from lookat to camera? Actually 'front' is camera viewing direction.
    # We'll set front = (lookat - cam_pos) normalized.
    front = (lookat - cam_pos).astype(np.float64)
    front /= (np.linalg.norm(front) + 1e-12)

    # choose an up vector not parallel to front
    up = np.array([0, 0, 1], dtype=np.float64)
    if abs(float(np.dot(front, up))) > 0.95:
        up = np.array([0, 1, 0], dtype=np.float64)

    ctr.set_lookat(lookat.astype(np.float64))
    ctr.set_front(front)
    ctr.set_up(up)
    ctr.set_zoom(0.7)

def fill_missing_depths(dirs: np.ndarray,
                        t_hit: np.ndarray,
                        k: int = 8,
                        min_cos: float = 0.6,
                        tau: float = 0.08) -> np.ndarray:
    """
    dirs: (N,3) unit directions
    t_hit: (N,) scaled depth, np.nan for miss
    return t0: (N,) scaled depth, np.nan if cannot be inferred
    """
    N = dirs.shape[0]
    hit_mask = np.isfinite(t_hit)
    hit_dirs = dirs[hit_mask]
    hit_t = t_hit[hit_mask]
    if hit_dirs.shape[0] < 3:
        return np.full((N,), np.nan, dtype=np.float32)

    t0 = t_hit.copy().astype(np.float32)

    # Precompute dot products in a vectorized way per miss direction
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
        # softmax weights on cosine similarity
        c = cos[j_top]
        w = np.exp((c - c.max()) / max(tau, 1e-6))
        w = w / (w.sum() + 1e-12)
        t0[i] = float(np.sum(w * hit_t[j_top]))
    return t0

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
    Score = max_var_near_surface_depth * novelty(distance_to_pointcloud)
    """
    if not np.isfinite(t0_scaled):
        return 0.0

    # Clamp band range
    t1 = max(0.0, t0_scaled - band_halfwidth_scaled)
    t2 = min(t_max_scaled, t0_scaled + band_halfwidth_scaled)
    if t2 <= t1 + 1e-9:
        return 0.0

    ts = torch.linspace(t1, t2, n_band_samples, device=device)
    u = torch.tensor(dir_world, device=device, dtype=torch.float32)
    u = u / (torch.norm(u) + 1e-12)
    Xq = (ts[:, None] * u[None, :]).contiguous()  # scaled space

    _, var = gpis_predict(model, likelihood, Xq, batch=50000)
    max_var = float(var.max().item())

    # Novelty from distance to observed point cloud at the candidate surface point
    x0_world = center_world + (t0_scaled * radius) * dir_world
    _, _, dist2 = kdtree.search_knn_vector_3d(x0_world, 1)
    if len(dist2) == 0:
        d_nn = 1e6
    else:
        d_nn = math.sqrt(dist2[0])

    d0 = max(novelty_beta * median_1nn_world, 1e-6)
    novelty = 1.0 - math.exp(- (d_nn / d0) ** 2)  # in [0,1)

    return max_var * novelty

# ------------------------------ Main ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, default="/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/reconstruction_clean_color_1s.ply", help="Input point cloud: .ply/.pcd/.xyz/.txt")
    parser.add_argument("--voxel", type=float, default=0.0, help="Voxel downsample size in world units (0 to disable)")
    parser.add_argument("--max_train", type=int, default=2000)
    parser.add_argument("--train_iters", type=int, default=60)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--grid_res", type=int, default=35, help="Grid resolution per axis for uncertainty viz")
    parser.add_argument("--n_dirs", type=int, default=1000, help="Number of candidate view directions")
    parser.add_argument("--planar", action="store_true", help="Use 2D polar (XY plane) directions instead of full sphere")
    parser.add_argument("--hit_eps", type=float, default=0.0, help="Ray hit threshold in world units (0 => auto from 1NN)")
    args = parser.parse_args()

    set_seed(0)

    # Load & preprocess
    pcd = load_point_cloud(args.in_path)
    pcd.remove_non_finite_points()
    if args.voxel > 0:
        pcd = pcd.voxel_down_sample(args.voxel)

    pts = np.asarray(pcd.points)
    obb = pcd.get_oriented_bounding_box()
    center = np.asarray(obb.center, dtype=np.float32)

    # radius = max distance to center (world units)
    radius = float(np.max(np.linalg.norm(pts - center[None, :], axis=1)))
    radius = max(radius, 1e-6)

    median_1nn = compute_median_1nn(pcd, sample_n=2000)
    hit_eps = args.hit_eps if args.hit_eps > 0 else max(1.5 * median_1nn, 1e-4)

    print(f"[Info] points={len(pcd.points)}, center={center.tolist()}, radius={radius:.4f}, median_1nn={median_1nn:.6f}, hit_eps={hit_eps:.6f}")
    print(f"[Info] device={args.device}")

    gpcfg = GPISConfig(
        max_train=args.max_train,
        train_iters=args.train_iters,
        add_outer_samples=True,
        outer_offset_scale=1.0,
        noise=1e-4,
    )
    nbv = NBVConfig(
        n_dirs=args.n_dirs,
        n_ray_samples=64,
        ray_step=0.02,
        hit_eps=hit_eps,
        visibility_alpha=0.5,
        delta_mu=0.15,
        band_extend=0.05,
        t_max_scale=2.2,
    )

    # Build training set & train GPIS
    train_x, train_y = build_training_set(pcd, center, radius, median_1nn, gpcfg)
    model, likelihood = train_gpis(train_x, train_y, args.device, gpcfg)

    # Uncertainty visualization cloud
    aabb = pcd.get_axis_aligned_bounding_box()
    minb = np.asarray(aabb.get_min_bound(), dtype=np.float32)
    maxb = np.asarray(aabb.get_max_bound(), dtype=np.float32)
    # Add margin
    margin = 0.15 * (maxb - minb)
    minb2 = minb - margin
    maxb2 = maxb + margin

    pcd_u, _ = build_uncertainty_cloud(
        model, likelihood,
        center_world=center,
        radius=radius,
        aabb_min_world=minb2,
        aabb_max_world=maxb2,
        grid_res=args.grid_res,
        delta_mu=nbv.delta_mu,
        device=args.device,
    )

    # NBV: score directions
    if args.planar:
        dirs = planar_polar_dirs(nbv.n_dirs)
    else:
        dirs = fibonacci_sphere(nbv.n_dirs)

    kdtree = o3d.geometry.KDTreeFlann(pcd)

    scores = np.zeros((dirs.shape[0],), dtype=np.float32)
    hit_flags = np.zeros_like(scores, dtype=np.int32)

    for i in range(dirs.shape[0]):
        u = dirs[i]
        u = u / (np.linalg.norm(u) + 1e-12)
        t_hit = estimate_hit_depth_scaled(
            pcd, kdtree, center_world=center, radius=radius,
            dir_world=u, t_max_scaled=nbv.t_max_scale,
            step_scaled=nbv.ray_step, hit_eps_world=hit_eps
        )
        if t_hit is not None:
            hit_flags[i] = 1
        s = score_direction(
            model, likelihood,
            center_world=center, radius=radius,
            dir_world=u, t_hit_scaled=t_hit,
            nbv=nbv, device=args.device
        )
        scores[i] = s

    best_idx = int(np.argmax(scores))
    best_dir = dirs[best_idx] / (np.linalg.norm(dirs[best_idx]) + 1e-12)
    best_score = float(scores[best_idx])
    print(f"[NBV] best_idx={best_idx}, best_score={best_score:.6e}, best_dir={best_dir.tolist()}, hit={bool(hit_flags[best_idx])}")

    # Build geometries for visualization
    pcd_vis = pcd.paint_uniform_color([0.65, 0.65, 0.65])
    origin = center
    arrow_len = 1.2 * radius
    arrow = make_arrow(origin, best_dir, arrow_len)
    arrow.paint_uniform_color([0.1, 0.1, 0.95])

    cam_dist = 2.0 * radius
    cam_pos = center + cam_dist * best_dir
    cam_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03 * radius, resolution=20)
    cam_sphere.translate(cam_pos.astype(np.float64))
    cam_sphere.paint_uniform_color([0.95, 0.1, 0.1])

    # Direction-score points on a sphere (optional, helps debug NBV)
    score01 = normalize01(scores)
    sphere_pts = center[None, :] + (1.5 * radius) * dirs
    colors = np.stack([score01, np.zeros_like(score01), 1.0 - score01], axis=1).astype(np.float64)
    pcd_dirs = o3d.geometry.PointCloud()
    pcd_dirs.points = o3d.utility.Vector3dVector(sphere_pts.astype(np.float64))
    pcd_dirs.colors = o3d.utility.Vector3dVector(colors)

    # Visualize
    geoms = [pcd_vis, arrow, cam_sphere, pcd_dirs]
    if pcd_u is not None:
        geoms.append(pcd_u)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="GPIS Uncertainty + NBV", width=1280, height=800)
    for g in geoms:
        vis.add_geometry(g)

    # set camera to best view
    set_camera_look_from(vis, cam_pos=cam_pos, lookat=center)

    print("[UI] Gray: input point cloud | Colored cloud: uncertainty (variance) near surface band | Blue arrow: best view dir | Red sphere: camera pos | Score sphere: direction scores")
    print("[UI] Close window to exit.")
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
