#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import open3d as o3d
import torch
import gpytorch


# ---------------- IO ----------------
def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts"]:
        pcd = o3d.io.read_point_cloud(path)
        if pcd.is_empty():
            raise RuntimeError(f"Loaded empty point cloud: {path}")
        return pcd
    elif ext == ".npz":
        data = np.load(path)
        if "points" not in data:
            raise RuntimeError("npz must contain key 'points' with shape (N,3)")
        pts = data["points"]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
        if "colors" in data:
            cols = data["colors"]
            if cols.max() > 1.0:
                cols = cols / 255.0
            pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
        return pcd
    else:
        raise ValueError(f"Unsupported point cloud format: {path}")


def pcd_to_xyz(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    return np.asarray(pcd.points).astype(np.float32)


# ---------------- GPIS (Sparse Variational GP) ----------------
class GPIS_SVGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor, mean_const: float = 1.0, learn_mean: bool = False):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        with torch.no_grad():
            self.mean_module.constant.fill_(mean_const)
        self.mean_module.constant.requires_grad_(learn_mean)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=3)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_svgp(model, likelihood, train_x, train_y, iters=2000, lr=0.01, batch_size=2048, device="cpu"):
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {"params": model.parameters()},
        {"params": likelihood.parameters()},
    ], lr=lr)

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_x.size(0))

    N = train_x.size(0)
    indices = torch.arange(N, device=device)

    for it in range(1, iters + 1):
        if 0 < batch_size < N:
            batch_idx = indices[torch.randint(0, N, (batch_size,), device=device)]
            x_b = train_x[batch_idx]
            y_b = train_y[batch_idx]
        else:
            x_b, y_b = train_x, train_y

        optimizer.zero_grad(set_to_none=True)
        output = model(x_b)
        loss = -mll(output, y_b)
        loss.backward()
        optimizer.step()

        if it % 100 == 0 or it == 1:
            noise = likelihood.noise.item()
            ls = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().reshape(-1)
            oscale = model.covar_module.outputscale.item()
            print(f"[iter {it:4d}] loss={loss.item():.4f}  noise={noise:.6f}  ls={ls}  oscale={oscale:.4f}")


@torch.no_grad()
def predict_in_batches(model, likelihood, Xq, batch=65536, device="cpu"):
    model.eval()
    likelihood.eval()

    means = []
    vars_ = []
    for i in range(0, Xq.shape[0], batch):
        xb = torch.from_numpy(Xq[i:i+batch]).to(device)
        with gpytorch.settings.fast_pred_var():
            pred = likelihood(model(xb))
            mean = pred.mean.detach().cpu().numpy()
            var = pred.variance.detach().cpu().numpy()
        means.append(mean)
        vars_.append(var)
    mean = np.concatenate(means, axis=0)
    var = np.concatenate(vars_, axis=0)
    return mean, var


# ---------------- Sampling & Visualization ----------------
def sample_query_points_aabb(aabb_min, aabb_max, mode="grid", grid_res=64, n_rand=200000):
    """
    返回:
      pts: (N,3)
      meta: dict (grid 模式下提供 shape/step/xs/ys/zs, 方便反查 topk grid index)
    """
    aabb_min = np.asarray(aabb_min, dtype=np.float32)
    aabb_max = np.asarray(aabb_max, dtype=np.float32)
    meta = {}

    if mode == "grid":
        xs = np.linspace(aabb_min[0], aabb_max[0], grid_res, dtype=np.float32)
        ys = np.linspace(aabb_min[1], aabb_max[1], grid_res, dtype=np.float32)
        zs = np.linspace(aabb_min[2], aabb_max[2], grid_res, dtype=np.float32)

        # 用 indexing="ij" 让 unravel_index 对应 (ix,iy,iz) 更直观
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        pts = np.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=1)

        meta = {
            "shape": X.shape,                 # (grid_res, grid_res, grid_res)
            "xs": xs, "ys": ys, "zs": zs,
            "step": np.array([xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]], dtype=np.float32),
            "aabb_min": aabb_min, "aabb_max": aabb_max
        }
        return pts, meta
    else:
        pts = np.random.uniform(aabb_min, aabb_max, size=(n_rand, 3)).astype(np.float32)
        meta = {"aabb_min": aabb_min, "aabb_max": aabb_max}
        return pts, meta


def colorize_by_value(vals, vmin=None, vmax=None,
                      low_color=(0.1, 0.2, 1.0), high_color=(1.0, 0.2, 0.1)):
    v = vals.astype(np.float32)
    if vmin is None:
        vmin = float(np.percentile(v, 1))
    if vmax is None:
        vmax = float(np.percentile(v, 99))
    t = (v - vmin) / (vmax - vmin + 1e-12)
    t = np.clip(t, 0.0, 1.0)[:, None]
    low = np.array(low_color, dtype=np.float32)[None, :]
    high = np.array(high_color, dtype=np.float32)[None, :]
    col = (1.0 - t) * low + t * high
    return col


def utility(mu: np.ndarray, std: np.ndarray, eps: float) -> np.ndarray:
    # u(x) = std * exp(-|mu|/eps)
    return std * np.exp(-np.abs(mu) / (eps + 1e-12))


def select_topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = int(k)
    if k <= 0:
        return np.empty((0,), dtype=np.int64)
    k = min(k, scores.shape[0])
    # argpartition: O(N)
    idx = np.argpartition(-scores, kth=k-1)[:k]
    # 按 score 真实排序（可选）
    idx = idx[np.argsort(-scores[idx])]
    return idx.astype(np.int64)


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcd", type=str, default="./reconstruction.ply", help="path to point cloud (.ply/.pcd/.xyz/.npz)")
    ap.add_argument("--voxel", type=float, default=0.003, help="voxel downsample size (m)")
    ap.add_argument("--max_points", type=int, default=12000, help="cap training points after downsample")
    ap.add_argument("--num_inducing", type=int, default=512, help="number of inducing points (SVGP)")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch_size", type=int, default=2048)

    # Driess-style prior mean
    ap.add_argument("--mean_const", type=float, default=1.0, help="prior mean m=constant (default 1.0)")
    ap.add_argument("--learn_mean", action="store_true", help="allow learning mean constant")

    # query sampling
    ap.add_argument("--margin", type=float, default=0.02, help="AABB margin (m)")
    ap.add_argument("--query_mode", type=str, default="grid", choices=["grid", "rand"])
    ap.add_argument("--grid_res", type=int, default=64, help="grid resolution per axis (grid_res^3 points)")
    ap.add_argument("--n_rand", type=int, default=200000, help="random query points if mode=rand")

    # surface / uncertainty
    ap.add_argument("--eps", type=float, default=0.08, help="surface band: |mu| < eps")
    ap.add_argument("--unc_percentile", type=float, default=92.0,
                    help="high-unc threshold percentile within surface band")

    # ---- Top-K grids marking ----
    ap.add_argument("--topk", type=int, default=2000, help="mark Top-K grids by uncertainty/utility")
    ap.add_argument("--topk_field", type=str, default="utility", choices=["std", "utility"],
                    help="use std(x) or utility(x)=std*exp(-|mu|/eps)")
    ap.add_argument("--topk_scope", type=str, default="surface", choices=["surface", "all"],
                    help="select Top-K within surface band or over all grid queries")

    # saving
    ap.add_argument("--out_dir", type=str, default="gpis_viz_out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    # -------- load & downsample --------
    pcd = load_point_cloud(args.pcd)
    print(f"[load] points={len(pcd.points)}")

    if args.voxel > 0:
        pcd = pcd.voxel_down_sample(args.voxel)
        print(f"[downsample] voxel={args.voxel} -> points={len(pcd.points)}")

    # keep largest cluster (optional but usually helps)
    labels = np.array(pcd.cluster_dbscan(eps=5.0 * args.voxel, min_points=30, print_progress=False))
    if labels.size > 0 and np.any(labels >= 0):
        valid = labels[labels >= 0]
        if valid.size > 0:
            largest = np.argmax(np.bincount(valid))
            idx = np.where(labels == largest)[0]
            pcd = pcd.select_by_index(idx)
            print(f"[cluster] keep largest cluster -> points={len(pcd.points)}")

    xyz = pcd_to_xyz(pcd)

    # cap training size for speed
    if xyz.shape[0] > args.max_points:
        sel = np.random.choice(xyz.shape[0], args.max_points, replace=False)
        xyz = xyz[sel]
        pcd = pcd.select_by_index(sel.tolist())
        print(f"[cap] max_points={args.max_points} -> points={xyz.shape[0]}")

    # training data: surface-only y=0
    train_x = torch.from_numpy(xyz).to(device)
    train_y = torch.zeros(train_x.size(0), device=device)

    # inducing points
    M = min(args.num_inducing, train_x.size(0))
    ind_idx = torch.randperm(train_x.size(0), device=device)[:M]
    inducing = train_x[ind_idx].clone()

    model = GPIS_SVGP(inducing_points=inducing, mean_const=args.mean_const, learn_mean=args.learn_mean).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
    ).to(device)

    # init hyperparams
    model.covar_module.base_kernel.lengthscale = args.voxel * 5.0
    model.covar_module.outputscale = 1.0
    likelihood.noise = 1e-3

    # -------- train --------
    print("[train] SVGP ...")
    train_svgp(model, likelihood, train_x, train_y,
               iters=args.iters, lr=args.lr, batch_size=args.batch_size, device=device)

    # -------- query points in AABB --------
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb_min = np.asarray(aabb.min_bound, dtype=np.float32) - args.margin
    aabb_max = np.asarray(aabb.max_bound, dtype=np.float32) + args.margin
    print(f"[AABB] min={aabb_min}, max={aabb_max}, margin={args.margin}")

    Xq, qmeta = sample_query_points_aabb(aabb_min, aabb_max,
                                         mode=args.query_mode, grid_res=args.grid_res, n_rand=args.n_rand)
    print(f"[query] mode={args.query_mode}  num={Xq.shape[0]}")

    # -------- predict --------
    mu, var = predict_in_batches(model, likelihood, Xq, batch=65536, device=device)
    std = np.sqrt(np.maximum(var, 0.0))

    # define field for topk
    if args.topk_field == "std":
        field = std
    else:
        field = utility(mu, std, eps=args.eps)

    # -------- surface band & high uncertainty (for surface visualization) --------
    surface_mask = np.abs(mu) < args.eps
    surf_pts = Xq[surface_mask]
    surf_std = std[surface_mask]
    print(f"[surface] |mu|<{args.eps} -> {surf_pts.shape[0]} points")

    if surf_pts.shape[0] == 0:
        print("No surface points found. Try increasing --eps (e.g. 0.12) or adjust lengthscale/iters.")
        return

    thr = np.percentile(surf_std, args.unc_percentile)
    high_mask = surf_std >= thr
    high_pts = surf_pts[high_mask]
    print(f"[high-unc] >= P{args.unc_percentile} (thr={thr:.6f}) -> {high_pts.shape[0]} points")

    # -------- Top-K grids marking --------
    if args.topk_scope == "surface":
        cand_mask = surface_mask
    else:
        cand_mask = np.ones_like(surface_mask, dtype=bool)

    cand_field = field[cand_mask]
    cand_pts = Xq[cand_mask]

    topk_idx = select_topk_indices(cand_field, args.topk)
    topk_pts = cand_pts[topk_idx]
    topk_vals = cand_field[topk_idx]
    print(f"[TopK] scope={args.topk_scope}, field={args.topk_field}, K={len(topk_idx)}")
    print(f"       max={topk_vals[0]:.6f}, min_in_topk={topk_vals[-1]:.6f}")

    # 如果是 grid 模式，打印前几个 TopK 的 grid index (ix,iy,iz)
    if args.query_mode == "grid" and "shape" in qmeta:
        shape = qmeta["shape"]  # (R,R,R)
        # cand_mask 是从 Xq 过滤出来的，要映射回原 Xq 的 index
        cand_global_idx = np.where(cand_mask)[0]
        topk_global_idx = cand_global_idx[topk_idx]
        print("[TopK grid indices] show first 10:")
        for j in range(min(10, topk_global_idx.shape[0])):
            g = int(topk_global_idx[j])
            ix, iy, iz = np.unravel_index(g, shape)
            print(f"  #{j:02d}: (ix,iy,iz)=({ix},{iy},{iz}), val={topk_vals[j]:.6f}, xyz={Xq[g]}")

    # -------- build Open3D geometries --------
    # training pcd (gray)
    pcd_train = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.astype(np.float64)))
    pcd_train.paint_uniform_color([0.5, 0.5, 0.5])

    # predicted surface colored by uncertainty std (blue->red)
    surf_colors = colorize_by_value(surf_std)
    pcd_surf = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(surf_pts.astype(np.float64)))
    pcd_surf.colors = o3d.utility.Vector3dVector(surf_colors.astype(np.float64))

    # high-unc surface (red)
    pcd_high = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(high_pts.astype(np.float64)))
    pcd_high.paint_uniform_color([1.0, 0.0, 0.0])

    # TopK grids (bright green)
    pcd_topk = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(topk_pts.astype(np.float64)))
    pcd_topk.paint_uniform_color([0.0, 1.0, 0.2])

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

    # -------- save outputs --------
    o3d.io.write_point_cloud(os.path.join(args.out_dir, "train_downsample.ply"), pcd_train)
    o3d.io.write_point_cloud(os.path.join(args.out_dir, "pred_surface_unc.ply"), pcd_surf)
    o3d.io.write_point_cloud(os.path.join(args.out_dir, "high_unc_surface.ply"), pcd_high)
    o3d.io.write_point_cloud(os.path.join(args.out_dir, "topk_grids.ply"), pcd_topk)

    np.savez(
        os.path.join(args.out_dir, "query_pred.npz"),
        Xq=Xq, mu=mu, var=var, std=std, field=field
    )

    print(f"[saved] -> {args.out_dir}/train_downsample.ply")
    print(f"[saved] -> {args.out_dir}/pred_surface_unc.ply")
    print(f"[saved] -> {args.out_dir}/high_unc_surface.ply")
    print(f"[saved] -> {args.out_dir}/topk_grids.ply")
    print(f"[saved] -> {args.out_dir}/query_pred.npz")

    # -------- visualize --------
    print("\n[visualize]")
    print("Gray: input surface points (downsampled)")
    print("Blue->Red: predicted surface band points colored by std")
    print("Red: high-std surface points (percentile in band)")
    print("Green: Top-K grids by chosen field (std/utility) within chosen scope (surface/all)")
    o3d.visualization.draw_geometries(
        [axis, pcd_train, pcd_surf, pcd_high, pcd_topk],
        width=1280, height=720
    )


if __name__ == "__main__":
    main()
