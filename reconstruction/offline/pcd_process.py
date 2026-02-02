"""
@FileName：pcd_process.py
@Description：
@Author：Ferry
@Time：2026 1/30/26 9:38 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Point cloud denoise (Open3D):
- remove non-finite points
- optional voxel downsample
- statistical outlier removal
- radius outlier removal
Save denoised point cloud.

Example:
  python pcd_denoise.py \
    --input "C:/.../recon_0_000587.ply" \
    --output "C:/.../recon_0_000587_denoised.ply" \
    --denoise both --nb_neighbors 30 --std_ratio 1.5 --radius 0.01 --min_points 16

Optional:
  --voxel 0.002
"""

import argparse
from pathlib import Path
import numpy as np


def _remove_non_finite(pcd):
    res = pcd.remove_non_finite_points(remove_nan=True, remove_infinite=True)
    return res[0] if isinstance(res, tuple) else res


def denoise_pcd(pcd, method, nb_neighbors, std_ratio, radius, min_points, verbose=True):
    method = method.lower().strip()
    pcd = _remove_non_finite(pcd)

    if method == "none":
        return pcd

    if method in ("statistical", "both"):
        if verbose:
            print(f"[DENOISE] StatisticalOutlierRemoval: nb_neighbors={nb_neighbors}, std_ratio={std_ratio}")
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio))
        if verbose:
            print(f"[DENOISE] kept {len(ind)} points after statistical")

    if method in ("radius", "both"):
        if float(radius) <= 0:
            raise ValueError("For radius/both denoise, --radius must be > 0.")
        if verbose:
            print(f"[DENOISE] RadiusOutlierRemoval: radius={radius}, min_points={min_points}")
        pcd, ind = pcd.remove_radius_outlier(nb_points=int(min_points), radius=float(radius))
        if verbose:
            print(f"[DENOISE] kept {len(ind)} points after radius")

    return pcd


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", "-i", default="/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/result/offline_tracking/tetraprism/003_ICP/normal/recon_0_000600_xyz_normal.ply", help="Input point cloud (.ply/.pcd/...)")
    parser.add_argument("--output", "-o", default="/home/ferry/data/Code2/Research/Inhand_Activate/demo/results/process_pcd/Cross.ply", help="Output point cloud (.ply/.pcd/...)")

    parser.add_argument("--voxel", type=float, default=0.0, help="Voxel downsample size (0 = no downsample)")

    parser.add_argument("--denoise", type=str, default="both",
                        choices=["none", "statistical", "radius", "both"],
                        help="Denoise method")
    parser.add_argument("--nb_neighbors", type=int, default=30, help="Statistical: number of neighbors")
    parser.add_argument("--std_ratio", type=float, default=1.5, help="Statistical: std ratio")
    parser.add_argument("--radius", type=float, default=0.01, help="Radius: search radius")
    parser.add_argument("--min_points", type=int, default=16, help="Radius: min neighbors within radius")

    parser.add_argument("--keep_normals", action="store_true",
                        help="If input has normals, keep them (Open3D may drop if not present).")
    parser.add_argument("--keep_colors", action="store_true",
                        help="Keep colors if present (default True in Open3D read/write, but keep for clarity).")

    args = parser.parse_args()

    import open3d as o3d

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise FileNotFoundError(f"Not found: {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pcd = o3d.io.read_point_cloud(str(in_path))
    if pcd.is_empty():
        raise RuntimeError("Loaded point cloud is empty.")

    # cache attributes (optional)
    has_colors = pcd.has_colors()
    has_normals = pcd.has_normals()
    colors = np.asarray(pcd.colors) if (has_colors and args.keep_colors) else None
    normals = np.asarray(pcd.normals) if (has_normals and args.keep_normals) else None

    # downsample (if requested)
    if args.voxel > 0:
        pcd = pcd.voxel_down_sample(float(args.voxel))

    # denoise
    before_n = len(pcd.points)
    pcd = denoise_pcd(
        pcd,
        method=args.denoise,
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio,
        radius=args.radius,
        min_points=args.min_points,
        verbose=True
    )
    after_n = len(pcd.points)

    if pcd.is_empty():
        raise RuntimeError("Point cloud becomes empty after denoise. Loosen denoise params.")

    print(f"[INFO] points: {before_n} -> {after_n} (removed {before_n - after_n})")

    # NOTE:
    # After outlier removal, indices change, so we cannot safely "re-attach" old colors/normals
    # unless we track indices. Open3D's remove_* already preserves attributes in returned pcd.
    # So we just write the returned pcd.

    ok = o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False, compressed=False, print_progress=False)
    print(f"[OK] Saved denoised point cloud: {out_path} (success={ok})")


if __name__ == "__main__":
    main()
