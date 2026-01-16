"""
@FileName：plot_summary.py
@Description：
@Author：Ferry
@Time：2026 1/15/26 6:51 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot mesh_mesh F-score (@5mm) over 0–60s by aggregating all summary_*.csv under a directory.

- Reads summary_*.csv
- Extracts time_s and mesh_mesh_fscore@0.005000 (or closest)
- Interpolates each object curve onto a 1Hz grid (0..60s)
- Plots mean curve; optionally plots each object curve with your specified names
- Saves: figure PNG + aggregated CSV (mean/std/n + per-object curves)

Example:
python plot_q2_mesh_fscore_curve.py \
  --root /path/to/q2_summaries \
  --pattern "summary_*.csv" \
  --show_individual \
  --out_fig q2_mesh_fscore_5mm.png \
  --out_csv q2_mesh_fscore_5mm_mean.csv
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Object name mapping (1..6)
# ----------------------------
OBJ_NAME = {
    1: "Cube",
    2: "Corner Block",
    3: "Cross Block",
    4: "L-shaped Block",
    5: "Small Cylinder",
    6: "Big Cylinder",
}


def extract_obj_id_from_filename(path: str):
    """
    Extract object id from filename like:
      summary_1.csv, summary_06.csv, summary_object1.csv ...
    Returns int or None.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"(\d+)$", name)  # trailing digits
    if m:
        return int(m.group(1))
    m = re.search(r"summary[_\-]?(\d+)", name)
    if m:
        return int(m.group(1))
    return None


def find_tau_col(df: pd.DataFrame, tau_target: float = 0.005) -> str:
    """Find the closest mesh_mesh_fscore@{tau} column; prefer exact '@0.005000'."""
    exact = f"mesh_mesh_fscore@{tau_target:.6f}"
    if exact in df.columns:
        return exact

    cand = []
    for c in df.columns:
        if c.startswith("mesh_mesh_fscore@"):
            try:
                tau = float(c.split("@", 1)[1])
                cand.append((abs(tau - tau_target), c))
            except Exception:
                pass
    if not cand:
        raise ValueError("No mesh_mesh_fscore@* column found in the CSV.")
    cand.sort(key=lambda x: x[0])
    return cand[0][1]


def interp_with_nan(t_src: np.ndarray, y_src: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Linear interpolation onto t_grid; outside range -> NaN (no extrapolation)."""
    t_src = np.asarray(t_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)

    m = np.isfinite(t_src) & np.isfinite(y_src)
    t_src, y_src = t_src[m], y_src[m]
    if t_src.size < 2:
        out = np.full_like(t_grid, np.nan, dtype=float)
        if t_src.size == 1:
            out[np.isclose(t_grid, t_src[0])] = y_src[0]
        return out

    order = np.argsort(t_src)
    t_src, y_src = t_src[order], y_src[order]

    # remove duplicate timestamps (keep last)
    tmp = pd.DataFrame({"t": t_src, "y": y_src}).groupby("t", as_index=False).last()
    t_src = tmp["t"].to_numpy()
    y_src = tmp["y"].to_numpy()

    y = np.interp(t_grid, t_src, y_src)
    y[(t_grid < t_src.min()) | (t_grid > t_src.max())] = np.nan
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="./summary", help="Directory containing summary_*.csv files.")
    ap.add_argument("--pattern", type=str, default="summary_*.csv", help="Glob pattern under root.")
    ap.add_argument("--t_max", type=float, default=60.0)
    ap.add_argument("--dt", type=float, default=1.0, help="Grid step in seconds (default: 1Hz).")
    ap.add_argument("--tau_mm", type=float, default=5.0, help="Threshold in mm (default: 5mm).")
    ap.add_argument("--out_fig", type=str, default="q2_mesh_fscore_5mm.png")
    ap.add_argument("--out_csv", type=str, default="q2_mesh_fscore_5mm_mean.csv")
    ap.add_argument("--show_individual", default=True,
                    help="Plot each object's curve (thin) with names, plus mean curve (thick).")
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    files = sorted(glob.glob(os.path.join(root, args.pattern)))
    if not files:
        raise RuntimeError(f"No files found: {os.path.join(root, args.pattern)}")

    # sort by parsed object id if possible (1..6), otherwise fallback to name
    def sort_key(fp):
        oid = extract_obj_id_from_filename(fp)
        return (999 if oid is None else oid, fp)

    files = sorted(files, key=sort_key)

    tau_m = args.tau_mm / 1000.0
    t_grid = np.arange(0.0, args.t_max + 1e-9, args.dt, dtype=float)

    curves = []
    names = []
    obj_ids = []

    for fp in files:
        df = pd.read_csv(fp)
        if "time_s" not in df.columns:
            raise ValueError(f"[{fp}] missing required column: time_s")

        col = find_tau_col(df, tau_target=tau_m)
        sub = df[["time_s", col]].copy().sort_values("time_s")
        sub = sub[(sub["time_s"] >= 0.0) & (sub["time_s"] <= args.t_max)]

        y = interp_with_nan(sub["time_s"].to_numpy(), sub[col].to_numpy(), t_grid)

        oid = extract_obj_id_from_filename(fp)
        name = OBJ_NAME.get(oid, os.path.splitext(os.path.basename(fp))[0].replace("summary_", ""))

        curves.append(y)
        names.append(name)
        obj_ids.append(oid)

    Y = np.vstack(curves)  # (N_obj, T)
    mean = np.nanmean(Y, axis=0)
    std = np.nanstd(Y, axis=0)
    n = np.sum(np.isfinite(Y), axis=0).astype(int)

    # Save aggregated CSV (also dump per-object curves as columns)
    out_df = pd.DataFrame({
        "time_s": t_grid,
        "mean_fscore": mean,
        "std_fscore": std,
        "n_objects": n,
    })
    for i, nm in enumerate(names):
        safe_nm = re.sub(r"[^A-Za-z0-9_\-]+", "_", nm)
        out_df[f"obj_{i+1:02d}_{safe_nm}"] = Y[i]
    out_df.to_csv(args.out_csv, index=False)
    print(f"[OK] Saved aggregated CSV: {args.out_csv}")

    # Plot
    plt.figure(figsize=(5,3), dpi=180)
    ax = plt.gca()

    if args.show_individual:
        for i in range(Y.shape[0]):
            ax.plot(t_grid, Y[i], lw=1.0, alpha=0.65, label=names[i])

    ax.plot(t_grid, mean, lw=2.2, label="Mean")

    ax.set_xlim(0, args.t_max)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("time (secs)", fontsize=10)
    ax.set_ylabel(r"Mesh F-Score ($\tau$ = 5mm)", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)

    # Legend: if too crowded, you can disable individual legend by not using --show_individual
    ax.legend(loc="lower right", frameon=True, framealpha=1.0, fontsize=8, ncol=1)

    plt.tight_layout()
    plt.savefig(args.out_fig, bbox_inches="tight")
    print(f"[OK] Saved figure: {args.out_fig}")


if __name__ == "__main__":
    main()
