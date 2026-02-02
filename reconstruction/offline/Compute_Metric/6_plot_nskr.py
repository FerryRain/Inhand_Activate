#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@FileName：plot_nskr.py
@Description：
@Author：Ferry
@Time：2026 1/21/26 6:11 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def finite_mask(y: np.ndarray) -> np.ndarray:
    return np.isfinite(y)


def best_idx(t: np.ndarray, y: np.ndarray, mode: str):
    """mode: 'min' or 'max'"""
    m = finite_mask(y) & finite_mask(t)
    if not np.any(m):
        return None
    idxs = np.where(m)[0]
    yy = y[idxs]
    j = int(np.argmin(yy)) if mode == "min" else int(np.argmax(yy))
    return int(idxs[j])


def value_at_time_interp(t: np.ndarray, y: np.ndarray, t0: float):
    """Linear interpolation at exactly t0; None if out of range or insufficient finite samples."""
    m = finite_mask(y) & finite_mask(t)
    if np.sum(m) < 2:
        return None
    tt = t[m]
    yy = y[m]
    order = np.argsort(tt)
    tt = tt[order]
    yy = yy[order]
    if t0 < tt[0] or t0 > tt[-1]:
        return None
    return float(np.interp(t0, tt, yy))


def format_label(name: str, best_v, best_t, vmark, unit: str, t_mark: float):
    def fmt_v(v):
        return "NA" if v is None or (not np.isfinite(v)) else f"{v:.4f}"

    def fmt_t(v):
        return "NA" if v is None or (not np.isfinite(v)) else f"{v:.2f}s"

    u = f" {unit}" if unit else ""
    return f"{name} | best={fmt_v(best_v)}{u} @ {fmt_t(best_t)} | t={t_mark:.0f}s={fmt_v(vmark)}{u}"


def plot_group(
    df: pd.DataFrame,
    t: np.ndarray,
    cols: list[str],
    title: str,
    y_label: str,
    best_mode: str,      # 'min' or 'max'
    to_mm: bool,         # convert meters->mm
    out_path: str,
    t_mark: float = 30.0,
    t_max: float = 60.0,
    dpi: int = 200,
    legend_loc: str = "best",
    legend_fontsize: int = 8,
):
    fig, ax = plt.subplots()

    any_plotted = False
    line_handles = []

    for col in cols:
        if col not in df.columns:
            continue

        y = df[col].to_numpy(dtype=np.float64)
        unit = ""
        if to_mm:
            y = 1000.0 * y
            unit = "mm"

        idx_b = best_idx(t, y, mode=best_mode)
        best_v = float(y[idx_b]) if idx_b is not None else None
        best_t = float(t[idx_b]) if idx_b is not None else None
        v_mark = value_at_time_interp(t, y, t_mark)

        label = format_label(col, best_v, best_t, v_mark, unit, t_mark)

        line, = ax.plot(t, y, label=label)
        c = line.get_color()
        line_handles.append(line)
        any_plotted = True

        # mark global best '*'
        if idx_b is not None:
            ax.scatter([t[idx_b]], [y[idx_b]], marker="*", s=80, color=c, zorder=5)

    # vertical dashed line at t_mark
    ax.axvline(float(t_mark), linestyle="--", linewidth=1.2, color="gray", alpha=0.9)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.0, float(t_max))

    if any_plotted:
        marker_best = Line2D(
            [0], [0],
            marker="*",
            linestyle="None",
            markersize=10,
            color="black",
            label="*  global best"
        )
        handles = line_handles + [marker_best]
        ax.legend(handles=handles, loc=legend_loc, fontsize=legend_fontsize, framealpha=0.9)
    else:
        ax.text(0.5, 0.5, "No matching columns found.", transform=ax.transAxes,
                ha="center", va="center")

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/result/offline_tracking/Big_Cylinder/003_ICP/eval_strictref/summary.csv")
    ap.add_argument("--out_dir", default="", help="default: <csv_dir>/plots_marks_legend")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--t_mark", type=float, default=30.0, help="draw dashed line at this exact time (seconds)")
    ap.add_argument("--t_max", type=float, default=60.0, help="plot x-axis range: [0, t_max]")
    ap.add_argument("--legend_loc", default="best")
    ap.add_argument("--legend_fontsize", type=int, default= 8)
    args = ap.parse_args()

    csv_path = os.path.expanduser(args.csv)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "time_s" not in df.columns:
        raise RuntimeError("summary.csv missing required column: time_s")

    # restrict to [0, t_max]
    df = df.sort_values("time_s").reset_index(drop=True)
    df = df[(df["time_s"] >= 0.0) & (df["time_s"] <= float(args.t_max))].copy()
    df = df.sort_values("time_s").reset_index(drop=True)

    if len(df) < 2:
        raise RuntimeError("Not enough samples in [0, t_max] to plot/interpolate.")

    t = df["time_s"].to_numpy(dtype=np.float64)

    csv_dir = os.path.dirname(os.path.abspath(csv_path)) or "."
    out_dir = os.path.expanduser(args.out_dir) if args.out_dir else os.path.join(csv_dir, "plots_marks_legend")
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Column presets (support BOTH old summary + new NKSR summary)
    #
    # OLD:
    #   pcd_pcd_* (pcd vs gt)
    #   mesh_mesh_* (mesh vs gt or mesh vs mesh depending old script)
    #
    # NEW (your updated evaluator):
    #   online_pcd_vs_gt_*
    #   offline_pcd_vs_gt_*
    #   mesh_vs_gt_*
    #   nksr_vs_mesh_*
    # ------------------------------------------------------------------

    # Distances (mm): accuracy / completeness
    fig1_cols = [
        # old
        "pcd_pcd_accuracy_mean",
        "pcd_pcd_completeness_mean",
        "mesh_mesh_accuracy_mean",
        "mesh_mesh_completeness_mean",
        # new meshes
        "mesh_vs_gt_accuracy_mean",
        "mesh_vs_gt_completeness_mean",
        "nksr_vs_gt_accuracy_mean",
        "nksr_vs_gt_completeness_mean",
        # new pcds
        "online_pcd_vs_gt_accuracy_mean",
        "online_pcd_vs_gt_completeness_mean",
        "offline_pcd_vs_gt_accuracy_mean",
        "offline_pcd_vs_gt_completeness_mean",
    ]

    # F@5mm
    fig2_cols = [
        # old
        "pcd_pcd_fscore@0.005000",
        "mesh_mesh_fscore@0.005000",
        # new meshes
        "mesh_vs_gt_fscore@0.005000",
        "nksr_vs_gt_fscore@0.005000",
        # new pcds
        "online_pcd_vs_gt_fscore@0.005000",
        "offline_pcd_vs_gt_fscore@0.005000",
    ]

    # PCD F-scores @ 2/5/10mm (OLD + NEW online/offline)
    fig3_cols = [
        # old
        "pcd_pcd_fscore@0.002000",
        "pcd_pcd_fscore@0.005000",
        "pcd_pcd_fscore@0.010000",
        # new online
        "online_pcd_vs_gt_fscore@0.002000",
        "online_pcd_vs_gt_fscore@0.005000",
        "online_pcd_vs_gt_fscore@0.010000",
        # new offline
        "offline_pcd_vs_gt_fscore@0.002000",
        "offline_pcd_vs_gt_fscore@0.005000",
        "offline_pcd_vs_gt_fscore@0.010000",
    ]

    # Mesh metrics F-scores @ 2/5/10mm (old mesh_mesh + new mesh_vs_gt)
    fig4_cols = [
        # old mesh-mesh
        "mesh_mesh_fscore@0.002000",
        "mesh_mesh_fscore@0.005000",
        "mesh_mesh_fscore@0.010000",
        # new mesh vs GT
        "mesh_vs_gt_fscore@0.002000",
        "mesh_vs_gt_fscore@0.005000",
        "mesh_vs_gt_fscore@0.010000",
    ]

    # NKSR mesh2mesh F-scores (nksr_vs_mesh)
    fig5_cols = [
        "nksr_vs_gt_fscore@0.002000",
        "nksr_vs_gt_fscore@0.005000",
        "nksr_vs_gt_fscore@0.010000",
    ]

    # NEW: Online vs Offline (same thresholds) - makes comparison obvious
    fig6_cols = [
        "online_pcd_vs_gt_fscore@0.002000",
        "offline_pcd_vs_gt_fscore@0.002000",
        "online_pcd_vs_gt_fscore@0.005000",
        "offline_pcd_vs_gt_fscore@0.005000",
        "online_pcd_vs_gt_fscore@0.010000",
        "offline_pcd_vs_gt_fscore@0.010000",
    ]

    plot_group(
        df, t, fig1_cols,
        title="Accuracy / Completeness (distance)  [PCD + Mesh + NKSR]",
        y_label="Distance (mm)",
        best_mode="min",
        to_mm=True,
        out_path=os.path.join(out_dir, "fig1_acc_comp_mm.png"),
        t_mark=args.t_mark,
        t_max=args.t_max,
        dpi=args.dpi,
        legend_loc=args.legend_loc,
        legend_fontsize=args.legend_fontsize,
    )

    plot_group(
        df, t, fig2_cols,
        title="F-score @ 5mm  [PCD + Mesh + NKSR]",
        y_label="F-score",
        best_mode="max",
        to_mm=False,
        out_path=os.path.join(out_dir, "fig2_fscore_5mm.png"),
        t_mark=args.t_mark,
        t_max=args.t_max,
        dpi=args.dpi,
        legend_loc=args.legend_loc,
        legend_fontsize=args.legend_fontsize,
    )

    plot_group(
        df, t, fig3_cols,
        title="PCD F-score @ 2/5/10mm (old pcd_pcd + new online/offline)",
        y_label="F-score",
        best_mode="max",
        to_mm=False,
        out_path=os.path.join(out_dir, "fig3_pcd_fscores.png"),
        t_mark=args.t_mark,
        t_max=args.t_max,
        dpi=args.dpi,
        legend_loc=args.legend_loc,
        legend_fontsize=args.legend_fontsize,
    )

    plot_group(
        df, t, fig4_cols,
        title="Mesh metrics F-score @ 2/5/10mm (Mesh-vs-Mesh or Mesh-vs-GT)",
        y_label="F-score",
        best_mode="max",
        to_mm=False,
        out_path=os.path.join(out_dir, "fig4_mesh_fscores.png"),
        t_mark=args.t_mark,
        t_max=args.t_max,
        dpi=args.dpi,
        legend_loc=args.legend_loc,
        legend_fontsize=args.legend_fontsize,
    )

    plot_group(
        df, t, fig5_cols,
        title="NKSR Mesh2Mesh F-score @ 2/5/10mm (NKSR vs Original Mesh)",
        y_label="F-score",
        best_mode="max",
        to_mm=False,
        out_path=os.path.join(out_dir, "fig5_nksr_mesh2mesh_fscores.png"),
        t_mark=args.t_mark,
        t_max=args.t_max,
        dpi=args.dpi,
        legend_loc=args.legend_loc,
        legend_fontsize=args.legend_fontsize,
    )

    plot_group(
        df, t, fig6_cols,
        title="Online vs Offline PCD (F-score @ 2/5/10mm)",
        y_label="F-score",
        best_mode="max",
        to_mm=False,
        out_path=os.path.join(out_dir, "fig6_online_vs_offline_pcd_fscores.png"),
        t_mark=args.t_mark,
        t_max=args.t_max,
        dpi=args.dpi,
        legend_loc=args.legend_loc,
        legend_fontsize=args.legend_fontsize,
    )

    print(f"[SAVED] 6 figures -> {out_dir}")
    print("  - fig1_acc_comp_mm.png")
    print("  - fig2_fscore_5mm.png")
    print("  - fig3_pcd_fscores.png")
    print("  - fig4_mesh_fscores.png")
    print("  - fig5_nksr_mesh2mesh_fscores.png")
    print("  - fig6_online_vs_offline_pcd_fscores.png")


if __name__ == "__main__":
    main()
