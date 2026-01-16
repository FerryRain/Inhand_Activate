"""
@FileName：vs_baseline_fscore_0.5.py
@Description：Overlay our curves with NeuralFeels (Suresh) using consistent colors and add a vertical dashed line at the time our F-score matches NeuralFeels
@Author：Ferry
@Time：2026 1/15/26 4:00 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def find_match_time_ours_reach_nf_max(t_ours, f_ours, t_nf, f_nf):
    """
    Preferred definition (matches your paper text intent):
    - Take NeuralFeels' best F-score within the window as reference: f_nf_ref = max_t f_nf(t).
    - Return the earliest time t where our f_ours(t) >= f_nf_ref.
    Fallback:
    - If never reached, return the time where |f_ours(t) - f_nf(t)| is minimal (after interpolation).
    """
    t_ours = np.asarray(t_ours, dtype=float)
    f_ours = np.asarray(f_ours, dtype=float)
    t_nf = np.asarray(t_nf, dtype=float)
    f_nf = np.asarray(f_nf, dtype=float)

    # Reference: NF best within window
    f_ref = float(np.nanmax(f_nf))
    # Earliest time we reach it
    idx = np.where(f_ours >= f_ref)[0]
    if idx.size > 0:
        t_match = float(t_ours[idx[0]])
        return t_match, f_ref, "reach_nf_max"

    # Fallback: closest to NF curve (pointwise)
    f_nf_on_ours = np.interp(t_ours, t_nf, f_nf)
    i = int(np.nanargmin(np.abs(f_ours - f_nf_on_ours)))
    t_match = float(t_ours[i])
    f_ref2 = float(f_nf_on_ours[i])
    return t_match, f_ref2, "closest_point"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ours_csv",
        type=str,
        default="/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/result/green_cube_00_001/eval_strictref/summary.csv",
    )
    ap.add_argument(
        "--nf_csv",
        type=str,
        default="neuralfeels_digitized_tau5mm.csv",
        help="Digitized NeuralFeels curves (tau=5mm): time_s, nf_precision_tau5mm, nf_recall_tau5mm, nf_fscore_tau5mm",
    )
    ap.add_argument("--out", type=str, default="fig_q1_left.png")
    ap.add_argument("--smooth_win", type=int, default=5)
    ap.add_argument("--t_max", type=float, default=30.0)
    ap.add_argument("--vline_color", type=str, default="0.35", help="Vertical line color (matplotlib color)")
    ap.add_argument("--vline_lw", type=float, default=1.2)
    ap.add_argument("--label_fs", type=int, default=10, help="Font size for axis labels")
    ap.add_argument("--tick_fs", type=int, default=9, help="Font size for tick labels")
    ap.add_argument("--legend_fs", type=int, default=6.5, help="Font size for legend")

    args = ap.parse_args()

    # ----------------------------
    # Load CSVs
    # ----------------------------
    df = pd.read_csv(args.ours_csv).sort_values("time_s")
    nf = pd.read_csv(args.nf_csv).sort_values("time_s")

    # Crop to 0~t_max
    df = df[(df["time_s"] >= 0) & (df["time_s"] <= args.t_max)].copy()
    nf = nf[(nf["time_s"] >= 0) & (nf["time_s"] <= args.t_max)].copy()

    # tau = 5mm corresponds to @0.005000
    req_cols = [
        "time_s",
        "pcd_pcd_fscore@0.005000",
        "pcd_pcd_precision@0.005000",
        "pcd_pcd_recall@0.005000",
    ]
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"[ours_csv] missing column: {c}")

    req_nf_cols = ["time_s", "nf_fscore_tau5mm", "nf_precision_tau5mm", "nf_recall_tau5mm"]
    for c in req_nf_cols:
        if c not in nf.columns:
            raise ValueError(f"[nf_csv] missing column: {c}")

    t = df["time_s"].to_numpy(dtype=float)
    f = df["mesh_mesh_fscore@0.005000"].to_numpy(dtype=float)
    p = df["mesh_mesh_precision@0.005000"].to_numpy(dtype=float)
    r = df["mesh_mesh_recall@0.005000"].to_numpy(dtype=float)

    t_nf = nf["time_s"].to_numpy(dtype=float)
    f_nf = nf["nf_fscore_tau5mm"].to_numpy(dtype=float)
    p_nf = nf["nf_precision_tau5mm"].to_numpy(dtype=float)
    r_nf = nf["nf_recall_tau5mm"].to_numpy(dtype=float)

    # Optional smoothing for our curves (paper-like)
    if args.smooth_win and args.smooth_win > 1:
        def smooth(x):
            return pd.Series(x).rolling(args.smooth_win, center=True, min_periods=1).mean().to_numpy(dtype=float)
        f, p, r = smooth(f), smooth(p), smooth(r)

    # ----------------------------
    # Compute match time (vline)
    # ----------------------------
    t_match, f_ref, mode = find_match_time_ours_reach_nf_max(t, f, t_nf, f_nf)

    # y value at t_match (ours)
    f_match_ours = float(np.interp(t_match, t, f))

    # ----------------------------
    # Plot (paper-like styling)
    # ----------------------------
    plt.figure(figsize=(3.25, 3.0), dpi=160)
    ax = plt.gca()

    # Fixed colors to match the paper figure
    c_f = "black"      # F-score
    c_p = "#ff7f0e"    # precision (orange)
    c_r = "#2ca02c"    # recall (green)

    # Line styles: Solid = Ours, Dashed = NeuralFeels (Suresh)
    ax.plot(t, f, color=c_f, linestyle="-", lw=1.8, alpha=0.95)
    ax.plot(t, p, color=c_p, linestyle="-", lw=1.2, alpha=0.95)
    ax.plot(t, r, color=c_r, linestyle="-", lw=1.2, alpha=0.95)

    ax.plot(t_nf, f_nf, color=c_f, linestyle="--", lw=1.8, alpha=0.95)
    ax.plot(t_nf, p_nf, color=c_p, linestyle="--", lw=1.2, alpha=0.95)
    ax.plot(t_nf, r_nf, color=c_r, linestyle="--", lw=1.2, alpha=0.95)

    # Vertical dashed line at match time
    ax.axvline(t_match, color=args.vline_color, linestyle="--", lw=args.vline_lw, alpha=0.95)
    # Optional marker at intersection height (kept subtle)
    ax.plot([t_match], [f_match_ours], marker="o", markersize=3.0, color=args.vline_color, alpha=0.95)

    # Axes formatting (match screenshot)
    # Axes formatting (match screenshot)
    ax.set_xlim(0, args.t_max)
    ax.set_ylim(0.4, 1.0)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])

    ax.set_xlabel("time (secs)", fontsize=args.label_fs)
    ax.set_ylabel(r"Shape F-Score ($\tau$ = 5mm)", fontsize=args.label_fs)

    # tick label font size
    ax.tick_params(axis="both", which="major", labelsize=args.tick_fs)

    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)

    # Legend (use args.legend_fs)

    # Combined legend: colors = metrics, line styles = methods, plus vline
    handles_metrics = [
        Line2D([0], [0], color=c_f, lw=2.0, linestyle="-", label="F-score"),
        Line2D([0], [0], color=c_p, lw=2.0, linestyle="-", label="precision"),
        Line2D([0], [0], color=c_r, lw=2.0, linestyle="-", label="recall"),
    ]
    handles_style = [
        Line2D([0], [0], color="black", lw=2.0, linestyle="-", label="Ours"),
        Line2D([0], [0], color="black", lw=2.0, linestyle="--", label="NeuralFeels"),
    ]

    # Legend entry for the vertical line
    if mode == "reach_nf_max":
        vline_label = f"Match NF best F-score"
    else:
        vline_label = f"Closest F-score match @ t={t_match:.1f}s"
    handles_vline = [
        Line2D([0], [0], color=args.vline_color, lw=args.vline_lw, linestyle="--", label=vline_label)
    ]

    ax.legend(
        handles=handles_metrics + handles_style + handles_vline,
        loc="lower right",
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        borderpad=0.6,
        handlelength=2.6,
        labelspacing=0.4,
        fontsize=args.legend_fs,
    )

    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"[OK] Saved: {args.out}")
    print(f"[INFO] Vertical line: t_match={t_match:.3f}s, ref_f={f_ref:.4f}, mode={mode}")


if __name__ == "__main__":
    main()
