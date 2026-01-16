#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot Q1 (right): compare our Mesh--Mesh F-score-vs-tau curve against NeuralFeels/Suresh baseline
(using digitized real-world yellow-solid curve).

Inputs:
- --ours_csv: CSV with columns at least: tau_mm, fscore
- --nf_csv  : CSV with columns: tau_mm, nf_realworld_fscore

Output:
- --out: saved plot path (PNG)

Style: follows your left-figure aesthetic (small canvas, dashed grid, legend box).
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours_csv", type=str, default="/home/ferry/data/Code2/Research/Inhand_Activate/plot/Q1/mesh_mesh_fscore_vs_tau.csv",
                    help="Our mesh-mesh curve CSV (tau_mm,fscore,...) from mesh_mesh_fscore_vs_tau_align.py")
    ap.add_argument("--nf_csv", type=str, default="/home/ferry/data/Code2/Research/Inhand_Activate/plot/Q1/suresh_realworld_fscore_vs_tau_digitized.csv",
                    help="Digitized NeuralFeels real-world curve CSV (tau_mm,nf_realworld_fscore)")
    ap.add_argument("--out", type=str, default="fig_q1_right.png")
    ap.add_argument("--tau_max", type=float, default=10.0, help="Plot range [0, tau_max] mm")
    ap.add_argument("--label_fs", type=int, default=10)
    ap.add_argument("--tick_fs", type=int, default=9)
    ap.add_argument("--legend_fs", type=int, default=6.5)
    args = ap.parse_args()

    ours = pd.read_csv(args.ours_csv).sort_values("tau_mm")
    nf = pd.read_csv(args.nf_csv).sort_values("tau_mm")

    for c in ["tau_mm", "fscore"]:
        if c not in ours.columns:
            raise ValueError(f"[ours_csv] missing column: {c}")
    for c in ["tau_mm", "nf_realworld_fscore"]:
        if c not in nf.columns:
            raise ValueError(f"[nf_csv] missing column: {c}")

    ours = ours[(ours["tau_mm"] >= 0) & (ours["tau_mm"] <= args.tau_max)].copy()
    nf = nf[(nf["tau_mm"] >= 0) & (nf["tau_mm"] <= args.tau_max)].copy()

    plt.figure(figsize=(3.25, 3.0), dpi=160)
    ax = plt.gca()

    # Colors: keep Suresh curve in yellow; ours in black (paper-friendly).
    c_ours = "black"
    c_nf = "#f1c40f"  # yellow

    ax.plot(ours["tau_mm"], ours["fscore"], color=c_ours, lw=1.8, linestyle="-", alpha=0.95)
    ax.plot(nf["tau_mm"], nf["nf_realworld_fscore"], color=c_nf, lw=1.8, linestyle="--", alpha=0.95)

    ax.set_xlim(0, args.tau_max)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Threshold $\tau$ (mm)", fontsize=args.label_fs)
    ax.set_ylabel("Shape F-Score", fontsize=args.label_fs)
    ax.tick_params(axis="both", which="major", labelsize=args.tick_fs)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)

    handles = [
        Line2D([0],[0], color=c_ours, lw=2.0, linestyle="-", label="Ours (Mesh--Mesh)"),
        Line2D([0],[0], color=c_nf, lw=2.0, linestyle="--", label="NeuralFeels (real world)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, fancybox=True, framealpha=1.0, fontsize=args.legend_fs)

    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"[OK] Saved: {args.out}")


if __name__ == "__main__":
    main()
