#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot F-score vs threshold tau from a summary CSV (new or legacy naming).

Supports columns like:
  nksr_vs_gt_fscore@5mm
  nksr_vs_gt_fscore@0.005m
  nksr_vs_gt_fscore@0.005000   (interpreted as meters by default if <= 0.05)

Selects the frame closest to target_time_s (default 30s) and plots tau in [0,10] mm.

Example:
python mesh_mesh_Fscore_diff_threshold.py \
  --in_csv /path/to/summary.csv \
  --prefix nksr_vs_gt_fscore \
  --target_time_s 30 \
  --out_fig fig_tau_t30.png
"""

import argparse
import csv
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        return list(r)


def safe_float(x: Optional[str]) -> float:
    if x is None:
        return float("nan")
    x = str(x).strip()
    if x == "" or x.lower() == "nan":
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def discover_tau_columns(header: List[str], prefix: str) -> List[Tuple[float, str]]:
    """
    Detect columns whose names look like:
      {prefix}@{num}mm
      {prefix}@{num}m
      {prefix}@{num}
    Return list of (tau_mm, colname), sorted by tau_mm.
    """
    out: List[Tuple[float, str]] = []
    p = prefix + "@"
    for k in header:
        kk = k.strip()
        if not kk.startswith(p):
            continue
        suf = kk[len(p):].strip()

        unit = None
        if suf.endswith("mm"):
            unit = "mm"
            num_str = suf[:-2]
        elif suf.endswith("m"):
            unit = "m"
            num_str = suf[:-1]
        else:
            num_str = suf

        # allow commas/spaces
        num_str = num_str.strip()
        try:
            val = float(num_str)
        except Exception:
            continue

        # interpret unit
        if unit == "mm":
            tau_mm = val
        elif unit == "m":
            tau_mm = val * 1000.0
        else:
            # legacy style often stores tau in meters like 0.005000 (no suffix)
            # heuristic: if val <= 0.05 -> treat as meters, else millimeters
            tau_mm = val * 1000.0 if val <= 0.05 else val

        out.append((float(tau_mm), kk))

    out.sort(key=lambda x: x[0])
    return out


def row_tau_curve(row: Dict[str, str], tau_cols: List[Tuple[float, str]]) -> Tuple[np.ndarray, np.ndarray]:
    taus = []
    vals = []
    for tau_mm, col in tau_cols:
        taus.append(tau_mm)
        vals.append(safe_float(row.get(col)))
    return np.asarray(taus, dtype=float), np.asarray(vals, dtype=float)


def select_row_by_time(rows: List[Dict[str, str]], target_time_s: float, tol_s: float) -> Tuple[int, Dict[str, str], float]:
    times = np.asarray([safe_float(r.get("time_s")) for r in rows], dtype=float)
    valid = np.isfinite(times)
    if not np.any(valid):
        raise RuntimeError("All rows have invalid time_s; cannot select by time.")

    diffs = np.abs(times - float(target_time_s))
    diffs[~valid] = np.inf
    i = int(np.argmin(diffs))
    err = float(diffs[i])

    if err > float(tol_s):
        print(f"[WARN] Closest time_s is {times[i]:.3f}s (|Δ|={err:.3f}s) > tol_s={tol_s:.3f}s")

    return i, rows[i], err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="/home/ferry/data/Code2/Research/Inhand_Activate/plot/Q3/summary_nksr_vs_gt.csv", help="Path to summary CSV")
    ap.add_argument("--prefix", type=str, default="nksr_vs_gt_fscore",
                    help="Column prefix before '@', e.g. nksr_vs_gt_fscore or mesh_vs_gt_fscore")
    ap.add_argument("--out_fig", default="fig_tau_t30.png")
    ap.add_argument("--out_csv", default="", help="Optional: save selected tau-curve CSV")

    ap.add_argument("--target_time_s", type=float, default=26.0)
    ap.add_argument("--tol_s", type=float, default=1.0)

    ap.add_argument("--tau_min_mm", type=float, default=0.0)
    ap.add_argument("--tau_max_mm", type=float, default=10.0)

    # plot style
    ap.add_argument("--label_fs", type=int, default=10)
    ap.add_argument("--tick_fs", type=int, default=9)
    ap.add_argument("--legend_fs", type=int, default=8)
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--fig_w", type=float, default=3.25)
    ap.add_argument("--fig_h", type=float, default=3.0)

    args = ap.parse_args()

    rows = read_csv_rows(args.in_csv)
    if not rows:
        raise RuntimeError("CSV has no data rows.")

    header = list(rows[0].keys())
    tau_cols = discover_tau_columns(header, prefix=args.prefix)

    # if still empty, print helpful hint
    if not tau_cols:
        cand = [k for k in header if "fscore@" in k or args.prefix.split("@")[0] in k]
        msg = (
            f"Cannot find tau columns with prefix='{args.prefix}@...'.\n"
            f"Hint: your CSV header may use a different prefix or format.\n"
            f"Some related columns: {cand[:30]}"
        )
        raise RuntimeError(msg)

    # restrict tau range
    tau_cols = [(t, c) for (t, c) in tau_cols if (t >= args.tau_min_mm - 1e-12 and t <= args.tau_max_mm + 1e-12)]
    if not tau_cols:
        raise RuntimeError("No tau columns remain after applying --tau_min_mm/--tau_max_mm.")

    # select row at target time
    idx, row, err = select_row_by_time(rows, args.target_time_s, args.tol_s)

    fid = int(safe_float(row.get("frame_id")))
    ts = safe_float(row.get("time_s"))

    taus_mm, fvals = row_tau_curve(row, tau_cols)
    m = np.isfinite(fvals)
    taus_mm = taus_mm[m]
    fvals = fvals[m]
    if taus_mm.size == 0:
        raise RuntimeError("Selected row has all-NaN F-scores in the requested tau range.")

    # save curve CSV
    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tau_mm", "fscore"])
            for t, v in zip(taus_mm.tolist(), fvals.tolist()):
                w.writerow([f"{t:.3f}", f"{v:.6f}"])
        print(f"[SAVED] curve csv -> {args.out_csv}")

    # plot
    plt.figure(figsize=(args.fig_w, args.fig_h), dpi=args.dpi)
    ax = plt.gca()

    label = f"{args.prefix.replace('_fscore','')} (t={ts:.2f}s, id={fid:06d}, |Δ|={err:.2f}s)"
    ax.plot(taus_mm, fvals, lw=1.8, linestyle="-", label=label)

    ax.set_xlim(args.tau_min_mm, args.tau_max_mm)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Threshold $\tau$ (mm)", fontsize=args.label_fs)
    ax.set_ylabel("Shape F-Score", fontsize=args.label_fs)
    ax.tick_params(axis="both", which="major", labelsize=args.tick_fs)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc="lower right", frameon=True, fancybox=True, framealpha=1.0, fontsize=args.legend_fs)

    plt.tight_layout()
    plt.savefig(args.out_fig, bbox_inches="tight")
    print(f"[SAVED] fig -> {args.out_fig}")
    print(f"[INFO] selected_row_index={idx}, frame_id={fid}, time_s={ts}, target_time_s={args.target_time_s}, abs_err_s={err:.6f}")


if __name__ == "__main__":
    main()
