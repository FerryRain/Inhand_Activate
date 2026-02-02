"""
@FileName：compare_uncertainty.py
@Description：
@Author：Ferry
@Time：2026 1/26/26 3:07 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare mean curves between two CSV files.

Required columns (both CSVs):
  - time_s
  - mean_total_unc_rf
  - mean_total_score

Outputs:
  - out_dir/mean_unc_compare.png
  - out_dir/mean_score_compare.png
  - out_dir/mean_unc_diff.png
  - out_dir/mean_score_diff.png
  - out_dir/compare_mean_curves.csv
  - printed trend summary in stdout
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REQ_COLS = ["time_s", "mean_total_unc_rf", "mean_total_score"]


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[{path}] missing required columns: {missing}. "
                         f"Need {REQ_COLS}.")
    df = df[REQ_COLS].copy()
    for c in REQ_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time_s"])
    df = df.sort_values("time_s").drop_duplicates("time_s", keep="last")
    return df


def _time_window(df: pd.DataFrame, tmin: Optional[float], tmax: Optional[float]) -> pd.DataFrame:
    if tmin is not None:
        df = df[df["time_s"] >= float(tmin)]
    if tmax is not None:
        df = df[df["time_s"] <= float(tmax)]
    return df


def _rolling(y: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return y
    s = pd.Series(y)
    return s.rolling(window=win, min_periods=max(1, win // 3), center=True).mean().to_numpy()


def _interp_to_grid(t_src: np.ndarray, y_src: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    # Only interpolate where src is finite
    m = np.isfinite(t_src) & np.isfinite(y_src)
    t = t_src[m]
    y = y_src[m]
    if t.size < 2:
        return np.full_like(t_grid, np.nan, dtype=np.float64)

    # For extrapolation, we keep NaN (safer than extrapolating)
    y_grid = np.interp(t_grid, t, y, left=np.nan, right=np.nan)

    # np.interp doesn't support NaN left/right directly in older numpy; emulate:
    # find bounds
    tmin, tmax = t.min(), t.max()
    y_grid[(t_grid < tmin) | (t_grid > tmax)] = np.nan
    return y_grid


def _build_grid(
    t_a: np.ndarray,
    t_b: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "inner":
        # intersection points only
        s = set(np.round(t_a, 10).tolist()).intersection(set(np.round(t_b, 10).tolist()))
        grid = np.array(sorted(s), dtype=np.float64)
        return grid
    if mode == "union":
        s = set(np.round(t_a, 10).tolist()).union(set(np.round(t_b, 10).tolist()))
        return np.array(sorted(s), dtype=np.float64)
    if mode == "to_a":
        return np.array(t_a, dtype=np.float64)
    if mode == "to_b":
        return np.array(t_b, dtype=np.float64)
    raise ValueError(f"Unknown align mode: {mode}")


def _trend_stats(t: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    m = np.isfinite(t) & np.isfinite(y)
    t = t[m]
    y = y[m]
    if t.size < 2:
        return {
            "n": float(t.size),
            "slope": np.nan,
            "delta": np.nan,
            "delta_pct": np.nan,
            "auc": np.nan,
            "mono_up_ratio": np.nan,
        }

    # Linear slope (least squares): y = a*t + b
    A = np.vstack([t, np.ones_like(t)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]

    y0, y1 = y[0], y[-1]
    delta = float(y1 - y0)
    delta_pct = float(delta / (abs(y0) + 1e-12) * 100.0)

    # Area under curve
    auc = float(np.trapz(y, t))

    # Monotonic “up” ratio (fraction of positive diffs among finite consecutive points)
    dy = np.diff(y)
    if dy.size == 0:
        mono = np.nan
    else:
        mono = float(np.mean(dy > 0))

    return {
        "n": float(t.size),
        "slope": float(a),
        "delta": delta,
        "delta_pct": delta_pct,
        "auc": auc,
        "mono_up_ratio": mono,
    }


def _print_stats(name: str, stats_unc: Dict[str, float], stats_score: Dict[str, float]):
    def fmt(d: Dict[str, float]) -> str:
        return (f"n={int(d['n'])}  slope={d['slope']:.6g}  "
                f"delta={d['delta']:.6g}  delta%={d['delta_pct']:.3g}%  "
                f"AUC={d['auc']:.6g}  mono_up={d['mono_up_ratio']:.3f}")

    print(f"\n[{name}] Unc : {fmt(stats_unc)}")
    print(f"[{name}] Score: {fmt(stats_score)}")


def _plot_compare(
    t: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
    label_a: str,
    label_b: str,
    title: str,
    ylabel: str,
    out_path: Path,
    y_lim: Optional[Tuple[float, float]],
    smooth_win: int,
):
    plt.figure()
    plt.plot(t, y_a, label=label_a)
    plt.plot(t, y_b, label=label_b)

    if smooth_win > 1:
        plt.plot(t, _rolling(y_a, smooth_win), label=f"{label_a} (smooth{smooth_win})", linewidth=2.0)
        plt.plot(t, _rolling(y_b, smooth_win), label=f"{label_b} (smooth{smooth_win})", linewidth=2.0)

    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    if y_lim is not None:
        plt.ylim(float(y_lim[0]), float(y_lim[1]))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def _plot_diff(
    t: np.ndarray,
    diff: np.ndarray,
    title: str,
    ylabel: str,
    out_path: Path,
):
    plt.figure()
    plt.plot(t, diff, label="B - A")
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_a", default="/home/ferry/data/Code2/Research/Inhand_Activate/Active/pcd/pcd_active/nbv_rf_curve_root.csv", help="First run CSV (baseline).")
    ap.add_argument("--csv_b", default="/home/ferry/data/Code2/Research/Inhand_Activate/Active/pcd/pcd_offline/xyz/nbv_rf_curve_root.csv", help="Second run CSV (to compare).")
    ap.add_argument("--label_a", default="run_A")
    ap.add_argument("--label_b", default="run_B")

    ap.add_argument("--align", default="union", choices=["union", "inner", "to_a", "to_b"],
                    help="How to align time axis. union=default (interpolate), inner=common times only.")
    ap.add_argument("--tmin", type=float, default=None)
    ap.add_argument("--tmax", type=float, default=None)

    ap.add_argument("--smooth", type=int, default=1, help="Rolling mean window for display only (>=1).")

    # Optional y-limits (to reproduce your locked axes if desired)
    ap.add_argument("--ylim_unc", type=float, nargs=2, default=None, metavar=("YMIN", "YMAX"))
    ap.add_argument("--ylim_score", type=float, nargs=2, default=None, metavar=("YMIN", "YMAX"))

    ap.add_argument("--out_dir", default="./compare_out")

    args = ap.parse_args()

    df_a = _time_window(_read_csv(args.csv_a), args.tmin, args.tmax)
    df_b = _time_window(_read_csv(args.csv_b), args.tmin, args.tmax)

    t_a = df_a["time_s"].to_numpy(dtype=np.float64)
    t_b = df_b["time_s"].to_numpy(dtype=np.float64)

    t = _build_grid(t_a, t_b, args.align)

    # Interpolate means onto grid
    unc_a = _interp_to_grid(t_a, df_a["mean_total_unc_rf"].to_numpy(np.float64), t)
    unc_b = _interp_to_grid(t_b, df_b["mean_total_unc_rf"].to_numpy(np.float64), t)
    score_a = _interp_to_grid(t_a, df_a["mean_total_score"].to_numpy(np.float64), t)
    score_b = _interp_to_grid(t_b, df_b["mean_total_score"].to_numpy(np.float64), t)

    diff_unc = unc_b - unc_a
    diff_score = score_b - score_a

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save aligned comparison CSV
    out_df = pd.DataFrame({
        "time_s": t,
        f"{args.label_a}_mean_total_unc_rf": unc_a,
        f"{args.label_b}_mean_total_unc_rf": unc_b,
        "diff_unc(B-A)": diff_unc,
        f"{args.label_a}_mean_total_score": score_a,
        f"{args.label_b}_mean_total_score": score_b,
        "diff_score(B-A)": diff_score,
    })
    out_csv = out_dir / "compare_mean_curves.csv"
    out_df.to_csv(out_csv, index=False)

    # Trend stats
    stats_a_unc = _trend_stats(t, unc_a)
    stats_b_unc = _trend_stats(t, unc_b)
    stats_a_score = _trend_stats(t, score_a)
    stats_b_score = _trend_stats(t, score_b)

    _print_stats(args.label_a, stats_a_unc, stats_a_score)
    _print_stats(args.label_b, stats_b_unc, stats_b_score)

    # Compare slopes directly (simple headline)
    def _cmp(name, a, b):
        if np.isfinite(a) and np.isfinite(b):
            print(f"[Compare] {name} slope: {args.label_b} - {args.label_a} = {b - a:.6g}")
        else:
            print(f"[Compare] {name} slope: insufficient finite data.")

    _cmp("Unc", stats_a_unc["slope"], stats_b_unc["slope"])
    _cmp("Score", stats_a_score["slope"], stats_b_score["slope"])

    # Plots
    _plot_compare(
        t, unc_a, unc_b,
        args.label_a, args.label_b,
        title="Mean Total UncRF over Time",
        ylabel="mean_total_unc_rf",
        out_path=out_dir / "mean_unc_compare.png",
        y_lim=tuple(args.ylim_unc) if args.ylim_unc is not None else None,
        smooth_win=max(1, int(args.smooth)),
    )

    _plot_compare(
        t, score_a, score_b,
        args.label_a, args.label_b,
        title="Mean Total Score over Time",
        ylabel="mean_total_score",
        out_path=out_dir / "mean_score_compare.png",
        y_lim=tuple(args.ylim_score) if args.ylim_score is not None else None,
        smooth_win=max(1, int(args.smooth)),
    )

    _plot_diff(
        t, diff_unc,
        title="Difference in Mean UncRF (B - A)",
        ylabel="diff_unc",
        out_path=out_dir / "mean_unc_diff.png",
    )

    _plot_diff(
        t, diff_score,
        title="Difference in Mean Score (B - A)",
        ylabel="diff_score",
        out_path=out_dir / "mean_score_diff.png",
    )

    print(f"\n[Done] outputs in: {out_dir.resolve()}")
    print(f"[Done] aligned CSV: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
