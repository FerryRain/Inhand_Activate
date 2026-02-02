#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare curves over time for F-score@5mm: precision/recall/fscore, within [t_min, t_max].

Suresh CSV header (required):
time_s,nf_precision_tau5mm,nf_recall_tau5mm,nf_fscore_tau5mm

Your CSV header (required fields depend on --ours_prefix):
time_s
{prefix}_precision@0.005000
{prefix}_recall@0.005000
{prefix}_fscore@0.005000

If --ours_path is a directory, average OUR curves over all *.csv in that directory
(after resampling to a common time grid).
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SURESH_COLS = {
    "t": "time_s",
    "p": "nf_precision_tau5mm",
    "r": "nf_recall_tau5mm",
    "f": "nf_fscore_tau5mm",
}


def _coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _dedup_by_time_mean(df: pd.DataFrame, tcol: str, ycols):
    df = df.dropna(subset=[tcol])
    if df.empty:
        return df
    gcols = [tcol] + list(ycols)
    df = df[gcols].groupby(tcol, as_index=False).mean(numeric_only=True)
    return df


def load_suresh(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in SURESH_COLS.values() if c not in df.columns]
    if missing:
        raise ValueError(f"[Suresh CSV] missing columns: {missing}\nGot: {list(df.columns)}")

    df = _coerce_numeric(df, [SURESH_COLS["t"], SURESH_COLS["p"], SURESH_COLS["r"], SURESH_COLS["f"]])
    df = _dedup_by_time_mean(df, SURESH_COLS["t"], [SURESH_COLS["p"], SURESH_COLS["r"], SURESH_COLS["f"]])
    df = df.sort_values(SURESH_COLS["t"])
    return df


def detect_default_prefix(ours_df: pd.DataFrame) -> str:
    preferred = ["online_pcd_vs_gt", "offline_pcd_vs_gt", "mesh_vs_gt", "nksr_vs_gt"]

    def ok(prefix: str) -> bool:
        return (
            f"{prefix}_precision@0.005000" in ours_df.columns
            and f"{prefix}_recall@0.005000" in ours_df.columns
            and f"{prefix}_fscore@0.005000" in ours_df.columns
        )

    for p in preferred:
        if ok(p):
            return p

    candidates = []
    for col in ours_df.columns:
        if col.endswith("_precision@0.005000"):
            prefix = col.replace("_precision@0.005000", "")
            if ok(prefix):
                candidates.append(prefix)

    if not candidates:
        raise ValueError(
            "Cannot detect a valid --ours_prefix. Need columns:\n"
            "  {prefix}_precision@0.005000, {prefix}_recall@0.005000, {prefix}_fscore@0.005000\n"
            f"Got columns: {list(ours_df.columns)}"
        )
    return candidates[0]


def load_ours_one(csv_path: Path, prefix: str | None) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(csv_path)
    if "time_s" not in df.columns:
        raise ValueError(f"[Your CSV] missing column: time_s\nFile: {csv_path}\nGot: {list(df.columns)}")

    if prefix is None or prefix.strip() == "":
        prefix = detect_default_prefix(df)

    cols = {
        "t": "time_s",
        "p": f"{prefix}_precision@0.005000",
        "r": f"{prefix}_recall@0.005000",
        "f": f"{prefix}_fscore@0.005000",
    }
    missing = [c for c in cols.values() if c not in df.columns]
    if missing:
        raise ValueError(
            f"[Your CSV] missing columns for prefix='{prefix}': {missing}\n"
            f"File: {csv_path}\nGot: {list(df.columns)}"
        )

    df = _coerce_numeric(df, [cols["t"], cols["p"], cols["r"], cols["f"]])
    df = _dedup_by_time_mean(df, cols["t"], [cols["p"], cols["r"], cols["f"]])
    df = df.sort_values(cols["t"])
    return df, prefix


def smooth_series(y: np.ndarray, window: int) -> np.ndarray:
    if window is None or window <= 1:
        return y
    s = pd.Series(y)
    return s.rolling(window=window, center=True, min_periods=max(1, window // 2)).mean().to_numpy()


def resample_interp(t: np.ndarray, y: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if t.size < 2:
        return np.full_like(t_grid, np.nan, dtype=float)
    order = np.argsort(t)
    t, y = t[order], y[order]
    return np.interp(t_grid, t, y, left=np.nan, right=np.nan)


def clip_range(t: np.ndarray, y: np.ndarray, tmin: float, tmax: float):
    m = np.isfinite(t) & np.isfinite(y) & (t >= tmin) & (t <= tmax)
    return t[m], y[m]


def load_ours_from_path(ours_path: Path, prefix: str | None) -> tuple[list[pd.DataFrame], str, list[Path]]:
    """
    Returns:
      dfs: list of per-file dataframes (already cleaned/deduped)
      prefix: resolved prefix
      files: list of csv paths actually loaded
    """
    if ours_path.is_file():
        df, pref = load_ours_one(ours_path, prefix)
        return [df], pref, [ours_path]

    if not ours_path.is_dir():
        raise ValueError(f"--ours_path must be a CSV file or a directory. Got: {ours_path}")

    files = sorted(ours_path.glob("*.csv"))
    if not files:
        raise ValueError(f"No *.csv found in directory: {ours_path}")

    dfs: list[pd.DataFrame] = []
    resolved_prefix: str | None = prefix

    # resolve prefix from first readable file if not provided
    first_df = None
    for fp in files:
        try:
            tmp = pd.read_csv(fp, nrows=5)
            first_df = tmp
            break
        except Exception:
            continue
    if first_df is None:
        raise ValueError(f"All CSV files failed to read in: {ours_path}")

    if resolved_prefix is None or resolved_prefix.strip() == "":
        resolved_prefix = detect_default_prefix(first_df)

    ok_files = []
    for fp in files:
        try:
            df, _ = load_ours_one(fp, resolved_prefix)
            dfs.append(df)
            ok_files.append(fp)
        except Exception as e:
            print(f"[WARN] Skip file due to error: {fp}\n  -> {e}", file=sys.stderr)

    if not dfs:
        raise ValueError(f"All CSV files failed (or were skipped) in: {ours_path}")

    return dfs, resolved_prefix, ok_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suresh_csv", type=Path,
                    default="/home/ferry/data/Code2/Research/Inhand_Activate/plot/Q3/neuralfeels_digitized_tau5mm.csv")
    # NOTE: now supports file OR directory
    ap.add_argument("--ours_path", type=Path,
                    default="/home/ferry/data/Code2/Research/Inhand_Activate/plot/Q3/ours_csv",
                    help="Path to a CSV file OR a directory containing multiple CSVs.")
    ap.add_argument("--ours_prefix", type=str, default="nksr_vs_gt",
                    help="Prefix for your metrics, e.g., online_pcd_vs_gt / mesh_vs_gt / nksr_vs_gt")
    ap.add_argument("--dt", type=float, default=1.0,
                    help="Resample onto a uniform time grid with step dt (seconds). REQUIRED for directory averaging.")
    ap.add_argument("--t_min", type=float, default=0.0)
    ap.add_argument("--t_max", type=float, default=30.0)
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("compare_prf5mm_mean_0_20s.png"))
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    # time window
    tmin, tmax = float(args.t_min), float(args.t_max)
    if not (np.isfinite(tmin) and np.isfinite(tmax) and tmax > tmin):
        raise ValueError(f"Invalid time range: tmin={tmin}, tmax={tmax}")

    # suresh
    df_s = load_suresh(args.suresh_csv)
    ts = df_s[SURESH_COLS["t"]].to_numpy(dtype=float)
    ps = df_s[SURESH_COLS["p"]].to_numpy(dtype=float)
    rs = df_s[SURESH_COLS["r"]].to_numpy(dtype=float)
    fs = df_s[SURESH_COLS["f"]].to_numpy(dtype=float)

    ts, ps = clip_range(ts, ps, tmin, tmax)
    _,  rs = clip_range(df_s[SURESH_COLS["t"]].to_numpy(dtype=float), rs, tmin, tmax)
    _,  fs = clip_range(df_s[SURESH_COLS["t"]].to_numpy(dtype=float), fs, tmin, tmax)

    # ours (file or directory)
    ours_dfs, prefix, used_files = load_ours_from_path(args.ours_path, args.ours_prefix)

    # averaging requires common grid
    if args.dt is None or args.dt <= 0:
        if args.ours_path.is_dir():
            raise ValueError("Directory averaging requires --dt > 0 to build a common time grid.")
        # single-file mode can work without dt, but we keep code simple: still build grid if dt<=0? no.
        # Here, we fall back to using the file's own timestamps (no averaging anyway).
    # build common grid always when dt>0 (recommended)
    if args.dt is not None and args.dt > 0:
        t_grid = np.arange(tmin, tmax + 1e-9, args.dt, dtype=float)
    else:
        t_grid = None

    # prepare ours stacks (resampled)
    P_list, R_list, F_list = [], [], []
    for df in ours_dfs:
        to = df["time_s"].to_numpy(dtype=float)
        po = df[f"{prefix}_precision@0.005000"].to_numpy(dtype=float)
        ro = df[f"{prefix}_recall@0.005000"].to_numpy(dtype=float)
        fo = df[f"{prefix}_fscore@0.005000"].to_numpy(dtype=float)

        to, po = clip_range(to, po, tmin, tmax)
        _,  ro = clip_range(df["time_s"].to_numpy(dtype=float), ro, tmin, tmax)
        _,  fo = clip_range(df["time_s"].to_numpy(dtype=float), fo, tmin, tmax)

        if t_grid is not None:
            P_list.append(resample_interp(to, po, t_grid))
            R_list.append(resample_interp(to, ro, t_grid))
            F_list.append(resample_interp(to, fo, t_grid))
        else:
            # single-file no-grid fallback
            P_list.append(po)
            R_list.append(ro)
            F_list.append(fo)

    # ours mean
    if t_grid is not None:
        P = np.nanmean(np.vstack(P_list), axis=0)
        R = np.nanmean(np.vstack(R_list), axis=0)
        F = np.nanmean(np.vstack(F_list), axis=0)
        t_plot = t_grid
    else:
        # single file mode only
        P, R, F = P_list[0], R_list[0], F_list[0]
        t_plot = ours_dfs[0]["time_s"].to_numpy(dtype=float)

    # suresh resample to same grid (if grid exists)
    if t_grid is not None:
        ps = resample_interp(ts, ps, t_grid)
        rs = resample_interp(ts, rs, t_grid)
        fs = resample_interp(ts, fs, t_grid)

    # smoothing (apply after averaging / resampling)
    P_s = smooth_series(P, args.smooth)
    R_s = smooth_series(R, args.smooth)
    F_s = smooth_series(F, args.smooth)

    ps_s = smooth_series(ps, args.smooth)
    rs_s = smooth_series(rs, args.smooth)
    fs_s = smooth_series(fs, args.smooth)

    # ---------------- Plot: 1 figure (P/R/F), y in [0.5, 1.0] ----------------
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    def plot_triplet(ax, t, p, r, f, method_label: str, method_ls: str):
        ax.plot(t, p, linestyle=method_ls, linewidth=1.5, label=f"{method_label}  Precision")
        ax.plot(t, r, linestyle=method_ls, linewidth=1.5, label=f"{method_label}  Recall")
        ax.plot(t, f, linestyle=method_ls, linewidth=2.5, label=f"{method_label}  F-score")

    plot_triplet(ax, t_plot, P_s, R_s, F_s, f"Ours", method_ls="-")
    plot_triplet(ax, t_plot if t_grid is not None else ts, ps_s, rs_s, fs_s, "Neural Feels", method_ls="--")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("Metric value @ 5mm")
    ax.set_xlim(tmin, tmax)
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

    title = f"P/R/F @ 5mm vs time ({tmin:.0f}–{tmax:.0f}s) (Ours mean vs Neural Feels)"
    if t_grid is not None:
        title += f"  [dt={args.dt}s]"
    if args.smooth and args.smooth > 1:
        title += f"  [smooth={args.smooth}]"
    fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"[OK] Saved figure to: {args.out.resolve()}")
    print(f"[OK] Ours files used (N={len(used_files)}): {args.ours_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
