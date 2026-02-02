"""
@FileName：30s_value_get.py
@Description：
@Author：Ferry
@Time：2026 1/27/26 2:34 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@FileName: extract_fscore_at_time_all_objects.py
@Description:
  Read all per-object CSVs under a folder, and extract NKSR-vs-GT F-score
  at a given time (default 30s), for tau=2/5/10mm.

  Required columns in each CSV:
    - frame_id
    - time_s
    - nksr_vs_gt_fscore@0.002000
    - nksr_vs_gt_fscore@0.005000
    - nksr_vs_gt_fscore@0.010000

  By default, the script picks the NEAREST time_s to target_time among rows
  where the requested columns are finite.

Usage:
  python extract_fscore_at_time_all_objects.py \
    --root_dir /path/to/folder_with_csvs \
    --time_s 30 \
    --out /path/to/out.csv

  # If your CSVs are nested in subfolders, keep default --recursive.
  # To require "almost exact" time matching:
  python extract_fscore_at_time_all_objects.py \
    --root_dir /path/to/folder \
    --time_s 30 \
    --time_tol 0.05 \
    --pick exact
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REQ_COLS = [
    "frame_id",
    "time_s",
    "nksr_vs_gt_fscore@0.002000",
    "nksr_vs_gt_fscore@0.005000",
    "nksr_vs_gt_fscore@0.010000",
]


def die(msg: str, code: int = 2) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


def to_num(df: pd.DataFrame, cols) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def pick_row_at_time(
    df: pd.DataFrame,
    target_time: float,
    pick_mode: str = "nearest",  # "nearest" or "exact"
    time_tol: float = 0.0,
) -> pd.Series:
    """
    Pick a row at target time.
    - nearest: choose row minimizing |time_s - target_time| among valid rows
    - exact: choose rows with |time_s - target_time| <= time_tol, then pick earliest (or first)
    Valid rows mean requested fscore columns are finite.
    """
    # Ensure numeric
    to_num(df, ["time_s"] + [c for c in REQ_COLS if c not in ("frame_id", "time_s")])

    if "time_s" not in df.columns:
        raise ValueError("missing column: time_s")

    # Valid rows: time finite + fscores finite
    m = np.isfinite(df["time_s"].to_numpy())
    for c in REQ_COLS:
        if c in ("frame_id", "time_s"):
            continue
        m = m & np.isfinite(df[c].to_numpy())

    valid = df[m].copy()
    if valid.empty:
        raise ValueError("no valid rows (time_s/fscores contain no finite values)")

    t = valid["time_s"].to_numpy(dtype=np.float64)

    if pick_mode == "exact":
        tol = float(time_tol)
        mm = np.abs(t - float(target_time)) <= tol
        exact = valid[mm].copy()
        if exact.empty:
            raise ValueError(f"no row within time_tol={tol} of target_time={target_time}")
        # if multiple, pick the one with smallest |dt|, tie-break by earliest time
        exact["_dt"] = np.abs(exact["time_s"] - float(target_time))
        exact = exact.sort_values(["_dt", "time_s"], ascending=[True, True])
        return exact.iloc[0].drop(labels=["_dt"], errors="ignore")

    # nearest
    dt = np.abs(t - float(target_time))
    j = int(np.argmin(dt))
    return valid.iloc[j]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root_dir",
        type=str,
        default="/home/ferry/data/Code2/Research/Inhand_Activate/plot/Q1/csv/summary",
        help="Folder that contains per-object CSVs (can be nested).",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Filename pattern to search for CSVs (default: *.csv).",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search CSVs in subfolders (recommended).",
    )
    ap.add_argument(
        "--time_s",
        type=float,
        default=30.0,
        help="Target time in seconds (default 30.0).",
    )
    ap.add_argument(
        "--pick",
        choices=["nearest", "exact"],
        default="nearest",
        help="How to pick the row at time_s (nearest or exact).",
    )
    ap.add_argument(
        "--time_tol",
        type=float,
        default=0.0,
        help="Tolerance for exact time matching (used when --pick exact).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output CSV path to save the summary.",
    )
    ap.add_argument(
        "--also_online",
        action="store_true",default=True,
        help="Also extract online_pcd_vs_gt_fscore@2/5/10 if present.",
    )
    args = ap.parse_args()

    root = Path(args.root_dir) if args.root_dir else None
    if root is None or not root.exists():
        die(f"--root_dir not found: {args.root_dir}")

    # gather csv paths
    if args.recursive:
        paths = sorted(root.rglob(args.pattern))
    else:
        paths = sorted(root.glob(args.pattern))

    paths = [p for p in paths if p.is_file()]
    if len(paths) == 0:
        die(f"No CSV files found under {root} with pattern {args.pattern} (recursive={args.recursive}).")

    # optional columns
    online_cols = [
        "online_pcd_vs_gt_fscore@0.002000",
        "online_pcd_vs_gt_fscore@0.005000",
        "online_pcd_vs_gt_fscore@0.010000",
    ]

    rows = []
    n_ok, n_skip = 0, 0

    for p in paths:
        obj_name = p.stem  # default: filename without suffix
        # if your structure is .../<obj>/<something>.csv, you may prefer parent folder name:
        # obj_name = p.parent.name

        try:
            df = pd.read_csv(p)

            missing = [c for c in REQ_COLS if c not in df.columns]
            if missing:
                raise ValueError(f"missing required columns: {missing}")

            # pick row at time
            row = pick_row_at_time(df, target_time=args.time_s, pick_mode=args.pick, time_tol=args.time_tol)

            out = {
                "object": obj_name,
                "csv_path": str(p),
                "frame_id": row.get("frame_id", np.nan),
                "time_s": float(row["time_s"]),
                "nksr_vs_gt_fscore@0.002000": float(row["nksr_vs_gt_fscore@0.002000"]),
                "nksr_vs_gt_fscore@0.005000": float(row["nksr_vs_gt_fscore@0.005000"]),
                "nksr_vs_gt_fscore@0.010000": float(row["nksr_vs_gt_fscore@0.010000"]),
            }

            if args.also_online:
                for c in online_cols:
                    out[c] = float(row[c]) if (c in row.index and np.isfinite(row[c])) else np.nan

            rows.append(out)
            n_ok += 1

        except Exception as e:
            n_skip += 1
            print(f"[SKIP] {p} ({type(e).__name__}: {e})", file=sys.stderr)
            continue

    if n_ok == 0:
        die("All CSVs failed/ignored. Check required columns / time selection.")

    out_df = pd.DataFrame(rows)

    # sort by object name for readability
    out_df = out_df.sort_values("object", ascending=True)

    # print
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 220)
    print(f"[INFO] Found CSVs: {len(paths)} | ok: {n_ok} | skipped: {n_skip}")
    print(f"[INFO] Target time: {args.time_s}s | pick={args.pick} | time_tol={args.time_tol}")
    print(out_df.to_string(index=False))

    # save
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"[INFO] Saved summary to: {out_path}")


if __name__ == "__main__":
    main()
