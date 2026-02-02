"""
@FileName：max_value_get.py
@Description：
@Author：Ferry
@Time：2026 1/23/26 1:44 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract the row(s) where nksr_vs_gt_fscore@0.005000 (5mm) reaches its maximum,
and report:
  - nksr_vs_gt_fscore@{0.002000, 0.005000, 0.010000}
  - online_pcd_vs_gt_fscore@{0.002000, 0.005000, 0.010000}
  - frame_id, time_s

Usage:
  python extract_best_nksr_fscore.py --csv /path/to/summary.csv
  python extract_best_nksr_fscore.py --csv summary.csv --mode all --out best_rows.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_5MM = "nksr_vs_gt_fscore@0.005000"

REQ_COLS = [
    "frame_id",
    "time_s",
    "nksr_vs_gt_fscore@0.002000",
    "nksr_vs_gt_fscore@0.005000",
    "nksr_vs_gt_fscore@0.010000",
    "online_pcd_vs_gt_fscore@0.002000",
    "online_pcd_vs_gt_fscore@0.005000",
    "online_pcd_vs_gt_fscore@0.010000",
]


def die(msg: str, code: int = 2) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


def to_num(df: pd.DataFrame, cols) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/home/ferry/data/Code2/Research/Inhand_Activate/plot/summary/metric/offline_tracking/Cube.csv", help="Path to summary.csv")
    ap.add_argument(
        "--mode",
        choices=["first", "all"],
        default="first",
        help="If multiple rows tie for max 5mm NKSR F-score, output the first (earliest time_s) or all.",
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=0.0,
        help="Tolerance for tie matching (useful if values are numerically very close). 0 means exact equality.",
    )
    ap.add_argument(
        "--out",
        default="",
        help="Optional output CSV path to save the extracted row(s).",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        die(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        die(f"Missing required columns: {missing}")

    # Ensure numeric
    to_num(df, [c for c in REQ_COLS if c not in ("frame_id",)])

    # Drop rows where target is NaN
    valid = df[np.isfinite(df[TARGET_5MM].to_numpy())].copy()
    if valid.empty:
        die(f"No valid numeric values in column: {TARGET_5MM}")

    max_val = valid[TARGET_5MM].max()

    if args.tol > 0:
        m = np.isclose(valid[TARGET_5MM].to_numpy(), max_val, rtol=0.0, atol=args.tol)
        best = valid[m]
    else:
        best = valid[valid[TARGET_5MM] == max_val]

    if best.empty:
        die("Internal error: best selection became empty.")

    # If "first": choose the earliest time_s among ties (fallback to first row order)
    if args.mode == "first":
        if best["time_s"].notna().any():
            best = best.sort_values("time_s", ascending=True).head(1)
        else:
            best = best.head(1)

    out_cols = [
        "frame_id",
        "time_s",
        "nksr_vs_gt_fscore@0.002000",
        "nksr_vs_gt_fscore@0.005000",
        "nksr_vs_gt_fscore@0.010000",
        "online_pcd_vs_gt_fscore@0.002000",
        "online_pcd_vs_gt_fscore@0.005000",
        "online_pcd_vs_gt_fscore@0.010000",
    ]
    result = best[out_cols].copy()

    # Pretty print
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 200)

    print(f"[INFO] max({TARGET_5MM}) = {max_val}")
    print(result.to_string(index=False))

    # Save if requested
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        print(f"[INFO] Saved to: {out_path}")


if __name__ == "__main__":
    main()
