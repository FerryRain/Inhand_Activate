"""
@FileName：plot_right_tau.py
@Description：
@Author：Ferry
@Time：2026 1/23/26 4:40 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_tau_curve_at_time.py

在指定秒数 t_eval 下：
- 从 ours_csv 中按 time_s 对每个 tau 的 F-score 进行线性插值，得到 F(t_eval, tau) 曲线；
- 从 suresh_csv 读取 (tau_mm, fscore) 曲线；
- 画对比图：F-score vs tau_mm。

用法示例：
  python compare_tau_curve_at_time.py \
    --ours_csv /path/to/ours_summary.csv \
    --suresh_csv /path/to/suresh_tau.csv \
    --times 10,20,30 \
    --out_png out_tau_compare.png \
    --out_pdf out_tau_compare.pdf

备注：
- ours_csv 必须包含 time_s 列，以及类似 nksr_vs_gt_fscore@5mm 的列（任意顺序都可）。
- suresh_csv 必须包含 tau_mm 和 fscore 列。
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TAU_COL_RE = re.compile(r"nksr_vs_gt_fscore@([0-9]+(?:\.[0-9]+)?)mm$")


def parse_times(s: str) -> List[float]:
    # 支持 "30" 或 "10,20,30"
    s = s.strip()
    if not s:
        return []
    return [float(x) for x in s.split(",") if x.strip()]


def finite_mask(a: np.ndarray) -> np.ndarray:
    return np.isfinite(a)


def interp_1d(t: np.ndarray, y: np.ndarray, t0: float) -> float:
    """
    对 y(t) 在 t0 处线性插值。
    若 t0 超出范围，则取端点值。
    自动跳过 NaN/Inf（若有效点不足则返回 NaN）。
    """
    m = finite_mask(t) & finite_mask(y)
    if np.count_nonzero(m) < 2:
        # 有效点不足，退化为最近有效点（如果有）
        if np.count_nonzero(m) == 1:
            return float(y[m][0])
        return float("nan")

    tt = t[m]
    yy = y[m]
    order = np.argsort(tt)
    tt = tt[order]
    yy = yy[order]

    if t0 <= tt[0]:
        return float(yy[0])
    if t0 >= tt[-1]:
        return float(yy[-1])
    return float(np.interp(t0, tt, yy))


def load_ours(ours_csv: str) -> Tuple[np.ndarray, Dict[float, np.ndarray]]:
    df = pd.read_csv(ours_csv)
    if "time_s" not in df.columns:
        raise ValueError(f"[ours_csv] missing required column: time_s")

    # 解析 tau -> column
    tau_to_col: Dict[float, str] = {}
    for c in df.columns:
        m = TAU_COL_RE.match(c)
        if m:
            tau = float(m.group(1))
            tau_to_col[tau] = c

    if not tau_to_col:
        raise ValueError(
            "[ours_csv] no tau columns found. Expect columns like: nksr_vs_gt_fscore@5mm"
        )

    t = df["time_s"].to_numpy(dtype=float)

    tau_to_y: Dict[float, np.ndarray] = {}
    for tau, col in tau_to_col.items():
        tau_to_y[tau] = df[col].to_numpy(dtype=float)

    return t, tau_to_y


def load_suresh(suresh_csv: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(suresh_csv)
    need = ["tau_mm", "fscore"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"[suresh_csv] missing required column: {c}")

    tau = df["tau_mm"].to_numpy(dtype=float)
    f = df["fscore"].to_numpy(dtype=float)

    m = finite_mask(tau) & finite_mask(f)
    tau = tau[m]
    f = f[m]

    order = np.argsort(tau)
    return tau[order], f[order]


def build_ours_curve_at_time(
    t: np.ndarray, tau_to_y: Dict[float, np.ndarray], t_eval: float
) -> Tuple[np.ndarray, np.ndarray]:
    taus = np.array(sorted(tau_to_y.keys()), dtype=float)
    fs = np.zeros_like(taus, dtype=float)
    for i, tau in enumerate(taus):
        fs[i] = interp_1d(t, tau_to_y[tau], t_eval)
    return taus, fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours_csv", type=str, default="/home/ferry/data/Code2/Research/Inhand_Activate/plot/Q3/summary_nksr_vs_gt.csv", help="我们的 summary CSV（含 time_s 与 fscore@kmm 列）")
    ap.add_argument("--suresh_csv", type=str, default="suresh_yellow_realworld_solid.csv", help="Suresh/NeuralFeels 的 tau_mm 曲线 CSV")
    ap.add_argument("--times", type=str, default="26", help="评测秒数，形如 '30' 或 '10,20,30'")
    ap.add_argument("--tau_min", type=float, default=0, help="只画 tau >= tau_min（可选）")
    ap.add_argument("--tau_max", type=float, default=10, help="只画 tau <= tau_max（可选）")
    ap.add_argument("--title", type=str, default="F-score vs tau at specified time(s)")
    ap.add_argument("--label_ours", type=str, default="Ours")
    ap.add_argument("--label_suresh", type=str, default="Neural Feels")
    ap.add_argument("--out_png", type=str, default="tau_compare.png")
    ap.add_argument("--out_pdf", type=str, default="")
    args = ap.parse_args()

    times = parse_times(args.times)
    if not times:
        raise ValueError("No valid --times provided.")

    t, tau_to_y = load_ours(args.ours_csv)
    tau_s, f_s = load_suresh(args.suresh_csv)

    # 过滤 tau 范围（两边同时过滤，保持对齐视觉）
    def apply_tau_range(x, y):
        m = np.ones_like(x, dtype=bool)
        if args.tau_min is not None:
            m &= x >= args.tau_min
        if args.tau_max is not None:
            m &= x <= args.tau_max
        return x[m], y[m]

    tau_s, f_s = apply_tau_range(tau_s, f_s)

    # ---------------- Plot ----------------
    plt.figure( figsize=(5, 5))

    # 固定颜色
    color_suresh = "saddlebrown"  # 棕色
    color_ours = "green"  # 绿色

    # 不同时间用不同线型（颜色保持一致）
    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]

    # 先画 suresh（棕色，无marker）
    plt.plot(tau_s, f_s, linestyle="-", linewidth=2.2, color=color_suresh,
             label=args.label_suresh)

    # 再画 ours（绿色，无marker）
    for i, t_eval in enumerate(times):
        tau_o, f_o = build_ours_curve_at_time(t, tau_to_y, t_eval)
        tau_o, f_o = apply_tau_range(tau_o, f_o)

        ls = line_styles[i % len(line_styles)]
        plt.plot(tau_o, f_o, linestyle=ls, linewidth=2.2, color=color_ours,
                 label=f"{args.label_ours} @ {t_eval:g}s")

    plt.xlabel("tau_mm")
    plt.ylabel("F-score")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_png), dpi=200)

    if args.out_pdf:
        out_pdf = Path(args.out_pdf)
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_pdf))

    print(f"[OK] Saved: {out_png}")
    if args.out_pdf:
        print(f"[OK] Saved: {args.out_pdf}")


if __name__ == "__main__":
    main()
