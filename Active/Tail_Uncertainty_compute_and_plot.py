#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@FileName: Uncertainty_tail_trend_plot.py
@Description:
  Compute "information-theoretic" tail trends for RF uncertainty / score over time,
  avoiding domination by large low-uncertainty regions.

  NOTE (NEW):
    - Ignore first N seconds (default N=3) for each object, because early frames may have extreme spikes.
    - We skip evaluation for t < ignore_first_s and keep them as NaN (so they do not affect mean/plots).
    - Failure-fill "first frame" is treated as t = ignore_first_s (not t=0).

Outputs:
  - CSV: <root_dir>/nbv_rf_tail_curve_root.csv
  - Plots:
      <prefix>_unc_tail.png
      <prefix>_score_tail.png
@author: Ferry
"""

import os
import re
import glob
import csv
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import matplotlib.pyplot as plt

from NBV_gpis_Field_for_metric import GPISNBVv3


_RECON_RE = re.compile(r"recon_0_(\d+)\.ply$", re.IGNORECASE)


def _parse_frame_id(path: str) -> Optional[int]:
    m = _RECON_RE.search(os.path.basename(path))
    if m is None:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def collect_recon_files(in_dir: str) -> Dict[int, str]:
    paths = sorted(glob.glob(os.path.join(in_dir, "recon_0_*.ply")))
    mp: Dict[int, str] = {}
    for p in paths:
        fid = _parse_frame_id(p)
        if fid is None:
            continue
        mp[fid] = p
    return mp


def pick_frame_path_in_range(frame2path: Dict[int, str], target_frame: int) -> Optional[Tuple[int, str]]:
    if not frame2path:
        return None
    frames = np.array(sorted(frame2path.keys()), dtype=np.int64)
    min_f = int(frames[0])
    max_f = int(frames[-1])

    if target_frame < min_f or target_frame > max_f:
        return None

    if target_frame in frame2path:
        return target_frame, frame2path[target_frame]

    j = int(np.argmin(np.abs(frames - int(target_frame))))
    pf = int(frames[j])
    return pf, frame2path[pf]


def list_object_dirs(root_dir: str):
    root = Path(root_dir)
    if not root.is_dir():
        raise ValueError(f"root_dir is not a directory: {root_dir}")
    obj_dirs = []
    for p in sorted(root.iterdir()):
        if p.is_dir():
            obj_dirs.append((p.name, str(p)))
    if len(obj_dirs) == 0:
        raise ValueError(f"No subdirectories found under: {root_dir}")
    return obj_dirs


# ---------- failure fill ----------
def fill_failed_by_rule(
    arr: np.ndarray,
    fail_mask: np.ndarray,
    first_fail_value: float = 300.0,
    fallback_value: float = 300.0,
    start_idx: int = 0,  # NEW: treat start_idx as "first time point"
) -> np.ndarray:
    """
    Fill failed points using:
      - i == start_idx => first_fail_value
      - else           => |arr[next_valid] - arr[prev_valid]|

    valid means: (not failed) and finite.
    prev/next search only within [start_idx, n).
    """
    out = arr.copy()
    n = int(out.shape[0])
    start_idx = int(max(0, min(start_idx, n)))  # clamp

    def prev_valid(i: int) -> Optional[int]:
        j = i - 1
        while j >= start_idx:
            if (not fail_mask[j]) and np.isfinite(out[j]):
                return j
            j -= 1
        return None

    def next_valid(i: int) -> Optional[int]:
        j = i + 1
        while j < n:
            if (not fail_mask[j]) and np.isfinite(out[j]):
                return j
            j += 1
        return None

    for i in np.where(fail_mask)[0]:
        i = int(i)
        if i < start_idx:
            # ignored region; keep as-is (usually NaN)
            continue
        if i == start_idx:
            out[i] = float(first_fail_value)
            continue

        L = prev_valid(i)
        R = next_valid(i)

        if L is not None and R is not None:
            out[i] = float(abs(out[R] - out[L]))
        elif L is not None and (L - 1) >= start_idx and np.isfinite(out[L - 1]) and (not fail_mask[L - 1]):
            out[i] = float(abs(out[L] - out[L - 1]))
        elif R is not None and (R + 1) < n and np.isfinite(out[R + 1]) and (not fail_mask[R + 1]):
            out[i] = float(abs(out[R + 1] - out[R]))
        else:
            out[i] = float(fallback_value)

    return out


# ---------- tail metrics ----------
def tail_metrics(x: np.ndarray, q95: float = 0.95, q99: float = 0.99, top_frac: float = 0.01):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(q95=np.nan, q99=np.nan, xmax=np.nan, top_mean=np.nan, n=0)

    q95v = float(np.quantile(x, q95))
    q99v = float(np.quantile(x, q99))
    xmax = float(np.max(x))

    k = int(np.ceil(x.size * float(top_frac)))
    k = max(1, min(k, x.size))
    topk = np.partition(x, -k)[-k:]
    top_mean = float(np.mean(topk))

    return dict(q95=q95v, q99=q99v, xmax=xmax, top_mean=top_mean, n=int(x.size))


def _parse_mark_times(s: str) -> List[float]:
    s = (s or "").strip()
    if not s:
        return []
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except Exception:
            pass
    return out


# ---------- per-object compute ----------
def run_one_object_tail_curve(
    obj_dir: str,
    fps: int,
    t_end_s: int,
    base_seed: int,
    est: GPISNBVv3,
    q95: float,
    q99: float,
    top_frac: float,
    first_fail_value: float,
    ignore_first_s: int = 3,  # NEW
    verbose: bool = True,
):
    frame2path = collect_recon_files(obj_dir)
    obj_name = os.path.basename(obj_dir.rstrip("/"))
    if verbose:
        print(f"[Obj] {obj_name}: found {len(frame2path)} recon files.")

    times = np.arange(0, int(t_end_s) + 1, dtype=np.int32)
    picked_frames = np.full_like(times, -1, dtype=np.int32)

    # uncertainty tail metrics
    unc_q95 = np.full_like(times, np.nan, dtype=np.float64)
    unc_q99 = np.full_like(times, np.nan, dtype=np.float64)
    unc_max = np.full_like(times, np.nan, dtype=np.float64)
    unc_top = np.full_like(times, np.nan, dtype=np.float64)

    # score tail metrics
    sc_q95 = np.full_like(times, np.nan, dtype=np.float64)
    sc_q99 = np.full_like(times, np.nan, dtype=np.float64)
    sc_max = np.full_like(times, np.nan, dtype=np.float64)
    sc_top = np.full_like(times, np.nan, dtype=np.float64)

    fail_mask = np.zeros_like(times, dtype=bool)
    ok_count, fail_count = 0, 0

    ignore_first_s = int(max(0, ignore_first_s))

    # -----------------------------
    # NEW: fit kernel ONCE per object (use LAST frame)
    # -----------------------------
    frames_sorted = sorted(frame2path.keys())
    if len(frames_sorted) == 0:
        raise ValueError(f"[Obj] {obj_name}: no recon files.")

    pf_ref = int(frames_sorted[-1])  # last frame id
    path_ref = frame2path[pf_ref]

    if verbose:
        print(f"[Obj] kernel fit uses LAST frame: {pf_ref} ({os.path.basename(path_ref)})")

    seed_ref = int(base_seed) + int(pf_ref)

    # IMPORTANT: the ONLY time we optimize kernel hypers for this object
    kernel_state = est.fit_kernel_once(path_ref, seed=seed_ref, verbose=verbose)

    for k, t in enumerate(times):
        # NEW: ignore early seconds entirely
        if int(t) < ignore_first_s:
            continue

        target_frame = int(t) * int(fps)
        picked = pick_frame_path_in_range(frame2path, target_frame)
        if picked is None:
            if verbose:
                print(f"[Obj] t={t:02d}s target_frame={target_frame}: out of range -> NaN")
            continue

        pf, path = picked
        picked_frames[k] = pf
        if verbose:
            print(f"[Obj] t={t:02d}s target={target_frame} -> picked={pf} ({os.path.basename(path)})")

        seed = int(base_seed) + int(pf)

        try:
            _ = est.estimate(path, seed=seed, verbose=False, kernel_state=kernel_state)

            unc_rf = getattr(est, "_unc_rf", None)
            scores = getattr(est, "scores", None)
            if unc_rf is None or scores is None:
                raise RuntimeError("Missing est._unc_rf / est.scores")

            unc_rf = np.asarray(unc_rf)
            scores = np.asarray(scores)

            um = tail_metrics(unc_rf, q95=q95, q99=q99, top_frac=top_frac)
            sm = tail_metrics(scores, q95=q95, q99=q99, top_frac=top_frac)

            unc_q95[k], unc_q99[k], unc_max[k], unc_top[k] = um["q95"], um["q99"], um["xmax"], um["top_mean"]
            sc_q95[k], sc_q99[k], sc_max[k], sc_top[k] = sm["q95"], sm["q99"], sm["xmax"], sm["top_mean"]

            ok_count += 1

        except Exception as e:
            fail_mask[k] = True
            fail_count += 1
            if verbose:
                print(f"[Fail] t={t:02d}s frame={pf} {os.path.basename(path)} ({type(e).__name__}: {e})")

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # fill failures (out-of-range NaN remains NaN)
    start_idx = ignore_first_s  # NEW: treat t=ignore_first_s as the "first" index for failure rule

    def _fill(a):
        return fill_failed_by_rule(
            a,
            fail_mask,
            first_fail_value=first_fail_value,
            fallback_value=first_fail_value,
            start_idx=start_idx,
        )

    unc_q95 = _fill(unc_q95)
    unc_q99 = _fill(unc_q99)
    unc_max = _fill(unc_max)
    unc_top = _fill(unc_top)

    sc_q95 = _fill(sc_q95)
    sc_q99 = _fill(sc_q99)
    sc_max = _fill(sc_max)
    sc_top = _fill(sc_top)

    if verbose:
        print(f"[Obj] {obj_name}: ok={ok_count}, fail(filled)={fail_count}, ignored t<={ignore_first_s-1}s, out-of-range kept NaN")

    return dict(
        times_s=times.astype(np.float32),
        picked_frames=picked_frames.astype(np.int32),
        unc_q95=unc_q95,
        unc_q99=unc_q99,
        unc_max=unc_max,
        unc_top=unc_top,
        score_q95=sc_q95,
        score_q99=sc_q99,
        score_max=sc_max,
        score_top=sc_top,
        ok_count=ok_count,
        fail_count=fail_count,
        ignore_first_s=ignore_first_s,
    )


# ---------- save / plot ----------
def save_root_csv(out_csv: str, times_s: np.ndarray, per_obj: Dict[str, dict], mean_curves: dict):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    obj_names = sorted(per_obj.keys())

    header = ["time_s"]
    for k in ["unc_q95", "unc_q99", "unc_max", "unc_top", "score_q95", "score_q99", "score_max", "score_top"]:
        header.append(f"mean_{k}")
    for name in obj_names:
        header += [
            f"{name}_picked_frame",
            f"{name}_unc_q95", f"{name}_unc_q99", f"{name}_unc_max", f"{name}_unc_top",
            f"{name}_score_q95", f"{name}_score_q99", f"{name}_score_max", f"{name}_score_top",
        ]

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        T = len(times_s)
        for i in range(T):
            row = [float(times_s[i])]
            for k in ["unc_q95", "unc_q99", "unc_max", "unc_top", "score_q95", "score_q99", "score_max", "score_top"]:
                v = mean_curves[k][i]
                row.append(float(v) if np.isfinite(v) else np.nan)

            for name in obj_names:
                d = per_obj[name]
                row.append(int(d["picked_frames"][i]))
                for k in ["unc_q95", "unc_q99", "unc_max", "unc_top", "score_q95", "score_q99", "score_max", "score_top"]:
                    v = d[k][i]
                    row.append(float(v) if np.isfinite(v) else np.nan)
            w.writerow(row)


def plot_mean_tail(out_prefix: str, times_s: np.ndarray, mean_curves: dict, mark_times: List[float], ignore_first_s: int):
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ignore_first_s = int(max(0, ignore_first_s))

    # Uncertainty tail (mean over objects)
    plt.figure()
    plt.plot(times_s, mean_curves["unc_q95"], label="mean unc q95", linewidth=2.0)
    plt.plot(times_s, mean_curves["unc_q99"], label="mean unc q99", linewidth=2.0)
    plt.plot(times_s, mean_curves["unc_max"], label="mean unc max", linewidth=2.0)
    plt.plot(times_s, mean_curves["unc_top"], label="mean unc top-mean", linewidth=2.0)
    for mt in mark_times:
        plt.axvline(mt, linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Tail Uncertainty (RF)")
    plt.title(f"RF Uncertainty Tail Trend (mean over objects, ignore first {ignore_first_s}s)")
    plt.grid(True)
    plt.legend()
    # NEW: focus on [ignore_first_s, end]
    if ignore_first_s > 0:
        plt.xlim(ignore_first_s, float(times_s[-1]))
    plt.tight_layout()
    plt.savefig(out_prefix + "_unc_tail.png", dpi=200)

    # Score tail (mean over objects)
    plt.figure()
    plt.plot(times_s, mean_curves["score_q95"], label="mean score q95", linewidth=2.0)
    plt.plot(times_s, mean_curves["score_q99"], label="mean score q99", linewidth=2.0)
    plt.plot(times_s, mean_curves["score_max"], label="mean score max", linewidth=2.0)
    plt.plot(times_s, mean_curves["score_top"], label="mean score top-mean", linewidth=2.0)
    for mt in mark_times:
        plt.axvline(mt, linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Tail Score (UncRF * NovRF)")
    plt.title(f"RF Score Tail Trend (mean over objects, ignore first {ignore_first_s}s)")
    plt.grid(True)
    plt.legend()
    if ignore_first_s > 0:
        plt.xlim(ignore_first_s, float(times_s[-1]))
    plt.tight_layout()
    plt.savefig(out_prefix + "_score_tail.png", dpi=200)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root_dir",
        type=str,
        default="/home/ferry/data/Code2/Research/Inhand_Activate/Active/pcd/offline_active",
        help="Root folder containing subfolders per object, each with recon_0_*.ply.",
    )
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--t_end", type=int, default=30, help="End time in seconds (inclusive).")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--out_csv", type=str, default="", help="Default: <root_dir>/nbv_rf_tail_curve_root.csv")
    ap.add_argument("--out_png_prefix", type=str, default="", help="Default: <root_dir>/nbv_rf_tail_curve_root")

    # tail knobs
    ap.add_argument("--q95", type=float, default=0.95)
    ap.add_argument("--q99", type=float, default=0.99)
    ap.add_argument("--top_frac", type=float, default=0.01, help="Top fraction for top-mean (default top 1%).")

    # failure rule
    ap.add_argument("--first_fail_value", type=float, default=0.00025)

    # NEW: ignore first seconds
    ap.add_argument("--ignore_first_s", type=int, default=6, help="Ignore first N seconds for each object (default 3).")

    # RF params
    ap.add_argument("--rf_k", type=int, default=32)
    ap.add_argument("--rf_min_cos", type=float, default=0.96)
    ap.add_argument("--unc_rf_reduce", type=str, default="max", choices=["max", "mean"])
    ap.add_argument("--nov_rf_reduce", type=str, default="mean", choices=["mean", "max"])

    # speed/quality
    ap.add_argument("--voxel", type=float, default=0.01)
    ap.add_argument("--max_train", type=int, default=2000)
    ap.add_argument("--train_iters", type=int, default=60)

    # optional: mark replanning times
    ap.add_argument("--mark_times", type=str, default="10,20,30", help="Vertical markers in seconds, comma-separated.")

    args = ap.parse_args()

    root_dir = args.root_dir
    out_csv = args.out_csv if args.out_csv else os.path.join(root_dir, "nbv_rf_tail_curve_root.csv")
    out_prefix = args.out_png_prefix if args.out_png_prefix else os.path.join(root_dir, "nbv_rf_tail_curve_root")
    mark_times = _parse_mark_times(args.mark_times)

    est = GPISNBVv3(
        voxel=args.voxel,
        max_train=args.max_train,
        train_iters=args.train_iters,
        rf_k=args.rf_k,
        rf_min_cos=args.rf_min_cos,
        unc_rf_reduce=args.unc_rf_reduce,
        nov_rf_reduce=args.nov_rf_reduce,
    )

    obj_dirs = list_object_dirs(root_dir)

    per_obj: Dict[str, dict] = {}
    total_ok, total_fail = 0, 0

    for obj_name, obj_dir in obj_dirs:
        print(f"\n========== Processing object: {obj_name} ==========")
        stats = run_one_object_tail_curve(
            obj_dir=obj_dir,
            fps=args.fps,
            t_end_s=args.t_end,
            base_seed=args.seed,
            est=est,
            q95=args.q95,
            q99=args.q99,
            top_frac=args.top_frac,
            first_fail_value=args.first_fail_value,
            ignore_first_s=args.ignore_first_s,
            verbose=True,
        )
        per_obj[obj_name] = stats
        total_ok += int(stats.get("ok_count", 0))
        total_fail += int(stats.get("fail_count", 0))

    times_s = next(iter(per_obj.values()))["times_s"]

    # stack + mean (NaN in first ignore_first_s seconds will be ignored)
    obj_names_sorted = sorted(per_obj.keys())
    mean_curves = {}
    for k in ["unc_q95", "unc_q99", "unc_max", "unc_top", "score_q95", "score_q99", "score_max", "score_top"]:
        stack = np.stack([per_obj[n][k] for n in obj_names_sorted], axis=0)
        mean_curves[k] = np.nanmean(stack, axis=0)

    save_root_csv(out_csv, times_s, per_obj, mean_curves)
    plot_mean_tail(out_prefix, times_s, mean_curves, mark_times, ignore_first_s=args.ignore_first_s)

    print(f"\n[Summary] ok={total_ok}, fail(filled)={total_fail}")
    print(f"[Done] CSV saved to: {out_csv}")
    print(f"[Done] Plots saved to: {out_prefix}_unc_tail.png and {out_prefix}_score_tail.png")
