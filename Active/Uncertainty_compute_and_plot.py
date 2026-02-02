"""
@FileName：Uncertainty_compute_and_plot.py
@Description：
    Root folder structure:
      root_dir/
        cube/
          recon_0_00xxxx.ply
        tetraprism/
          recon_0_00xxxx.ply
        ...

    For each object:
      - sample time t = 0..t_end (inclusive)
      - fps = 20 frames per second (default)
      - target_frame = t * fps
      - pick nearest available recon_0_*.ply within [min_frame, max_frame]
      - run GPISNBVv3.estimate(ply)
      - record:
          total_unc_rf(t)   = sum_i UncRF(u_i)
          total_score(t)    = sum_i UncRF(u_i) * NovRF(u_i)  (i.e., sum(scores))

    Failure value rule (your request):
      - if estimate fails at the FIRST time point (index 0): set value = 300
      - else: set value = | value(next_valid) - value(prev_valid) |
        (prev/next are the closest valid time points around this failure)

    Missing (out-of-range) time points remain NaN (not treated as failure).

    Outputs:
      - CSV: <root_dir>/nbv_rf_curve_root.csv (unless overridden)
      - Plots:
          <prefix>_unc.png
          <prefix>_score.png
@Author：Ferry
@Time：2026-01-24
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""

import os
import re
import glob
import csv
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from NBV_gpis_Field import GPISNBVv3


# ------------------------------ File parsing ------------------------------
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
    """Return dict: frame_id -> filepath for recon_0_*.ply in directory."""
    paths = sorted(glob.glob(os.path.join(in_dir, "recon_0_*.ply")))
    mp: Dict[int, str] = {}
    for p in paths:
        fid = _parse_frame_id(p)
        if fid is None:
            continue
        mp[fid] = p
    return mp


def pick_frame_path_in_range(frame2path: Dict[int, str], target_frame: int) -> Optional[Tuple[int, str]]:
    """
    Prefer exact match; otherwise choose nearest frame within [min_frame, max_frame].
    If target_frame is outside available range, return None (avoid extrapolating beyond data).
    """
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
    """One-level subfolders under root_dir are treated as objects."""
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


def safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return float(default)
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return v
    except Exception:
        return float(default)


# ------------------------------ Failure fill rule ------------------------------
def fill_failed_by_rule(
    arr: np.ndarray,
    fail_mask: np.ndarray,
    first_fail_value: float = 300.0,
    fallback_value: float = 300.0,
) -> np.ndarray:
    """
    Fill failed points using:
      - idx==0 => first_fail_value
      - else   => |arr[next_valid] - arr[prev_valid]|

    prev_valid/next_valid are the closest valid indices around idx
    (valid = finite and not failed). If cannot find both, fallback gracefully.
    """
    out = arr.copy()
    n = int(out.shape[0])

    def prev_valid(i: int) -> Optional[int]:
        j = i - 1
        while j >= 0:
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

    fail_idxs = np.where(fail_mask)[0]
    for i in fail_idxs:
        if i == 0:
            out[i] = float(first_fail_value)
            continue

        L = prev_valid(int(i))
        R = next_valid(int(i))

        if L is not None and R is not None:
            out[i] = float(abs(out[R] - out[L]))
        elif L is not None and (L - 1) >= 0 and np.isfinite(out[L - 1]) and (not fail_mask[L - 1]):
            out[i] = float(abs(out[L] - out[L - 1]))
        elif R is not None and (R + 1) < n and np.isfinite(out[R + 1]) and (not fail_mask[R + 1]):
            out[i] = float(abs(out[R + 1] - out[R]))
        else:
            out[i] = float(fallback_value)

    return out


# ------------------------------ Per-object computation ------------------------------
def run_one_object_curve(
    obj_dir: str,
    fps: int,
    t_end_s: int,
    base_seed: int,
    est: GPISNBVv3,
    first_fail_value: float = 300.0,
    verbose: bool = True,
):
    """
    For t = 0..t_end_s:
      target_frame = t * fps
      pick nearest available recon file within range
      run estimate()

    Missing (out-of-range) => NaN.
    estimate-failure => marked in fail_mask, then filled by your rule.
    """
    frame2path = collect_recon_files(obj_dir)
    obj_name = os.path.basename(obj_dir.rstrip("/"))
    if verbose:
        print(f"[Obj] {obj_name}: found {len(frame2path)} recon files.")

    times = np.arange(0, int(t_end_s) + 1, dtype=np.int32)
    picked_frames = np.full_like(times, -1, dtype=np.int32)

    total_unc = np.full_like(times, np.nan, dtype=np.float64)
    total_score = np.full_like(times, np.nan, dtype=np.float64)
    hit_ratio = np.full_like(times, np.nan, dtype=np.float64)

    fail_mask_unc = np.zeros_like(times, dtype=bool)
    fail_mask_score = np.zeros_like(times, dtype=bool)

    ok_count = 0
    fail_count = 0

    for k, t in enumerate(times):
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
            nbv = est.estimate(path, seed=seed, verbose=False)

            unc_rf = getattr(est, "_unc_rf", None)
            scores = getattr(est, "scores", None)

            if unc_rf is None or scores is None:
                raise RuntimeError("Missing _unc_rf / scores from estimator.")

            unc_rf = np.asarray(unc_rf)
            scores = np.asarray(scores)

            if (not np.all(np.isfinite(unc_rf))) or (not np.all(np.isfinite(scores))):
                raise ValueError("Non-finite unc_rf/scores.")

            total_unc[k] = float(np.sum(unc_rf))
            total_score[k] = float(np.sum(scores))
            hit_ratio[k] = safe_float(nbv.get("hit_ratio", np.nan), np.nan)

            ok_count += 1

        except Exception as e:
            fail_mask_unc[k] = True
            fail_mask_score[k] = True
            hit_ratio[k] = 0.0
            fail_count += 1
            if verbose:
                print(
                    f"[Fail] t={t:02d}s frame={pf} {os.path.basename(path)} "
                    f"({type(e).__name__}: {e})"
                )

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total_unc = fill_failed_by_rule(
        total_unc,
        fail_mask_unc,
        first_fail_value=first_fail_value,
        fallback_value=first_fail_value,
    )
    total_score = fill_failed_by_rule(
        total_score,
        fail_mask_score,
        first_fail_value=first_fail_value,
        fallback_value=first_fail_value,
    )

    if verbose:
        print(f"[Obj] {obj_name}: ok={ok_count}, fail(filled by rule)={fail_count} (out-of-range kept as NaN)")

    return {
        "times_s": times.astype(np.float32),
        "picked_frames": picked_frames.astype(np.int32),
        "total_unc_rf": total_unc.astype(np.float64),
        "total_score": total_score.astype(np.float64),
        "hit_ratio": hit_ratio.astype(np.float64),
        "ok_count": ok_count,
        "fail_count": fail_count,
    }


# ------------------------------ Output ------------------------------
def save_root_csv(out_csv: str, times_s, per_obj, mean_unc, mean_score):
    """
    Wide CSV:
      time_s, mean_total_unc_rf, mean_total_score,
      then per-object columns: picked_frame, total_unc_rf, total_score
    """
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    obj_names = sorted(per_obj.keys())

    header = ["time_s", "mean_total_unc_rf", "mean_total_score"]
    for name in obj_names:
        header += [f"{name}_picked_frame", f"{name}_total_unc_rf", f"{name}_total_score"]

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(times_s)):
            row = [float(times_s[i]), float(mean_unc[i]), float(mean_score[i])]
            for name in obj_names:
                d = per_obj[name]
                row += [
                    int(d["picked_frames"][i]),
                    float(d["total_unc_rf"][i]),
                    float(d["total_score"][i]),
                ]
            w.writerow(row)


def plot_root_curves(out_png_prefix: str, times_s, per_obj, mean_unc, mean_score):
    """Two figures: uncertainty and score, each with per-object lines + mean."""
    out_dir = os.path.dirname(out_png_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    obj_names = sorted(per_obj.keys())

    # --- Uncertainty ---
    plt.figure()
    for name in obj_names:
        plt.plot(times_s, per_obj[name]["total_unc_rf"], label=name)
    plt.plot(times_s, mean_unc, label="mean", linewidth=2.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Total UncRF  (sum over directions)")
    plt.title("Per-object Total RF Uncertainty + Mean")
    plt.grid(True)
    plt.legend()
    plt.ylim(0.20, 0.25)  # <<< lock y-axis for unc
    plt.xlim(0,30)
    plt.tight_layout()
    plt.savefig(out_png_prefix + "_unc.png", dpi=200)

    # --- Score ---
    plt.figure()
    for name in obj_names:
        plt.plot(times_s, per_obj[name]["total_score"], label=name)
    plt.plot(times_s, mean_score, label="mean", linewidth=2.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Total UncRF * NovRF  (sum over directions)")
    plt.title("Per-object Total RF Score + Mean")
    plt.grid(True)
    plt.legend()
    plt.ylim(0.02, 0.05)  # <<< lock y-axis for score
    plt.xlim(0,30)
    plt.tight_layout()
    plt.savefig(out_png_prefix + "_score.png", dpi=200)


# ------------------------------ CLI entry ------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root_dir",
        type=str,
        default="/home/ferry/data/Code2/Research/Inhand_Activate/Active/pcd/offline_active",
        help="Root folder containing subfolders per object, each with recon_0_*.ply.",
    )

    ap.add_argument("--fps", type=int, default=20, help="Frames per second (default: 20).")
    ap.add_argument("--t_end", type=int, default=50, help="End time in seconds (inclusive), default: 50.")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--out_csv", type=str, default="", help="Output CSV (default: <root_dir>/nbv_rf_curve_root.csv)")
    ap.add_argument(
        "--out_png_prefix",
        type=str,
        default="",
        help="Output PNG prefix (default: <root_dir>/nbv_rf_curve_root)",
    )

    ap.add_argument("--first_fail_value", type=float, default=0.1, help="Failure value used at index 0.")

    # RF params
    ap.add_argument("--rf_k", type=int, default=32)
    ap.add_argument("--rf_min_cos", type=float, default=0.96)
    ap.add_argument("--unc_rf_reduce", type=str, default="max", choices=["max", "mean"])
    ap.add_argument("--nov_rf_reduce", type=str, default="mean", choices=["mean", "max"])

    # speed/quality knobs
    ap.add_argument("--voxel", type=float, default=0.01)
    ap.add_argument("--max_train", type=int, default=2000)
    ap.add_argument("--train_iters", type=int, default=60)

    args = ap.parse_args()

    root_dir = args.root_dir
    out_csv = args.out_csv if args.out_csv else os.path.join(root_dir, "nbv_rf_curve_root.csv")
    out_png_prefix = args.out_png_prefix if args.out_png_prefix else os.path.join(root_dir, "nbv_rf_curve_root")

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

    per_obj = {}
    total_ok = 0
    total_fail = 0

    for obj_name, obj_dir in obj_dirs:
        print(f"\n========== Processing object: {obj_name} ==========")
        stats = run_one_object_curve(
            obj_dir=obj_dir,
            fps=args.fps,
            t_end_s=args.t_end,
            base_seed=args.seed,
            est=est,
            first_fail_value=args.first_fail_value,
            verbose=True,
        )
        per_obj[obj_name] = stats
        total_ok += int(stats.get("ok_count", 0))
        total_fail += int(stats.get("fail_count", 0))

    times_s = next(iter(per_obj.values()))["times_s"]

    obj_names_sorted = sorted(per_obj.keys())
    unc_stack = np.stack([per_obj[k]["total_unc_rf"] for k in obj_names_sorted], axis=0)
    score_stack = np.stack([per_obj[k]["total_score"] for k in obj_names_sorted], axis=0)

    mean_unc = np.nanmean(unc_stack, axis=0)
    mean_score = np.nanmean(score_stack, axis=0)

    save_root_csv(out_csv, times_s, per_obj, mean_unc, mean_score)
    plot_root_curves(out_png_prefix, times_s, per_obj, mean_unc, mean_score)

    print(f"\n[Summary] ok={total_ok}, fail(filled by rule)={total_fail}")
    print(f"[Done] CSV saved to: {out_csv}")
    print(f"[Done] Plots saved to: {out_png_prefix}_unc.png and {out_png_prefix}_score.png")
