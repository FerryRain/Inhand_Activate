"""
@FileName：depth_compare.py
@Description：
@Author：Ferry
@Time：2025 12/17/25 3:23 PM
@Copyright：©2024-2025 ShanghaiTech University-RIMLAB
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np

try:
    import cv2
except ImportError:
    raise ImportError("请先安装 opencv-python: pip install opencv-python")

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".exr"}
NPY_EXTS = {".npy"}


def iter_files(path, recursive=False):
    if os.path.isfile(path):
        return [path]
    files = []
    for root, _, fnames in os.walk(path):
        for f in fnames:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMG_EXTS or ext in NPY_EXTS:
                files.append(os.path.join(root, f))
        if not recursive:
            break
    files.sort()
    return files


def load_depth(fp):
    ext = os.path.splitext(fp)[1].lower()
    if ext in NPY_EXTS:
        d = np.load(fp)
        return d
    d = cv2.imread(fp, cv2.IMREAD_UNCHANGED)  # 保留原 dtype (uint16/float32 等)
    if d is None:
        raise RuntimeError(f"读不到文件: {fp}")
    return d


def robust_stats(arr):
    a = arr
    if a.ndim == 3:
        # 如果是多通道，先取第0通道做分析（大多数 depth 应该是单通道）
        a = a[..., 0]
    a = np.asarray(a)

    flat = a.reshape(-1)

    info = {}
    info["dtype"] = str(a.dtype)
    info["shape"] = list(a.shape)

    if flat.size == 0:
        info.update(dict(min=None, max=None, p1=None, p50=None, p99=None,
                         zero_ratio=None, nan_ratio=None, inf_ratio=None,
                         nonzero_min=None, nonzero_p50=None, nonzero_p99=None, nonzero_max=None))
        return info

    is_float = np.issubdtype(a.dtype, np.floating)
    nan_ratio = float(np.isnan(flat).mean()) if is_float else 0.0
    inf_ratio = float(np.isinf(flat).mean()) if is_float else 0.0

    valid = flat.copy()
    if is_float:
        valid = valid[np.isfinite(valid)]

    if valid.size == 0:
        info.update(dict(min=None, max=None, p1=None, p50=None, p99=None,
                         zero_ratio=None, nan_ratio=nan_ratio, inf_ratio=inf_ratio,
                         nonzero_min=None, nonzero_p50=None, nonzero_p99=None, nonzero_max=None))
        return info

    info["min"] = float(valid.min())
    info["max"] = float(valid.max())
    info["p1"] = float(np.percentile(valid, 1))
    info["p50"] = float(np.percentile(valid, 50))
    info["p99"] = float(np.percentile(valid, 99))

    zero_ratio = float((valid == 0).mean()) if valid.size else None
    info["zero_ratio"] = zero_ratio
    info["nan_ratio"] = nan_ratio
    info["inf_ratio"] = inf_ratio

    nz = valid[valid > 0]
    if nz.size:
        info["nonzero_min"] = float(nz.min())
        info["nonzero_p50"] = float(np.percentile(nz, 50))
        info["nonzero_p99"] = float(np.percentile(nz, 99))
        info["nonzero_max"] = float(nz.max())
    else:
        info["nonzero_min"] = None
        info["nonzero_p50"] = None
        info["nonzero_p99"] = None
        info["nonzero_max"] = None

    return info


def guess_unit(info):
    """
    非严格，只做“经验提示”：
    - uint16 且非零中位数/最大值在几千到几万：很像毫米(mm)
    - float 且非零中位数在 0~10：很像米(m)
    """
    dtype = info["dtype"]
    nz50 = info.get("nonzero_p50", None)
    nz99 = info.get("nonzero_p99", None)

    if nz50 is None:
        return "unknown (no nonzero depth)"

    if "uint16" in dtype or "uint32" in dtype or "int" in dtype:
        if nz50 > 100 and (nz99 is not None and nz99 > 1000):
            return "likely millimeters (integer depth)"
        return "likely integer depth (unit unclear)"
    if "float" in dtype:
        if 0 < nz50 < 20:
            return "likely meters (float depth)"
        return "likely float depth (unit unclear)"
    return "unknown"


def analyze_path(name, path, recursive=False, max_files=50):
    files = iter_files(path, recursive=recursive)
    if not files:
        raise RuntimeError(f"[{name}] 没找到任何 depth 文件: {path}")

    files = files[:max_files]
    infos = []
    for fp in files:
        d = load_depth(fp)
        infos.append(robust_stats(d))

    # 汇总：dtype 统计 + 中位尺度
    dtypes = {}
    nz50_list = []
    nz99_list = []
    zero_list = []
    for inf in infos:
        dtypes[inf["dtype"]] = dtypes.get(inf["dtype"], 0) + 1
        if inf["nonzero_p50"] is not None:
            nz50_list.append(inf["nonzero_p50"])
        if inf["nonzero_p99"] is not None:
            nz99_list.append(inf["nonzero_p99"])
        if inf["zero_ratio"] is not None:
            zero_list.append(inf["zero_ratio"])

    summary = {
        "name": name,
        "path": path,
        "n_files": len(files),
        "dtype_counts": dtypes,
        "nonzero_p50_median": float(np.median(nz50_list)) if nz50_list else None,
        "nonzero_p99_median": float(np.median(nz99_list)) if nz99_list else None,
        "zero_ratio_median": float(np.median(zero_list)) if zero_list else None,
    }

    # 用“第一个文件”的信息做形状/范围参考
    ref = infos[0]
    summary["ref_shape"] = ref["shape"]
    summary["ref_min"] = ref["min"]
    summary["ref_max"] = ref["max"]
    summary["ref_nonzero_p50"] = ref["nonzero_p50"]
    summary["unit_guess"] = guess_unit({"dtype": list(dtypes.keys())[0], **ref}) if dtypes else "unknown"
    return summary


def print_summary(s):
    print("=" * 90)
    print(f"[{s['name']}] {s['path']}")
    print(f"  checked_files: {s['n_files']}")
    print(f"  dtype_counts: {s['dtype_counts']}")
    print(f"  ref_shape: {s['ref_shape']}")
    print(f"  ref_min/max: {s['ref_min']} / {s['ref_max']}")
    print(f"  median(nonzero_p50): {s['nonzero_p50_median']}")
    print(f"  median(nonzero_p99): {s['nonzero_p99_median']}")
    print(f"  median(zero_ratio): {s['zero_ratio_median']}")
    print(f"  unit_guess: {s['unit_guess']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="/home/ferry/data/Code2/Research/PoseEstimation/BundleTrack/YCBInEOAT/mustard0/depth", help="对比对象A：depth 文件或文件夹")
    ap.add_argument("--b", default="/home/ferry/data/Code2/Research/PoseEstimation/BundleTrack/YCBInEOAT/run_1765955006/depth", help="对比对象B：depth 文件或文件夹")
    ap.add_argument("--recursive", action="store_true", help="递归扫描子目录")
    ap.add_argument("--max-files", type=int, default=50, help="每个路径最多检查多少个文件")
    args = ap.parse_args()

    A = analyze_path("A", args.a, recursive=args.recursive, max_files=args.max_files)
    B = analyze_path("B", args.b, recursive=args.recursive, max_files=args.max_files)

    print_summary(A)
    print_summary(B)

    # 估计尺度比：用非零深度中位数对齐
    a50 = A["nonzero_p50_median"]
    b50 = B["nonzero_p50_median"]
    if a50 is not None and b50 is not None and b50 != 0:
        ratio = a50 / b50
        print("\n" + "-" * 90)
        print(f"Estimated scale ratio: A/B ≈ {ratio:.6f}")
        print("  例：如果 ratio≈1000，常见含义是 A=毫米、B=米（或反之）。")
    else:
        print("\n无法估计尺度比（缺少非零深度统计或 b50=0）。")


if __name__ == "__main__":
    main()
