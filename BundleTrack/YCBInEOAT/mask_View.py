#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np

try:
    import cv2
except ImportError:
    raise ImportError("请先安装 opencv-python: pip install opencv-python")

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
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

def load_mask(fp):
    ext = os.path.splitext(fp)[1].lower()
    if ext in NPY_EXTS:
        m = np.load(fp)
        return m
    m = cv2.imread(fp, cv2.IMREAD_UNCHANGED)  # 保留原 dtype (uint8/uint16)
    if m is None:
        raise RuntimeError(f"读不到文件: {fp}")
    return m

def summarize_values(arr, max_unique_print=30):
    # 如果是 3 通道 mask，很多时候是“伪彩色”，先给整体 unique（逐元素）也能判断 0/255
    flat = arr.reshape(-1)

    # unique 可能很大，先粗略统计
    uniq = np.unique(flat)
    info = {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "min": int(flat.min()) if flat.size else None,
        "max": int(flat.max()) if flat.size else None,
        "unique_count": int(uniq.size),
        "unique_preview": uniq[:max_unique_print].tolist() if uniq.size <= max_unique_print else uniq[:max_unique_print].tolist(),
        "is_binary_01": set(uniq.tolist()).issubset({0, 1}),
        "is_binary_0255": set(uniq.tolist()).issubset({0, 255}),
    }
    # 额外：统计 0 和 非0 占比
    if flat.size:
        nonzero = int(np.count_nonzero(flat))
        info["nonzero_ratio"] = float(nonzero / flat.size)
    else:
        info["nonzero_ratio"] = None
    return info, uniq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/home/ferry/data/Code2/Research/PoseEstimation/BundleTrack/YCBInEOAT/mustard0/masks", help="mask 文件路径 或 mask 文件夹路径")
    ap.add_argument("--recursive", action="store_true", help="递归扫描子目录")
    ap.add_argument("--max-files", type=int, default=50, help="最多检查多少个文件（防止太慢）")
    ap.add_argument("--max-unique-print", type=int, default=30, help="最多打印多少个 unique 值")
    args = ap.parse_args()

    files = iter_files(args.path, recursive=args.recursive)
    if not files:
        print("没找到任何 mask 文件（支持 png/jpg/tiff/npy 等）")
        return

    files = files[:args.max_files]
    global_uniq = set()
    binary01_cnt = 0
    binary0255_cnt = 0

    for i, fp in enumerate(files):
        m = load_mask(fp)

        # 如果是 3 通道，顺便也看一下单通道（取第0通道）是否更像二值
        info_all, uniq_all = summarize_values(m, max_unique_print=args.max_unique_print)
        global_uniq.update(uniq_all.tolist())

        # 打印
        print("=" * 80)
        print(f"[{i+1}/{len(files)}] {fp}")
        print(f"  dtype={info_all['dtype']} shape={info_all['shape']} min={info_all['min']} max={info_all['max']}")
        print(f"  unique_count={info_all['unique_count']}")
        print(f"  unique_preview={info_all['unique_preview']}"
              + ("" if info_all["unique_count"] <= args.max_unique_print else " ..."))
        print(f"  nonzero_ratio={info_all['nonzero_ratio']:.6f}" if info_all["nonzero_ratio"] is not None else "  nonzero_ratio=None")
        print(f"  binary(0/1)={info_all['is_binary_01']}  binary(0/255)={info_all['is_binary_0255']}")

        if info_all["is_binary_01"]:
            binary01_cnt += 1
        if info_all["is_binary_0255"]:
            binary0255_cnt += 1

        if m.ndim == 3 and m.shape[2] >= 1:
            ch0 = m[..., 0]
            info_ch0, _ = summarize_values(ch0, max_unique_print=args.max_unique_print)
            print(f"  [channel0] min={info_ch0['min']} max={info_ch0['max']} "
                  f"binary(0/1)={info_ch0['is_binary_01']} binary(0/255)={info_ch0['is_binary_0255']}")

    print("\n" + "#" * 80)
    print(f"Checked files: {len(files)}")
    print(f"Binary 0/1 files:   {binary01_cnt}")
    print(f"Binary 0/255 files: {binary0255_cnt}")
    global_uniq_sorted = sorted(global_uniq)
    preview = global_uniq_sorted[:60]
    print(f"Global unique values preview (first {len(preview)}): {preview}"
          + ("" if len(global_uniq_sorted) <= 60 else " ..."))
    print("#" * 80)

if __name__ == "__main__":
    main()
