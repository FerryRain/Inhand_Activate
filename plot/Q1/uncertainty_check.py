#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

# 你的NBV类
from Active.NBV_gpis import GPISNBVv2

try:
    import open3d as o3d
except Exception:
    o3d = None





RECON_FULL_RE = re.compile(r"^recon_0_(\d+)\.ply$")

def list_recon0_ply(in_dir: str, recursive: bool = True) -> List[Tuple[int, Path]]:
    root = Path(in_dir)
    it = root.rglob("recon_0_*.ply") if recursive else root.glob("recon_0_*.ply")

    out: List[Tuple[int, Path]] = []
    for p in it:
        if not p.is_file():
            continue
        m = RECON_FULL_RE.match(p.name)  # 注意：对完整文件名匹配
        if not m:
            continue
        frame_id = int(m.group(1))
        out.append((frame_id, p))

    out.sort(key=lambda x: x[0])
    return out



def pick_by_time(frames: List[Tuple[int, Path]], fps: float, t_sec: float) -> Tuple[int, Path, int]:
    """
    目标帧 target = round(t*fps)。返回最近帧 (frame_id, path, target_frame)。
    """
    if not frames:
        raise RuntimeError("No recon_0_*.ply files found.")

    target = int(round(t_sec * fps))

    # 二分找最近
    ids = [fid for fid, _ in frames]
    import bisect
    j = bisect.bisect_left(ids, target)
    if j == 0:
        fid, p = frames[0]
    elif j >= len(frames):
        fid, p = frames[-1]
    else:
        fid_l, p_l = frames[j - 1]
        fid_r, p_r = frames[j]
        if abs(fid_l - target) <= abs(fid_r - target):
            fid, p = fid_l, p_l
        else:
            fid, p = fid_r, p_r
    return fid, p, target


def _try_get_vis_pcd(est) -> Optional["o3d.geometry.PointCloud"]:
    """
    尝试从 est 内部抓到用于 viz 的点云（期望已按 uncertainty 上色）。
    如果你在 GPISNBVv2 里把可视化点云保存为 self.pcd_vis / self.vis_pcd 等，这里就能自动截图。
    """
    if o3d is None:
        return None

    for name in ["pcd_vis", "vis_pcd", "pcd_uncert", "uncertainty_pcd", "pcd_colored", "colored_pcd", "pcd"]:
        if hasattr(est, name):
            obj = getattr(est, name)
            if isinstance(obj, o3d.geometry.PointCloud):
                return obj
    return None


def save_pcd_screenshot(pcd, out_png: str, w: int = 1280, h: int = 720, point_size: float = 2.0) -> bool:
    if o3d is None or pcd is None:
        return False
    try:
        vis = o3d.visualization.Visualizer()
        try:
            vis.create_window(width=w, height=h, visible=False)
        except TypeError:
            vis.create_window(width=w, height=h)
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        if opt is not None:
            opt.point_size = float(point_size)
        vis.poll_events()
        vis.update_renderer()
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        vis.capture_screen_image(out_png, do_render=True)
        vis.destroy_window()
        return True
    except Exception as e:
        print(f"[WARN] Screenshot failed: {e}")
        try:
            vis.destroy_window()
        except Exception:
            pass
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/result/green_cube_00_001/pcd")
    ap.add_argument("--fps", type=float, default=4)
    ap.add_argument("--times", type=float, nargs="+", default=[10, 20, 30])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_viz", action="store_true")
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--recursive", action="store_true")
    args = ap.parse_args()

    frames = list_recon0_ply(args.in_dir, recursive=args.recursive or True)
    if not frames:
        raise RuntimeError(f"No recon_0_*.ply found under: {args.in_dir}")

    print(f"[INFO] Found {len(frames)} recon_0_*.ply files.")
    print(f"[INFO] First: frame={frames[0][0]}  file={frames[0][1]}")
    print(f"[INFO] Last : frame={frames[-1][0]}  file={frames[-1][1]}")

    for t in args.times:
        fid, ply_path, target = pick_by_time(frames, args.fps, t)
        print(f"\n[INFO] t={t:.2f}s -> target_frame={target} -> picked_frame={fid} -> {ply_path.name}")

        est = GPISNBVv2()
        nbv = est.estimate(str(ply_path), seed=args.seed, verbose=True)
        print(f"[INFO] NBV best_dir = {nbv.get('best_dir', None)}")

        saved = False
        if args.out_dir:
            out_png = str(Path(args.out_dir) / f"uncertainty_t{int(round(t)):02d}s_target{target:06d}_pick{fid:06d}.png")
            pcd = _try_get_vis_pcd(est)
            if pcd is not None:
                saved = save_pcd_screenshot(pcd, out_png)
                if saved:
                    print(f"[INFO] Saved: {out_png}")
            if not saved:
                print("[WARN] Auto-save failed (pcd not found on est or Open3D screenshot failed).")

        if (not args.no_viz) and (not saved):
            est.viz_2()

    print("\n[DONE]")


if __name__ == "__main__":
    main()
