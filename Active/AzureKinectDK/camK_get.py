"""
@FileName：camK_get.py
@Description：
@Author：Ferry
@Time：2025 12/25/25 3:35 PM
@Copyright：©2024-2025 ShanghaiTech University-RIMLAB
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dump Azure Kinect DK camera intrinsics (K matrix) to cam_K.txt

Usage:
  python3 dump_k4a_K.py --out cam_K.txt --color_res 720P --depth_mode NFOV_UNBINNED --fps 30
"""

import argparse
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="cam_K.txt", help="Output path of K matrix")
    ap.add_argument("--color_res", type=str, default="720P",
                    choices=["720P", "1080P", "1536P", "2160P"],
                    help="Color resolution")
    ap.add_argument("--depth_mode", type=str, default="NFOV_UNBINNED",
                    choices=["NFOV_UNBINNED", "NFOV_2X2BINNED", "WFOV_UNBINNED", "WFOV_2X2BINNED"],
                    help="Depth mode")
    ap.add_argument("--fps", type=int, default=30, choices=[5, 15, 30], help="Camera FPS")
    ap.add_argument("--which", type=str, default="color", choices=["color", "depth"],
                    help="Dump color or depth camera intrinsics")
    return ap.parse_args()


def main():
    args = parse_args()

    import pyk4a
    from pyk4a import PyK4A, Config, CalibrationType

    # --- map enums ---
    cr_map = {
        "720P": pyk4a.ColorResolution.RES_720P,
        "1080P": pyk4a.ColorResolution.RES_1080P,
        "1536P": pyk4a.ColorResolution.RES_1536P,
        "2160P": pyk4a.ColorResolution.RES_2160P,
    }
    dm_map = {
        "NFOV_UNBINNED": pyk4a.DepthMode.NFOV_UNBINNED,
        "NFOV_2X2BINNED": pyk4a.DepthMode.NFOV_2X2BINNED,
        "WFOV_UNBINNED": pyk4a.DepthMode.WFOV_UNBINNED,
        "WFOV_2X2BINNED": pyk4a.DepthMode.WFOV_2X2BINNED,
    }
    fps_map = {5: pyk4a.FPS.FPS_5, 15: pyk4a.FPS.FPS_15, 30: pyk4a.FPS.FPS_30}

    k4a = PyK4A(
        Config(
            color_resolution=cr_map[args.color_res],
            depth_mode=dm_map[args.depth_mode],
            camera_fps=fps_map[args.fps],
            color_format=pyk4a.ImageFormat.COLOR_BGRA32,
            synchronized_images_only=True,
        )
    )

    try:
        k4a.start()

        calib_type = CalibrationType.COLOR if args.which == "color" else CalibrationType.DEPTH
        K = k4a.calibration.get_camera_matrix(calib_type).astype(np.float64)  # 3x3

        # save as 3 lines x 3 cols
        np.savetxt(args.out, K, fmt="%.10f")
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        print(f"[OK] Saved {args.which} intrinsics to: {args.out}")
        print(f"K=\n{K}")
        print(f"fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")

    finally:
        try:
            k4a.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
