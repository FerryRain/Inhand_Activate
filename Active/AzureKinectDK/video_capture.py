"""
@FileName：video_capture.py
@Description：
@Author：Ferry
@Time：2026 1/12/26 3:53 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Record Azure Kinect DK RGB + Depth using AzureKinectDK wrapper.

Outputs (default):
  out_dir/
    meta.json
    K_color.npy
    rgb_video.mp4              (if --save_rgb_video)
    rgb/000000.png ...         (if --save_rgb_frames)
    depth/000000.png ...       (if --save_depth_frames and depth_format=png16)
    depth_npy/000000.npy ...   (if --save_depth_frames and depth_format=npy)

Press 'q' to quit.
"""

import os
import json
import time
import argparse
from datetime import datetime

import cv2
import numpy as np

import pyk4a
from pyk4a import PyK4A, Config, CalibrationType


class AzureKinectDK:
    def __init__(self, color_res, depth_mode, fps=30):
        if color_res == "720P":
            self.cr = pyk4a.ColorResolution.RES_720P
        elif color_res == "1080P":
            self.cr = pyk4a.ColorResolution.RES_1080P
        elif color_res == "1536P":
            self.cr = pyk4a.ColorResolution.RES_1536P
        elif color_res == "2160P":
            self.cr = pyk4a.ColorResolution.RES_2160P
        else:
            raise ValueError(f"Unsupported color_res={color_res}")

        if depth_mode == "NFOV_UNBINNED":
            self.dm = pyk4a.DepthMode.NFOV_UNBINNED
        elif depth_mode == "NFOV_2X2BINNED":
            self.dm = pyk4a.DepthMode.NFOV_2X2BINNED
        elif depth_mode == "WFOV_UNBINNED":
            self.dm = pyk4a.DepthMode.WFOV_UNBINNED
        elif depth_mode == "WFOV_2X2BINNED":
            self.dm = pyk4a.DepthMode.WFOV_2X2BINNED
        else:
            raise ValueError(f"Unsupported depth_mode={depth_mode}")

        if fps == 30:
            self.cfps = pyk4a.FPS.FPS_30
        elif fps == 15:
            self.cfps = pyk4a.FPS.FPS_15
        elif fps == 5:
            self.cfps = pyk4a.FPS.FPS_5
        else:
            raise ValueError(f"Unsupported fps={fps}")

        self.k4a = PyK4A(
            Config(
                color_resolution=self.cr,
                depth_mode=self.dm,
                camera_fps=self.cfps,
                color_format=pyk4a.ImageFormat.COLOR_BGRA32,
                synchronized_images_only=True,
            )
        )
        self.K_color = None

    def start_init(self):
        self.k4a.start()
        self.K_color = self.k4a.calibration.get_camera_matrix(CalibrationType.COLOR).astype(np.float32)

        color_w = color_h = None
        for _ in range(8):
            cap = self.k4a.get_capture()
            if cap is not None and cap.color is not None and np.any(cap.color):
                color_h, color_w = cap.color.shape[:2]
                break
            time.sleep(0.02)

        if color_w is None:
            cap = self.k4a.get_capture()
            if cap is None or cap.color is None:
                raise RuntimeError("Azure Kinect: cannot get color frame. Check device & permissions.")
            color_h, color_w = cap.color.shape[:2]

        print("[AzureKinectDK] init ok")
        print(f"[K4A] Started: color={color_w}x{color_h}@{self.cfps}, depth_mode={self.dm}")
        print(f"[K4A] K_color: fx={self.K_color[0,0]:.2f}, fy={self.K_color[1,1]:.2f}, "
              f"cx={self.K_color[0,2]:.2f}, cy={self.K_color[1,2]:.2f}")

    def stop(self):
        try:
            self.k4a.stop()
        except Exception:
            pass

    def get_k4a_frame(self, out_size=None, require_aligned_depth=True):
        cap = self.k4a.get_capture()
        if cap is None:
            return None, None

        color = cap.color  # BGRA
        if color is None or (not np.any(color)):
            return None, None

        # BGRA -> BGR
        if color.ndim == 3 and color.shape[2] == 4:
            color_bgr = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
        else:
            color_bgr = color.copy()

        # Depth in mm (uint16). Prefer transformed_depth if available.
        depth_aligned = None
        if hasattr(cap, "transformed_depth"):
            try:
                depth_aligned = cap.transformed_depth
            except Exception:
                depth_aligned = None

        if depth_aligned is not None and np.any(depth_aligned):
            depth_mm = depth_aligned.astype(np.uint16)
        else:
            depth_mm = cap.depth
            if depth_mm is None:
                return None, None
            depth_mm = depth_mm.astype(np.uint16)

            if require_aligned_depth:
                print("[WARN] cap.transformed_depth not available; depth may NOT be aligned to color.")
                print("       Your pipeline assumes depth aligned to color.")

        if out_size is not None:
            out_w, out_h = int(out_size[0]), int(out_size[1])
            if color_bgr.shape[1] != out_w or color_bgr.shape[0] != out_h:
                color_bgr = cv2.resize(color_bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                depth_mm = cv2.resize(depth_mm, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        return color_bgr, depth_mm


def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def record_rgbd(
    out_dir: str,
    color_res: str,
    depth_mode: str,
    fps: int,
    out_w: int | None,
    out_h: int | None,
    duration_sec: float | None,
    max_frames: int | None,
    preview: bool,
    save_rgb_video: bool,
    save_rgb_frames: bool,
    save_depth_frames: bool,
    depth_format: str,  # "png16" or "npy"
    require_aligned_depth: bool,
):
    _ensure_dir(out_dir)

    # init camera
    cam = AzureKinectDK(color_res=color_res, depth_mode=depth_mode, fps=fps)
    cam.start_init()

    # grab one frame to get size after resize policy
    out_size = None
    if out_w is not None and out_h is not None:
        out_size = (out_w, out_h)

    color0, depth0 = cam.get_k4a_frame(out_size=out_size, require_aligned_depth=require_aligned_depth)
    if color0 is None or depth0 is None:
        cam.stop()
        raise RuntimeError("Failed to fetch initial frame. Check camera connection / permissions.")

    H, W = color0.shape[:2]

    # write meta
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "color_res": color_res,
        "depth_mode": depth_mode,
        "fps": fps,
        "output_size": [W, H],
        "duration_sec": duration_sec,
        "max_frames": max_frames,
        "save_rgb_video": save_rgb_video,
        "save_rgb_frames": save_rgb_frames,
        "save_depth_frames": save_depth_frames,
        "depth_format": depth_format,
        "require_aligned_depth": require_aligned_depth,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if cam.K_color is not None:
        np.save(os.path.join(out_dir, "K_color.npy"), cam.K_color)

    # output dirs
    rgb_dir = os.path.join(out_dir, "rgb")
    depth_dir = os.path.join(out_dir, "depth")
    depth_npy_dir = os.path.join(out_dir, "depth_npy")

    if save_rgb_frames:
        _ensure_dir(rgb_dir)
    if save_depth_frames and depth_format == "png16":
        _ensure_dir(depth_dir)
    if save_depth_frames and depth_format == "npy":
        _ensure_dir(depth_npy_dir)

    # rgb video writer
    vw = None
    if save_rgb_video:
        # mp4v is generally available; if not, try XVID/MJPG (avi).
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        rgb_video_path = os.path.join(out_dir, "rgb_video.mp4")
        vw = cv2.VideoWriter(rgb_video_path, fourcc, float(fps), (W, H))
        if not vw.isOpened():
            # fallback to AVI
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            rgb_video_path = os.path.join(out_dir, "rgb_video.avi")
            vw = cv2.VideoWriter(rgb_video_path, fourcc, float(fps), (W, H))
        if not vw.isOpened():
            cam.stop()
            raise RuntimeError("Failed to open VideoWriter. Try installing ffmpeg or use --save_rgb_frames only.")

    # recording loop
    t0 = time.time()
    idx = 0
    print(f"[REC] start -> {out_dir}")
    print("[REC] press 'q' to quit")

    try:
        while True:
            if duration_sec is not None and (time.time() - t0) >= duration_sec:
                break
            if max_frames is not None and idx >= max_frames:
                break

            color_bgr, depth_mm = cam.get_k4a_frame(out_size=out_size, require_aligned_depth=require_aligned_depth)
            if color_bgr is None or depth_mm is None:
                # occasional drop; just continue
                continue

            # save rgb video
            if vw is not None:
                vw.write(color_bgr)

            # save rgb frames
            if save_rgb_frames:
                rgb_path = os.path.join(rgb_dir, f"{idx:06d}.png")
                cv2.imwrite(rgb_path, color_bgr)

            # save depth frames
            if save_depth_frames:
                if depth_format == "png16":
                    # 16-bit PNG; values are in mm
                    dpath = os.path.join(depth_dir, f"{idx:06d}.png")
                    cv2.imwrite(dpath, depth_mm)  # uint16 preserved
                elif depth_format == "npy":
                    dpath = os.path.join(depth_npy_dir, f"{idx:06d}.npy")
                    np.save(dpath, depth_mm)
                else:
                    raise ValueError(f"Unsupported depth_format={depth_format}")

            # preview
            if preview:
                # visualize depth for display only (do NOT affect saved depth)
                depth_vis = depth_mm.astype(np.float32)
                # clamp to 0~4000mm for visualization
                depth_vis = np.clip(depth_vis, 0, 4000)
                depth_vis = (depth_vis / 4000.0 * 255.0).astype(np.uint8)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

                vis = np.hstack([color_bgr, depth_vis])
                cv2.imshow("AzureKinectDK RGB | Depth(vis)", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            idx += 1

    finally:
        if vw is not None:
            vw.release()
        cam.stop()
        if preview:
            cv2.destroyAllWindows()

    dt = time.time() - t0
    print(f"[REC] done. frames={idx}, elapsed={dt:.2f}s, avg_fps={idx / max(dt, 1e-6):.2f}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="azure_record_out", help="output directory")
    ap.add_argument("--color_res", type=str, default="720P",
                    choices=["720P", "1080P", "1536P", "2160P"])
    ap.add_argument("--depth_mode", type=str, default="NFOV_UNBINNED",
                    choices=["NFOV_UNBINNED", "NFOV_2X2BINNED", "WFOV_UNBINNED", "WFOV_2X2BINNED"])
    ap.add_argument("--fps", type=int, default=30, choices=[5, 15, 30])

    ap.add_argument("--out_w", type=int, default=None, help="resize width (optional)")
    ap.add_argument("--out_h", type=int, default=None, help="resize height (optional)")

    ap.add_argument("--duration", type=float, default=None, help="record seconds (optional)")
    ap.add_argument("--max_frames", type=int, default=None, help="max frames (optional)")

    ap.add_argument("--preview", action="store_true", help="show preview window")
    ap.add_argument("--save_rgb_video", action="store_true", help="save rgb as video (mp4/avi)")
    ap.add_argument("--save_rgb_frames", action="store_true", help="save rgb frames as png")
    ap.add_argument("--save_depth_frames", action="store_true", help="save depth frames")
    ap.add_argument("--depth_format", type=str, default="png16", choices=["png16", "npy"],
                    help="depth per-frame format")
    ap.add_argument("--require_aligned_depth", action="store_true",
                    help="warn if depth not aligned to color (recommended)")

    return ap.parse_args()


def main():
    args = parse_args()

    # sensible defaults: save frames if user did not specify any saving flags
    if (not args.save_rgb_video) and (not args.save_rgb_frames) and (not args.save_depth_frames):
        args.save_rgb_frames = True
        args.save_depth_frames = True

    # If only one of out_w/out_h is provided, ignore resize to avoid aspect mismatch.
    if (args.out_w is None) ^ (args.out_h is None):
        print("[WARN] Only one of --out_w/--out_h provided; ignoring resize.")
        args.out_w = None
        args.out_h = None

    record_rgbd(
        out_dir=args.out_dir,
        color_res=args.color_res,
        depth_mode=args.depth_mode,
        fps=args.fps,
        out_w=args.out_w,
        out_h=args.out_h,
        duration_sec=args.duration,
        max_frames=args.max_frames,
        preview=args.preview,
        save_rgb_video=args.save_rgb_video,
        save_rgb_frames=args.save_rgb_frames,
        save_depth_frames=args.save_depth_frames,
        depth_format=args.depth_format,
        require_aligned_depth=args.require_aligned_depth,
    )


if __name__ == "__main__":
    main()
