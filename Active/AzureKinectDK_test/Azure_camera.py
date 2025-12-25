"""
@FileName：Azure_camera.py
@Description：
@Author：Ferry
@Time：2025 12/25/25 4:23 PM
@Copyright：©2024-2025 ShanghaiTech University-RIMLAB
"""


import pyk4a
from pyk4a import PyK4A, Config, CalibrationType
import cv2
import numpy as np

class AzureKinectDK():
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


    def start_init(self):
        self.k4a.start()
        self.K_color = self.k4a.calibration.get_camera_matrix(CalibrationType.COLOR).astype(np.float32)
        color_w = color_h = None
        for _ in range(5):
            cap = self.k4a.get_capture()
            if cap.color is not None and np.any(cap.color):
                color_h, color_w = cap.color.shape[:2]
                break
            time.sleep(0.02)

        if color_w is None:
            cap = self.k4a.get_capture()
            if cap.color is None:
                raise RuntimeError("Azure Kinect: cannot get color frame. Check device & permissions.")
            color_h, color_w = cap.color.shape[:2]

        print("[AzureKinectDK] init ok]")
        print(f"[K4A] Started Azure Kinect DK: color={color_w}x{color_h}@{self.cfps}, depth_mode={self.dm}")
        print(f"[K4A] K_color: fx={self.K_color[0,0]:.2f}, fy={self.K_color[1,1]:.2f}, cx={self.K_color[0,2]:.2f}, cy={self.K_color[1,2]:.2f}")


    def get_k4a_frame(self, out_size=None, require_aligned_depth=True):
        cap = self.k4a.get_capture()
        if cap is None:
            return None, None

        color = cap.color  # usually BGRA :contentReference[oaicite:7]{index=7}
        if color is None or (not np.any(color)):
            return None, None

        # BGRA -> BGR
        if color.ndim == 3 and color.shape[2] == 4:
            color_bgr = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
        else:
            # uncommon cases
            color_bgr = color.copy()

        # Depth in mm (uint16). Use transformed_depth to align depth->color if available. :contentReference[oaicite:8]{index=8}
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
                # 这里直接给强提醒：你现在的 pipeline 需要 depth 对齐到 color
                print("[WARN] cap.transformed_depth not available; depth may NOT be aligned to color.")
                print("       Your mask->depth logic assumes alignment. Consider enabling transformed_depth / transformation.")

        # optional resize (keep your old 640x480 behavior if you want)
        if out_size is not None:
            out_w, out_h = int(out_size[0]), int(out_size[1])
            if color_bgr.shape[1] != out_w or color_bgr.shape[0] != out_h:
                color_bgr = cv2.resize(color_bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                depth_mm = cv2.resize(depth_mm, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        return color_bgr, depth_mm