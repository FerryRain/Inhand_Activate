"""
@FileName：real_time_pic_get.py
@Description：
@Author：Ferry
@Time：2025 12/19/25 3:18 PM
@Copyright：©2024-2025 ShanghaiTech University-RIMLAB
"""


import os
import re
import json
import time
import zmq
import numpy as np
import cv2

# ---------------- ZMQ utils ----------------
def bbox_from_mask(mask, pad=5):
    """mask: HxW (0/1 or 0/255). Return roi [x0,x1,y0,y1] or None if empty."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(mask.shape[1] - 1, x1 + pad)
    y1 = min(mask.shape[0] - 1, y1 + pad)
    return [int(x0), int(x1), int(y0), int(y1)]


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def make_client(addr="tcp://127.0.0.1:5550", timeout_ms=10000):
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.connect(addr)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    sock.setsockopt(zmq.LINGER, 0)
    return sock


def send_frame_with_sock(
        sock,
        color_bgr=None,        # uint8 HxWx3 (BGR)
        depth=None,            # uint16 HxW (mm) or float32 HxW (m)
        mask=None,             # uint8 HxW (0/1 or 0/255). Optional
        depth_type="uint16",
        id_str="00000",
        roi=None,              # [x0,x1,y0,y1] optional
        ob_in_cam=None,        # 4x4 float32 optional
        has_init=False,
):
    assert color_bgr is not None and depth is not None
    H, W = color_bgr.shape[:2]
    assert depth.shape[0] == H and depth.shape[1] == W

    # normalize mask (if provided)
    has_mask = mask is not None
    if has_mask:
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        mask01 = (mask > 0).astype(np.uint8)
    else:
        mask01 = None

    header = {
        "H": int(H),
        "W": int(W),
        "color_ch": 3,
        "depth_type": str(depth_type),   # "uint16" or "float32"
        "id_str": str(id_str),
        "has_init": bool(has_init),
        "has_mask": bool(has_mask),
    }

    if roi is not None:
        header["roi"] = [float(x) for x in roi]
    elif has_mask:
        auto_roi = bbox_from_mask(mask01, pad=5)
        if auto_roi is not None:
            header["roi"] = [float(x) for x in auto_roi]
            header["roi_pad"] = 5

    if has_init and ob_in_cam is not None:
        header["ob_in_cam"] = [float(x) for x in ob_in_cam.reshape(-1).tolist()]

    parts = [
        json.dumps(header).encode("utf-8"),
        color_bgr.tobytes(),
        depth.tobytes(),
    ]
    if has_mask:
        parts.append(mask01.tobytes())

    sock.send_multipart(parts)
    resp = json.loads(sock.recv().decode("utf-8"))
    T = np.array(resp["ob_in_cam"], dtype=np.float32).reshape(4, 4)
    return T, resp


# ---------------- Pose init from mask+depth ----------------
def compute_center3d_from_mask_depth_mm(mask01, depth_mm, fx, fy, cx, cy):
    """
    mask01: 0/1
    depth_mm: uint16, millimeters (aligned to color)
    return center3d (meters) in camera frame, or None
    """
    ys, xs = np.nonzero(mask01 > 0)
    if xs.size == 0:
        return None

    u = float(xs.mean())
    v = float(ys.mean())

    dvals = depth_mm[mask01 > 0].astype(np.float32)
    dvals = dvals[np.isfinite(dvals)]
    dvals = dvals[dvals > 0]
    if dvals.size == 0:
        return None

    z = float(np.median(dvals)) / 1000.0  # m
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    return np.array([X, Y, z], dtype=np.float32)


def make_ob_in_cam_from_center(center3d_m):
    T = np.eye(4, dtype=np.float32)
    # rotation = I (默认标准坐标系)
    T[:3, 3] = center3d_m.reshape(3)
    return T


# ---------------- SAM2 wrapper (bbox prompt -> mask) ----------------
class SamSegmenter:
    """
    bbox prompt (x0,y0,x1,y1) -> mask01
    """
    def __init__(self, checkpoint, model_cfg, device="cuda"):
        self.device = device
        self.predictor = None
        self._init_model(checkpoint, model_cfg)

    def _init_model(self, checkpoint, model_cfg):
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as e:
            raise RuntimeError(
                "Cannot import SAM2. Make sure sam2 is installed and available.\n"
                "Expected imports:\n"
                "  from sam2.build_sam import build_sam2\n"
                "  from sam2.sam2_image_predictor import SAM2ImagePredictor\n"
                f"Original error: {e}"
            )

        if self.device == "cuda":
            if not torch.cuda.is_available():
                print("[WARN] CUDA not available, fallback to CPU.")
                self.device = "cpu"

        model = build_sam2(model_cfg, checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(model)
        print(f"[SAM2] Loaded. device={self.device}, ckpt={checkpoint}, cfg={model_cfg}")

    def segment_from_box(self, bgr, box_xyxy):
        """
        bgr: HxWx3 uint8
        box_xyxy: [x0,y0,x1,y1]
        return mask01 uint8 HxW
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)

        x0, y0, x1, y1 = box_xyxy
        box = np.array([x0, y0, x1, y1], dtype=np.float32)

        masks, scores, _ = self.predictor.predict(
            box=box[None, :],
            multimask_output=False
        )
        # masks: [N, H, W] bool
        m = masks[0].astype(np.uint8)
        return m


# ---------------- Realsense D455 ----------------
def start_realsense(width=640, height=480, fps=30):
    try:
        import pyrealsense2 as rs
    except Exception as e:
        raise RuntimeError(
            "Cannot import pyrealsense2. Please install librealsense python bindings.\n"
            f"Original error: {e}"
        )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(config)

    # align depth to color
    align = rs.align(rs.stream.color)

    # intrinsics (color)
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    fx, fy, cx, cy = float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy)

    # depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())  # meters per unit

    print(f"[RS] Started D455: {width}x{height}@{fps}")
    print(f"[RS] intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print(f"[RS] depth_scale={depth_scale} m/unit")

    return pipeline, align, (fx, fy, cx, cy), depth_scale


def get_rs_frame(pipeline, align, depth_scale):
    """
    return color_bgr (uint8), depth_mm (uint16), both aligned
    """
    import pyrealsense2 as rs

    frames = pipeline.wait_for_frames()
    frames = align.process(frames)

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None

    color = np.asanyarray(color_frame.get_data())  # BGR
    depth_raw = np.asanyarray(depth_frame.get_data())  # uint16 in sensor units

    # convert to mm uint16, to match your previous pipeline expectation
    # depth_m = depth_raw * depth_scale
    # depth_mm = depth_m * 1000
    depth_mm = (depth_raw.astype(np.float32) * depth_scale * 1000.0).round().astype(np.uint16)

    return color, depth_mm


def overlay_mask(bgr, mask01, alpha=0.45):
    if mask01 is None:
        return bgr
    vis = bgr.copy()
    m = (mask01 > 0)
    if m.any():
        # use green overlay (no need to specify exact color if you prefer, but here is practical)
        overlay = vis.copy()
        overlay[m] = (0, 255, 0)
        vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
    return vis


if __name__ == "__main__":
    # ---------------- user configs ----------------
    addr = "tcp://127.0.0.1:5550"

    # SAM2
    checkpoint = "./YCBInEOAT/sam2.1_hiera_tiny.pt"
    model_cfg  = "configs/sam2.1/sam2.1_hiera_t.yaml"
    device = "cuda"   # "cuda" or "cpu"

    # Realsense stream
    rs_w, rs_h, rs_fps = 640, 480, 30

    # ---------------- init ----------------
    sock = make_client(addr=addr, timeout_ms=10000)
    segmenter = SamSegmenter(checkpoint=checkpoint, model_cfg=model_cfg, device=device)

    pipeline, align, (fx, fy, cx, cy), depth_scale = start_realsense(rs_w, rs_h, rs_fps)

    has_init = False
    init_done = False
    i = 0

    prev_mask01 = None
    prev_box_xyxy = None  # [x0,y0,x1,y1]

    win = "D455 (press 's' init | 'r' reset | 'q' quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        t_last = time.time()
        while True:
            color, depth_mm = get_rs_frame(pipeline, align, depth_scale)
            if color is None:
                continue

            # preview
            disp = overlay_mask(color, prev_mask01) if init_done else color.copy()
            now = time.time()
            fps = 1.0 / max(1e-6, (now - t_last))
            t_last = now
            cv2.putText(disp, f"fps={fps:.1f}  init={init_done}  i={i}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if key == ord('r'):
                print("[RESET] wait for next 's' to initialize.")
                init_done = False
                i = 0
                prev_mask01 = None
                prev_box_xyxy = None
                continue

            # ---- press 's' to initialize on current frame ----
            if (key == ord('s')) and (not init_done):
                # 1) manual ROI select
                print("[INIT] Select ROI (press ENTER/SPACE to confirm, ESC to cancel)...")
                x, y, w, h = cv2.selectROI("Select SAM Prompt ROI", color, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("Select SAM Prompt ROI")
                if w <= 1 or h <= 1:
                    print("[INIT] ROI canceled / too small.")
                    continue

                x0, y0, x1, y1 = int(x), int(y), int(x + w - 1), int(y + h - 1)
                box_xyxy = [x0, y0, x1, y1]

                # 2) SAM segmentation on this frame
                t0 = time.time()
                mask01 = segmenter.segment_from_box(color, box_xyxy)
                dt = time.time() - t0
                if mask01 is None or mask01.sum() == 0:
                    print("[INIT] SAM returned empty mask. Try again.")
                    continue
                print(f"[INIT] SAM ok. infer={dt*1000:.1f}ms, mask_pixels={int(mask01.sum())}")

                # 3) compute center3D from mask+depth -> initial ob_in_cam (R=I)
                center3d = compute_center3d_from_mask_depth_mm(mask01, depth_mm, fx, fy, cx, cy)
                if center3d is None:
                    print("[INIT] Cannot compute center3D (mask empty or invalid depth). Try again.")
                    continue
                ob_in_cam = make_ob_in_cam_from_center(center3d)
                print(f"[INIT] center3D(m): X={center3d[0]:.4f}, Y={center3d[1]:.4f}, Z={center3d[2]:.4f}")

                # 4) send as first frame i==0, has_init=True
                id_str = f"{i:06d}"
                t_send0 = time.time()
                T, resp = send_frame_with_sock(
                    sock,
                    color_bgr=color,
                    depth=depth_mm,
                    mask=mask01,
                    depth_type="uint16",
                    id_str=id_str,
                    roi=None,                # let it auto from mask
                    has_init=True,
                    ob_in_cam=ob_in_cam
                )
                print(f"[SEND][INIT] i={i} ok={resp.get('ok')} cost={time.time()-t_send0:.3f}s")
                # update state
                init_done = True
                has_init = True
                prev_mask01 = mask01
                prev_box_xyxy = bbox_from_mask(mask01, pad=5)
                if prev_box_xyxy is not None:
                    # bbox_from_mask gives [x0,x1,y0,y1] -> convert to xyxy
                    prev_box_xyxy = [prev_box_xyxy[0], prev_box_xyxy[2], prev_box_xyxy[1], prev_box_xyxy[3]]
                i += 1
                continue

            # ---- after init: each frame -> auto prompt by previous bbox -> SAM mask -> send ----
            if init_done:
                # build prompt box
                if prev_box_xyxy is None:
                    # fallback: use whole image (not ideal but safe)
                    box_xyxy = [0, 0, color.shape[1] - 1, color.shape[0] - 1]
                else:
                    box_xyxy = prev_box_xyxy

                # segment
                t0 = time.time()
                mask01 = segmenter.segment_from_box(color, box_xyxy)
                infer_ms = (time.time() - t0) * 1000.0

                # if SAM fails, keep previous mask (optional)
                if mask01 is None or mask01.sum() == 0:
                    print(f"[WARN] empty mask at i={i}, keep previous mask.")
                    mask01 = prev_mask01

                # update bbox for next frame
                if mask01 is not None:
                    bb = bbox_from_mask(mask01, pad=5)
                    if bb is not None:
                        prev_box_xyxy = [bb[0], bb[2], bb[1], bb[3]]
                    prev_mask01 = mask01

                # send
                id_str = f"{i:06d}"
                t_send0 = time.time()
                T, resp = send_frame_with_sock(
                    sock,
                    color_bgr=color,
                    depth=depth_mm,
                    mask=mask01,
                    depth_type="uint16",
                    id_str=id_str,
                    roi=None,            # auto ROI from mask
                    has_init=False,
                    ob_in_cam=None
                )
                cost = time.time() - t_send0
                print(f"[SEND] i={i:06d} ok={resp.get('ok')} infer_ms={infer_ms:.1f} send_cost={cost:.3f}s")
                i += 1

    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        try:
            sock.close()
        except Exception:
            pass
        print("[EXIT] Done.")
