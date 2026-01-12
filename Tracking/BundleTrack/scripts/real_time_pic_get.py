#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@FileName：real_time_pic_get.py
@Description：RealSense -> SAM2ImagePredictor(bbox prompt per-frame) -> ZMQ -> local 6D pose viz
@Author：Ferry (modified)
@Time：2025 12/19/25
"""

import json
import time
import math
import zmq
import numpy as np
import cv2


# =========================
# ZMQ + ROI utils
# =========================
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
        color_bgr,              # uint8 HxWx3 (BGR)
        depth,                  # uint16 HxW (mm)
        mask=None,              # uint8 HxW (0/1 or 0/255). Optional
        depth_type="uint16",
        id_str="00000",
        roi=None,               # [x0,x1,y0,y1] optional
        ob_in_cam=None,         # 4x4 float32 optional
        has_init=False,
):
    assert color_bgr is not None and depth is not None
    H, W = color_bgr.shape[:2]
    assert depth.shape[0] == H and depth.shape[1] == W

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
        "depth_type": str(depth_type),   # "uint16"
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


# =========================
# Pose init from mask+depth
# =========================
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
    T[:3, 3] = center3d_m.reshape(3)
    return T


# =========================
# Local visualization (mask + 6D pose)
# =========================
def overlay_mask(bgr, mask01, alpha=0.45):
    if mask01 is None:
        return bgr
    vis = bgr.copy()
    m = (mask01 > 0)
    if m.any():
        overlay = vis.copy()
        overlay[m] = (0, 255, 0)
        vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
    return vis


def project_cam_point(Pc, K):
    z = float(Pc[2])
    if z <= 1e-6:
        return None
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    u = fx * (float(Pc[0]) / z) + cx
    v = fy * (float(Pc[1]) / z) + cy
    return (u, v)


def rpy_from_R_zyx(R):
    """
    对齐 Eigen::Matrix3f::eulerAngles(2,1,0):
    yaw(z), pitch(y), roll(x), 最终显示 [roll, pitch, yaw]
    """
    r00, r01, r02 = R[0, 0], R[0, 1], R[0, 2]
    r10, r11, r12 = R[1, 0], R[1, 1], R[1, 2]
    r20, r21, r22 = R[2, 0], R[2, 1], R[2, 2]

    yaw = math.atan2(r10, r00)
    pitch = math.atan2(-r20, math.sqrt(r21 * r21 + r22 * r22))
    roll = math.atan2(r21, r22)

    roll_deg = roll * 180.0 / math.pi
    pitch_deg = pitch * 180.0 / math.pi
    yaw_deg = yaw * 180.0 / math.pi
    return roll_deg, pitch_deg, yaw_deg


def draw_pose6d_on_image(img_bgr, T_obj_in_cam, K, axis_len=0.05, thickness=2, text_org=(5, 60)):
    R = T_obj_in_cam[:3, :3].astype(np.float32)
    t = T_obj_in_cam[:3, 3].astype(np.float32)

    O = t
    X = t + R @ np.array([axis_len, 0, 0], dtype=np.float32)
    Y = t + R @ np.array([0, axis_len, 0], dtype=np.float32)
    Z = t + R @ np.array([0, 0, axis_len], dtype=np.float32)

    o2 = project_cam_point(O, K)
    if o2 is not None:
        o2i = (int(round(o2[0])), int(round(o2[1])))
        cv2.circle(img_bgr, o2i, 3, (0, 255, 255), -1, cv2.LINE_AA)

        x2 = project_cam_point(X, K)
        if x2 is not None:
            x2i = (int(round(x2[0])), int(round(x2[1])))
            cv2.line(img_bgr, o2i, x2i, (0, 0, 255), thickness, cv2.LINE_AA)

        y2 = project_cam_point(Y, K)
        if y2 is not None:
            y2i = (int(round(y2[0])), int(round(y2[1])))
            cv2.line(img_bgr, o2i, y2i, (0, 255, 0), thickness, cv2.LINE_AA)

        z2 = project_cam_point(Z, K)
        if z2 is not None:
            z2i = (int(round(z2[0])), int(round(z2[1])))
            cv2.line(img_bgr, o2i, z2i, (255, 0, 0), thickness, cv2.LINE_AA)

    roll, pitch, yaw = rpy_from_R_zyx(R)
    ss1 = f"t = [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]"
    ss2 = f"rpy(deg) = [{roll:.2f}, {pitch:.2f}, {yaw:.2f}]"
    x0, y0 = int(text_org[0]), int(text_org[1])
    cv2.putText(img_bgr, ss1, (x0, y0), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img_bgr, ss2, (x0, y0 + 22), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 1, cv2.LINE_AA)


# =========================
# SAM2 Image predictor (bbox prompt)
# =========================
class SamSegmenter:
    """
    bbox prompt (x0,y0,x1,y1) -> mask01
    """
    def __init__(self, checkpoint, model_cfg, device="cuda", use_amp=True):
        self.device = device
        self.use_amp = use_amp
        self.predictor = None
        self._init_model(checkpoint, model_cfg)

    def _init_model(self, checkpoint, model_cfg):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if self.device == "cuda" and (not torch.cuda.is_available()):
            print("[WARN] CUDA not available, fallback to CPU.")
            self.device = "cpu"

        model = build_sam2(model_cfg, checkpoint, device=self.device)
        model.eval()
        self.predictor = SAM2ImagePredictor(model)
        print(f"[SAM2] Loaded ImagePredictor. device={self.device}, ckpt={checkpoint}, cfg={model_cfg}")

    def segment_from_box(self, bgr, box_xyxy):
        """
        bgr: HxWx3 uint8
        box_xyxy: [x0,y0,x1,y1]
        return mask01 uint8 HxW
        """
        import torch

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # set_image 会重新算当前帧 embedding（不累积缓存）
        with torch.inference_mode():
            if (self.device == "cuda") and self.use_amp:
                # 降显存占用
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    self.predictor.set_image(rgb)
                    box = np.array(box_xyxy, dtype=np.float32)
                    masks, scores, _ = self.predictor.predict(
                        box=box[None, :],
                        multimask_output=False
                    )
            else:
                self.predictor.set_image(rgb)
                box = np.array(box_xyxy, dtype=np.float32)
                masks, scores, _ = self.predictor.predict(
                    box=box[None, :],
                    multimask_output=False
                )

        m = masks[0].astype(np.uint8)  # HxW
        return m


# =========================
# RealSense
# =========================
def start_realsense(width=640, height=480, fps=30):
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    fx, fy, cx, cy = float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())  # meters per unit

    print(f"[RS] Started D455: {width}x{height}@{fps}")
    print(f"[RS] intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print(f"[RS] depth_scale={depth_scale} m/unit")

    return pipeline, align, (fx, fy, cx, cy), depth_scale


def get_rs_frame(pipeline, align, depth_scale):
    import pyrealsense2 as rs

    frames = pipeline.wait_for_frames()
    frames = align.process(frames)

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None

    color = np.asanyarray(color_frame.get_data())      # BGR uint8
    depth_raw = np.asanyarray(depth_frame.get_data())  # uint16 sensor units
    depth_mm = (depth_raw.astype(np.float32) * depth_scale * 1000.0).round().astype(np.uint16)
    return color, depth_mm


# =========================
# Main
# =========================
if __name__ == "__main__":
    # ---------- configs ----------
    addr = "tcp://127.0.0.1:5550"

    checkpoint = "./YCBInEOAT/sam2.1_hiera_tiny.pt"
    model_cfg  = "configs/sam2.1/sam2.1_hiera_t.yaml"

    # 如果你还 OOM：把 device 改成 "cpu"
    device = "cuda"   # "cuda" or "cpu"
    use_amp = True    # cuda 下建议 True（更省显存）

    rs_w, rs_h, rs_fps = 640, 480, 30
    axis_len = 0.05

    # ---------- init ----------
    sock = make_client(addr=addr, timeout_ms=10000)
    print(f"[ZMQ] connected: {addr}")

    segmenter = SamSegmenter(checkpoint=checkpoint, model_cfg=model_cfg, device=device, use_amp=use_amp)

    pipeline, align, (fx, fy, cx, cy), depth_scale = start_realsense(rs_w, rs_h, rs_fps)
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)

    win = "D455 + SAM2 bbox prompt (s:init ROI | r:reset | q:quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    init_done = False
    i = 0

    prev_mask01 = None
    prev_box_xyxy = None  # [x0,y0,x1,y1]
    last_T = None

    t_last = time.time()

    try:
        while True:
            color, depth_mm = get_rs_frame(pipeline, align, depth_scale)
            if color is None:
                continue

            disp = color.copy()
            if prev_mask01 is not None:
                disp = overlay_mask(disp, prev_mask01)
            if last_T is not None:
                draw_pose6d_on_image(disp, last_T, K, axis_len=axis_len, thickness=2, text_org=(5, 60))

            now = time.time()
            fps = 1.0 / max(1e-6, (now - t_last))
            t_last = now
            cv2.putText(disp, f"fps={fps:.1f}  init={init_done}  i={i}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

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
                last_T = None
                continue

            # ---------- init: press s, select ROI once ----------
            if (key == ord('s')) and (not init_done):
                print("[INIT] Select ROI (ENTER/SPACE confirm, ESC cancel)...")
                x, y, w, h = cv2.selectROI("Select bbox prompt ROI", color, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("Select bbox prompt ROI")
                if w <= 1 or h <= 1:
                    print("[INIT] ROI canceled / too small.")
                    continue

                x0, y0, x1, y1 = int(x), int(y), int(x + w - 1), int(y + h - 1)
                box_xyxy = [x0, y0, x1, y1]

                # 1) SAM bbox prompt on init frame
                t0 = time.time()
                try:
                    mask01 = segmenter.segment_from_box(color, box_xyxy)
                except RuntimeError as e:
                    print(f"[INIT] SAM failed: {e}")
                    print("If CUDA OOM: set device='cpu' or close other GPU programs.")
                    continue

                dt = (time.time() - t0) * 1000.0
                if mask01 is None or mask01.sum() == 0:
                    print("[INIT] SAM returned empty mask. Try again.")
                    continue
                print(f"[INIT] SAM ok. infer={dt:.1f}ms, mask_pixels={int(mask01.sum())}")

                # 2) compute center3D -> ob_in_cam init
                center3d = compute_center3d_from_mask_depth_mm(mask01, depth_mm, fx, fy, cx, cy)
                if center3d is None:
                    print("[INIT] Cannot compute center3D; send init without ob_in_cam.")
                    has_init = False
                    ob_in_cam = None
                else:
                    has_init = True
                    ob_in_cam = make_ob_in_cam_from_center(center3d)
                    print(f"[INIT] center3D(m): X={center3d[0]:.4f}, Y={center3d[1]:.4f}, Z={center3d[2]:.4f}")

                # 3) send first frame
                id_str = f"{i:06d}"
                t_send0 = time.time()
                T, resp = send_frame_with_sock(
                    sock,
                    color_bgr=color,
                    depth=depth_mm,
                    mask=mask01,
                    depth_type="uint16",
                    id_str=id_str,
                    roi=None,
                    has_init=has_init,
                    ob_in_cam=ob_in_cam
                )
                print(f"[SEND][INIT] i={i} ok={resp.get('ok')} cost={time.time()-t_send0:.3f}s")

                # update state
                init_done = True
                prev_mask01 = mask01
                bb = bbox_from_mask(mask01, pad=5)
                if bb is not None:
                    # bb: [x0,x1,y0,y1] -> xyxy
                    prev_box_xyxy = [bb[0], bb[2], bb[1], bb[3]]
                else:
                    prev_box_xyxy = box_xyxy

                last_T = T
                i += 1
                continue

            # ---------- after init: each frame bbox prompt from previous mask bbox ----------
            if init_done:
                if prev_box_xyxy is None:
                    box_xyxy = [0, 0, color.shape[1] - 1, color.shape[0] - 1]
                else:
                    box_xyxy = prev_box_xyxy

                # segment
                t0 = time.time()
                try:
                    mask01 = segmenter.segment_from_box(color, box_xyxy)
                except RuntimeError as e:
                    print(f"[WARN] SAM failed at i={i}: {e}")
                    mask01 = prev_mask01

                infer_ms = (time.time() - t0) * 1000.0

                if mask01 is None or mask01.sum() == 0:
                    # fallback
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
                    roi=None,
                    has_init=False,
                    ob_in_cam=None
                )
                cost = time.time() - t_send0

                last_T = T
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
