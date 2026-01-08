"""
@FileName: realtime_get.py
@Description:
@Author: Ferry
@Time: 2025/12/25 3:57 PM
"""

import json
import time
import math
import zmq
import numpy as np
import cv2

from Azure_camera import AzureKinectDK


# -------------------------
# ROI / ZMQ
# -------------------------
def bbox_from_mask(mask01: np.ndarray, pad: int = 5):
    """mask01: HxW uint8 {0,1}. Return [x0,x1,y0,y1] or None."""
    ys, xs = np.where(mask01 > 0)
    if xs.size == 0:
        return None

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(mask01.shape[1] - 1, x1 + pad)
    y1 = min(mask01.shape[0] - 1, y1 + pad)
    return [x0, x1, y0, y1]


def make_client(addr: str = "tcp://127.0.0.1:5550", timeout_ms: int = 10000):
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.connect(addr)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    sock.setsockopt(zmq.LINGER, 0)
    return sock


def send_frame_with_sock(
        sock,
        color_bgr: np.ndarray,          # uint8 HxWx3
        depth_mm: np.ndarray,           # uint16 HxW
        mask: np.ndarray | None = None, # uint8 HxW (0/1 or 0/255)
        id_str: str = "000000",
        roi=None,                       # [x0,x1,y0,y1]
        ob_in_cam: np.ndarray | None = None,
        has_init: bool = False,
):
    H, W = color_bgr.shape[:2]
    assert depth_mm.shape[:2] == (H, W)

    has_mask = mask is not None
    mask01 = None
    if has_mask:
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask01 = (mask > 0).astype(np.uint8)

    header = {
        "H": int(H),
        "W": int(W),
        "color_ch": 3,
        "depth_type": "uint16",
        "id_str": str(id_str),
        "has_init": bool(has_init),
        "has_mask": bool(has_mask),
    }

    # roi：优先使用显式 roi；否则从 mask 自动生成
    if roi is not None:
        header["roi"] = [float(x) for x in roi]
    elif mask01 is not None:
        auto_roi = bbox_from_mask(mask01, pad=5)
        if auto_roi is not None:
            header["roi"] = [float(x) for x in auto_roi]
            header["roi_pad"] = 5

    if has_init and ob_in_cam is not None:
        header["ob_in_cam"] = [float(x) for x in ob_in_cam.reshape(-1).tolist()]

    parts = [
        json.dumps(header).encode("utf-8"),
        color_bgr.tobytes(),
        depth_mm.tobytes(),
    ]
    if mask01 is not None:
        parts.append(mask01.tobytes())

    sock.send_multipart(parts)
    resp = json.loads(sock.recv().decode("utf-8"))
    T = np.array(resp["ob_in_cam"], dtype=np.float32).reshape(4, 4)
    return T, resp


# -------------------------
# Init pose from mask+depth
# -------------------------
def compute_center3d_from_mask_depth_mm(mask01, depth_mm, fx, fy, cx, cy):
    """Return center3d in camera frame (meters), or None."""
    ys, xs = np.nonzero(mask01 > 0)
    if xs.size == 0:
        return None

    u, v = float(xs.mean()), float(ys.mean())

    d = depth_mm[mask01 > 0].astype(np.float32)
    d = d[np.isfinite(d)]
    d = d[d > 0]
    if d.size == 0:
        return None

    z = float(np.median(d)) / 1000.0
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    return np.array([X, Y, z], dtype=np.float32)


def make_ob_in_cam_from_center(center3d_m):
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = center3d_m.reshape(3)
    return T


# -------------------------
# Visualization
# -------------------------
def overlay_mask(bgr, mask01, alpha=0.45):
    if mask01 is None:
        return bgr
    vis = bgr.copy()
    m = mask01 > 0
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
    """Match Eigen eulerAngles(2,1,0): yaw(z), pitch(y), roll(x)."""
    r00, r01, r02 = R[0, 0], R[0, 1], R[0, 2]
    r10, r11, r12 = R[1, 0], R[1, 1], R[1, 2]
    r20, r21, r22 = R[2, 0], R[2, 1], R[2, 2]

    yaw = math.atan2(r10, r00)
    pitch = math.atan2(-r20, math.sqrt(r21 * r21 + r22 * r22))
    roll = math.atan2(r21, r22)

    return (
        roll * 180.0 / math.pi,
        pitch * 180.0 / math.pi,
        yaw * 180.0 / math.pi,
    )


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

        for P, col in [(X, (0, 0, 255)), (Y, (0, 255, 0)), (Z, (255, 0, 0))]:
            p2 = project_cam_point(P, K)
            if p2 is not None:
                p2i = (int(round(p2[0])), int(round(p2[1])))
                cv2.line(img_bgr, o2i, p2i, col, thickness, cv2.LINE_AA)

    roll, pitch, yaw = rpy_from_R_zyx(R)
    ss1 = f"t = [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]"
    ss2 = f"rpy(deg) = [{roll:.2f}, {pitch:.2f}, {yaw:.2f}]"
    x0, y0 = int(text_org[0]), int(text_org[1])
    cv2.putText(img_bgr, ss1, (x0, y0), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img_bgr, ss2, (x0, y0 + 22), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 1, cv2.LINE_AA)


# -------------------------
# SAM2 bbox prompt wrapper
# -------------------------
class SamSegmenter:
    """bbox prompt -> mask01"""

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
        print(f"[SAM2] Loaded. device={self.device}")

    def segment_from_box(self, bgr, box_xyxy):
        import torch

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        box = np.array(box_xyxy, dtype=np.float32)[None, :]

        with torch.inference_mode():
            if (self.device == "cuda") and self.use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    self.predictor.set_image(rgb)
                    masks, _, _ = self.predictor.predict(box=box, multimask_output=False)
            else:
                self.predictor.set_image(rgb)
                masks, _, _ = self.predictor.predict(box=box, multimask_output=False)

        return masks[0].astype(np.uint8)


# -------------------------
# Main
# -------------------------
def main():
    addr = "tcp://127.0.0.1:5550"

    checkpoint = "sam_model/sam2.1_hiera_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    device = "cuda"
    use_amp = True

    axis_len = 0.05

    # Azure Kinect
    k4a_color_res = "720P"
    k4a_depth_mode = "NFOV_UNBINNED"
    k4a_fps = 30

    sock = make_client(addr=addr, timeout_ms=10000)
    print(f"[ZMQ] connected: {addr}")

    segmenter = SamSegmenter(checkpoint=checkpoint, model_cfg=model_cfg, device=device, use_amp=use_amp)

    camera = AzureKinectDK(color_res=k4a_color_res, fps=k4a_fps, depth_mode=k4a_depth_mode)
    camera.start_init()
    K = camera.K_color.astype(np.float32)

    win = "Azure Kinect DK + SAM2 (s:init ROI | r:reset | q:quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    init_done = False
    i = 0

    prev_mask01 = None
    prev_box_xyxy = None
    last_T = None

    t_last = time.time()

    try:
        while True:
            color, depth_mm = camera.get_k4a_frame(require_aligned_depth=True)
            if color is None or depth_mm is None:
                continue

            fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

            disp = color.copy()
            if prev_mask01 is not None:
                disp = overlay_mask(disp, prev_mask01)
            if last_T is not None:
                draw_pose6d_on_image(disp, last_T, K, axis_len=axis_len, thickness=2, text_org=(5, 60))

            now = time.time()
            fps_show = 1.0 / max(1e-6, (now - t_last))
            t_last = now
            cv2.putText(disp, f"fps={fps_show:.1f}  init={init_done}  i={i}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if key == ord('r'):
                print("[RESET] Press 's' to re-init.")
                init_done = False
                i = 0
                prev_mask01 = None
                prev_box_xyxy = None
                last_T = None
                continue

            # -------- init with ROI (press s once) --------
            if (key == ord('s')) and (not init_done):
                print("[INIT] Select ROI (ENTER/SPACE confirm, ESC cancel)...")
                x, y, w, h = cv2.selectROI("Select bbox prompt ROI", color, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("Select bbox prompt ROI")
                if w <= 1 or h <= 1:
                    print("[INIT] ROI canceled / too small.")
                    continue

                box_xyxy = [int(x), int(y), int(x + w - 1), int(y + h - 1)]

                try:
                    t0 = time.time()
                    mask01 = segmenter.segment_from_box(color, box_xyxy)
                    infer_ms = (time.time() - t0) * 1000.0
                except RuntimeError as e:
                    print(f"[INIT] SAM failed: {e}")
                    continue

                if mask01 is None or mask01.sum() == 0:
                    print("[INIT] Empty mask. Try again.")
                    continue

                print(f"[INIT] SAM ok. infer={infer_ms:.1f}ms, mask_pixels={int(mask01.sum())}")

                center3d = compute_center3d_from_mask_depth_mm(mask01, depth_mm, fx, fy, cx, cy)
                has_init = (center3d is not None)
                ob_in_cam = make_ob_in_cam_from_center(center3d) if has_init else None
                if has_init:
                    print(f"[INIT] center3D(m): X={center3d[0]:.4f}, Y={center3d[1]:.4f}, Z={center3d[2]:.4f}")
                else:
                    print("[INIT] center3D failed; send init without ob_in_cam.")

                id_str = f"{i:06d}"
                t_send0 = time.time()
                T, resp = send_frame_with_sock(
                    sock,
                    color_bgr=color,
                    depth_mm=depth_mm,
                    mask=mask01,
                    id_str=id_str,
                    roi=None,
                    has_init=has_init,
                    ob_in_cam=ob_in_cam,
                )
                print(f"[SEND][INIT] i={i} ok={resp.get('ok')} cost={time.time()-t_send0:.3f}s")

                init_done = True
                prev_mask01 = mask01

                bb = bbox_from_mask(mask01, pad=5)
                prev_box_xyxy = [bb[0], bb[2], bb[1], bb[3]] if bb is not None else box_xyxy

                last_T = T
                i += 1
                continue

            # -------- tracking phase --------
            if init_done:
                box_xyxy = prev_box_xyxy if prev_box_xyxy is not None else [0, 0, color.shape[1] - 1, color.shape[0] - 1]

                try:
                    t0 = time.time()
                    mask01 = segmenter.segment_from_box(color, box_xyxy)
                    infer_ms = (time.time() - t0) * 1000.0
                except RuntimeError as e:
                    print(f"[WARN] SAM failed at i={i}: {e}")
                    mask01 = prev_mask01
                    infer_ms = -1.0

                if mask01 is None or mask01.sum() == 0:
                    mask01 = prev_mask01

                if mask01 is not None:
                    bb = bbox_from_mask(mask01, pad=5)
                    if bb is not None:
                        prev_box_xyxy = [bb[0], bb[2], bb[1], bb[3]]
                    prev_mask01 = mask01

                id_str = f"{i:06d}"
                t_send0 = time.time()
                T, resp = send_frame_with_sock(
                    sock,
                    color_bgr=color,
                    depth_mm=depth_mm,
                    mask=mask01,
                    id_str=id_str,
                    roi=None,
                    has_init=False,
                    ob_in_cam=None,
                )
                cost = time.time() - t_send0
                last_T = T

                print(f"[SEND] i={i:06d} ok={resp.get('ok')} infer_ms={infer_ms:.1f} send_cost={cost:.3f}s")
                i += 1

    finally:
        cv2.destroyAllWindows()
        try:
            sock.close()
        except Exception:
            pass

        # 尝试关闭相机（按你 AzureKinectDK 的实现可能叫 stop/close）
        for fn in ["stop", "close", "shutdown"]:
            if hasattr(camera, fn):
                try:
                    getattr(camera, fn)()
                    break
                except Exception:
                    pass

        print("[EXIT] Done.")


if __name__ == "__main__":
    main()
