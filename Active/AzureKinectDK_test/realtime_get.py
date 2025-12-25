"""
@FileName：realtime_get.py
@Description：
@Author：Ferry
@Time：2025 12/25/25 3:57 PM
@Copyright：©2024-2025 ShanghaiTech University-RIMLAB
"""

import json
import time
import math
import zmq
import numpy as np
import cv2
from Azure_camera import AzureKinectDK


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
    depth_mm: uint16, millimeters (ALIGNED to color!)
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

        with torch.inference_mode():
            if (self.device == "cuda") and self.use_amp:
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

        m = masks[0].astype(np.uint8)
        return m


# =========================
# Azure Kinect DK (pyk4a)
# =========================
def start_azure_kinect(color_res="720P", fps=30, depth_mode="NFOV_UNBINNED"):
    """
    return: k4a, K_color(3x3), (color_w, color_h)
    """
    import pyk4a
    from pyk4a import PyK4A, Config, CalibrationType

    # --- enums ---
    # pyk4a.ColorResolution.RES_720P ... :contentReference[oaicite:4]{index=4}
    if color_res == "720P":
        cr = pyk4a.ColorResolution.RES_720P
    elif color_res == "1080P":
        cr = pyk4a.ColorResolution.RES_1080P
    elif color_res == "1536P":
        cr = pyk4a.ColorResolution.RES_1536P
    elif color_res == "2160P":
        cr = pyk4a.ColorResolution.RES_2160P
    else:
        raise ValueError(f"Unsupported color_res={color_res}")

    if depth_mode == "NFOV_UNBINNED":
        dm = pyk4a.DepthMode.NFOV_UNBINNED
    elif depth_mode == "NFOV_2X2BINNED":
        dm = pyk4a.DepthMode.NFOV_2X2BINNED
    elif depth_mode == "WFOV_UNBINNED":
        dm = pyk4a.DepthMode.WFOV_UNBINNED
    elif depth_mode == "WFOV_2X2BINNED":
        dm = pyk4a.DepthMode.WFOV_2X2BINNED
    else:
        raise ValueError(f"Unsupported depth_mode={depth_mode}")

    if fps == 30:
        cfps = pyk4a.FPS.FPS_30
    elif fps == 15:
        cfps = pyk4a.FPS.FPS_15
    elif fps == 5:
        cfps = pyk4a.FPS.FPS_5
    else:
        raise ValueError(f"Unsupported fps={fps}")

    # 建议用 BGRA32，省掉解码麻烦（pyk4a 示例里也这么用） :contentReference[oaicite:5]{index=5}
    k4a = PyK4A(
        Config(
            color_resolution=cr,
            depth_mode=dm,
            camera_fps=cfps,
            color_format=pyk4a.ImageFormat.COLOR_BGRA32,
            synchronized_images_only=True,
        )
    )
    k4a.start()

    # 取 color 内参 3x3（像素单位） :contentReference[oaicite:6]{index=6}
    K_color = k4a.calibration.get_camera_matrix(CalibrationType.COLOR).astype(np.float32)

    # 拿一帧确认分辨率 & warmup（有时前几帧 color 可能为空）
    color_w = color_h = None
    for _ in range(5):
        cap = k4a.get_capture()
        if cap.color is not None and np.any(cap.color):
            color_h, color_w = cap.color.shape[:2]
            break
        time.sleep(0.02)

    if color_w is None:
        # fallback：从 K 推不出分辨率，只能再等
        cap = k4a.get_capture()
        if cap.color is None:
            raise RuntimeError("Azure Kinect: cannot get color frame. Check device & permissions.")
        color_h, color_w = cap.color.shape[:2]

    print(f"[K4A] Started Azure Kinect DK: color={color_w}x{color_h}@{fps}, depth_mode={depth_mode}")
    print(f"[K4A] K_color: fx={K_color[0,0]:.2f}, fy={K_color[1,1]:.2f}, cx={K_color[0,2]:.2f}, cy={K_color[1,2]:.2f}")

    return k4a, K_color, (color_w, color_h)


def get_k4a_frame(k4a, out_size=None, require_aligned_depth=True):
    """
    return color_bgr(uint8 HxWx3), depth_mm(uint16 HxW)
    out_size: (W,H) optional, will resize both color and depth.
    require_aligned_depth: True 时优先要求 depth 已对齐到 color（transformed_depth）
    """
    cap = k4a.get_capture()
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


def scale_K_for_resize(K, src_wh, dst_wh):
    """If you resize image from src_wh(W,H) to dst_wh(W,H), scale K accordingly."""
    src_w, src_h = float(src_wh[0]), float(src_wh[1])
    dst_w, dst_h = float(dst_wh[0]), float(dst_wh[1])
    sx = dst_w / src_w
    sy = dst_h / src_h
    K2 = K.copy().astype(np.float32)
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] *= sx
    K2[1, 2] *= sy
    return K2


# =========================
# Main
# =========================
if __name__ == "__main__":
    # ---------- configs ----------
    addr = "tcp://127.0.0.1:5550"

    checkpoint = "../sam2.1_hiera_tiny.pt"
    model_cfg  = "configs/sam2.1/sam2.1_hiera_t.yaml"

    device = "cuda"   # "cuda" or "cpu"
    use_amp = True

    axis_len = 0.05

    # Azure Kinect config
    k4a_color_res = "720P"          # 720P/1080P/1536P/2160P
    k4a_depth_mode = "NFOV_UNBINNED"  # NFOV_UNBINNED/WFOV_UNBINNED/...
    k4a_fps = 30


    # ---------- init ----------
    sock = make_client(addr=addr, timeout_ms=10000)
    print(f"[ZMQ] connected: {addr}")

    segmenter = SamSegmenter(checkpoint=checkpoint, model_cfg=model_cfg, device=device, use_amp=use_amp)

    camera = AzureKinectDK(color_res=k4a_color_res, fps=k4a_fps, depth_mode=k4a_depth_mode)
    camera.start_init()
    # k4a, K_native, native_wh = start_azure_kinect(color_res=k4a_color_res, fps=k4a_fps, depth_mode=k4a_depth_mode)
    # if out_size is not None:
    #     K = scale_K_for_resize(K_native, src_wh=native_wh, dst_wh=out_size)
    #     print(f"[K4A] Resize enabled: {native_wh[0]}x{native_wh[1]} -> {out_size[0]}x{out_size[1]}")
    # else:
    #     K = K_native
    K = camera.K_color

    win = "Azure Kinect DK + SAM2 bbox prompt (s:init ROI | r:reset | q:quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    init_done = False
    i = 0

    prev_mask01 = None
    prev_box_xyxy = None
    last_T = None

    t_last = time.time()

    try:
        while True:
            # color, depth_mm = get_k4a_frame(k4a, out_size=out_size, require_aligned_depth=True)
            color, depth_mm = camera.get_k4a_frame(require_aligned_depth=True)
            if color is None:
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
            k4a.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        try:
            sock.close()
        except Exception:
            pass
        print("[EXIT] Done.")
