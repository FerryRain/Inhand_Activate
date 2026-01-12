"""
@FileName：tracker.py
@Description：
@Author：Ferry
@Time：2026 1/9/26 3:50 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""
import json
import math
import time
import inspect

import cv2
import numpy as np
import zmq
import torch

from Active.AzureKinectDK.Azure_camera import AzureKinectDK
from ultralytics.models.sam import SAM3VideoPredictor


# ------------------------------ Helpers ------------------------------
class _DummyVideoDataset:
    """Minimal dataset stub for Ultralytics SAM3 video predictor."""
    def __init__(self, frames: int = 1_000_000):
        self.mode = "video"
        self.frame = 0
        self.frames = frames


# ------------------------------ Tracker ------------------------------
class Tracker:
    def __init__(self,
                 axis_len=0.05, k4a_color_res="720P", k4a_depth_mode="NFOV_UNBINNED", k4a_fps=30,
                 bundletrack_addr="tcp://127.0.0.1:5550",

                 # -------- SAM3 params --------
                 sam3_model="/home/ferry/data/Code2/Research/Inhand_Activate/sam_model/sam3.pt",
                 sam3_conf=0.25,
                 sam3_imgsz=640,
                 sam3_half=True,
                 sam3_device="cuda",
                 sam3_compile_mode="default",   # ✅ 必须是字符串: default / reduce-overhead / max-autotune ...

                 # UI
                 show_tracker=True,
                 ):
        # ---------------- Camera ----------------
        self.axis_len = axis_len
        self.k4a_color_res = k4a_color_res
        self.k4a_depth_mode = k4a_depth_mode
        self.k4a_fps = k4a_fps
        self.show_tracker = show_tracker

        self.camera = AzureKinectDK(color_res=k4a_color_res, fps=k4a_fps, depth_mode=k4a_depth_mode)
        self.camera.start_init()
        self.K = self.camera.K_color.astype(np.float32)

        self.fx, self.fy, self.cx, self.cy = float(self.K[0, 0]), float(self.K[1, 1]), float(self.K[0, 2]), float(self.K[1, 2])

        # ---------------- ZMQ ----------------
        self.bundletrack_addr = bundletrack_addr
        self.sock = self.make_client(addr=self.bundletrack_addr, timeout_ms=10000)
        print(f"[ZMQ] connected: {self.bundletrack_addr}")

        # ---------------- SAM3 Video Predictor ----------------
        overrides = dict(
            conf=float(sam3_conf), task="segment", mode="predict",
            imgsz=int(sam3_imgsz),
            model=sam3_model,
            half=bool(sam3_half),
            save=False,
            verbose=False,
            compile=str(sam3_compile_mode),  # ✅ 修复你遇到的 mode=False 报错
        )
        self.video_pred = SAM3VideoPredictor(overrides=overrides)
        self.video_pred.setup_model()

        # 给 predictor 一个 “video 模式”dataset stub，允许我们用实时帧逐帧推进 frame index
        self.video_pred.dataset = _DummyVideoDataset(frames=1_000_000)
        self._reset_sam3_state()

        # cache inference signature for robust kwargs mapping
        self._infer_sig = inspect.signature(self.video_pred.inference)

        # ---------------- State ----------------
        self.init_done = False
        self.i = 0
        self.sam_frame = 0

        self.prev_mask01 = None
        self.prev_box_xyxy = None
        self.last_T = None
        self.t_last = time.time()

        # optional gate
        self.gate_mask01 = None
        self.gate_ready = False

        self.win = "Azure Kinect + SAM3(points init -> video track) | s:init | g:gate | r:reset | q:quit"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)

    # ====================== Init / Tracking ======================
    def init_tracker(self):
        self.color, self.depth_mm = self.camera.get_k4a_frame(require_aligned_depth=True)
        disp = self._make_disp(self.color)

        cv2.imshow(self.win, disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            self._shutdown()
            print("[EXIT] Done.")
            return "quit"

        if key == ord('g'):
            gate = self.select_gate_mask_polygon(self.color)
            if gate is None or gate.sum() == 0:
                print("[GATE] canceled / empty.")
                return
            self.gate_mask01 = gate.astype(np.uint8)
            self.gate_ready = True
            print(f"[GATE] ok. pixels={int(self.gate_mask01.sum())}")
            return

        if (key == ord('s')) and (not self.init_done):
            # 1) 选点：LMB 正点，RMB 负点
            pts, lbls = self.select_points(self.color)
            if pts is None or lbls is None:
                print("[INIT] point selection canceled.")
                return
            if int(np.sum(lbls == 1)) == 0:
                print("[INIT] need at least one positive point.")
                return

            # 2) reset SAM3 state, then bootstrap on frame 0 using points
            self._reset_sam3_state()
            self.sam_frame = 0

            try:
                mask01 = self._sam3_bootstrap_points(self.color, pts, lbls)
            except Exception as e:
                print(f"[INIT] SAM3 bootstrap(points) failed: {e}")
                return

            if mask01 is None or int(mask01.sum()) == 0:
                print("[INIT] got empty mask from points; try adding more positive/negative points.")
                return

            if self.gate_mask01 is not None:
                mask01 = (mask01 & self.gate_mask01).astype(np.uint8)

            self.prev_mask01 = mask01
            bb = self.bbox_from_mask(mask01, pad=5)
            self.prev_box_xyxy = [bb[0], bb[2], bb[1], bb[3]] if bb is not None else None

            # 3) init bundletrack
            center3d = self.compute_center3d_from_mask_depth_mm(
                mask01, self.depth_mm, self.fx, self.fy, self.cx, self.cy
            )
            has_init = (center3d is not None)
            ob_in_cam = self.make_ob_in_cam_from_center(center3d) if has_init else None
            if has_init:
                print(f"[INIT] center3D(m): X={center3d[0]:.4f}, Y={center3d[1]:.4f}, Z={center3d[2]:.4f}")
            else:
                print("[INIT] center3D failed; send init without ob_in_cam.")

            self.T, resp = self.send_frame_with_sock(
                self.sock,
                color_bgr=self.color,
                depth_mm=self.depth_mm,
                mask=mask01,
                id_str=f"{self.i:06d}",
                roi=None,
                has_init=has_init,
                ob_in_cam=ob_in_cam,
            )
            self.last_T = self.T
            self.init_done = True

            self.i += 1
            self.sam_frame += 1
            print(f"[SEND][INIT] ok={resp.get('ok')}")
            return

    def tracking(self):
        if not self.init_done:
            return

        self.color, self.depth_mm = self.camera.get_k4a_frame(require_aligned_depth=True)

        if self.show_tracker:
            disp = self._make_disp(self.color)
            cv2.imshow(self.win, disp)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                self._shutdown()
                print("[EXIT] Done.")
                return "quit"

            if key == ord('r'):
                print("[RESET] reset (keep gate). Press 's' to init again.")
                self.init_done = False
                self.prev_mask01 = None
                self.prev_box_xyxy = None
                self.last_T = None
                self.i = 0
                self.sam_frame = 0
                self._reset_sam3_state()
                return

            if key == ord('g'):
                gate = self.select_gate_mask_polygon(self.color)
                if gate is None or gate.sum() == 0:
                    print("[GATE] canceled / empty.")
                else:
                    self.gate_mask01 = gate.astype(np.uint8)
                    self.gate_ready = True
                    print(f"[GATE] ok. pixels={int(self.gate_mask01.sum())}")
                return

        # SAM3 video step
        try:
            mask01 = self._sam3_step(self.color, self.sam_frame)
        except Exception as e:
            print(f"[WARN] SAM3 step failed at i={self.i} sam_frame={self.sam_frame}: {e}")
            mask01 = None

        if mask01 is None or int(mask01.sum()) == 0:
            mask01 = self.prev_mask01

        if mask01 is not None and self.gate_mask01 is not None:
            mask01 = (mask01 & self.gate_mask01).astype(np.uint8)

        self.prev_mask01 = mask01

        # send to bundletrack
        self.T, resp = self.send_frame_with_sock(
            self.sock,
            color_bgr=self.color,
            depth_mm=self.depth_mm,
            mask=self.prev_mask01,
            id_str=f"{self.i:06d}",
            roi=None,
            has_init=False,
            ob_in_cam=None,
        )
        self.last_T = self.T

        self.i += 1
        self.sam_frame += 1

    # ====================== SAM3 Internals ======================
    def _reset_sam3_state(self):
        # important: reset inference_state for new sequence
        self.video_pred.inference_state = self.video_pred._init_state(num_frames=1_000_000)
        self.video_pred.prompts = {}
        self.video_pred.dataset.frame = 0

    def _prep_im(self, bgr: np.ndarray):
        # set batch to let predictor know original image size
        self.video_pred.batch = (None, [bgr], None, None, None)
        im = self.video_pred.preprocess(self.video_pred.pre_transform([bgr]))
        return im

    def _call_inference(self, im, **kwargs):
        """Call inference with only supported kwargs (robust across versions)."""
        call_kwargs = {}
        for k, v in kwargs.items():
            if v is None:
                continue
            if k in self._infer_sig.parameters:
                call_kwargs[k] = v
        return self.video_pred.inference(im, **call_kwargs)

    def _sam3_bootstrap_points(self, bgr: np.ndarray, pts_xy: np.ndarray, lbls: np.ndarray) -> np.ndarray | None:
        """
        Bootstrap at current frame index using point prompts.
        pts_xy: (N,2) float32 in pixel coords (x,y)
        lbls: (N,) int64 where 1=positive, 0=negative
        """
        im = self._prep_im(bgr)
        dev = im.device

        # common SAM-style shapes: (1,N,2) and (1,N)
        pts_t = torch.from_numpy(pts_xy.astype(np.float32))[None, ...].to(dev)
        lbl_t = torch.from_numpy(lbls.astype(np.int64))[None, ...].to(dev)

        self.video_pred.dataset.frame = int(self.sam_frame)

        # try multiple kw names depending on ultralytics version
        kw = {}
        if "points" in self._infer_sig.parameters:
            kw["points"] = pts_t
        elif "point_coords" in self._infer_sig.parameters:
            kw["point_coords"] = pts_t

        if "labels" in self._infer_sig.parameters:
            kw["labels"] = lbl_t
        elif "point_labels" in self._infer_sig.parameters:
            kw["point_labels"] = lbl_t

        with torch.inference_mode():
            pred_masks, pred_scores = self._call_inference(im, **kw)

        return self._mask01_from_pred_masks(pred_masks, bgr.shape[:2])

    def _sam3_step(self, bgr: np.ndarray, frame_idx: int) -> np.ndarray | None:
        im = self._prep_im(bgr)
        self.video_pred.dataset.frame = int(frame_idx)

        with torch.inference_mode():
            pred_masks, pred_scores = self.video_pred.inference(im)

        return self._mask01_from_pred_masks(pred_masks, bgr.shape[:2])

    def _mask01_from_pred_masks(self, pred_masks, hw) -> np.ndarray | None:
        if pred_masks is None or len(pred_masks) == 0:
            return None
        mk = pred_masks[0].detach().float().cpu().numpy()
        mk01 = (mk > 0.5).astype(np.uint8)
        H, W = hw
        if mk01.shape != (H, W):
            mk01 = cv2.resize(mk01, (W, H), interpolation=cv2.INTER_NEAREST)
        return mk01

    # ====================== Point UI ======================
    def select_points(self, bgr, win_name="Select points: LMB=+  RMB=-  u=undo  c=clear  ENTER=OK  ESC=cancel"):
        """
        Return:
          pts_xy: (N,2) float32 (x,y)
          lbls:   (N,) int64  (1=positive, 0=negative)
        """
        H, W = bgr.shape[:2]
        pts = []   # list of (x,y,label)
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        def redraw():
            vis = bgr.copy()
            cv2.putText(vis, win_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            for (x, y, lb) in pts:
                col = (0, 255, 0) if lb == 1 else (0, 0, 255)
                cv2.circle(vis, (x, y), 5, col, -1, cv2.LINE_AA)
            return vis

        def on_mouse(event, x, y, flags, param):
            nonlocal pts
            if event == cv2.EVENT_LBUTTONDOWN:
                pts.append((int(x), int(y), 1))
            elif event == cv2.EVENT_RBUTTONDOWN:
                pts.append((int(x), int(y), 0))

        cv2.setMouseCallback(win_name, on_mouse)

        while True:
            vis = redraw()
            cv2.imshow(win_name, vis)
            k = cv2.waitKey(20) & 0xFF
            if k in (13, 10, 32):  # Enter/Space
                if len(pts) == 0:
                    print("[POINT] add at least one point.")
                    continue
                break
            if k == 27:  # ESC
                cv2.destroyWindow(win_name)
                return None, None
            if k == ord('u') and len(pts) > 0:
                pts.pop()
            if k == ord('c'):
                pts = []

        cv2.destroyWindow(win_name)

        pts_xy = np.array([[p[0], p[1]] for p in pts], dtype=np.float32)
        lbls = np.array([p[2] for p in pts], dtype=np.int64)
        return pts_xy, lbls

    # ====================== Gate polygon UI ======================
    def select_gate_mask_polygon(self, bgr, win_name="Select GATE polygon: LMB=add  u=undo  c=clear  ENTER=OK  ESC=cancel"):
        H, W = bgr.shape[:2]
        pts = []
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        def redraw():
            vis = bgr.copy()
            cv2.putText(vis, win_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            if len(pts) > 0:
                for p in pts:
                    cv2.circle(vis, p, 4, (0, 255, 255), -1, cv2.LINE_AA)
                if len(pts) >= 2:
                    cv2.polylines(vis, [np.array(pts, np.int32)], False, (0, 255, 255), 2, cv2.LINE_AA)
            return vis

        def on_mouse(event, x, y, flags, param):
            nonlocal pts
            if event == cv2.EVENT_LBUTTONDOWN:
                pts.append((int(x), int(y)))

        cv2.setMouseCallback(win_name, on_mouse)

        while True:
            vis = redraw()
            cv2.imshow(win_name, vis)
            k = cv2.waitKey(20) & 0xFF
            if k in (13, 10, 32):
                if len(pts) < 3:
                    print("[GATE] Need at least 3 vertices.")
                    continue
                break
            if k == 27:
                cv2.destroyWindow(win_name)
                return None
            if k == ord('u') and len(pts) > 0:
                pts.pop()
            if k == ord('c'):
                pts = []

        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(pts, np.int32)], 1)
        cv2.destroyWindow(win_name)
        return mask

    # ====================== Visualization ======================
    def _make_disp(self, bgr: np.ndarray) -> np.ndarray:
        disp = bgr.copy()

        if self.gate_mask01 is not None:
            gm = self.gate_mask01 > 0
            if gm.any():
                overlay = disp.copy()
                overlay[gm] = (255, 0, 0)
                disp = cv2.addWeighted(overlay, 0.12, disp, 0.88, 0)

        if self.prev_mask01 is not None:
            disp = self.overlay_mask(disp, self.prev_mask01)

        if self.last_T is not None:
            self.draw_pose6d_on_image(disp, self.last_T, self.K, axis_len=self.axis_len, thickness=2, text_org=(5, 60))

        now = time.time()
        fps_show = 1.0 / max(1e-6, (now - self.t_last))
        self.t_last = now
        cv2.putText(disp, f"fps={fps_show:.1f}  init={self.init_done}  i={self.i}  sam_frame={self.sam_frame}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        return disp

    # ====================== ZMQ ======================
    def make_client(self, addr: str = "tcp://127.0.0.1:5550", timeout_ms: int = 10000):
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REQ)
        sock.connect(addr)
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        sock.setsockopt(zmq.LINGER, 0)
        return sock

    def _shutdown(self):
        try:
            self.sock.close()
        except Exception:
            pass
        for fn in ["stop", "close", "shutdown"]:
            if hasattr(self.camera, fn):
                try:
                    getattr(self.camera, fn)()
                    break
                except Exception:
                    pass

    def send_frame_with_sock(self,
                             sock,
                             color_bgr: np.ndarray,
                             depth_mm: np.ndarray,
                             mask: np.ndarray | None = None,
                             id_str: str = "000000",
                             roi=None,
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

        if roi is not None:
            header["roi"] = [float(x) for x in roi]
        elif mask01 is not None:
            auto_roi = self.bbox_from_mask(mask01, pad=5)
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

    # ====================== Overlay / Pose ======================
    def overlay_mask(self, bgr, mask01, alpha=0.45):
        if mask01 is None:
            return bgr
        vis = bgr.copy()
        m = mask01 > 0
        if m.any():
            overlay = vis.copy()
            overlay[m] = (0, 255, 0)
            vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
        return vis

    def draw_pose6d_on_image(self, img_bgr, T_obj_in_cam, K, axis_len=0.05, thickness=2, text_org=(5, 60)):
        R = T_obj_in_cam[:3, :3].astype(np.float32)
        t = T_obj_in_cam[:3, 3].astype(np.float32)

        O = t
        X = t + R @ np.array([axis_len, 0, 0], dtype=np.float32)
        Y = t + R @ np.array([0, axis_len, 0], dtype=np.float32)
        Z = t + R @ np.array([0, 0, axis_len], dtype=np.float32)

        o2 = self.project_cam_point(O, K)
        if o2 is not None:
            o2i = (int(round(o2[0])), int(round(o2[1])))
            cv2.circle(img_bgr, o2i, 3, (0, 255, 255), -1, cv2.LINE_AA)
            for P, col in [(X, (0, 0, 255)), (Y, (0, 255, 0)), (Z, (255, 0, 0))]:
                p2 = self.project_cam_point(P, K)
                if p2 is not None:
                    p2i = (int(round(p2[0])), int(round(p2[1])))
                    cv2.line(img_bgr, o2i, p2i, col, thickness, cv2.LINE_AA)

        roll, pitch, yaw = self.rpy_from_R_zyx(R)
        ss1 = f"t = [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]"
        ss2 = f"rpy(deg) = [{roll:.2f}, {pitch:.2f}, {yaw:.2f}]"
        x0, y0 = int(text_org[0]), int(text_org[1])
        cv2.putText(img_bgr, ss1, (x0, y0), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_bgr, ss2, (x0, y0 + 22), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 1, cv2.LINE_AA)

    def project_cam_point(self, Pc, K):
        z = float(Pc[2])
        if z <= 1e-6:
            return None
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        u = fx * (float(Pc[0]) / z) + cx
        v = fy * (float(Pc[1]) / z) + cy
        return (u, v)

    def rpy_from_R_zyx(self, R):
        r00, r01, r02 = R[0, 0], R[0, 1], R[0, 2]
        r10, r11, r12 = R[1, 0], R[1, 1], R[1, 2]
        r20, r21, r22 = R[2, 0], R[2, 1], R[2, 2]
        yaw = math.atan2(r10, r00)
        pitch = math.atan2(-r20, math.sqrt(r21 * r21 + r22 * r22))
        roll = math.atan2(r21, r22)
        return (roll * 180.0 / math.pi, pitch * 180.0 / math.pi, yaw * 180.0 / math.pi)

    def compute_center3d_from_mask_depth_mm(self, mask01, depth_mm, fx, fy, cx, cy):
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

    def make_ob_in_cam_from_center(self, center3d_m):
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = center3d_m.reshape(3)
        return T

    def bbox_from_mask(self, mask01: np.ndarray, pad: int = 5):
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


if __name__ == '__main__':
    tracker = Tracker()

    while True:
        if tracker.init_done:
            ret = tracker.tracking()
            if ret == "quit":
                break
        else:
            ret = tracker.init_tracker()
            if ret == "quit":
                break
