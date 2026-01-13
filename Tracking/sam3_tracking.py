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
from typing import Optional

import cv2
import numpy as np
import zmq

from Active.AzureKinectDK.Azure_camera import AzureKinectDK

# -------- SAM3 (Ultralytics) --------
from ultralytics.models.sam import SAM3SemanticPredictor


class Sam3TextSegmenter:
    """
    一个很薄的封装：输入一张 BGR 图 + text prompt，输出单个 mask01 (HxW, uint8 {0,1})
    - 支持 gate_mask01：只在 gate 内保留结果（AND）
    - 支持 prev_mask01：用 IoU 做“稳定选择”（在多实例时挑最像上一帧的那个）
    """

    def __init__(
        self,
        model_path: str,
        imgsz: int = 640,
        conf: float = 0.25,
        device: str = "cuda",
        half: bool = True,
        verbose: bool = False,
        save: bool = False,  # 实时建议 False
    ):
        self.model_path = model_path
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.device = device
        self.half = bool(half)
        self.verbose = bool(verbose)
        self.save = bool(save)

        overrides = dict(
            conf=self.conf,
            task="segment",
            mode="predict",
            imgsz=self.imgsz,
            model=self.model_path,
            device=self.device,
            half=self.half,
            verbose=self.verbose,
            save=self.save,
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
        self._tmp_img_path = "/tmp/sam3_tmp_frame.jpg"

    @staticmethod
    def _iou(a01: np.ndarray, b01: np.ndarray) -> float:
        if a01 is None or b01 is None:
            return 0.0
        a = a01.astype(bool)
        b = b01.astype(bool)
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return float(inter) / float(union) if union > 0 else 0.0

    @staticmethod
    def _extract_masks_from_results(results):
        """
        兼容式解析：
        - 常见：results 是 list[Results]，results[0].masks.data 是 (N,H,W)
        - 或 results 直接是 Results
        返回 numpy: (N,H,W) bool/uint8
        """
        r = None
        if results is None:
            return None

        if hasattr(results, "masks"):
            r = results
        elif isinstance(results, (list, tuple)) and len(results) > 0 and hasattr(results[0], "masks"):
            r = results[0]

        if r is None or r.masks is None:
            return None

        # Ultralytics Results.masks.data: torch.Tensor (N,H,W)
        data = getattr(r.masks, "data", None)
        if data is None:
            return None

        try:
            masks = data.detach().float().cpu().numpy()
        except Exception:
            try:
                masks = data.cpu().numpy()
            except Exception:
                return None

        # 转成 bool
        masks = masks > 0.5
        return masks

    def segment_from_text(
        self,
        bgr: np.ndarray,
        text_prompt: str,
        prev_mask01: Optional[np.ndarray] = None,
        gate_mask01: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        返回 mask01 (HxW uint8 {0,1}) 或 None
        """
        if bgr is None:
            return None

        # 1) set_image：优先直接喂 numpy；如果你的 ultralytics 版本不支持，就落回写 tmp 文件
        try:
            self.predictor.set_image(bgr)
        except Exception:
            cv2.imwrite(self._tmp_img_path, bgr)
            self.predictor.set_image(self._tmp_img_path)

        # 2) 推理：文本概念分割（可能返回多实例 masks）
        results = self.predictor(text=[text_prompt])

        masks = self._extract_masks_from_results(results)
        if masks is None or masks.shape[0] == 0:
            return None

        H, W = bgr.shape[:2]
        gate = None
        if gate_mask01 is not None:
            gate = (gate_mask01.astype(np.uint8) > 0)
            if gate.shape != (H, W):
                gate = cv2.resize(gate.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

        # 3) 多实例选择策略：
        #    - 有 gate：只保留在 gate 内有像素的候选；并最终 AND gate
        #    - 有 prev：在候选中挑 IoU 最大
        #    - 否则：挑 gate 内面积最大（或整体面积最大）
        best = None
        best_iou = -1.0
        best_area = -1

        for k in range(masks.shape[0]):
            mk = masks[k]
            if mk.shape != (H, W):
                mk = cv2.resize(mk.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

            if gate is not None:
                mk_in = np.logical_and(mk, gate)
                area_in = int(mk_in.sum())
                if area_in <= 0:
                    continue
                mk_use = mk_in
                area_use = area_in
            else:
                mk_use = mk
                area_use = int(mk_use.sum())
                if area_use <= 0:
                    continue

            if prev_mask01 is not None:
                iou = self._iou(mk_use.astype(np.uint8), prev_mask01.astype(np.uint8))
                if iou > best_iou:
                    best_iou = iou
                    best_area = area_use
                    best = mk_use
            else:
                # 没有 prev，就挑面积最大的
                if area_use > best_area:
                    best_area = area_use
                    best = mk_use

        if best is None:
            return None

        return best.astype(np.uint8)


class Tracker:
    def __init__(
        self,
        axis_len=0.05,
        k4a_color_res="720P",
        k4a_depth_mode="NFOV_UNBINNED",
        k4a_fps=30,
        bundletrack_addr="tcp://127.0.0.1:5550",
        # -------- SAM3 params --------
        sam3_model="/home/ferry/data/Code2/Research/Inhand_Activate/sam_model/sam3.pt",
        sam3_device="cuda",
        sam3_imgsz=640,
        sam3_conf=0.25,
        sam3_half=True,
        # 文本提示
        text_prompt="An green object with colorful stickers attached",
        # UI
        show_tracker=True,
    ):
        # set AzureKinectDK camera
        self.axis_len = axis_len
        self.k4a_color_res = k4a_color_res
        self.k4a_depth_mode = k4a_depth_mode
        self.k4a_fps = k4a_fps
        self.show_tracker = show_tracker

        self.camera = AzureKinectDK(color_res=k4a_color_res, fps=k4a_fps, depth_mode=k4a_depth_mode)
        self.camera.start_init()
        self.K = self.camera.K_color.astype(np.float32)

        # set bundletrack zmq
        self.bundletrack_addr = bundletrack_addr
        self.sock = self.make_client(addr=self.bundletrack_addr, timeout_ms=10000)
        print(f"[ZMQ] connected: {self.bundletrack_addr}")

        # set SAM3 segmenter
        self.text_prompt = text_prompt
        self.segmenter = Sam3TextSegmenter(
            model_path=sam3_model,
            imgsz=int(sam3_imgsz),
            conf=float(sam3_conf),
            device=str(sam3_device),
            half=bool(sam3_half),
            verbose=False,
            save=False,  # 实时建议 False
        )
        print(f"[SAM3] model={sam3_model} device={sam3_device} prompt='{self.text_prompt}'")

        # tracking params
        self.init_done = False
        self.i = 0
        self.prev_mask01 = None
        self.prev_box_xyxy = None
        self.last_T = None
        self.t_last = time.time()

        self.gate_mask01 = None
        self.gate_ready = False

        self.win = "Azure Kinect DK + SAM3(text) (g:gate | s:init | r:reset | q:quit)"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)

        self.fx, self.fy, self.cx, self.cy = (
            float(self.K[0, 0]),
            float(self.K[1, 1]),
            float(self.K[0, 2]),
            float(self.K[1, 2]),
        )

    def init_tracker(self):
        self.color, self.depth_mm = self.camera.get_k4a_frame(require_aligned_depth=True)
        disp = self.color.copy()

        # 可视化 gate（红色）+ prev mask（绿色）
        if self.gate_mask01 is not None:
            disp = self.overlay_mask_color(disp, self.gate_mask01, color=(0, 0, 255), alpha=0.25)
        if self.prev_mask01 is not None:
            disp = self.overlay_mask_color(disp, self.prev_mask01, color=(0, 255, 0), alpha=0.45)
        if self.last_T is not None:
            self.draw_pose6d_on_image(disp, self.last_T, self.K, axis_len=self.axis_len, thickness=2, text_org=(5, 60))

        now = time.time()
        fps_show = 1.0 / max(1e-6, (now - self.t_last))
        self.t_last = now
        cv2.putText(
            disp,
            f"fps={fps_show:.1f}  init={self.init_done}  i={self.i}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
        )

        cv2.imshow(self.win, disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            self._shutdown()
            print("[EXIT] Done.")
            return "quit"

        # g：选择 gate polygon mask（可重复选择）
        if key == ord("g"):
            print("[GATE] Select a gate mask (polygon).")
            gate = self.select_gate_mask_polygon(self.color)
            if gate is None or gate.sum() == 0:
                print("[GATE] canceled / empty.")
                return
            self.gate_mask01 = gate
            self.gate_ready = True
            print(f"[GATE] ok. pixels={int(gate.sum())}")
            return

        # s：初始化（需要先有 gate；没有就先让你选）
        if (key == ord("s")) and (not self.init_done):
            if self.gate_mask01 is None:
                print("[GATE] Please select a gate mask (polygon) first.")
                gate = self.select_gate_mask_polygon(self.color)
                if gate is None or gate.sum() == 0:
                    print("[GATE] canceled / empty.")
                    return
                self.gate_mask01 = gate
                self.gate_ready = True
                print(f"[GATE] ok. pixels={int(gate.sum())}")

            print(f"[INIT] SAM3 text prompt = '{self.text_prompt}'")

            try:
                t0 = time.time()
                mask01 = self.segmenter.segment_from_text(
                    self.color,
                    self.text_prompt,
                    prev_mask01=None,
                    gate_mask01=self.gate_mask01,
                )
                infer_ms = (time.time() - t0) * 1000.0
            except Exception as e:
                print(f"[INIT] SAM3 failed: {e}")
                return

            if mask01 is None or int(mask01.sum()) == 0:
                print("[INIT] Empty mask from SAM3 text prompt. Consider changing prompt.")
                return

            self.mask01 = mask01
            print(f"[INIT] SAM3 ok. infer={infer_ms:.1f}ms, mask_pixels={int(self.mask01.sum())}")

            center3d = self.compute_center3d_from_mask_depth_mm(
                self.mask01, self.depth_mm, self.fx, self.fy, self.cx, self.cy
            )
            self.has_init = center3d is not None
            self.ob_in_cam = self.make_ob_in_cam_from_center(center3d) if self.has_init else None
            if self.has_init:
                print(f"[INIT] center3D(m): X={center3d[0]:.4f}, Y={center3d[1]:.4f}, Z={center3d[2]:.4f}")
            else:
                print("[INIT] center3D failed; send init without ob_in_cam.")

            self.id_str = f"{self.i:06d}"
            t_send0 = time.time()
            self.T, resp = self.send_frame_with_sock(
                self.sock,
                color_bgr=self.color,
                depth_mm=self.depth_mm,
                mask=self.mask01,
                id_str=self.id_str,
                roi=None,
                has_init=self.has_init,
                ob_in_cam=self.ob_in_cam,
            )
            print(f"[SEND][INIT] i={self.i} ok={resp.get('ok')} cost={time.time() - t_send0:.3f}s")

            self.init_done = True
            self.prev_mask01 = self.mask01

            bb = self.bbox_from_mask(self.mask01, pad=5)
            self.prev_box_xyxy = [bb[0], bb[2], bb[1], bb[3]] if bb is not None else None

            self.last_T = self.T
            self.i += 1

        return

    def tracking(self):
        if not self.init_done:
            return

        self.color, self.depth_mm = self.camera.get_k4a_frame(require_aligned_depth=True)

        if self.show_tracker:
            disp = self.color.copy()

            if self.gate_mask01 is not None:
                disp = self.overlay_mask_color(disp, self.gate_mask01, color=(0, 0, 255), alpha=0.25)
            if self.prev_mask01 is not None:
                disp = self.overlay_mask_color(disp, self.prev_mask01, color=(0, 255, 0), alpha=0.45)
            if self.last_T is not None:
                self.draw_pose6d_on_image(
                    disp, self.last_T, self.K, axis_len=self.axis_len, thickness=2, text_org=(5, 60)
                )

            now = time.time()
            fps_show = 1.0 / max(1e-6, (now - self.t_last))
            self.t_last = now
            cv2.putText(
                disp,
                f"fps={fps_show:.1f}  init={self.init_done}  i={self.i}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
            )

            cv2.imshow(self.win, disp)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                self._shutdown()
                print("[EXIT] Done.")
                return "quit"
            if key == ord("r"):
                return "reconstruct"
            if key == ord("a"):
                return "active"
            if key == ord("g"):
                print("[GATE] Re-select gate mask (polygon).")
                gate = self.select_gate_mask_polygon(self.color)
                if gate is None or gate.sum() == 0:
                    print("[GATE] canceled / empty.")
                else:
                    self.gate_mask01 = gate
                    self.gate_ready = True
                    print(f"[GATE] ok. pixels={int(gate.sum())}")
                return

        # --- SAM3 text prompt segmentation ---
        try:
            t0 = time.time()
            mask01 = self.segmenter.segment_from_text(
                self.color,
                self.text_prompt,
                prev_mask01=self.prev_mask01,
                gate_mask01=self.gate_mask01,
            )
            infer_ms = (time.time() - t0) * 1000.0
        except Exception as e:
            print(f"[WARN] SAM3 failed at i={self.i}: {e}")
            mask01 = self.prev_mask01
            infer_ms = -1.0

        if mask01 is None or int(mask01.sum()) == 0:
            mask01 = self.prev_mask01

        self.mask01 = mask01

        if self.mask01 is not None:
            bb = self.bbox_from_mask(self.mask01, pad=5)
            if bb is not None:
                self.prev_box_xyxy = [bb[0], bb[2], bb[1], bb[3]]
            self.prev_mask01 = self.mask01

        self.id_str = f"{self.i:06d}"
        t_send0 = time.time()
        self.T, resp = self.send_frame_with_sock(
            self.sock,
            color_bgr=self.color,
            depth_mm=self.depth_mm,
            mask=self.mask01,
            id_str=self.id_str,
            roi=None,
            has_init=False,
            ob_in_cam=None,
        )

        self.last_T = self.T
        # print(f"[SEND] i={self.i:06d} ok={resp.get('ok')} infer_ms={infer_ms:.1f} send_cost={time.time() - t_send0:.3f}s")

        self.i += 1

    def select_gate_mask_polygon(self, bgr, win_name="Select GATE mask (polygon)"):
        """
        左键：添加多边形顶点
        u：撤销最后一个点
        c：清空
        Enter/Space：完成并生成mask
        ESC：取消
        """
        H, W = bgr.shape[:2]
        pts = []

        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        def redraw():
            vis = bgr.copy()
            cv2.putText(
                vis,
                "LMB=add | u=undo | c=clear | ENTER/SPACE=OK | ESC=cancel",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
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
            if k in (13, 10, 32):  # Enter/Space
                if len(pts) < 3:
                    print("[GATE] Need at least 3 vertices.")
                    continue
                break
            if k == 27:  # ESC
                cv2.destroyWindow(win_name)
                return None
            if k == ord("u"):
                if len(pts) > 0:
                    pts.pop()
            if k == ord("c"):
                pts = []

        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(pts, np.int32)], 1)
        cv2.destroyWindow(win_name)
        return mask

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

    def make_client(self, addr: str = "tcp://127.0.0.1:5550", timeout_ms: int = 10000):
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REQ)
        sock.connect(addr)
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        sock.setsockopt(zmq.LINGER, 0)
        return sock

    def overlay_mask_color(self, bgr, mask01, color=(0, 255, 0), alpha=0.45):
        if mask01 is None:
            return bgr
        vis = bgr.copy()
        m = mask01 > 0
        if m.any():
            overlay = vis.copy()
            overlay[m] = color
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

    def compute_center3d_from_mask_depth_mm(self, mask01, depth_mm, fx, fy, cx, cy):
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

    def make_ob_in_cam_from_center(self, center3d_m):
        T = np.eye(4, dtype=np.float32)
        cos = np.cos(115.0 * math.pi / 180.0)
        sin = np.sin(115.0 * math.pi / 180.0)
        rotation_x = np.array(
            [
                [1, 0, 0],
                [0, cos, -sin],
                [0, sin, cos],
            ],
            dtype=np.float32,
        )
        T[:3, :3] = rotation_x
        T[:3, 3] = center3d_m.reshape(3)
        return T

    def bbox_from_mask(self, mask01: np.ndarray, pad: int = 5):
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

    def send_frame_with_sock(
        self,
        sock,
        color_bgr: np.ndarray,  # uint8 HxWx3
        depth_mm: np.ndarray,  # uint16 HxW
        mask: np.ndarray | None = None,  # uint8 HxW (0/1 or 0/255)
        id_str: str = "000000",
        roi=None,  # [x0,x1,y0,y1]
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


if __name__ == "__main__":
    tracker = Tracker()

    while True:
        if tracker.init_done:
            quit_flag = tracker.tracking()
            if quit_flag == "quit":
                break
        else:
            quit_flag = tracker.init_tracker()
            if quit_flag == "quit":
                break
