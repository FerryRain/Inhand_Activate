#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@FileName：tracking_offline.py
@Description：Record one frame per call (to disk), then offline tracking from disk with resume.
@Author：Ferry
@Time：2026/01/19
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""

import os
import re
import json
import math
import time
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np
import zmq

from Active.AzureKinectDK.Azure_camera import AzureKinectDK
from ultralytics.models.sam import SAM3SemanticPredictor


# ------------------------ Small IO helpers ------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def safe_imwrite(path: str, img: np.ndarray, params: Optional[List[int]] = None):
    """
    Safe-ish write: write to tmp with SAME extension, then os.replace().
    Fixes OpenCV error "could not find a writer for .tmp".
    """
    root, ext = os.path.splitext(path)
    if ext == "":
        raise ValueError(f"safe_imwrite: path has no extension: {path}")
    tmp = root + ".tmp" + ext
    ok = cv2.imwrite(tmp, img, params or [])
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {tmp}")
    os.replace(tmp, path)

def safe_imread_color(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img

def safe_imread_depth_u16(path: str) -> np.ndarray:
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"Failed to read depth: {path}")
    if d.dtype != np.uint16:
        # 有些情况下会读成 uint8 或 int32，直接转成 uint16
        d = d.astype(np.uint16, copy=False)
    return d


# ------------------------ SAM3 (Ultralytics) ------------------------

class Sam3TextSegmenter:
    """
    输入一张 BGR 图 + text prompt，输出单个 mask01 (HxW, uint8 {0,1})
    - 支持 gate_mask01：只在 gate 内保留结果（AND）
    - 支持 prev_mask01：用 IoU 做“稳定选择”
    """

    def __init__(
        self,
        model_path: str,
        imgsz: int = 640,
        conf: float = 0.25,
        device: str = "cuda",
        half: bool = True,
        verbose: bool = False,
        save: bool = False,
    ):
        overrides = dict(
            conf=float(conf),
            task="segment",
            mode="predict",
            imgsz=int(imgsz),
            model=str(model_path),
            device=str(device),
            half=bool(half),
            verbose=bool(verbose),
            save=bool(save),
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
        r = None
        if results is None:
            return None

        if hasattr(results, "masks"):
            r = results
        elif isinstance(results, (list, tuple)) and len(results) > 0 and hasattr(results[0], "masks"):
            r = results[0]

        if r is None or r.masks is None:
            return None

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

        masks = masks > 0.5
        return masks

    def segment_from_text(
        self,
        bgr: np.ndarray,
        text_prompt: str,
        prev_mask01: Optional[np.ndarray] = None,
        gate_mask01: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        if bgr is None:
            return None

        # set_image: 优先 numpy；不支持则落回写 tmp 文件
        try:
            self.predictor.set_image(bgr)
        except Exception:
            cv2.imwrite(self._tmp_img_path, bgr)
            self.predictor.set_image(self._tmp_img_path)

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
                if area_use > best_area:
                    best_area = area_use
                    best = mk_use

        if best is None:
            return None
        return best.astype(np.uint8)


# ------------------------ Disk-based Recorder + Offline Tracking ------------------------

class OfflineDiskRecorderTracker:
    """
    落盘版本（不吃内存）：
    - record_frame(): 每调用一次录制一帧 -> 写本地 color/depth
    - draw_gate(): 在已录制帧上画 gate -> 保存 gate.png
    - track(): 从本地读取，第一次 idx=0 init；后续从 last_tracked_idx 开始（用 last_T 作为 init）继续
    """

    def __init__(
        self,
        root_dir: str,
        axis_len=0.05,
        k4a_color_res="720P",
        k4a_depth_mode="NFOV_UNBINNED",
        k4a_fps=30,
        bundletrack_addr="tcp://127.0.0.1:5550",
        # SAM3
        sam3_model="/home/ferry/data/Code2/Research/Inhand_Activate/sam_model/sam3.pt",
        sam3_device="cuda",
        sam3_imgsz=640,
        sam3_conf=0.25,
        sam3_half=True,
        text_prompt="An green object with colorful stickers attached",
        show_ui=True,
        jpg_quality: int = 90,
    ):
        self.axis_len = float(axis_len)
        self.show_ui = bool(show_ui)
        self.jpg_quality = int(jpg_quality)

        # directories
        self.root_dir = os.path.abspath(root_dir)
        self.color_dir = os.path.join(self.root_dir, "color")
        self.depth_dir = os.path.join(self.root_dir, "depth")
        self.state_path = os.path.join(self.root_dir, "state.json")
        self.gate_path = os.path.join(self.root_dir, "gate.png")
        ensure_dir(self.root_dir)
        ensure_dir(self.color_dir)
        ensure_dir(self.depth_dir)

        # Azure Kinect
        self.camera = AzureKinectDK(color_res=k4a_color_res, fps=k4a_fps, depth_mode=k4a_depth_mode)
        self.camera.start_init()
        self.K = self.camera.K_color.astype(np.float32)
        self.fx, self.fy, self.cx, self.cy = (
            float(self.K[0, 0]),
            float(self.K[1, 1]),
            float(self.K[0, 2]),
            float(self.K[1, 2]),
        )

        # ZMQ
        self.bundletrack_addr = str(bundletrack_addr)
        self.sock = self.make_client(addr=self.bundletrack_addr, timeout_ms=10000)
        print(f"[ZMQ] connected: {self.bundletrack_addr}")

        # SAM3
        self.text_prompt = str(text_prompt)
        self.segmenter = Sam3TextSegmenter(
            model_path=sam3_model,
            imgsz=int(sam3_imgsz),
            conf=float(sam3_conf),
            device=str(sam3_device),
            half=bool(sam3_half),
            verbose=False,
            save=False,
        )
        print(f"[SAM3] model={sam3_model} device={sam3_device} prompt='{self.text_prompt}'")

        # state
        self.idx_next: int = 0
        self.init_done: bool = False
        self.last_tracked_idx: int = -1
        self.last_T: Optional[np.ndarray] = None
        self.prev_mask01: Optional[np.ndarray] = None  # 不落盘（只在一次 track() 调用内部用）
        self.gate_mask01: Optional[np.ndarray] = None

        # load disk state if exists
        self._load_state()
        self._load_gate_if_exists()
        self._sync_idx_next_from_disk()

        # UI
        self.win = "OfflineDiskRecorderTracker (r:record | g:gate | t:track | q:quit)"
        if self.show_ui:
            cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)

    # ---------------- lifecycle ----------------

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
        if self.show_ui:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    # ---------------- disk paths ----------------

    def _color_path(self, idx: int) -> str:
        return os.path.join(self.color_dir, f"{idx:06d}.jpg")

    def _depth_path(self, idx: int) -> str:
        return os.path.join(self.depth_dir, f"{idx:06d}.png")

    def _frame_exists(self, idx: int) -> bool:
        return os.path.isfile(self._color_path(idx)) and os.path.isfile(self._depth_path(idx))

    def _list_indices_on_disk(self) -> List[int]:
        pat = re.compile(r"^(\d{6})\.(jpg|jpeg|png)$", re.IGNORECASE)
        idxs = []
        for fn in os.listdir(self.color_dir):
            m = pat.match(fn)
            if m:
                idxs.append(int(m.group(1)))
        idxs = sorted(set(idxs))
        # depth 可能缺失，过滤掉不完整的
        idxs = [i for i in idxs if self._frame_exists(i)]
        return idxs

    def _sync_idx_next_from_disk(self):
        idxs = self._list_indices_on_disk()
        if len(idxs) == 0:
            self.idx_next = max(self.idx_next, 0)
            return
        self.idx_next = max(self.idx_next, idxs[-1] + 1)

    # ---------------- state persistence ----------------

    def _load_state(self):
        if not os.path.isfile(self.state_path):
            return
        try:
            with open(self.state_path, "r") as f:
                st = json.load(f)
            self.idx_next = int(st.get("idx_next", 0))
            self.init_done = bool(st.get("init_done", False))
            self.last_tracked_idx = int(st.get("last_tracked_idx", -1))
            lt = st.get("last_T", None)
            if lt is not None and isinstance(lt, list) and len(lt) == 16:
                self.last_T = np.array(lt, dtype=np.float32).reshape(4, 4)
            print(f"[STATE] loaded: init_done={self.init_done}, last_tracked_idx={self.last_tracked_idx}, idx_next={self.idx_next}")
        except Exception as e:
            print(f"[STATE] failed to load {self.state_path}: {e}")

    def _save_state(self):
        st = dict(
            idx_next=int(self.idx_next),
            init_done=bool(self.init_done),
            last_tracked_idx=int(self.last_tracked_idx),
            last_T=(self.last_T.reshape(-1).tolist() if self.last_T is not None else None),
        )
        try:
            tmp = self.state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(st, f, indent=2)
            os.replace(tmp, self.state_path)
        except Exception as e:
            print(f"[STATE] failed to save {self.state_path}: {e}")

    def _load_gate_if_exists(self):
        if not os.path.isfile(self.gate_path):
            return
        try:
            g = cv2.imread(self.gate_path, cv2.IMREAD_UNCHANGED)
            if g is None:
                return
            if g.ndim == 3:
                g = g[..., 0]
            self.gate_mask01 = (g > 0).astype(np.uint8)
            print(f"[GATE] loaded from {self.gate_path}, pixels={int(self.gate_mask01.sum())}")
        except Exception as e:
            print(f"[GATE] failed to load {self.gate_path}: {e}")

    def _save_gate(self):
        if self.gate_mask01 is None:
            return
        try:
            # 保存为 0/255 便于肉眼查看
            vis = (self.gate_mask01.astype(np.uint8) * 255)
            safe_imwrite(self.gate_path, vis)
            print(f"[GATE] saved to {self.gate_path}")
        except Exception as e:
            print(f"[GATE] failed to save {self.gate_path}: {e}")

    # ---------------- public APIs ----------------

    def record_frame(self) -> int:
        """
        录制一帧：采集 (color_bgr, depth_mm) 并写入本地：
          color/xxxxxx.jpg
          depth/xxxxxx.png  (uint16, mm)
        """
        color, depth_mm = self.camera.get_k4a_frame(require_aligned_depth=True)
        if depth_mm.dtype != np.uint16:
            depth_mm = depth_mm.astype(np.uint16)

        idx = int(self.idx_next)
        c_path = self._color_path(idx)
        d_path = self._depth_path(idx)

        # color: jpg（体积更小），depth: png（保留16位）
        safe_imwrite(c_path, color, params=[int(cv2.IMWRITE_JPEG_QUALITY), self.jpg_quality])
        safe_imwrite(d_path, depth_mm)

        self.idx_next += 1
        self._save_state()
        # print(f"[RECORD] idx={idx} -> {c_path}, {d_path}")
        return idx

    def draw_gate(self, idx: Optional[int] = None) -> Optional[np.ndarray]:
        """
        在某个已录制帧上绘制 gate polygon，并保存 gate.png。
        默认 idx=0；若 0 不存在则用最新一帧。
        """
        idxs = self._list_indices_on_disk()
        if len(idxs) == 0:
            print("[GATE] No recorded frames on disk. Call record_frame() first.")
            return None

        if idx is None:
            idx = 0 if 0 in idxs else idxs[-1]
        if idx not in idxs:
            print(f"[GATE] idx={idx} not found on disk.")
            return None

        color = safe_imread_color(self._color_path(idx))
        print(f"[GATE] Draw gate on recorded frame idx={idx}")
        gate = self.select_gate_mask_polygon(color, win_name="Select GATE mask (polygon)")
        if gate is None or int(gate.sum()) == 0:
            print("[GATE] canceled / empty.")
            return None

        self.gate_mask01 = gate.astype(np.uint8)
        print(f"[GATE] ok. pixels={int(self.gate_mask01.sum())}")
        self._save_gate()
        return self.gate_mask01

    def track(self) -> None:
        """
        从本地读取，做 SAM3 + bundletrack。
        - 第一次：idx=0 init（mask+depth求center3D -> ob_in_cam）
        - 后续：从 last_tracked_idx 开始（用 last_T 作为 init 再送一次），然后跑到最新帧
        """
        idxs = self._list_indices_on_disk()
        if len(idxs) == 0:
            print("[TRACK] No recorded frames on disk.")
            return

        max_idx = idxs[-1]

        # gate 必须在第一次 init 前设置
        if (not self.init_done) and (self.gate_mask01 is None):
            # 默认让你在第一帧画 gate
            gate = self.draw_gate(idx=0 if 0 in idxs else None)
            if gate is None:
                print("[TRACK] gate not set. Abort.")
                return

        # 起始 idx
        if not self.init_done:
            start_idx = 0
        else:
            start_idx = self.last_tracked_idx  # 按你的要求：从上次结束的帧开始再init

        if start_idx < 0:
            start_idx = 0

        if start_idx > max_idx:
            print(f"[TRACK] start_idx={start_idx} > max_idx={max_idx}, nothing to do.")
            return

        # 只处理磁盘上存在的 idx（防止中间缺帧）
        idx_set = set(idxs)
        print(f"[TRACK] start_idx={start_idx}, max_idx={max_idx}, init_done={self.init_done}")

        # 本次 track 调用内，prev_mask 从 None 开始（或沿用也行）
        # 这里沿用上次调用的 prev_mask01 能更稳定，但跨次调用你没落盘，因此默认 None
        # 如果你希望跨 track() 保持 prev_mask，请不要清空 self.prev_mask01
        # self.prev_mask01 = None

        for idx in range(start_idx, max_idx + 1):
            if idx not in idx_set:
                continue

            color = safe_imread_color(self._color_path(idx))
            depth_mm = safe_imread_depth_u16(self._depth_path(idx))

            # ---------- SAM3 ----------
            try:
                mask01 = self.segmenter.segment_from_text(
                    color,
                    self.text_prompt,
                    prev_mask01=self.prev_mask01 if self.init_done else None,
                    gate_mask01=self.gate_mask01,
                )
            except Exception as e:
                print(f"[WARN] SAM3 failed at idx={idx}: {e}")
                mask01 = self.prev_mask01

            if mask01 is None or int(mask01.sum()) == 0:
                mask01 = self.prev_mask01

            if mask01 is None or int(mask01.sum()) == 0:
                print(f"[WARN] Empty mask at idx={idx}. Skip frame.")
                continue

            # ---------- init logic ----------
            has_init = False
            ob_in_cam = None

            if (not self.init_done) and idx == 0:
                center3d = self.compute_center3d_from_mask_depth_mm(
                    mask01, depth_mm, self.fx, self.fy, self.cx, self.cy
                )
                has_init = center3d is not None
                ob_in_cam = self.make_ob_in_cam_from_center(center3d) if has_init else None
                self.init_pose = ob_in_cam

            elif self.init_done and idx == start_idx:
                # 断点续追：从上次结束那帧开始重新 init 一次（用 last_T）
                has_init = self.last_T is not None
                ob_in_cam = self.last_T.copy() if self.last_T is not None else None

            # ---------- send ----------
            id_str = f"{idx:06d}"
            try:
                T, resp = self.send_frame_with_sock(
                    self.sock,
                    color_bgr=color,
                    depth_mm=depth_mm,
                    mask=mask01,
                    id_str=id_str,
                    roi=None,
                    has_init=has_init,
                    ob_in_cam=ob_in_cam,
                )
            except Exception as e:
                print(f"[ERROR] send_frame_with_sock failed at idx={idx}: {e}")
                return

            self.last_T = T
            self.prev_mask01 = mask01
            self.last_tracked_idx = idx

            if not self.init_done:
                self.init_done = True
                print(f"[INIT] done at idx={idx}, ok={resp.get('ok')}")

            self._save_state()

            # UI（可选）
            if self.show_ui:
                disp = color.copy()
                if self.gate_mask01 is not None:
                    disp = self.overlay_mask_color(disp, self.gate_mask01, color=(0, 0, 255), alpha=0.25)
                disp = self.overlay_mask_color(disp, mask01, color=(0, 255, 0), alpha=0.35)
                self.draw_pose6d_on_image(disp, T, self.K, axis_len=self.axis_len, thickness=2, text_org=(5, 60))
                cv2.putText(
                    disp,
                    f"TRACK idx={idx}/{max_idx} init={has_init} (disk)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow(self.win, disp)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    self._shutdown()
                    raise SystemExit

        print(f"[TRACK] finished. last_tracked_idx={self.last_tracked_idx}")

    # ---------------- ZMQ + vision utils ----------------

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
        cos = np.cos(135.0 * math.pi / 180.0)
        sin = np.sin(135.0 * math.pi / 180.0)
        rotation_x = np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]], dtype=np.float32)
        T[:3, :3] = rotation_x
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

    def send_frame_with_sock(
        self,
        sock,
        color_bgr: np.ndarray,
        depth_mm: np.ndarray,
        mask: Optional[np.ndarray] = None,
        id_str: str = "000000",
        roi=None,
        ob_in_cam: Optional[np.ndarray] = None,
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

        if has_init and (ob_in_cam is not None):
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

    def select_gate_mask_polygon(self, bgr, win_name="Select GATE mask (polygon)"):
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


# ------------------------ Simple interactive demo ------------------------

def main():
    # 你可以改成你想要的存储目录
    out_dir = "/home/ferry/data/Code2/Research/Inhand_Activate/Tracking/offline_cache"

    trk = OfflineDiskRecorderTracker(
        root_dir=out_dir,
        text_prompt="An green object",
        show_ui=True,
        jpg_quality=90,
    )

    print("\n[USAGE]")
    print("  r : record one frame -> write to disk")
    print("  g : draw gate on frame0 (or latest if 0 not exist) -> save gate.png")
    print("  t : track from disk (first time init on idx=0; later resume from last_tracked_idx)")
    print("  q : quit\n")

    try:
        while True:
            # 只做一个轻量 live 预览（不缓存）
            color, _ = trk.camera.get_k4a_frame(require_aligned_depth=True)
            disp = color.copy()
            if trk.gate_mask01 is not None:
                disp = trk.overlay_mask_color(disp, trk.gate_mask01, color=(0, 0, 255), alpha=0.25)
            cv2.putText(
                disp,
                f"disk_dir={os.path.basename(trk.root_dir)}  idx_next={trk.idx_next}  init_done={trk.init_done}  last_tracked={trk.last_tracked_idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )
            cv2.imshow(trk.win, disp)
            key = cv2.waitKey(10) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                trk.record_frame()
            elif key == ord("g"):
                trk.draw_gate(idx=0)
            elif key == ord("t"):
                trk.track()

    finally:
        trk._shutdown()


if __name__ == "__main__":
    main()
