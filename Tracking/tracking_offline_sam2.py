#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@FileName：tracking_offline.py
@Description：Record one frame per call (to disk), then offline tracking from disk with resume.
             Segmentation: SAM2 bbox prompt (via your SamSegmenter wrapper).
             Gate mask (polygon) and prompt bbox (rectangle) are separated:
               - gate: AND constraint on output mask
               - bbox: SAM2 prompt box (and optional constraint on prompt box region)
@Author：Ferry
@Time：2026/01/19
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""

import os
import re
import json
import math
from typing import Optional, List, Tuple

import cv2
import numpy as np
import zmq

from Active.AzureKinectDK.Azure_camera import AzureKinectDK
from Tracking.sam2_class import SamSegmenter


# ------------------------ Small IO helpers ------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def safe_imwrite(path: str, img: np.ndarray, params: Optional[List[int]] = None):
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
        d = d.astype(np.uint16, copy=False)
    return d


# ------------------------ Mask/Box helpers ------------------------

def sample_center_points_from_mask(mask01: np.ndarray, n: int = 5) -> Optional[np.ndarray]:
    """
    从 mask 的“中心区域”取 n 个点（用 distance transform 取最大值点）。
    返回 (n,2) 的 float32，坐标是 (x,y)。
    """
    if mask01 is None:
        return None
    m = (mask01.astype(np.uint8) > 0).astype(np.uint8)
    if int(m.sum()) == 0:
        return None

    # distanceTransform 需要 0/255
    dist = cv2.distanceTransform((m * 255).astype(np.uint8), cv2.DIST_L2, 5)
    if dist is None:
        return None

    ys, xs = np.nonzero(m > 0)
    if xs.size == 0:
        return None

    vals = dist[ys, xs]
    k = min(int(n), int(vals.size))
    if k <= 0:
        return None

    # 取 top-k 最大的距离点（最靠内）
    idx = np.argpartition(vals, -k)[-k:]
    sel_x = xs[idx].astype(np.float32)
    sel_y = ys[idx].astype(np.float32)

    pts = np.stack([sel_x, sel_y], axis=1)  # (k,2)
    return pts


def filter_points_in_box(points_xy: np.ndarray, box_xyxy: Optional[List[int]]) -> np.ndarray:
    if points_xy is None or box_xyxy is None:
        return points_xy
    x0, y0, x1, y1 = [int(v) for v in box_xyxy]
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    keep = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    return points_xy[keep]


def box_xyxy_from_mask(mask01: np.ndarray, pad: int = 8) -> Optional[List[int]]:
    """mask01 -> [x0, y0, x1, y1]"""
    if mask01 is None:
        return None
    ys, xs = np.where(mask01 > 0)
    if xs.size == 0:
        return None
    H, W = mask01.shape[:2]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W - 1, x1 + pad)
    y1 = min(H - 1, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def intersect_xyxy(a: Optional[List[int]], b: Optional[List[int]]) -> Optional[List[int]]:
    """intersection of two xyxy boxes"""
    if a is None:
        return b
    if b is None:
        return a
    x0 = max(int(a[0]), int(b[0]))
    y0 = max(int(a[1]), int(b[1]))
    x1 = min(int(a[2]), int(b[2]))
    y1 = min(int(a[3]), int(b[3]))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def roi_x0x1y0y1_from_mask(mask01: np.ndarray, pad: int = 5) -> Optional[List[int]]:
    """For bundletrack header ROI: [x0, x1, y0, y1]"""
    if mask01 is None:
        return None
    ys, xs = np.where(mask01 > 0)
    if xs.size == 0:
        return None
    H, W = mask01.shape[:2]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W - 1, x1 + pad)
    y1 = min(H - 1, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, x1, y0, y1]


def mask_and_gate(mask01: Optional[np.ndarray], gate01: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask01 is None:
        return None
    if gate01 is None:
        return mask01.astype(np.uint8)
    if mask01.shape != gate01.shape:
        H, W = gate01.shape[:2]
        mask01 = cv2.resize(mask01.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    return (mask01.astype(np.uint8) & (gate01.astype(np.uint8) > 0).astype(np.uint8))


# ------------------------ Disk-based Recorder + Offline Tracking ------------------------

class OfflineDiskRecorderTracker:
    """
    落盘版本（不吃内存）：
    - record_frame(): 每调用一次录制一帧 -> 写本地 color/depth
    - draw_gate(): 在已录制帧上画 gate polygon -> 保存 gate.png
    - draw_prompt_box(): 在已录制帧上画 bbox -> 保存 prompt_box.json
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
        # SAM2
        sam2_checkpoint: str = "/path/to/sam2_checkpoint.pt",
        sam2_model_cfg: str = "configs/sam2/sam2_hiera_l.yaml",
        sam2_device: str = "cuda",
        sam2_use_amp: bool = True,
        # prompt policy
        box_pad_prev: int = 14,          # prev_mask bbox pad
        constrain_prev_by_prompt_box: bool = True,  # prev bbox 是否被你手画 prompt_box 限制
        show_ui=True,
        jpg_quality: int = 90,
    ):
        self.axis_len = float(axis_len)
        self.show_ui = bool(show_ui)
        self.jpg_quality = int(jpg_quality)

        self.box_pad_prev = int(box_pad_prev)
        self.constrain_prev_by_prompt_box = bool(constrain_prev_by_prompt_box)

        # directories
        self.root_dir = os.path.abspath(root_dir)
        self.color_dir = os.path.join(self.root_dir, "color")
        self.depth_dir = os.path.join(self.root_dir, "depth")
        self.state_path = os.path.join(self.root_dir, "state.json")
        self.gate_path = os.path.join(self.root_dir, "gate.png")
        self.prompt_box_path = os.path.join(self.root_dir, "prompt_box.json")
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

        # SAM2
        self.segmenter = SamSegmenter(
            checkpoint=str(sam2_checkpoint),
            model_cfg=str(sam2_model_cfg),
            device=str(sam2_device),
            use_amp=bool(sam2_use_amp),
        )
        print(f"[SAM2] checkpoint={sam2_checkpoint} cfg={sam2_model_cfg} device={sam2_device} amp={sam2_use_amp}")

        # state
        self.idx_next: int = 0
        self.init_done: bool = False
        self.last_tracked_idx: int = -1
        self.last_T: Optional[np.ndarray] = None
        self.prev_mask01: Optional[np.ndarray] = None  # 不落盘
        self.gate_mask01: Optional[np.ndarray] = None
        self.prompt_box_xyxy: Optional[List[int]] = None  # [x0,y0,x1,y1] separate from gate

        # load disk state if exists
        self._load_state()
        self._load_gate_if_exists()
        self._load_prompt_box_if_exists()
        self._sync_idx_next_from_disk()

        # UI
        self.win = "OfflineDiskRecorderTracker (r:record | g:gate | b:bbox | t:track | q:quit)"
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
            vis = (self.gate_mask01.astype(np.uint8) * 255)
            safe_imwrite(self.gate_path, vis)
            print(f"[GATE] saved to {self.gate_path}")
        except Exception as e:
            print(f"[GATE] failed to save {self.gate_path}: {e}")

    def _load_prompt_box_if_exists(self):
        if not os.path.isfile(self.prompt_box_path):
            return
        try:
            with open(self.prompt_box_path, "r") as f:
                d = json.load(f)
            box = d.get("prompt_box_xyxy", None)
            if isinstance(box, list) and len(box) == 4:
                self.prompt_box_xyxy = [int(x) for x in box]
                print(f"[BBOX] loaded from {self.prompt_box_path}: {self.prompt_box_xyxy}")
        except Exception as e:
            print(f"[BBOX] failed to load {self.prompt_box_path}: {e}")

    def _save_prompt_box(self):
        if self.prompt_box_xyxy is None:
            return
        try:
            tmp = self.prompt_box_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"prompt_box_xyxy": [int(x) for x in self.prompt_box_xyxy]}, f, indent=2)
            os.replace(tmp, self.prompt_box_path)
            print(f"[BBOX] saved to {self.prompt_box_path}: {self.prompt_box_xyxy}")
        except Exception as e:
            print(f"[BBOX] failed to save {self.prompt_box_path}: {e}")

    # ---------------- public APIs ----------------

    def record_frame(self) -> int:
        color, depth_mm = self.camera.get_k4a_frame(require_aligned_depth=True)
        if depth_mm.dtype != np.uint16:
            depth_mm = depth_mm.astype(np.uint16)

        idx = int(self.idx_next)
        c_path = self._color_path(idx)
        d_path = self._depth_path(idx)

        safe_imwrite(c_path, color, params=[int(cv2.IMWRITE_JPEG_QUALITY), self.jpg_quality])
        safe_imwrite(d_path, depth_mm)

        self.idx_next += 1
        self._save_state()
        return idx

    def draw_gate(self, idx: Optional[int] = None) -> Optional[np.ndarray]:
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

    def draw_prompt_box(self, idx: Optional[int] = None) -> Optional[List[int]]:
        """
        画一个矩形 bbox 作为 SAM2 prompt（与 gate 无关）
        保存到 prompt_box.json
        """
        idxs = self._list_indices_on_disk()
        if len(idxs) == 0:
            print("[BBOX] No recorded frames on disk. Call record_frame() first.")
            return None

        if idx is None:
            idx = 0 if 0 in idxs else idxs[-1]
        if idx not in idxs:
            print(f"[BBOX] idx={idx} not found on disk.")
            return None

        color = safe_imread_color(self._color_path(idx))
        print(f"[BBOX] Draw bbox on recorded frame idx={idx}")
        box = self.select_prompt_box_rectangle(color, win_name="Select PROMPT BBOX (drag or 2 clicks)")
        if box is None:
            print("[BBOX] canceled.")
            return None

        self.prompt_box_xyxy = [int(x) for x in box]
        self._save_prompt_box()
        return self.prompt_box_xyxy

    def track(self) -> None:
        idxs = self._list_indices_on_disk()
        if len(idxs) == 0:
            print("[TRACK] No recorded frames on disk.")
            return

        max_idx = idxs[-1]

        # 第一次 init 前：gate / bbox 可以分别设置（互不相同）
        if (not self.init_done) and (self.gate_mask01 is None):
            gate = self.draw_gate(idx=0 if 0 in idxs else None)
            if gate is None:
                print("[TRACK] gate not set. Abort.")
                return

        if self.prompt_box_xyxy is None and self.prev_mask01 is None:
            # 没有 prev_mask 的情况下，必须有 bbox prompt
            box = self.draw_prompt_box(idx=0 if 0 in idxs else None)
            if box is None:
                print("[TRACK] prompt bbox not set. Abort.")
                return

        # 起始 idx
        if not self.init_done:
            start_idx = 0
        else:
            start_idx = self.last_tracked_idx  # resume：从上次结束帧再 init 一次

        start_idx = max(0, int(start_idx))

        if start_idx > max_idx:
            print(f"[TRACK] start_idx={start_idx} > max_idx={max_idx}, nothing to do.")
            return

        idx_set = set(idxs)
        print(f"[TRACK] start_idx={start_idx}, max_idx={max_idx}, init_done={self.init_done}")

        for idx in range(start_idx, max_idx + 1):
            if idx not in idx_set:
                continue

            color = safe_imread_color(self._color_path(idx))
            depth_mm = safe_imread_depth_u16(self._depth_path(idx))

            # ---------- SAM2 prompt ----------
            # 你的要求：tracking 时，prompt 用上一帧 mask 的中心区域若干点
            # 第 0 帧 / prev_mask 不存在：用你手画的 bbox 初始化

            mask01 = None
            box_for_sam2 = self.prompt_box_xyxy if self.constrain_prev_by_prompt_box else None

            try:
                if self.prev_mask01 is not None and int(self.prev_mask01.sum()) > 0:
                    pts = sample_center_points_from_mask(self.prev_mask01, n=5)

                    # 若你希望点也被 prompt_box 限制（与你原先 constrain 逻辑一致）
                    if self.constrain_prev_by_prompt_box and self.prompt_box_xyxy is not None and pts is not None:
                        pts = filter_points_in_box(pts, self.prompt_box_xyxy)
                        if pts is not None and pts.shape[0] == 0:
                            pts = None

                    if pts is not None and pts.shape[0] > 0:
                        labels = np.ones((pts.shape[0],), dtype=np.int32)  # 全正点
                        mask01 = self.segmenter.segment_from_points(
                            color,
                            points_xy=pts,
                            point_labels=labels,
                            box_xyxy=box_for_sam2,  # 你可以设为 None 表示纯点 prompt
                        )
                    else:
                        # 兜底：点采样失败则用 bbox
                        if self.prompt_box_xyxy is not None:
                            mask01 = self.segmenter.segment_from_box(color, box_xyxy=self.prompt_box_xyxy)
                        else:
                            mask01 = self.prev_mask01

                else:
                    # 第一次：用你画的 bbox 初始化
                    if self.prompt_box_xyxy is None:
                        raise RuntimeError("prompt_box_xyxy is None (need bbox for init)")
                    mask01 = self.segmenter.segment_from_box(color, box_xyxy=self.prompt_box_xyxy)

                # gate 仍然是独立 AND 约束
                mask01 = mask_and_gate(mask01, self.gate_mask01)

            except Exception as e:
                print(f"[WARN] SAM2 failed at idx={idx}: {e}")
                mask01 = mask_and_gate(self.prev_mask01, self.gate_mask01) if self.prev_mask01 is not None else None



            if mask01 is None or int(mask01.sum()) == 0:
                mask01 = mask_and_gate(self.prev_mask01, self.gate_mask01) if self.prev_mask01 is not None else None

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

                # show gate (red overlay)
                if self.gate_mask01 is not None:
                    disp = self.overlay_mask_color(disp, self.gate_mask01, color=(0, 0, 255), alpha=0.20)

                # show bbox (yellow rectangle) -- independent from gate
                if self.prompt_box_xyxy is not None:
                    x0, y0, x1, y1 = [int(x) for x in self.prompt_box_xyxy]
                    cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 255), 2)

                # show current mask (green)
                disp = self.overlay_mask_color(disp, mask01, color=(0, 255, 0), alpha=0.35)

                self.draw_pose6d_on_image(disp, T, self.K, axis_len=self.axis_len, thickness=2, text_org=(5, 60))

                # draw sampled prompt points (cyan dots)
                if self.prev_mask01 is not None and int(self.prev_mask01.sum()) > 0:
                    pts_dbg = sample_center_points_from_mask(self.prev_mask01, n=5)
                    if self.constrain_prev_by_prompt_box and self.prompt_box_xyxy is not None and pts_dbg is not None:
                        pts_dbg = filter_points_in_box(pts_dbg, self.prompt_box_xyxy)
                    if pts_dbg is not None:
                        for (x, y) in pts_dbg:
                            cv2.circle(disp, (int(x), int(y)), 4, (255, 255, 0), -1, cv2.LINE_AA)

                cv2.putText(
                    disp,
                    f"TRACK idx={idx}/{max_idx} init={has_init} (gate AND, bbox prompt)",
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
            auto_roi = roi_x0x1y0y1_from_mask(mask01, pad=5)
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

    # ------------------------ UI selectors ------------------------

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

    def select_prompt_box_rectangle(self, bgr, win_name="Select PROMPT BBOX (drag or 2 clicks)"):
        H, W = bgr.shape[:2]
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        state = {
            "dragging": False,
            "p0": None,
            "p1": None,
            "box": None,  # xyxy
        }

        def clamp_pt(x, y):
            x = max(0, min(W - 1, int(x)))
            y = max(0, min(H - 1, int(y)))
            return x, y

        def current_vis():
            vis = bgr.copy()
            cv2.putText(
                vis,
                "Drag LMB or click 2 points | ENTER/SPACE=OK | ESC=cancel | c=clear",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            if state["p0"] is not None:
                cv2.circle(vis, state["p0"], 4, (0, 255, 255), -1, cv2.LINE_AA)
            if state["p1"] is not None:
                cv2.circle(vis, state["p1"], 4, (0, 255, 255), -1, cv2.LINE_AA)

            if state["box"] is not None:
                x0, y0, x1, y1 = state["box"]
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)
            return vis

        def update_box_from_p0p1():
            if state["p0"] is None or state["p1"] is None:
                state["box"] = None
                return
            x0, y0 = state["p0"]
            x1, y1 = state["p1"]
            xa, xb = (min(x0, x1), max(x0, x1))
            ya, yb = (min(y0, y1), max(y0, y1))
            if xb <= xa or yb <= ya:
                state["box"] = None
                return
            state["box"] = [xa, ya, xb, yb]

        def on_mouse(event, x, y, flags, param):
            x, y = clamp_pt(x, y)
            if event == cv2.EVENT_LBUTTONDOWN:
                state["dragging"] = True
                state["p0"] = (x, y)
                state["p1"] = (x, y)
                update_box_from_p0p1()
            elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
                state["p1"] = (x, y)
                update_box_from_p0p1()
            elif event == cv2.EVENT_LBUTTONUP and state["dragging"]:
                state["dragging"] = False
                state["p1"] = (x, y)
                update_box_from_p0p1()

        cv2.setMouseCallback(win_name, on_mouse)

        while True:
            cv2.imshow(win_name, current_vis())
            k = cv2.waitKey(20) & 0xFF
            if k in (13, 10, 32):  # Enter/Space
                if state["box"] is None:
                    print("[BBOX] Need a valid rectangle.")
                    continue
                break
            if k == 27:  # ESC
                cv2.destroyWindow(win_name)
                return None
            if k == ord("c"):
                state["dragging"] = False
                state["p0"] = None
                state["p1"] = None
                state["box"] = None

        cv2.destroyWindow(win_name)
        return state["box"]


# ------------------------ Simple interactive demo ------------------------

def main():
    out_dir = "/home/ferry/data/Code2/Research/Inhand_Activate/Tracking/offline_cache"

    trk = OfflineDiskRecorderTracker(
        root_dir=out_dir,
        sam2_checkpoint="sam_model/sam2.1_hiera_tiny.pt",
        sam2_model_cfg="configs/sam2.1/sam2.1_hiera_t.yaml",
        sam2_device="cuda",
        sam2_use_amp=True,
        show_ui=True,
        jpg_quality=90,
        box_pad_prev=14,
        constrain_prev_by_prompt_box=True,
    )

    print("\n[USAGE]")
    print("  r : record one frame -> write to disk")
    print("  g : draw gate polygon (output mask AND constraint) -> save gate.png")
    print("  b : draw bbox rectangle (SAM2 prompt bbox, independent from gate) -> save prompt_box.json")
    print("  t : track from disk (init idx=0; later resume from last_tracked_idx)")
    print("  q : quit\n")

    try:
        while True:
            color, _ = trk.camera.get_k4a_frame(require_aligned_depth=True)
            disp = color.copy()

            if trk.gate_mask01 is not None:
                disp = trk.overlay_mask_color(disp, trk.gate_mask01, color=(0, 0, 255), alpha=0.20)

            if trk.prompt_box_xyxy is not None:
                x0, y0, x1, y1 = [int(x) for x in trk.prompt_box_xyxy]
                cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 255), 2)

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
            elif key == ord("b"):
                trk.draw_prompt_box(idx=0)
            elif key == ord("t"):
                trk.track()

    finally:
        trk._shutdown()


if __name__ == "__main__":
    main()
