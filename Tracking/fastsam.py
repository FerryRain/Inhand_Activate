"""
@FileName：fastsam_class.py
@Description：FastSAM text-prompt segmentation wrapper (Ultralytics)
@Author：Ferry
@Time：2026 1/12/26
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class FastSAMCfg:
    weights: str = "FastSAM-s.pt"   # or "FastSAM-x.pt"
    imgsz: int = 1024
    conf: float = 0.4
    iou: float = 0.9
    retina_masks: bool = True
    device: str = "cuda"            # "cuda" or "cpu"
    verbose: bool = False
    # 选择 mask 的策略
    min_area_ratio: float = 0.0005  # 过滤极小碎片
    max_area_ratio: float = 0.60    # 过滤过大（容易把整只手/背景吃进去）
    iou_keep_th: float = 0.02       # 与 prev_mask 的 IoU 太小则可选择回退

    gate_min_overlap_px: int = 2000  # 候选mask与gate的最小交集像素，否则认为不属于gate区域
    gate_inside_ratio_th: float = 0.20  # 候选mask中至少多少比例落在gate内，否则丢弃


class FastSamSegmenter:
    """FastSAM text prompt -> mask01 (HxW uint8 {0,1})"""

    def __init__(self, cfg: FastSAMCfg):
        self.cfg = cfg
        self.model = None
        self._init_model()

    def _init_model(self):
        # Ultralytics FastSAM API
        from ultralytics import FastSAM

        self.model = FastSAM(self.cfg.weights)
        # model 会在调用时根据 device 放置；这里不强制 .to()

    @staticmethod
    def _to_mask01(m: np.ndarray) -> np.ndarray:
        # m 可能是 float/0-1，也可能是 0/255
        return (m > 0).astype(np.uint8)

    @staticmethod
    def _mask_iou(a01: np.ndarray, b01: np.ndarray) -> float:
        if a01 is None or b01 is None:
            return 0.0
        inter = np.logical_and(a01 > 0, b01 > 0).sum()
        union = np.logical_or(a01 > 0, b01 > 0).sum()
        return float(inter) / float(union + 1e-6)

    def _pick_best_mask(
        self,
        masks01: np.ndarray,          # (N,H,W) uint8 {0,1}
        prev_mask01: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if masks01 is None or masks01.size == 0:
            return None

        N, H, W = masks01.shape
        areas = masks01.reshape(N, -1).sum(axis=1).astype(np.float32)
        area_ratio = areas / float(H * W + 1e-6)

        # 先过滤面积过小/过大
        keep = np.where((area_ratio >= self.cfg.min_area_ratio) & (area_ratio <= self.cfg.max_area_ratio))[0]
        if keep.size == 0:
            keep = np.arange(N)

        if prev_mask01 is None:
            # init：选面积最大的（但避免极端大）
            k = keep[int(np.argmax(areas[keep]))]
            return masks01[k]

        # tracking：优先选与 prev IoU 最大的
        ious = np.array([self._mask_iou(masks01[i], prev_mask01) for i in keep], dtype=np.float32)
        best_local = int(np.argmax(ious))
        best_i = int(keep[best_local])
        best_iou = float(ious[best_local])

        if best_iou < self.cfg.iou_keep_th:
            # IoU 太小：说明 prompt 可能漂了；你可以选择回退 prev_mask
            # 这里默认回退，防止抖动/跳目标
            return prev_mask01

        return masks01[best_i]

    def segment_from_text(
            self,
            bgr: np.ndarray,
            text_prompt: str,
            prev_mask01: Optional[np.ndarray] = None,
            gate_mask01: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Returns:
            mask01: HxW uint8 {0,1} or None
        """
        if bgr is None:
            return None

        kwargs = dict(
            device=self.cfg.device,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            retina_masks=self.cfg.retina_masks,
            verbose=self.cfg.verbose,
            texts=text_prompt,
        )

        results = self.model(bgr, **kwargs)
        if results is None or len(results) == 0:
            return prev_mask01 if prev_mask01 is not None else None

        r0 = results[0]
        if r0.masks is None:
            return prev_mask01 if prev_mask01 is not None else None

        masks = r0.masks.data
        try:
            masks = masks.detach().float().cpu().numpy()
        except Exception:
            masks = np.asarray(masks)

        if masks.ndim == 2:
            masks = masks[None, ...]

        masks01 = (masks > 0).astype(np.uint8)  # (N,H,W)

        # -------- gate filtering & selection --------
        if gate_mask01 is not None:
            gate = (gate_mask01 > 0).astype(np.uint8)
        else:
            gate = None

        # prev 也在 gate 内比较更稳定
        prev_gate = None
        if prev_mask01 is not None and gate is not None:
            prev_gate = ((prev_mask01 > 0).astype(np.uint8) & gate)
        elif prev_mask01 is not None:
            prev_gate = (prev_mask01 > 0).astype(np.uint8)

        keep_idx = []
        masked_candidates = []
        for i in range(masks01.shape[0]):
            m = masks01[i]
            if gate is not None:
                inter = (m & gate)
                inter_area = int(inter.sum())
                if inter_area < int(self.cfg.gate_min_overlap_px):
                    continue
                inside_ratio = float(inter_area) / float(int(m.sum()) + 1e-6)
                if inside_ratio < float(self.cfg.gate_inside_ratio_th):
                    continue
                m = inter  # 直接将候选限制在 gate 内
            else:
                if int(m.sum()) == 0:
                    continue
            keep_idx.append(i)
            masked_candidates.append(m)

        if len(masked_candidates) == 0:
            # gate 太严/文本漂移：回退上一帧（保证稳定）
            return prev_gate if prev_gate is not None else None

        # 选择策略：
        # - 有 prev：选 IoU 最大
        # - 无 prev：选面积最大
        if prev_gate is not None and int(prev_gate.sum()) > 0:
            def iou(a, b):
                inter = np.logical_and(a > 0, b > 0).sum()
                union = np.logical_or(a > 0, b > 0).sum()
                return float(inter) / float(union + 1e-6)

            ious = [iou(m, prev_gate) for m in masked_candidates]
            best = masked_candidates[int(np.argmax(ious))]
            # 如果 IoU 太小，仍回退 prev，避免跳目标
            if float(np.max(ious)) < float(self.cfg.iou_keep_th):
                return prev_gate
        else:
            areas = [int(m.sum()) for m in masked_candidates]
            best = masked_candidates[int(np.argmax(areas))]

        # 轻度平滑（可选）
        best = best.astype(np.uint8)
        best = cv2.medianBlur(best * 255, 5)
        best = (best > 0).astype(np.uint8)

        return best

