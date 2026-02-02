"""
@FileName：sam2_class.py
@Description：SAM2 bbox prompt wrapper
@Author：Ferry
@Time：2026 1/9/26 3:57 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""
import torch
import cv2
import numpy as np
from typing import Optional, List, Tuple

# sam2_checkpoint: str = "/path/to/sam2_checkpoint.pt",
# sam2_model_cfg: str = "configs/sam2/sam2_hiera_l.yaml",

class SamSegmenter:
    """bbox prompt -> mask01"""

    def __init__(self, checkpoint, model_cfg, device="cuda", use_amp=True):
        self.device = device
        self.use_amp = use_amp
        self.predictor = None
        self._init_model(checkpoint, model_cfg)

    def _init_model(self, checkpoint, model_cfg):

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

    def segment_from_points(
        self,
        bgr,
        points_xy: np.ndarray,        # (N,2) in (x,y)
        point_labels: np.ndarray,     # (N,) 1=pos, 0=neg
        box_xyxy: Optional[List[int]] = None,  # optional, can be None
    ):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        pts = np.asarray(points_xy, dtype=np.float32)
        lbs = np.asarray(point_labels, dtype=np.int32)

        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"points_xy must be (N,2), got {pts.shape}")
        if lbs.ndim != 1 or lbs.shape[0] != pts.shape[0]:
            raise ValueError(f"point_labels must be (N,), got {lbs.shape} vs N={pts.shape[0]}")

        # SAM2 expects shape (1, N, 2) and (1, N)
        point_coords = pts[None, :, :]
        point_labels = lbs[None, :]

        box = None
        if box_xyxy is not None:
            box = np.array(box_xyxy, dtype=np.float32)[None, :]  # (1,4)

        with torch.inference_mode():
            if (self.device == "cuda") and self.use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    self.predictor.set_image(rgb)
                    masks, _, _ = self.predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=box,
                        multimask_output=False,
                    )
            else:
                self.predictor.set_image(rgb)
                masks, _, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=False,
                )

        return masks[0].astype(np.uint8)
