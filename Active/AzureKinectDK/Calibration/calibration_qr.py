#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@FileName：calibration_qr.py
@Description：
  Azure Kinect DK (720P) + A4 QR board (W0/W1/W2/W3/WC) -> one-click extrinsic calibration.

  World frame definition (on the printed sheet):
    Origin: page center
    +X: right on the sheet
    +Y: up on the sheet
    +Z: out of the sheet (up from the table)

  Output:
    - T_world_in_cam (4x4): X_cam = R * X_world + t
      (same naming convention as T_ob_in_cam)
    - also prints cam_in_world (inverse)

  Controls:
    - 'c': collect N frames, calibrate, save
    - 'q' or ESC: quit

@Author：Ferry (refactor by ChatGPT)
@Time：2026-01-13
"""

import os
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

# -------- Azure Kinect --------
import pyk4a
from pyk4a import PyK4A, Config, CalibrationType


# =========================
# Board / IDs
# =========================
VALID_IDS = ["W0", "W1", "W2", "W3", "WC"]


def order_corners_tl_tr_br_bl(c: np.ndarray) -> np.ndarray:
    """
    Ensure the 4 corners are ordered as:
      [top-left, top-right, bottom-right, bottom-left]
    c: (4,2)
    """
    c = c.astype(np.float64).reshape(4, 2)
    s = c.sum(axis=1)            # x+y
    d = c[:, 0] - c[:, 1]        # x-y
    tl = c[np.argmin(s)]
    br = c[np.argmax(s)]
    tr = c[np.argmax(d)]
    bl = c[np.argmin(d)]
    return np.stack([tl, tr, br, bl], axis=0)


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / (n + eps)


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rodrigues: axis-angle -> R"""
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = w / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]], dtype=np.float64)
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.astype(np.float64)
    T[:3, 3] = t.reshape(3).astype(np.float64)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


# =========================
# Robust QR detection
# =========================
def detect_qr_multi(detector: cv2.QRCodeDetector, bgr: np.ndarray) -> List[Dict]:
    """
    Robust multi-preprocess + multi-scale QR detection.
    Return list: [{"id": str, "corners": (4,2) float64}, ...]
    """
    def _run_once(img_bgr):
        out = []
        try:
            ok, decoded_info, points, _ = detector.detectAndDecodeMulti(img_bgr)
        except Exception:
            ok, decoded_info, points = detector.detectAndDecodeMulti(img_bgr)

        if (not ok) or (points is None) or (decoded_info is None):
            return out

        pts = np.array(points, dtype=np.float64)  # (N,4,2)
        decoded_info = list(decoded_info)

        for i, txt in enumerate(decoded_info):
            if txt is None:
                continue
            txt = txt.strip()
            if txt not in VALID_IDS:
                continue
            corners = pts[i].reshape(4, 2).astype(np.float64)
            corners = order_corners_tl_tr_br_bl(corners)
            out.append({"id": txt, "corners": corners})
        return out

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE improves local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g1 = clahe.apply(gray)

    # adaptive threshold for uneven lighting
    g2 = cv2.adaptiveThreshold(
        g1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )

    variants = [
        bgr,
        cv2.cvtColor(g1, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR),
    ]
    scales = [0.75, 1.0, 1.25, 1.5]

    best_by_id: Dict[str, Tuple[float, np.ndarray]] = {}
    for v in variants:
        for s in scales:
            if abs(s - 1.0) < 1e-6:
                img = v
                scale = 1.0
            else:
                img = cv2.resize(v, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                scale = s

            dets = _run_once(img)
            for d in dets:
                cid = d["id"]
                corners = d["corners"] / scale
                area = float(abs(cv2.contourArea(corners.astype(np.float32))))
                if (cid not in best_by_id) or (area > best_by_id[cid][0]):
                    best_by_id[cid] = (area, corners)

    # subpixel refinement on original gray
    refined = []
    if len(best_by_id) > 0:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
        for cid, (area, corners) in best_by_id.items():
            c0 = corners.astype(np.float32).reshape(-1, 1, 2)
            try:
                c_ref = cv2.cornerSubPix(gray, c0, winSize=(7, 7), zeroZone=(-1, -1), criteria=criteria)
                corners_ref = c_ref.reshape(4, 2).astype(np.float64)
            except Exception:
                corners_ref = corners.astype(np.float64)
            refined.append({"id": cid, "corners": corners_ref})
    return refined


# =========================
# Template -> World 3D points
# =========================
def build_board_world_corners_from_template(template_path: str) -> Dict[str, np.ndarray]:
    """
    Load the A4 template PNG, detect QR corners in template pixels,
    then convert pixel coords to world coords (meters) using A4 size.

    Assumption:
      - Template image is a full A4 landscape page raster (ratio ~ sqrt(2))
      - Printed at 100% scale (no fit-to-page)

    Returns:
      dict id -> (4,3) corners in world coords (z=0 plane), order TL,TR,BR,BL.
    """
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")

    img = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read template: {template_path}")

    h, w = img.shape[:2]
    # A4 landscape: 297mm x 210mm
    A4_W_MM = 297.0
    A4_H_MM = 210.0

    sx = A4_W_MM / float(w)  # mm/px
    sy = A4_H_MM / float(h)

    # simple sanity check
    ratio = float(w) / float(h)
    if abs(ratio - (A4_W_MM / A4_H_MM)) > 0.03:
        print(f"[WARN] Template aspect ratio={ratio:.3f} not close to A4 landscape={(A4_W_MM/A4_H_MM):.3f}. "
              f"Ensure the template is full-page A4 landscape raster.")

    detector = cv2.QRCodeDetector()
    dets = detect_qr_multi(detector, img)

    out: Dict[str, np.ndarray] = {}
    for d in dets:
        cid = d["id"]
        corners_px = d["corners"]  # (4,2) in template pixels, TL,TR,BR,BL

        # pixel -> mm, origin at page center, +X right, +Y up
        xy_mm = np.zeros((4, 2), dtype=np.float64)
        xy_mm[:, 0] = (corners_px[:, 0] - 0.5 * w) * sx
        xy_mm[:, 1] = (0.5 * h - corners_px[:, 1]) * sy  # invert y: up

        xyz_m = np.zeros((4, 3), dtype=np.float64)
        xyz_m[:, 0] = xy_mm[:, 0] / 1000.0
        xyz_m[:, 1] = xy_mm[:, 1] / 1000.0
        xyz_m[:, 2] = 0.0
        out[cid] = xyz_m

    missing = [mid for mid in VALID_IDS if mid not in out]
    if len(missing) > 0:
        raise RuntimeError(f"Template QR decode missing IDs: {missing}. "
                           f"Please use the exact generated A4 template PNG (not a screenshot/cropped).")
    return out


# =========================
# Azure Kinect intrinsics
# =========================
def get_color_K_dist_from_device(k4a: PyK4A) -> Tuple[np.ndarray, np.ndarray]:
    """
    Try to obtain K and distortion from Azure Kinect calibration for COLOR camera.
    Different pyk4a builds expose slightly different APIs; we try several.

    Returns:
      K: (3,3)
      dist: (1,5) [k1,k2,p1,p2,k3] (OpenCV)
    """
    calib = k4a.calibration

    # --- K ---
    K = None
    try:
        K = calib.get_camera_matrix(CalibrationType.COLOR).astype(np.float64)
    except Exception:
        pass

    if K is None:
        # fallback: access intrinsics struct
        try:
            intr = calib.get_intrinsics(CalibrationType.COLOR)
            # common layouts:
            # intr.param.fx / intr.param.cx ...
            if hasattr(intr, "param"):
                fx = float(intr.param.fx)
                fy = float(intr.param.fy)
                cx = float(intr.param.cx)
                cy = float(intr.param.cy)
            else:
                fx = float(intr.fx)
                fy = float(intr.fy)
                cx = float(intr.cx)
                cy = float(intr.cy)
            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float64)
        except Exception as e:
            raise RuntimeError(f"Cannot get COLOR intrinsics from device via pyk4a. "
                               f"Use --K_path to provide K manually. Details: {e}")

    # --- Distortion ---
    dist = np.zeros((1, 5), dtype=np.float64)
    got_dist = False
    try:
        dc = calib.get_distortion_coefficients(CalibrationType.COLOR)
        dc = np.array(dc, dtype=np.float64).reshape(-1)
        if dc.size >= 5:
            dist[0, :] = dc[:5]
            got_dist = True
    except Exception:
        pass

    if not got_dist:
        # some builds expose intrinsics.param.k1..k6, p1,p2
        try:
            intr = calib.get_intrinsics(CalibrationType.COLOR)
            p = intr.param if hasattr(intr, "param") else intr
            k1 = float(getattr(p, "k1", 0.0))
            k2 = float(getattr(p, "k2", 0.0))
            p1 = float(getattr(p, "p1", 0.0))
            p2 = float(getattr(p, "p2", 0.0))
            k3 = float(getattr(p, "k3", 0.0))
            dist[0, :] = [k1, k2, p1, p2, k3]
        except Exception:
            # ok to proceed with zeros, but accuracy may drop
            print("[WARN] Cannot read distortion; using zeros. "
                  "If pose is biased, provide distortion manually or update pyk4a.")
            dist = np.zeros((1, 5), dtype=np.float64)

    return K, dist


def load_K_txt(path: str) -> np.ndarray:
    K = np.loadtxt(path).astype(np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {K.shape} from {path}")
    return K


# =========================
# PnP
# =========================
def solve_pnp_world_in_cam(obj_pts: np.ndarray, img_pts: np.ndarray,
                           K: np.ndarray, dist: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    obj_pts = np.ascontiguousarray(obj_pts, dtype=np.float64).reshape(-1, 3)
    img_pts = np.ascontiguousarray(img_pts, dtype=np.float64).reshape(-1, 2)

    if obj_pts.shape[0] < 4 or img_pts.shape[0] != obj_pts.shape[0]:
        return False, None, None, None

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts, img_pts, K, dist,
        iterationsCount=2000,
        reprojectionError=8.0,   # for 720p this is usually safe
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if ok:
        return True, rvec, tvec, inliers

    # fallback: non-RANSAC
    ok2, rvec2, tvec2 = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if ok2:
        return True, rvec2, tvec2, None

    return False, None, None, None


# =========================
# Corner buffer (temporal median)
# =========================
class CornerBuffer:
    def __init__(self, maxlen: int = 30):
        self.maxlen = int(maxlen)
        self.buf: Dict[str, deque] = {mid: deque(maxlen=self.maxlen) for mid in VALID_IDS}

    def push(self, dets: List[Dict]):
        for d in dets:
            cid = d["id"]
            if cid in self.buf:
                self.buf[cid].append(d["corners"].astype(np.float64))

    def available_ids(self, min_samples: int = 10) -> List[str]:
        out = []
        for mid in VALID_IDS:
            if len(self.buf[mid]) >= int(min_samples):
                out.append(mid)
        return out

    def median_corners(self, mid: str) -> np.ndarray:
        arr = np.stack(list(self.buf[mid]), axis=0)  # (T,4,2)
        return np.median(arr, axis=0)                # (4,2)


# =========================
# Optional: anti-flicker settings
# =========================
def try_set_anti_flicker(k4a: PyK4A, powerline_hz: int = 50):
    """
    Disable auto exposure and set exposure to a multiple of powerline half-period.
    Not all pyk4a builds support set_color_control; failure is OK.
    """
    if powerline_hz == 50:
        exposure_us = 20000   # 20ms
    else:
        exposure_us = 16667   # 16.67ms

    try:
        k4a.set_color_control(
            pyk4a.ColorControlCommand.EXPOSURE_TIME_ABSOLUTE,
            pyk4a.ColorControlMode.MANUAL,
            int(exposure_us)
        )
        k4a.set_color_control(
            pyk4a.ColorControlCommand.GAIN,
            pyk4a.ColorControlMode.MANUAL,
            0
        )
        k4a.set_color_control(
            pyk4a.ColorControlCommand.WHITEBALANCE,
            pyk4a.ColorControlMode.MANUAL,
            4500
        )
        print(f"[AntiFlicker] MANUAL exposure={exposure_us}us, gain=0, wb=4500K")
    except Exception as e:
        print(f"[AntiFlicker] set_color_control not supported: {e}")


# =========================
# Main
# =========================
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=str, required=True,
                    help="A4 template PNG path (full-page raster), e.g. hand_world_qr_a4_landscape-1.png")
    ap.add_argument("--out", type=str, default="./T_world_in_cam.txt",
                    help="output 4x4 txt (world->cam)")
    ap.add_argument("--out_cam_in_world", type=str, default="./T_cam_in_world.txt",
                    help="also save inverse (cam->world)")
    ap.add_argument("--use_manual_K", action="store_true", default=False,
                    help="if set, use --K_path instead of device intrinsics")
    ap.add_argument("--K_path", type=str, default="",
                    help="3x3 intrinsic matrix txt (fallback)")
    ap.add_argument("--powerline_hz", type=int, default=50, choices=[50, 60],
                    help="anti-flicker hint for exposure setting")
    ap.add_argument("--collect_frames", type=int, default=20,
                    help="how many frames to aggregate for one calibration")
    ap.add_argument("--min_samples", type=int, default=10,
                    help="min samples per marker for median")
    ap.add_argument("--min_markers", type=int, default=3,
                    help="min markers required for solvePnP")
    ap.add_argument("--show_axes_len", type=float, default=0.06,
                    help="axis length (meters) for drawFrameAxes")
    args = ap.parse_args()

    # 1) build world 3D corners from template
    world_corners_by_id = build_board_world_corners_from_template(args.template)
    print("[Board] Loaded world corners from template for IDs:", list(world_corners_by_id.keys()))

    # 2) start Azure Kinect DK in 720P color-only
    cfg = Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        depth_mode=pyk4a.DepthMode.OFF,
        camera_fps=pyk4a.FPS.FPS_30,
        synchronized_images_only=False,  # IMPORTANT: avoid the error you hit
    )
    k4a = PyK4A(cfg)
    k4a.start()
    time.sleep(0.2)

    try_set_anti_flicker(k4a, powerline_hz=args.powerline_hz)

    # 3) K, dist
    if args.use_manual_K:
        if not args.K_path:
            raise RuntimeError("--use_manual_K requires --K_path")
        K = load_K_txt(args.K_path)
        dist = np.zeros((1, 5), dtype=np.float64)
        print("[K] Using manual K from:", args.K_path)
    else:
        K, dist = get_color_K_dist_from_device(k4a)
        print("[K] Using device intrinsics.")
    print("[K] fx,fy,cx,cy =", float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
    print("[dist] =", dist.reshape(-1).tolist())

    detector = cv2.QRCodeDetector()
    buf = CornerBuffer(maxlen=max(args.collect_frames, 30))

    print("\n[UI] Press 'c' to calibrate (collect frames -> solvePnP -> save). Press 'q' or ESC to quit.\n")

    last_T = None

    while True:
        cap = k4a.get_capture()
        color = cap.color
        if color is None:
            continue

        # Azure color often arrives as BGRA
        if color.ndim == 3 and color.shape[2] == 4:
            bgr = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
        else:
            bgr = color.copy()

        dets = detect_qr_multi(detector, bgr)
        buf.push(dets)

        vis = bgr.copy()

        # draw detections
        for d in dets:
            cid = d["id"]
            c = d["corners"].astype(np.int32)
            cv2.polylines(vis, [c.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
            cx = int(np.mean(c[:, 0]))
            cy = int(np.mean(c[:, 1]))
            cv2.putText(vis, cid, (cx - 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # if we already have a pose, draw axes
        if last_T is not None:
            R = last_T[:3, :3]
            t = last_T[:3, 3]
            rvec, _ = cv2.Rodrigues(R)
            tvec = t.reshape(3, 1)
            try:
                cv2.drawFrameAxes(vis, K, dist, rvec, tvec, args.show_axes_len, 2)
            except Exception:
                pass

        cv2.imshow("AzureKinect(720P) QR Calibration", vis)
        key = cv2.waitKey(1) & 0xFF

        if key in [27, ord('q')]:
            break

        if key == ord('c'):
            # collect N frames worth of data (buffer already accumulating)
            valid_ids = buf.available_ids(min_samples=args.min_samples)
            if len(valid_ids) < args.min_markers:
                print(f"[WARN] Not enough stable markers. Have {len(valid_ids)} ids with >= {args.min_samples} samples: {valid_ids}")
                continue

            # build correspondences
            obj_list, img_list = [], []
            used_ids = []
            for mid in valid_ids:
                # 3D corners in world (meters), order TL,TR,BR,BL
                obj_c = world_corners_by_id[mid].astype(np.float64)  # (4,3)
                # 2D corners in image (pixels), median over time
                img_c = buf.median_corners(mid).astype(np.float64)    # (4,2)
                img_c = order_corners_tl_tr_br_bl(img_c)

                obj_list.append(obj_c)
                img_list.append(img_c)
                used_ids.append(mid)

            obj_pts = np.concatenate(obj_list, axis=0)
            img_pts = np.concatenate(img_list, axis=0)

            ok, rvec, tvec, inliers = solve_pnp_world_in_cam(obj_pts, img_pts, K, dist)
            if not ok:
                print("[ERROR] Calibration failed: solvePnPRansac failed.")
                print("        Common causes: wrong K for resolution, corner order mismatch, too few/unstable markers.")
                continue

            R, _ = cv2.Rodrigues(rvec)
            t = tvec.reshape(3)
            T_world_in_cam = make_T(R, t)        # X_cam = R * X_world + t
            T_cam_in_world = invert_T(T_world_in_cam)

            last_T = T_world_in_cam.copy()

            np.savetxt(args.out, T_world_in_cam, fmt="%.10f")
            np.savetxt(args.out_cam_in_world, T_cam_in_world, fmt="%.10f")

            nin = int(inliers.shape[0]) if inliers is not None else -1
            print("===============================================")
            print(f"[OK] Used marker IDs: {used_ids}")
            print(f"[OK] Inliers: {nin} / {obj_pts.shape[0]}")
            print(f"[Saved] T_world_in_cam -> {args.out}")
            print(f"[Saved] T_cam_in_world -> {args.out_cam_in_world}")
            print("[T_world_in_cam]\n", T_world_in_cam)
            print("[cam position in world] t_WC =", T_cam_in_world[:3, 3].tolist())
            print("===============================================")

    cv2.destroyAllWindows()
    k4a.stop()


if __name__ == "__main__":
    main()
