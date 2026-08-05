"""
@FileName：samdataset.py
@Description：Use SAM2 to segment an image sequence and export per-frame masks for BundleTrack.
              Save 0/1 masks to masks/, 0/255 masks to masks_viz/, and save a pose txt for the prompt frame.
              Pose options:
                (A) center+standard axes (R=I, t=center3D) from mask+depth+intrinsics
                (B) user provided 4x4 pose txt

              + Timing (ONLY inference time):
                Record per-frame inference time and save to CSV:
                  - infer_ms: time spent to get next frame result from predictor iterator
                  (Mask saving / pose saving are NOT included)
"""

import os
import glob
import json
import argparse
import numpy as np
from PIL import Image

import time
import csv

import torch
import cv2

from sam2.build_sam import build_sam2_video_predictor


IMG_EXTS = ("*.jpg", "*.jpeg", "*.png")
DEPTH_EXTS = ("*.png", "*.tif", "*.tiff", "*.exr", "*.npy")


def sorted_frames(img_dir, exts=IMG_EXTS):
    files = []
    for e in exts:
        files += glob.glob(os.path.join(img_dir, e))
    files.sort()
    return files


def ensure_jpg_dir(frames, tmp_dir):
    """If input frames are png, convert to jpg into tmp_dir (same basename)."""
    if all(f.lower().endswith((".jpg", ".jpeg")) for f in frames):
        return os.path.dirname(frames[0])

    os.makedirs(tmp_dir, exist_ok=True)
    for f in frames:
        base = os.path.splitext(os.path.basename(f))[0]
        out = os.path.join(tmp_dir, base + ".jpg")
        if os.path.exists(out):
            continue
        Image.open(f).convert("RGB").save(out, quality=95)
    return tmp_dir


def collect_points_on_image(img_path, win_name="SAM2 - Click Points"):
    """
    LMB: positive point (label=1)
    RMB: negative point (label=0)
    u  : undo
    Enter/q: done
    Esc: cancel
    """
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Cannot read image: {img_path}")

    points, labels = [], []
    help_text = "LMB:+  RMB:-   u:undo   Enter/q:done   Esc:cancel"

    def redraw():
        vis = img_bgr.copy()
        for (x, y), lb in zip(points, labels):
            color = (0, 255, 0) if lb == 1 else (0, 0, 255)
            cv2.circle(vis, (int(x), int(y)), 5, color, -1)
        cv2.putText(vis, help_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, help_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.imshow(win_name, vis)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((float(x), float(y)))
            labels.append(1)
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append((float(x), float(y)))
            labels.append(0)
            redraw()

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, ord('q')):  # Enter or q
            break
        elif key == ord('u'):
            if points:
                points.pop()
                labels.pop()
                redraw()
        elif key == 27:  # Esc
            cv2.destroyWindow(win_name)
            raise RuntimeError("Canceled by user (ESC).")

    cv2.destroyWindow(win_name)

    if len(points) == 0:
        raise RuntimeError("No points selected. Please click at least one point.")

    return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)


def parse_points_str(s):
    """Format: 'x,y;x,y;...' """
    pts = []
    for item in s.split(";"):
        item = item.strip()
        if not item:
            continue
        x, y = item.split(",")
        pts.append((float(x), float(y)))
    if len(pts) == 0:
        raise ValueError("Empty --point string.")
    return np.array(pts, dtype=np.float32)


def parse_labels_str(s, n):
    arr = [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    if len(arr) != n:
        raise ValueError(f"--labels length {len(arr)} != num points {n}")
    return np.array(arr, dtype=np.int32)


def to_hw_uint8(mask_np: np.ndarray) -> np.ndarray:
    """
    Convert mask to 2D (H,W) uint8.
    Handles shapes like:
      (H,W)
      (1,H,W)
      (H,W,1)
      (C,H,W)  -> take first channel
      (H,W,C)  -> take first channel
    """
    m = np.asarray(mask_np)
    m = np.squeeze(m)  # remove dims of size 1

    if m.ndim == 2:
        return m.astype(np.uint8)

    if m.ndim == 3:
        # (C,H,W)
        if m.shape[0] <= 4 and m.shape[0] < m.shape[-1] and m.shape[0] < m.shape[-2]:
            m = m[0]
        else:
            # (H,W,C)
            m = m[..., 0]
        return m.astype(np.uint8)

    raise ValueError(f"Mask has unsupported shape after squeeze: {m.shape}")


def read_intrinsics(intrinsics_json=None, camK_txt=None):
    """Return fx, fy, cx, cy."""
    if intrinsics_json is not None:
        with open(intrinsics_json, "r") as f:
            d = json.load(f)
        fx, fy = float(d["fx"]), float(d["fy"])
        cx, cy = float(d["ppx"]), float(d["ppy"])
        return fx, fy, cx, cy

    if camK_txt is not None:
        K = np.loadtxt(camK_txt).reshape(3, 3)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        return fx, fy, cx, cy

    raise RuntimeError("Need intrinsics: provide --intrinsics_json or --camK_txt.")


def find_depth_file(depth_dir, base):
    """Find depth file by basename (base + ext). Try common extensions."""
    for ext in [".png", ".tif", ".tiff", ".exr", ".npy"]:
        fp = os.path.join(depth_dir, base + ext)
        if os.path.exists(fp):
            return fp
    cand = glob.glob(os.path.join(depth_dir, base + ".*"))
    return cand[0] if cand else None


def load_depth(depth_fp):
    ext = os.path.splitext(depth_fp)[1].lower()
    if ext == ".npy":
        return np.load(depth_fp)
    d = cv2.imread(depth_fp, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise RuntimeError(f"Cannot read depth: {depth_fp}")
    return d


def depth_to_meters(depth, depth_scale=0.001):
    """
    Convert depth array to float meters.
    - If depth is uint16/int: assume it's in "1/depth_scale" units (default mm => 0.001m)
    - If float: assume it's already meters
    """
    if np.issubdtype(depth.dtype, np.floating):
        return depth.astype(np.float32)
    return depth.astype(np.float32) * float(depth_scale)


def compute_mask_centroid_uv(mask01):
    ys, xs = np.nonzero(mask01 > 0)
    if len(xs) == 0:
        return None
    u = float(xs.mean())
    v = float(ys.mean())
    return u, v


def compute_center3d_from_mask_depth(mask01, depth_m, fx, fy, cx, cy):
    """
    Robust center:
    - compute centroid (u,v) from mask
    - take median depth over mask (valid > 0)
    - project to 3D (camera frame)
    """
    uv = compute_mask_centroid_uv(mask01)
    if uv is None:
        return None

    u, v = uv
    dvals = depth_m[mask01 > 0]
    dvals = dvals[np.isfinite(dvals)]
    dvals = dvals[dvals > 0]
    if dvals.size == 0:
        return None
    Z = float(np.median(dvals))

    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z
    return X, Y, Z


def read_pose_txt(pose_txt_path):
    T = np.loadtxt(pose_txt_path).reshape(4, 4)
    return T.astype(np.float64)


def write_pose_txt(T, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for r in range(4):
            f.write(" ".join(f"{float(v):.18e}" for v in T[r]) + "\n")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--img_dir",
                    default="/home/ferry/data/Code2/Research/Inhand_Activate/Tracking/BundleTrack/YCBInEOAT/002/rgb",
                    help="RGB frames dir")
    ap.add_argument("--out_dir",
                    default="/home/ferry/data/Code2/Research/Inhand_Activate/Tracking/BundleTrack/YCBInEOAT/002/masks",
                    help="Output masks dir (0/1)")
    ap.add_argument("--out_viz_dir",
                    default="/home/ferry/data/Code2/Research/Inhand_Activate/Tracking/BundleTrack/YCBInEOAT/002/masks_viz",
                    help="Output masks_viz dir (0/255)")

    ap.add_argument("--checkpoint", default="./sam2.1_hiera_tiny.pt", help="SAM2 checkpoint")
    ap.add_argument("--model_cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml", help="SAM2 config yaml")

    ap.add_argument("--prompt_frame", type=int, default=0)
    ap.add_argument("--obj_id", type=int, default=1)

    # prompt
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--point", type=str, default=None, help='non-GUI: "x,y;x,y;..."')
    ap.add_argument("--labels", type=str, default=None, help='non-GUI: "1,1,0,..."')
    ap.add_argument("--box", type=str, default=None, help='non-GUI: "x0,y0,x1,y1"')

    ap.add_argument("--tmp_jpg_dir", default=None)

    # pose saving
    ap.add_argument("--save_pose", action="store_true",default=True,
                    help="Save pose txt for prompt_frame into annotated_poses/xxxx.txt")
    ap.add_argument("--pose_out_dir",
                    default="/home/ferry/data/Code2/Research/Inhand_Activate/Tracking/BundleTrack/YCBInEOAT/002/annotated_poses",
                    help="Default: sibling of img_dir -> ../annotated_poses")
    ap.add_argument("--pose_txt", default=None,
                    help="If provided, directly write this 4x4 pose to prompt frame (mode B).")

    # for center+standard axes pose (mode A)
    ap.add_argument("--depth_dir",
                    default="/home/ferry/data/Code2/Research/Inhand_Activate/Tracking/BundleTrack/YCBInEOAT/002/depth",
                    help="Depth dir (needed for center+standard pose)")
    ap.add_argument("--intrinsics_json",
                    default=None,
                    help="intrinsics.json with fx/fy/ppx/ppy")
    ap.add_argument("--camK_txt", default="/home/ferry/data/Code2/Research/Inhand_Activate/Tracking/BundleTrack/YCBInEOAT/cube_purple_2/xyz/cam_K.txt", help="cam_K.txt (3x3)")
    ap.add_argument("--depth_scale", type=float, default=0.001,
                    help="Scale for integer depth to meters. Default 0.001 (mm->m).")

    # timing (ONLY inference time)
    ap.add_argument("--time_csv", default=None, help="Save per-frame inference timing CSV (default: out_dir/infer_times.csv)")
    ap.add_argument("--print_every", type=int, default=1, help="Print timing every N frames (0 disables)")
    ap.add_argument("--cuda_sync", action="store_true",
                    help="Call torch.cuda.synchronize() around timing for more accurate GPU timing (slower).")

    args = ap.parse_args()

    frames = sorted_frames(args.img_dir, IMG_EXTS)
    if len(frames) == 0:
        raise RuntimeError(f"No images found in {args.img_dir}")

    tmp_dir = args.tmp_jpg_dir or os.path.join(args.img_dir, "_sam2_jpg")
    video_dir = ensure_jpg_dir(frames, tmp_dir)
    video_frames = sorted_frames(video_dir, IMG_EXTS)  # align with predictor frame_idx

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.out_viz_dir, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")

    predictor = build_sam2_video_predictor(args.model_cfg, args.checkpoint)
    if hasattr(predictor, "fill_hole_area"):
        predictor.fill_hole_area = 0

    use_gui = args.interactive or ((args.point is None) and (args.box is None))
    pose_saved = False

    time_rows = []

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=video_dir)

        # prompt
        if use_gui:
            prompt_img_path = frames[args.prompt_frame]
            print(f"[GUI] Click points on: {prompt_img_path}")
            points, labels = collect_points_on_image(prompt_img_path)

            if hasattr(predictor, "add_new_points_or_box"):
                predictor.add_new_points_or_box(state, frame_idx=args.prompt_frame,
                                                obj_id=args.obj_id, points=points, labels=labels)
            else:
                predictor.add_new_points(state, frame_idx=args.prompt_frame,
                                         obj_id=args.obj_id, points=points, labels=labels)
        else:
            if (args.point is None) == (args.box is None):
                raise RuntimeError("Provide exactly one of --point or --box in non-interactive mode.")

            if args.point is not None:
                points = parse_points_str(args.point)
                if args.labels is None:
                    labels = np.ones((points.shape[0],), dtype=np.int32)
                else:
                    labels = parse_labels_str(args.labels, points.shape[0])

                if hasattr(predictor, "add_new_points_or_box"):
                    predictor.add_new_points_or_box(state, frame_idx=args.prompt_frame,
                                                    obj_id=args.obj_id, points=points, labels=labels)
                else:
                    predictor.add_new_points(state, frame_idx=args.prompt_frame,
                                             obj_id=args.obj_id, points=points, labels=labels)
            else:
                x0, y0, x1, y1 = [float(v) for v in args.box.split(",")]
                box = np.array([x0, y0, x1, y1], dtype=np.float32)
                predictor.add_new_points_or_box(state, frame_idx=args.prompt_frame,
                                                obj_id=args.obj_id, box=box)

        # propagate + save (only inference time measured)
        it = predictor.propagate_in_video(state)

        while True:
            # measure only the time spent to obtain the next result from iterator
            if args.cuda_sync:
                torch.cuda.synchronize()
            t_infer0 = time.perf_counter()

            try:
                frame_idx, obj_ids, mask_logits = next(it)
            except StopIteration:
                break

            if args.cuda_sync:
                torch.cuda.synchronize()
            t_infer1 = time.perf_counter()
            infer_dt = t_infer1 - t_infer0  # seconds

            # === post-processing & saving (NOT included in inference time) ===
            obj_ids = list(obj_ids)
            k = obj_ids.index(args.obj_id) if args.obj_id in obj_ids else 0

            m01 = (mask_logits[k] > 0).to(torch.uint8).cpu().numpy()
            m01 = to_hw_uint8(m01)  # ensure (H,W)
            m255 = (m01 * 255).astype(np.uint8)

            base = os.path.splitext(os.path.basename(video_frames[frame_idx]))[0]
            name = base + ".png"

            Image.fromarray(m01, mode="L").save(os.path.join(args.out_dir, name))
            Image.fromarray(m255, mode="L").save(os.path.join(args.out_viz_dir, name))

            # Save pose for prompt frame (only once)
            if args.save_pose and (not pose_saved) and (frame_idx == args.prompt_frame):
                pose_out_dir = args.pose_out_dir
                if pose_out_dir is None:
                    seq_dir = os.path.dirname(args.img_dir.rstrip("/"))
                    pose_out_dir = os.path.join(seq_dir, "annotated_poses")

                pose_path = os.path.join(pose_out_dir, base + ".txt")

                if args.pose_txt is not None:
                    # Mode B: user provided 4x4
                    T = read_pose_txt(args.pose_txt)
                    write_pose_txt(T, pose_path)
                    print(f"[POSE] wrote provided pose to: {pose_path}")
                else:
                    # Mode A: center+standard axes (requires depth+intrinsics)
                    if args.depth_dir is None:
                        raise RuntimeError("For center+standard pose, please provide --depth_dir (or use --pose_txt).")

                    fx, fy, cx, cy = read_intrinsics(args.intrinsics_json, args.camK_txt)

                    depth_fp = find_depth_file(args.depth_dir, base)
                    if depth_fp is None:
                        raise RuntimeError(f"Cannot find depth for {base} in {args.depth_dir}")

                    depth = load_depth(depth_fp)
                    depth_m = depth_to_meters(depth, depth_scale=args.depth_scale)

                    center3d = compute_center3d_from_mask_depth(m01, depth_m, fx, fy, cx, cy)
                    if center3d is None:
                        raise RuntimeError(f"Cannot compute center3D for frame {base} (mask empty or depth invalid).")

                    X, Y, Z = center3d
                    T = np.eye(4, dtype=np.float64)
                    T[0, 3] = X
                    T[1, 3] = Y
                    T[2, 3] = Z

                    write_pose_txt(T, pose_path)
                    print(f"[POSE] wrote center+standard pose to: {pose_path}")
                    print(f"       center3D (camera): X={X:.4f}, Y={Y:.4f}, Z={Z:.4f} (meters)")

                pose_saved = True

            # record only inference time
            infer_ms = infer_dt * 1000.0
            time_rows.append([int(frame_idx), base, infer_ms])

            if args.print_every > 0 and (frame_idx % args.print_every == 0):
                print(f"[TIME] frame={frame_idx:04d} infer={infer_ms:.2f}ms")

    # Save timing CSV (only inference time)
    csv_path = args.time_csv or os.path.join(args.out_dir, "infer_times.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "name", "infer_ms"])
        writer.writerows(time_rows)

    arr = np.array([r[2] for r in time_rows], dtype=np.float64)  # infer_ms
    if arr.size > 0:
        print(f"[TIME] saved: {csv_path}")
        print(f"[TIME] frames={arr.size}, mean={arr.mean():.2f}ms, median={np.median(arr):.2f}ms, p90={np.percentile(arr, 90):.2f}ms")

    print(f"[OK] masks (0/1) saved to: {args.out_dir}")
    print(f"[OK] masks_viz (0/255) saved to: {args.out_viz_dir}")
    if args.save_pose:
        print(f"[OK] pose saved for prompt_frame={args.prompt_frame} (once).")


if __name__ == "__main__":
    main()
