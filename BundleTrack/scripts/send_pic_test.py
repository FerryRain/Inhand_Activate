"""
@FileName：send_pic_folder.py
@Description：Send all frames (color/depth/mask) in folders to BundleTrack ZMQ server and receive pose.
"""
import os
import re
import glob
import json
import time
import zmq
import numpy as np
import cv2


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


# def natural_key(s: str):
#     """Sort like: 2 < 10, also works for '00019_1765....png'."""
#     return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]
def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

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
        color_bgr=None,        # uint8 HxWx3 (BGR)
        depth=None,            # uint16 HxW (mm) or float32 HxW (m)
        mask=None,             # uint8 HxW (0/1 or 0/255). Optional
        depth_type="uint16",
        id_str="00000",
        roi=None,              # [x0,x1,y0,y1] optional
        ob_in_cam=None,        # 4x4 float32 optional
        has_init=False,
):
    assert color_bgr is not None and depth is not None
    H, W = color_bgr.shape[:2]
    assert depth.shape[0] == H and depth.shape[1] == W

    # normalize mask (if provided)
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
        "depth_type": str(depth_type),   # "uint16" or "float32"
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


def iter_frames(rgb_dir, depth_dir, mask_dir=None, exts=("png", "jpg", "jpeg")):
    rgb_files = []
    for e in exts:
        rgb_files += glob.glob(os.path.join(rgb_dir, f"*.{e}"))
    rgb_files = sorted(rgb_files, key=natural_key)

    for rgb_path in rgb_files:
        base = os.path.splitext(os.path.basename(rgb_path))[0]
        # 默认 depth/mask 和 rgb 同名同后缀；如果你 depth 都是 png，也能自动兜底
        depth_path = None
        for e in exts:
            cand = os.path.join(depth_dir, base + f".{e}")
            if os.path.exists(cand):
                depth_path = cand
                break
        if depth_path is None:
            # 常见：depth固定png
            cand = os.path.join(depth_dir, base + ".png")
            if os.path.exists(cand):
                depth_path = cand

        mask_path = None
        if mask_dir is not None:
            for e in exts:
                cand = os.path.join(mask_dir, base + f".{e}")
                if os.path.exists(cand):
                    mask_path = cand
                    break
            if mask_path is None:
                cand = os.path.join(mask_dir, base + ".png")
                if os.path.exists(cand):
                    mask_path = cand

        yield base, rgb_path, depth_path, mask_path




def iter_rgb_files(rgb_dir):
    fs = glob.glob(os.path.join(rgb_dir, "*.png"))
    fs += glob.glob(os.path.join(rgb_dir, "*.jpg"))
    fs += glob.glob(os.path.join(rgb_dir, "*.jpeg"))
    return sorted(fs, key=natural_key)

if __name__ == "__main__":
    addr = "tcp://127.0.0.1:5550"

    root = "/home/ferry/data/Code2/Research/Inhand_Activate/BundleTrack/YCBInEOAT/mustard0"
    rgb_dir   = os.path.join(root, "rgb")
    depth_dir = os.path.join(root, "depth")
    mask_dir  = os.path.join(root, "masks")

    init_ob_in_cam = np.loadtxt(
        os.path.join(root, "annotated_poses/0000000.txt")
    ).astype(np.float32).reshape(4, 4)

    sock = make_client(addr=addr, timeout_ms=10000)

    for i, rgb_path in enumerate(iter_rgb_files(rgb_dir)):
        base = os.path.splitext(os.path.basename(rgb_path))[0]
        depth_path = os.path.join(depth_dir, base + ".png")
        mask_path  = os.path.join(mask_dir,  base + ".png")

        color = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        mask  = cv2.imread(mask_path,  cv2.IMREAD_UNCHANGED)

        if color is None or depth is None:
            print(f"[Skip] read failed: {base}")
            continue

        if i == 0:
            has_init = True
            ob_in_cam = init_ob_in_cam
        else:
            has_init = False
            ob_in_cam = None
        t = time.time()
        T, resp = send_frame_with_sock(
            sock,                      # ✅ 第一个参数是 sock
            color_bgr=color,
            depth=depth,
            mask=mask,
            depth_type="uint16",
            id_str=base,
            roi=None,
            has_init=has_init,
            ob_in_cam=ob_in_cam
        )

        print(f"[{i:04d}] {base} ok={resp.get('ok')} used_init={has_init} cost_time={time.time() - t:.3f}s")
