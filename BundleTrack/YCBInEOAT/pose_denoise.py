import os
import glob
import numpy as np
import cv2

# ===================== 配置区（按你数据改） =====================
ROOT = "./inhand_object2"
RGB_GLOB  = f"{ROOT}/rgb/*.png"       # 或 jpg
POSE_GLOB = f"{ROOT}/poses/*.txt"
K_PATH    = f"{ROOT}/cam_K.txt"

OUT_DIR = os.path.join(ROOT, "outputs_pose_opt")
OUT_POSE_DIR = os.path.join(OUT_DIR, "poses_smooth")
VIZ_RAW_DIR = os.path.join(OUT_DIR, "pose_viz_raw")
VIZ_SMOOTH_DIR = os.path.join(OUT_DIR, "pose_viz_smooth")

POSE_TRANSLATION_IN_MM = False  # 你的pose看起来像米(0.3m)，一般 False

# ============ 核心：去噪强度 ============
# right-invariant smoothing:
#   T_hat <- T_hat * Exp( [alpha_t*v, alpha_r*w] )  where  [v,w] = Log(inv(T_hat)*T_meas)
ALPHA_TRANS = 0.25    # 0~1，越小越稳
ALPHA_ROT   = 0.25    # 0~1，越小越稳

# 每帧最大允许更新（防止偶发抖动/跳变把轨迹拉飞）
CLAMP_TRANS_M = 0.05          # m
CLAMP_ROT_DEG = 15.0          # deg

# 如果检测到跳变（相对上一帧变化过大）就直接“跳过该测量”（更鲁棒）
SKIP_JUMPS = True
JUMP_TRANS_M = 0.08           # m
JUMP_ROT_DEG = 25.0           # deg

# 可视化
SAVE_VIZ = True
AXIS_LEN = 0.05
THICKNESS = 2


# ===================== Lie(SE3) 工具：优先用 SciPy，没有就 fallback =====================
try:
    from scipy.spatial.transform import Rotation as _R
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def skew(w):
    wx, wy, wz = w
    return np.array([[0, -wz, wy],
                     [wz, 0, -wx],
                     [-wy, wx, 0]], dtype=np.float64)

def so3_exp(w):
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    K = skew(k)
    return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)

def so3_log(Rm):
    cos_theta = (np.trace(Rm) - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-12:
        return np.zeros(3)
    w_hat = (Rm - Rm.T) / (2*np.sin(theta))
    return theta * np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]], dtype=np.float64)

def se3_exp(xi):
    v = xi[:3]
    w = xi[3:]
    if _HAS_SCIPY:
        Rm = _R.from_rotvec(w).as_matrix()
    else:
        Rm = so3_exp(w)

    theta = np.linalg.norm(w)
    W = skew(w)
    if theta < 1e-12:
        V = np.eye(3) + 0.5*W + (1.0/6.0)*(W@W)
    else:
        A = np.sin(theta)/theta
        B = (1-np.cos(theta))/(theta**2)
        C = (1-A)/(theta**2)
        V = np.eye(3) + B*W + C*(W@W)

    t = V @ v
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = Rm
    T[:3, 3] = t
    return T

def se3_log(T):
    Rm = T[:3,:3]
    t  = T[:3, 3]
    if _HAS_SCIPY:
        w = _R.from_matrix(Rm).as_rotvec()
    else:
        w = so3_log(Rm)

    theta = np.linalg.norm(w)
    W = skew(w)
    if theta < 1e-12:
        V_inv = np.eye(3) - 0.5*W + (1.0/12.0)*(W@W)
    else:
        A = np.sin(theta)/theta
        B = (1-np.cos(theta))/(theta**2)
        coeff = (1.0/(theta**2)) * (1.0 - A/(2.0*B))
        V_inv = np.eye(3) - 0.5*W + coeff*(W@W)

    v = V_inv @ t
    xi = np.zeros(6, dtype=np.float64)
    xi[:3] = v
    xi[3:] = w
    return xi

def reorthonormalize_R(T):
    Rm = T[:3, :3]
    U, _, Vt = np.linalg.svd(Rm)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    T2 = T.copy()
    T2[:3, :3] = Rn
    return T2


# ===================== IO + Viz =====================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_pose_4x4(path):
    T = np.loadtxt(path).astype(np.float64)
    if T.shape == (3, 4):
        T4 = np.eye(4, dtype=np.float64)
        T4[:3, :4] = T
        T = T4
    if T.shape != (4,4):
        raise ValueError(f"Bad pose shape {T.shape} in {path}")
    if POSE_TRANSLATION_IN_MM:
        T[:3, 3] *= 0.001
    return T

def rot_angle_deg(R):
    tr = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))

def project_cam_point(Pc, K):
    z = Pc[2]
    if z <= 1e-6:
        return None
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    u = fx * (Pc[0] / z) + cx
    v = fy * (Pc[1] / z) + cy
    return (float(u), float(v))

def draw_pose6d_on_image(img_bgr, T_obj_in_cam, K, axis_len=0.05, thickness=2, text_org=(5, 60)):
    Rm = T_obj_in_cam[:3,:3]
    t  = T_obj_in_cam[:3, 3]

    O = t
    X = t + Rm @ np.array([axis_len, 0, 0], dtype=np.float64)
    Y = t + Rm @ np.array([0, axis_len, 0], dtype=np.float64)
    Z = t + Rm @ np.array([0, 0, axis_len], dtype=np.float64)

    o2 = project_cam_point(O, K)
    if o2 is None:
        return img_bgr
    o2i = (int(round(o2[0])), int(round(o2[1])))
    cv2.circle(img_bgr, o2i, 3, (0,255,255), -1, cv2.LINE_AA)

    x2 = project_cam_point(X, K)
    y2 = project_cam_point(Y, K)
    z2 = project_cam_point(Z, K)
    if x2 is not None:
        cv2.line(img_bgr, o2i, (int(round(x2[0])), int(round(x2[1]))), (0,0,255), thickness, cv2.LINE_AA)
    if y2 is not None:
        cv2.line(img_bgr, o2i, (int(round(y2[0])), int(round(y2[1]))), (0,255,0), thickness, cv2.LINE_AA)
    if z2 is not None:
        cv2.line(img_bgr, o2i, (int(round(z2[0])), int(round(z2[1]))), (255,0,0), thickness, cv2.LINE_AA)

    ss1 = f"t = [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]"
    cv2.putText(img_bgr, ss1, text_org, cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,255), 1, cv2.LINE_AA)
    return img_bgr


# ===================== 核心：pose 去噪 =====================
def smooth_poses_right_invariant(T_list):
    clamp_rot = np.deg2rad(CLAMP_ROT_DEG)
    jump_rot  = np.deg2rad(JUMP_ROT_DEG)

    T_hat = T_list[0].copy()
    T_hat = reorthonormalize_R(T_hat)
    out = [T_hat.copy()]

    for i in range(1, len(T_list)):
        T_meas = reorthonormalize_R(T_list[i])

        # 跳变检测（基于相对变换）
        if SKIP_JUMPS:
            dT_meas = np.linalg.inv(T_list[i-1]) @ T_list[i]
            dt_meas = np.linalg.norm(dT_meas[:3, 3])
            da_meas = rot_angle_deg(dT_meas[:3, :3])
            if dt_meas > JUMP_TRANS_M or da_meas > JUMP_ROT_DEG:
                # 直接沿用上一估计（不吸收该帧测量）
                out.append(T_hat.copy())
                continue

        # 右不变误差：E = inv(T_hat) * T_meas
        E = np.linalg.inv(T_hat) @ T_meas
        xi = se3_log(E)
        v, w = xi[:3], xi[3:]

        # clamp 单步更新
        vn = np.linalg.norm(v)
        wn = np.linalg.norm(w)
        if vn > CLAMP_TRANS_M:
            v = v * (CLAMP_TRANS_M / (vn + 1e-12))
        if wn > clamp_rot:
            w = w * (clamp_rot / (wn + 1e-12))

        xi_f = np.zeros(6, dtype=np.float64)
        xi_f[:3] = ALPHA_TRANS * v
        xi_f[3:] = ALPHA_ROT   * w

        T_hat = T_hat @ se3_exp(xi_f)
        T_hat = reorthonormalize_R(T_hat)
        out.append(T_hat.copy())

    return out


def main():
    ensure_dir(OUT_DIR)
    ensure_dir(OUT_POSE_DIR)
    if SAVE_VIZ:
        ensure_dir(VIZ_RAW_DIR)
        ensure_dir(VIZ_SMOOTH_DIR)

    rgb_paths  = sorted(glob.glob(RGB_GLOB))
    pose_paths = sorted(glob.glob(POSE_GLOB))
    if len(pose_paths) == 0:
        raise RuntimeError("No pose files found.")
    if SAVE_VIZ and len(rgb_paths) == 0:
        raise RuntimeError("SAVE_VIZ=True but no rgb images found.")

    n = min(len(rgb_paths), len(pose_paths)) if SAVE_VIZ else len(pose_paths)
    rgb_paths = rgb_paths[:n]
    pose_paths = pose_paths[:n]

    K = np.loadtxt(K_PATH).reshape(3, 3).astype(np.float64)

    # load poses (BundleTrack saved: T_obj_in_cam)
    T_raw = [load_pose_4x4(p) for p in pose_paths]

    print("[Info] First pose:\n", T_raw[0])
    print("[Info] Last  pose:\n", T_raw[-1])
    print(f"[Info] Total poses: {len(T_raw)}")

    # smooth
    T_smooth = smooth_poses_right_invariant(T_raw)

    # save smoothed poses with same filename
    for pp, Ts in zip(pose_paths, T_smooth):
        name = os.path.basename(pp)
        out_path = os.path.join(OUT_POSE_DIR, name)
        np.savetxt(out_path, Ts, fmt="%.10f")

    print("[OK] Saved smoothed poses to:", OUT_POSE_DIR)

    # save viz
    if SAVE_VIZ:
        for i, (rp, T0, T1) in enumerate(zip(rgb_paths, T_raw, T_smooth)):
            bgr = cv2.imread(rp, cv2.IMREAD_COLOR)
            if bgr is None:
                continue

            raw_viz = draw_pose6d_on_image(bgr.copy(), T0, K, axis_len=AXIS_LEN, thickness=THICKNESS)
            sm_viz  = draw_pose6d_on_image(bgr.copy(), T1, K, axis_len=AXIS_LEN, thickness=THICKNESS)

            cv2.putText(raw_viz, f"{i:06d} RAW", (5, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 1, cv2.LINE_AA)
            cv2.putText(sm_viz,  f"{i:06d} SMOOTH", (5, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 1, cv2.LINE_AA)

            # cv2.imwrite(os.path.join(VIZ_RAW_DIR,    f"{i:06d}.jpg"), raw_viz, [cv2.IMWRITE_JPEG_QUALITY, 90])
            cv2.imwrite(os.path.join(VIZ_SMOOTH_DIR, f"{i:06d}.jpg"), sm_viz,  [cv2.IMWRITE_JPEG_QUALITY, 90])

        print("[OK] Saved pose visualization to:")
        print("  -", VIZ_RAW_DIR)
        print("  -", VIZ_SMOOTH_DIR)


if __name__ == "__main__":
    main()
