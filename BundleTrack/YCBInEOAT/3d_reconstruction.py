import os
import glob
import numpy as np
import cv2
import open3d as o3d

# ===================== 路径 =====================
ROOT = "./inhand_object2"
RGB_GLOB   = f"{ROOT}/rgb/*.png"
DEPTH_GLOB = f"{ROOT}/depth/*.png"
MASK_GLOB  = f"{ROOT}/masks/*.png"
POSE_GLOB  = f"{ROOT}/poses/*.txt"
K_PATH     = f"{ROOT}/cam_K.txt"

OUT_DIR = os.path.join(ROOT, "outputs_clean")
POSE_VIZ_DIR = os.path.join(OUT_DIR, "pose_viz")

# ===================== 基本参数 =====================
DEPTH_SCALE = 1000.0   # uint16(mm)->m: 1000；若depth已是米(float)则设1.0
DEPTH_TRUNC = 1.5      # TSDF depth_trunc（越小越不容易把背景融进去）

POSE_TRANSLATION_IN_MM = False  # 你的pose看起来像米(0.3m)，一般 False

# TSDF
VOXEL_LEN = 0.002
SDF_TRUNC = 0.01

# ===================== 关键：mask 清理（强烈推荐开启） =====================
KEEP_LARGEST_CC = True      # 只保留最大连通域，去掉零碎mask
ERODE_KERNEL = 5            # 5/7 都行
ERODE_ITERS  = 2            # 1~3 之间调，越大越“收紧”

# ===================== 关键：深度过滤（强烈推荐） =====================
Z_MIN, Z_MAX = 0.05, 1.2    # 按你的场景调：背景多就把Z_MAX调小
USE_GRAD_FILTER = True      # 去 depth discontinuity / flying pixels
GRAD_THRESH = 0.02          # 单位：m/px（起点值，按效果调）

# ===================== pose 去噪（可选） =====================
USE_POSE_SMOOTHING = True
ALPHA_TRANS = 0.25
ALPHA_ROT   = 0.25
CLAMP_TRANS_M = 0.05
CLAMP_ROT_DEG = 15.0

# ===================== pose 跳变剔除（可选） =====================
SKIP_POSE_JUMPS = True
MAX_DT = 0.03        # 3cm
MAX_DA_DEG = 10.0    # 10deg

# ===================== 输出/可视化 =====================
SAVE_POSE_VIZ = True
AXIS_LEN = 0.05
THICKNESS = 2

SAVE_MESH_PLY = True
SAVE_MESH_OBJ = True
SAVE_PCD_PLY  = True

KEEP_LARGEST_MESH_COMPONENT = True  # 最后只保留最大连通块（非常有效）


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

def smooth_poses_right_invariant(T_list):
    clamp_rot = np.deg2rad(CLAMP_ROT_DEG)
    T_hat = T_list[0].copy()
    out = [T_hat.copy()]

    for i in range(1, len(T_list)):
        T_meas = T_list[i]
        E = np.linalg.inv(T_hat) @ T_meas
        xi = se3_log(E)

        v = xi[:3]
        w = xi[3:]

        v_norm = np.linalg.norm(v)
        w_norm = np.linalg.norm(w)
        if v_norm > CLAMP_TRANS_M:
            v = v * (CLAMP_TRANS_M / (v_norm + 1e-12))
        if w_norm > clamp_rot:
            w = w * (clamp_rot / (w_norm + 1e-12))

        xi_f = np.zeros(6)
        xi_f[:3] = ALPHA_TRANS * v
        xi_f[3:] = ALPHA_ROT   * w

        T_hat = T_hat @ se3_exp(xi_f)
        out.append(T_hat.copy())

    return out


# ===================== 工具函数 =====================
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

def ensure_same_size(rgb, depth, mask):
    H, W = rgb.shape[:2]
    if depth.shape[:2] != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return depth, mask

def refine_mask(mask_raw):
    mask_u8 = (mask_raw > 0).astype(np.uint8)

    if KEEP_LARGEST_CC:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if num > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask_u8 = (labels == largest).astype(np.uint8)

    if ERODE_ITERS > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERODE_KERNEL, ERODE_KERNEL))
        mask_u8 = cv2.erode(mask_u8, k, iterations=ERODE_ITERS)

    return mask_u8.astype(bool)

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

def keep_largest_mesh_component(mesh: o3d.geometry.TriangleMesh):
    tri_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    tri_clusters = np.asarray(tri_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    if cluster_n_triangles.size == 0:
        return mesh
    keep = int(cluster_n_triangles.argmax())
    remove_mask = (tri_clusters != keep)
    mesh.remove_triangles_by_mask(remove_mask)
    mesh.remove_unreferenced_vertices()
    return mesh


# ===================== 主流程 =====================
def main():
    ensure_dir(OUT_DIR)
    if SAVE_POSE_VIZ:
        ensure_dir(POSE_VIZ_DIR)

    rgb_paths   = sorted(glob.glob(RGB_GLOB))
    depth_paths = sorted(glob.glob(DEPTH_GLOB))
    mask_paths  = sorted(glob.glob(MASK_GLOB))
    pose_paths  = sorted(glob.glob(POSE_GLOB))

    n = min(len(rgb_paths), len(depth_paths), len(mask_paths), len(pose_paths))
    if n == 0:
        raise RuntimeError("No data found. Check ROOT and subfolders.")
    rgb_paths, depth_paths, mask_paths, pose_paths = rgb_paths[:n], depth_paths[:n], mask_paths[:n], pose_paths[:n]

    K = np.loadtxt(K_PATH).reshape(3, 3).astype(np.float64)

    # intrinsics
    bgr0 = cv2.imread(rgb_paths[0], cv2.IMREAD_COLOR)
    H, W = bgr0.shape[:2]
    intr = o3d.camera.PinholeCameraIntrinsic(W, H, K[0,0], K[1,1], K[0,2], K[1,2])

    # load poses (BundleTrack saved ob_in_cam = T_obj_in_cam)
    T_raw = [load_pose_4x4(p) for p in pose_paths]
    T_use = smooth_poses_right_invariant(T_raw) if USE_POSE_SMOOTHING else T_raw

    # TSDF volume (world = object frame, extrinsic = world->camera = T_obj_in_cam)
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_LEN,
        sdf_trunc=SDF_TRUNC,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    used_prev = None
    integrated = 0

    for i in range(n):
        bgr = cv2.imread(rgb_paths[i], cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(depth_paths[i], cv2.IMREAD_UNCHANGED)
        mask_raw  = cv2.imread(mask_paths[i], cv2.IMREAD_UNCHANGED)
        if bgr is None or depth_raw is None or mask_raw is None:
            print(f"[{i}] read fail, skip")
            continue

        depth_raw, mask_raw = ensure_same_size(bgr, depth_raw, mask_raw)

        # depth -> meters
        if depth_raw.dtype.kind in ("u", "i"):
            depth_m = depth_raw.astype(np.float32) / float(DEPTH_SCALE)
        else:
            depth_m = depth_raw.astype(np.float32)

        # depth range filter
        depth_m[(depth_m < Z_MIN) | (depth_m > Z_MAX)] = 0.0

        # refine mask (largest CC + erosion)
        mask_bin = refine_mask(mask_raw)
        depth_m[~mask_bin] = 0.0

        # gradient filter (optional)
        if USE_GRAD_FILTER:
            dzdx = cv2.Sobel(depth_m, cv2.CV_32F, 1, 0, ksize=3)
            dzdy = cv2.Sobel(depth_m, cv2.CV_32F, 0, 1, ksize=3)
            grad = np.sqrt(dzdx*dzdx + dzdy*dzdy)
            depth_m[grad > GRAD_THRESH] = 0.0

        # pose jump skip (optional)
        T = T_use[i]
        if SKIP_POSE_JUMPS and used_prev is not None:
            dT = np.linalg.inv(used_prev) @ T
            dt = np.linalg.norm(dT[:3, 3])
            da = rot_angle_deg(dT[:3, :3])
            if dt > MAX_DT or da > MAX_DA_DEG:
                print(f"[skip jump] frame {i}: dt={dt:.3f}m da={da:.1f}deg")
                continue

        used_prev = T

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb.astype(np.uint8)),
            o3d.geometry.Image(depth_m.astype(np.float32)),
            depth_scale=1.0,
            depth_trunc=DEPTH_TRUNC,
            convert_rgb_to_intensity=False
        )

        vol.integrate(rgbd, intr, T)
        integrated += 1

        if SAVE_POSE_VIZ:
            viz = draw_pose6d_on_image(bgr.copy(), T, K, axis_len=AXIS_LEN, thickness=THICKNESS)
            cv2.putText(viz, f"{i:06d}", (5, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(POSE_VIZ_DIR, f"{i:06d}_pose_viz.jpg"), viz, [cv2.IMWRITE_JPEG_QUALITY, 90])

        if (i + 1) % 20 == 0 or (i + 1) == n:
            print(f"[{i+1}/{n}] processed, integrated={integrated}")

    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    # mesh cleanup
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    if KEEP_LARGEST_MESH_COMPONENT:
        mesh = keep_largest_mesh_component(mesh)
        mesh.compute_vertex_normals()

    pcd = vol.extract_point_cloud()

    # save
    if SAVE_MESH_PLY:
        o3d.io.write_triangle_mesh(os.path.join(OUT_DIR, "fused_mesh.ply"), mesh)
    if SAVE_MESH_OBJ:
        o3d.io.write_triangle_mesh(os.path.join(OUT_DIR, "fused_mesh.obj"), mesh, write_triangle_uvs=False)
    if SAVE_PCD_PLY:
        o3d.io.write_point_cloud(os.path.join(OUT_DIR, "fused_pcd.ply"), pcd)

    print("Saved to:", OUT_DIR)
    if SAVE_POSE_VIZ:
        print("Pose viz dir:", POSE_VIZ_DIR)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    o3d.visualization.draw_geometries([mesh, axis], window_name="TSDF Mesh (cleaned)")


if __name__ == "__main__":
    main()
