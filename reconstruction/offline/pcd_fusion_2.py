"""
@FileName：final_recon_dual_output.py
@Description：
    从 BundleTrack keyframes 生成序列化的累积融合点云。
    1. 逐帧累加点云，每增加一帧生成一个当前的融合结果。
    2. 严格保留原始去噪逻辑（SOR, ROR, DBSCAN）。
    3. 生成两个文件：标准颜色二进制 PLY 和 纯坐标 ASCII PLY。
    4. 输出保存至 pcd 子目录下，命名格式为 recon_0_000xxx.ply。
@Author：Ferry (Refactor by ChatGPT)
@Time：2026-02-03
"""

import os
import glob
import argparse
import numpy as np
import cv2
import open3d as o3d

# ---------------- IO (保持不变) ----------------
def load_K_txt(path: str) -> np.ndarray:
    K = np.loadtxt(path).astype(np.float64)
    return K

def load_T_txt(path: str) -> np.ndarray:
    T = np.loadtxt(path).astype(np.float64)
    if T.shape == (3, 4): T = np.vstack([T, [0, 0, 0, 1]])
    return T

def read_color_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None

def read_mask_u8_255(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None: return None
    if m.max() <= 1: m = (m * 255).astype(np.uint8)
    return m

def read_depth_any(path: str) -> np.ndarray:
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is not None and depth.ndim == 3: depth = depth[:, :, 0]
    return depth

def find_first_existing(paths):
    for p in paths:
        if os.path.exists(p): return p
    return None

def parse_frame_id(fid: str) -> int:
    digits = "".join([c for c in fid if c.isdigit()])
    return int(digits) if digits else 0

def make_xyz_only(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """提取纯坐标点云（剥离颜色）"""
    xyz = o3d.geometry.PointCloud()
    xyz.points = o3d.utility.Vector3dVector(np.asarray(pcd.points).astype(np.float64))
    return xyz

# ---------------- 去噪与预处理 (严格保留原始参数) ----------------
def erode_mask(mask: np.ndarray, k: int, iters: int) -> np.ndarray:
    if k <= 1 or iters <= 0: return mask
    ker = np.ones((k, k), np.uint8)
    return cv2.erode(mask, ker, iterations=iters)

def depth_range_filter(depth: np.ndarray, scale: float, zmin: float, zmax: float) -> np.ndarray:
    out = depth.copy()
    if out.dtype == np.uint16:
        zmin_r, zmax_r = int(zmin * scale), int(zmax * scale)
        out[(out < zmin_r) | (out > zmax_r)] = 0
    else:
        out[(out < zmin) | (out > zmax)] = 0.0
    return out

def post_clean_aggressive(pcd, args):
    """强力全局清洗 (逻辑与参数完全保留自源代码)"""
    if pcd.is_empty(): return pcd

    # 1. 统计离群点剔除 (移除飞点)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=0.8)
    
    # 2. 全局体素下采样 (合并重影)
    if args.voxel > 0:
        pcd = pcd.voxel_down_sample(args.voxel)
    
    # 3. 半径离群点剔除 (移除孤岛)
    pcd, ind = pcd.remove_radius_outlier(nb_points=20, radius=args.voxel * 5)
    
    # 4. DBSCAN 聚类 (只留最大连通体)
    labels = np.array(pcd.cluster_dbscan(eps=args.voxel * 4, min_points=20))
    if labels.size > 0 and labels.max() >= 0:
        counts = np.bincount(labels[labels >= 0])
        pcd = pcd.select_by_index(np.where(labels == np.argmax(counts))[0])
    
    return pcd

# ---------------- 核心流程：改为逐帧输出 ----------------
def reconstruct_and_save_all(fids, rgb_dir, dep_dir, msk_dir, pos_dir, intrinsic, args):
    # 创建输出子目录 pcd
    pcd_out_dir = os.path.join(args.out_dir, "pcd")
    os.makedirs(pcd_out_dir, exist_ok=True)
    
    fused_obj = o3d.geometry.PointCloud()
    guessed_scale = None

    for i, fid in enumerate(fids):
        rgb_path = find_first_existing([os.path.join(rgb_dir, f"{fid}.jpg"), os.path.join(rgb_dir, f"{fid}.png")])
        dep_path = find_first_existing([os.path.join(dep_dir, f"{fid}.png"), os.path.join(dep_dir, f"{fid}.exr")])
        msk_path = find_first_existing([os.path.join(msk_dir, f"{fid}.png"), os.path.join(msk_dir, f"{fid}.jpg")])
        pose_path = os.path.join(pos_dir, f"{fid}.txt")

        if not all([rgb_path, dep_path, msk_path, os.path.exists(pose_path)]):
            continue

        rgb, depth, mask = read_color_rgb(rgb_path), read_depth_any(dep_path), read_mask_u8_255(msk_path)
        T_cam_in_ob = np.linalg.inv(load_T_txt(pose_path))

        if args.depth_scale < 0:
            if guessed_scale is None: guessed_scale = 1000.0 if depth.dtype == np.uint16 else 1.0
            cur_scale = guessed_scale
        else: cur_scale = args.depth_scale

        mask = erode_mask(mask, args.mask_erode_k, args.mask_erode_iter)
        depth = depth_range_filter(depth, cur_scale, args.zmin, args.zmax)
        
        depth_masked = depth.copy()
        depth_masked[mask == 0] = 0
        
        o3d_color = o3d.geometry.Image(rgb)
        o3d_depth = o3d.geometry.Image(depth_masked.astype(np.uint16) if depth_masked.dtype==np.uint16 else depth_masked.astype(np.float32))
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, depth_scale=cur_scale, depth_trunc=args.depth_trunc, convert_rgb_to_intensity=False
        )
        pcd_cam = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        
        # 单帧初步去噪 (原始参数)
        pcd_cam, _ = pcd_cam.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # 将当前帧加入累积点云
        fused_obj += pcd_cam.transform(T_cam_in_ob)

        # --- 核心修改：每增加一帧，生成并保存当前的融合结果 ---
        # 1. 获取当前帧号并格式化为 6 位数字
        frame_id_val = parse_frame_id(fid)
        frame_str = str(frame_id_val).zfill(6)
        
        # 2. 执行原始去噪逻辑 (对当前累积的点云副本进行清洗)
        current_cleaned = post_clean_aggressive(fused_obj, args)

        if not current_cleaned.is_empty():
            # 输出 1: 标准颜色二进制文件
            out_bin = os.path.join(pcd_out_dir, f"recon_0_{frame_str}.ply")
            o3d.io.write_point_cloud(out_bin, current_cleaned, write_ascii=False)

            # 输出 2: 纯坐标 ASCII 文件
            xyz_pcd = make_xyz_only(current_cleaned)
            if args.xyz_down_voxel > 0:
                xyz_pcd = xyz_pcd.voxel_down_sample(args.xyz_down_voxel)
            
            out_ascii = os.path.join(pcd_out_dir, f"recon_0_{frame_str}_xyz_ascii.ply")
            o3d.io.write_point_cloud(out_ascii, xyz_pcd, write_ascii=True)
            
            print(f"进度: {i+1}/{len(fids)} | 已处理并保存帧: {frame_str} (点数: {len(current_cleaned.points)})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug_dir", default="/home/ferry/data/Code2/Research/Inhand_Activate/Tracking/BundleTrack/results/002", type=str)
    ap.add_argument("--out_dir", default="/home/ferry/data/Code2/Research/Inhand_Activate/Tracking/BundleTrack/YCBInEOAT/002", type=str)
    ap.add_argument("--K_path", default="/home/ferry/data/Code2/Research/Inhand_Activate/Tracking/BundleTrack/YCBInEOAT/002/cam_K.txt", type=str)
    
    ap.add_argument("--max_frame_id", type=int, default=1000)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--voxel", type=float, default=0.001)
    ap.add_argument("--depth_trunc", type=float, default=0.8)
    ap.add_argument("--mask_erode_k", type=int, default=7)
    ap.add_argument("--mask_erode_iter", type=int, default=1)
    ap.add_argument("--zmin", type=float, default=0.1)
    ap.add_argument("--zmax", type=float, default=1.0)
    ap.add_argument("--depth_scale", type=float, default=-1.0)
    ap.add_argument("--xyz_down_voxel", type=float, default=0.001)

    args = ap.parse_args()

    base = os.path.join(args.debug_dir, "keyframes")
    rgb_dir, dep_dir = os.path.join(base, "rgb_full"), os.path.join(base, "depth")
    msk_dir, pos_dir = os.path.join(base, "mask"), os.path.join(base, "poses")

    rgb_list = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")) + glob.glob(os.path.join(rgb_dir, "*.png")))
    
    fid_items = []
    for p in rgb_list:
        fid = os.path.splitext(os.path.basename(p))[0]
        frame_id = parse_frame_id(fid)
        if frame_id <= args.max_frame_id:
            fid_items.append((frame_id, fid))
    
    fid_items.sort(key=lambda x: x[0])
    
    if len(fid_items) == 0:
        print(f"[Error] 没有找到序号 <= {args.max_frame_id} 的帧。")
        return

    selected_fids = [item[1] for item in fid_items[::args.stride]]
    
    print(f"开始增量重建。总处理帧数: {len(selected_fids)}")

    K = load_K_txt(args.K_path)
    tmp_img = cv2.imread(rgb_list[0])
    intrinsic = o3d.camera.PinholeCameraIntrinsic(tmp_img.shape[1], tmp_img.shape[0], K[0,0], K[1,1], K[0,2], K[1,2])

    # 执行逐帧重建与保存
    reconstruct_and_save_all(selected_fids, rgb_dir, dep_dir, msk_dir, pos_dir, intrinsic, args)

if __name__ == "__main__":
    main()