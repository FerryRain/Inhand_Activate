"""
@FileName：pointscloud_Splicing.py
@Description：
@Author：Ferry
@Time：2025 8/19/25 2:30 PM
@Copyright：©2024-2025 ShanghaiTech University-RIMLAB
"""
import os

from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import open3d as o3d

def pos2uv(obj_pos, view_matrix, projection_matrix, width, height):
    point_world = np.array([*obj_pos, 1.0])

    point_cam = view_matrix @ point_world
    z_cam = point_cam[2]
    point_clip = projection_matrix @ point_cam

    point_ndc = point_clip[:3] / point_clip[3]
    if not (-1 <= point_ndc[0] <= 1 and -1 <= point_ndc[1] <= 1):
        print("Out of width!")
        return None, None, z_cam

    u = (point_ndc[0] + 1) * 0.5 * width
    v = (1 - point_ndc[1]) * 0.5 * height

    return u, v, z_cam

def get_prompt_neighborhood(center_uv, offsets=[(-35, 0), (35, 0), (0, -35), (0, 35),(-40, 0), (40, 0), (0, -40), (0, 40)]):
    u, v = center_uv
    points = [[u, v]] + [[u + dx, v + dy] for dx, dy in offsets]
    input_point = np.array(points)
    input_label = np.ones(len(points), dtype=int)
    return input_point, input_label


def depth_to_pointcloud_camera_coords(depth, fx, fy, cx, cy):
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    z = depth
    x = -(u - cx) * z / fx # flip x-axis
    y = (v - cy) * z / fy

    points = np.stack((x, y, z), axis=-1)
    return points.reshape(-1, 3)

def extract_intrinsics_from_proj_matrix(proj_matrix, width, height):
    fx = proj_matrix[0, 0] * width / 2
    fy = proj_matrix[1, 1] * height / 2
    cx = (1 - proj_matrix[0, 2]) * width / 2
    cy = (1 + proj_matrix[1, 2]) * height / 2
    return fx, fy, cx, cy

def inverse_view_matrix_to_cam_pose(view_matrix):
    view_inv = np.linalg.inv(view_matrix)
    R_world = view_inv[:3, :3]
    t_world = view_inv[:3, 3]
    return R_world, t_world


def transform_camera_to_world(points_cam, R_world, t_world):
    return (R_world @ points_cam.T).T + t_world


def depth_to_world_pointcloud(depth_tensor, proj_matrix, view_matrix, width, height):
    depth_np = depth_tensor.cpu().numpy()
    fx, fy, cx, cy = extract_intrinsics_from_proj_matrix(proj_matrix, width, height)
    points_cam = depth_to_pointcloud_camera_coords(depth_np, fx, fy, cx, cy)
    R_world, t_world = inverse_view_matrix_to_cam_pose(view_matrix)
    points_world = transform_camera_to_world(points_cam, R_world, t_world)
    return points_cam, points_world

def pose_to_matrix(pose):
    pos = pose[:3]
    quat = pose[3:]
    rot = R.from_quat(quat).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


def preprocess_depth(depth, rgb, predictor, obj_pos, view_matrix, projection_matrix, width, height):
    predictor.set_image(rgb)
    u, v, z = pos2uv(obj_pos, view_matrix, projection_matrix, width, height)

    center_uv = (u, v)
    input_point, input_label = get_prompt_neighborhood(center_uv)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )
    mask = masks[0]
    depth_masked = np.where(mask, depth, 0.0)

    depth_tensor = torch.tensor(depth_masked)
    height, width = depth_tensor.shape

    points_cam, points_world = depth_to_world_pointcloud(depth_tensor, projection_matrix, view_matrix, width, height)
    depth_object = depth_masked

    return depth_object


def prepocess_pcd(depth_object, projection_matrix, view_matrix, rgb):
    depth_tensor = torch.tensor(depth_object)
    H, W = depth_tensor.shape

    _, points_world = depth_to_world_pointcloud(depth_tensor, projection_matrix, view_matrix, W, H)

    mask = (depth_tensor != 0).reshape(-1).cpu().numpy()
    points_world = points_world[mask]


    if rgb.dtype != np.float32:
        colors = (rgb.astype(np.float32) / 255.0).reshape(-1, 3)[mask]
    else:
        colors = rgb.reshape(-1, 3)[mask]
        colors = np.clip(colors, 0.0, 1.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if points_world.shape[0] >= 100:
        nb = max(8, int(points_world.shape[0] * 0.01))
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=2.0)

    return pcd

# def prepocess_pcd(depth_object, projection_matrix, view_matrix):
#     depth_tensor = torch.tensor(depth_object)
#     height, width = depth_tensor.shape
#
#     _, points_world = depth_to_world_pointcloud(depth_tensor, projection_matrix, view_matrix, width, height)
#     mask = (depth_tensor != 0)
#     points_world = points_world[mask.reshape(-1)]
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points_world)
#     points_world_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=int(points_world.shape[0] * 0.01),
#                                                            std_ratio=2.0)
#     return points_world_pcd

def transfer_to_object_axis(pcd, pose):
    T_wo = pose_to_matrix(pose[0])
    points_world = np.asarray(pcd.points)
    T_obj_world = np.linalg.inv(T_wo)
    points_obj = (T_obj_world[:3, :3] @ points_world.T).T + T_obj_world[:3, 3]
    return points_obj

def pointscloud_splicing(pcd1, pcd2, pose1, pose2):
    T_wo1 = pose_to_matrix(pose1[0])
    T_wo2 = pose_to_matrix(pose2[0])

    points_world_1 = np.asarray(pcd1.points)
    points_world_2 = np.asarray(pcd2.points)

    T_obj1_world = np.linalg.inv(T_wo1)
    T_obj2_world = np.linalg.inv(T_wo2)

    points_obj0_1 = (T_obj1_world[:3, :3] @ points_world_1.T).T + T_obj1_world[:3, 3]
    points_obj0_2 = (T_obj2_world[:3, :3] @ points_world_2.T).T + T_obj2_world[:3, 3]

    points_combined = np.vstack([points_obj0_1, points_obj0_2])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_combined)
    o3d.visualization.draw_geometries([pcd])
    return points_combined, pcd


def process_one_frame(i, predictor, view_matrix, projection_matrix):
    rgb_path   = f"./images/rgb{i}.npy"
    depth_path = f"./images/depth{i}.npy"
    pose_path  = f"./images/object_pose{i}.npy"

    if not (os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(pose_path)):
        print(f"[Frame {i}] 缺少文件，已跳过。")
        return None

    img_rgb = np.load(rgb_path)[:, :, :3]
    predictor.set_image(img_rgb)

    depth_np = np.load(depth_path)
    depth_t  = torch.tensor(depth_np)
    H, W = depth_t.shape

    pose_i  = np.load(pose_path)
    obj_pos = pose_i[:, :3].reshape(-1)

    depth_object = preprocess_depth(depth_t, img_rgb, predictor, obj_pos,
                                    view_matrix, projection_matrix, W, H)

    pcd_world = prepocess_pcd(depth_object, projection_matrix, view_matrix, img_rgb)
    pts_world = np.asarray(pcd_world.points)
    if pts_world.size == 0:
        print(f"[Frame {i}] 没有有效点，已跳过。")
        return None

    pts_obj = transfer_to_object_axis(pcd_world, pose_i)  # (N,3)
    print(f"[Frame {i}] 有效点(物体系)：{pts_obj.shape[0]}")
    return pts_obj

    return pcd_world


def ICP_registration(source_pcd, target_pcd, threshold=0.02):
    # print("\n2. 预处理与参数选择...")

    # 【重要】动态计算Voxel Size，使其更具适应性
    # 我们取点云包围盒对角线长度的一个比例作为voxel_size
    bbox = target_pcd.get_axis_aligned_bounding_box()
    diag_len = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    voxel_size = diag_len / 40.0  # 关键参数！可以尝试 /30, /40, /50 来调整

    #print(f"点云包围盒对角线长度: {diag_len:.4f}")
    #print(f"动态计算的 Voxel Size: {voxel_size:.4f}")

    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)
    #print(f"降采样后, 源点云点数: {len(source_down.points)}, 目标点云点数: {len(target_down.points)}")

    # 估计法线
    radius_normal = voxel_size * 2
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # 计算FPFH特征
    radius_feature = voxel_size * 5
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    distance_threshold = voxel_size * 1

    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [ # 增加RANSAC迭代次数的健壮性
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(5000000, 0.999)) # 增加最大迭代次数

    # 【关键诊断】打印RANSAC结果
    # print("\n--- RANSAC 诊断信息 ---")
    # print(ransac_result)
    # print("------------------------")
    # if ransac_result.fitness < 0.1:
    #     print("警告: RANSAC fitness 非常低，粗配准可能已失败！")
    # else:
    #     print("RANSAC fitness 看起来不错，继续执行ICP。")

    # print("显示粗配准结果...")
    #draw_registration_result(source_down, target_down, ransac_result.transformation)

    # print("\n4. 执行Point-to-Plane ICP精配准...")
    init_transform = ransac_result.transformation
    icp_result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, distance_threshold, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    # 【关键诊断】打印ICP结果
    # print("\n--- ICP 诊断信息 ---")
    # print(icp_result)
    # print("--------------------")

    # print("显示精配准结果...")
    # #draw_registration_result(source_down, target_down, icp_result.transformation)

    # print("\n5. 在原始高分辨率点云上执行最终精炼ICP...")

    # 为原始高密度点云定义一套独立的、更精细的参数
    radius_normal_final = voxel_size * 1.0  
    final_threshold = voxel_size * 0.5      

    # 【关键修正】在调用 PointToPlane ICP 之前，必须为原始点云估计法线！
    # 错误正是因为缺少了下面这两行代码。
    # print(f":: 为原始点云估计法线 (使用精细半径: {radius_normal_final:.4f})...")
    source_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal_final, max_nn=30))
    target_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal_final, max_nn=30))

    # 使用上一步的结果作为极其优秀的初始值
    init_transform_final = icp_result.transformation

    # 现在可以安全地调用 PointToPlane ICP 了
    final_icp_result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, final_threshold, init_transform_final,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())


    # print("最终精炼ICP完成。最终变换矩阵：")
    # print(final_icp_result.transformation)

    # print("\n--- 最终ICP诊断信息 ---")
    # print(final_icp_result)

    # print("显示最终精炼配准结果 (这应该是你想要的效果)...")
    #draw_registration_result(source_pcd, target_pcd, final_icp_result.transformation)
    return source_pcd.transform(final_icp_result.transformation)+target_pcd



if __name__ == '__main__':
    model_type = 'vit_h'
    model_path = "./sam_vit_h_4b8939.pth"
    sam = sam_model_registry[model_type](checkpoint=model_path)
    sam.to("cuda")
    predictor = SamPredictor(sam)

    view_matrix = np.load("./view_matrix.npz.npy").reshape(4, 4).T
    projection_matrix = np.load("./proj_matrix.npz.npy").reshape(4, 4).T

    all_pts = []
    for i in range(1, 21):
        pts_obj = process_one_frame(i, predictor, view_matrix, projection_matrix)
        if pts_obj is not None:
            all_pts.append(pts_obj)

    if len(all_pts) == 0:
        raise RuntimeError("没有任何帧成功生成点云。")

    points_combined = np.vstack(all_pts)
    # combined_pcd = None     

    # for i in range(1, 21):
    #     pcd = process_one_frame(i, predictor, view_matrix, projection_matrix)
    #     if pcd is not None:
    #         if combined_pcd is None:
    #             combined_pcd = pcd
    #         else:
    #             combined_pcd = ICP_registration(combined_pcd, pcd)
    # points_combined = np.asarray(combined_pcd.points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_combined)

    if points_combined.shape[0] > 200000:
        pcd = pcd.voxel_down_sample(voxel_size=0.003)

    o3d.visualization.draw_geometries([pcd])


