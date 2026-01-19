"""
@FileName：recon.py
@Description：
@Author：Ferry
@Time：2026 1/19/26 3:43 AM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""

import torch
import numpy as np
import nksr
from pycg import vis
from examples.common import load_bunny_example, warning_on_low_memory
import open3d_pycg as o3d
from examples.common import load_spot_example, warning_on_low_memory
if __name__ == '__main__':
    warning_on_low_memory(1024.0)
    device = torch.device("cuda:0")
    # 要OBJ.1 2 5    3 4重测
    # bunny_geom = vis.from_file("/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/result/green_cube_00_001_down0.01/pcd/recon_0_000240.ply")
    # bunny_geom = o3d.io.read_point_cloud(
        # "/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/result/green_cube_00_001_down0.01/normal/recon_0_000244_xyz_normal.ply")
    # bunny_geom = o3d.io.read_point_cloud("/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/result/green_cube_00_002/normal/recon_0_000170_xyz_normal.ply")  #obj1
    # bunny_geom = o3d.io.read_point_cloud("/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/result/green_cube_02_000/normal/recon_0_000118_xyz_normal.ply")  #obj2
    # bunny_geom = o3d.io.read_point_cloud(
    #     "/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/result/purple_cube_01/normal/recon_0_000195_xyz_normal.ply")  # obj3
    bunny_geom = o3d.io.read_point_cloud(
        "/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/result/cube_obj1/003/normal/recon_0_000200_xyz_normal.ply")  # obj4
    # bunny_geom = o3d.io.read_point_cloud(
    #     "/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/result/yellow_cylinder_002/normal/recon_0_000140_xyz_normal.ply")  # obj5
    # bunny_geom = o3d.io.read_point_cloud(
    #     "/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/result/yellow_cylinder_small_001/normal/recon_0_000280_xyz_normal.ply")  # obj6



    input_xyz = torch.from_numpy(np.asarray(bunny_geom.points)).float().to(device)
    input_normal = torch.from_numpy(np.asarray(bunny_geom.normals)).float().to(device)

    reconstructor = nksr.Reconstructor(device)
    # field = reconstructor.reconstruct(input_xyz, input_normal,voxel_size=0.0050)
    field = reconstructor.reconstruct(input_xyz, input_normal, detail_level=0.4)
    mesh = field.extract_dual_mesh(mise_iter=1)

    vis.show_3d([vis.mesh(mesh.v, mesh.f)], [bunny_geom])
