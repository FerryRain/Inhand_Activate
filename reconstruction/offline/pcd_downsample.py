"""
@FileName：pcd_downsample.py
@Description：
@Author：Ferry
@Time：2026 1/7/26 10:34 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""
import open3d as o3d

in_path  = "/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/GT_data/purple_cube_01/GT/ply/GT.ply"          # 改成你的输入
out_path = "/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/GT_data/purple_cube_01/GT/ply/GT_down.ply"     # 输出
ratio    = 0.01                  # 保留比例：0.1=保留10%

pcd = o3d.io.read_point_cloud(in_path)
pcd_ds = pcd.random_down_sample(ratio)

o3d.io.write_point_cloud(out_path, pcd_ds, write_ascii=True)
print(f"Saved: {out_path}, points {len(pcd.points)} -> {len(pcd_ds.points)}")

