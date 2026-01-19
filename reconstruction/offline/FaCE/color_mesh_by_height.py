"""
@FileName：color_mesh_by_height.py
@Description：
@Author：Ferry
@Time：2025 11/29/25 12:16 AM
@Copyright：©2024-2025 ShanghaiTech University-RIMLAB
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import open3d as o3d


def main():
    if len(sys.argv) < 2:
        print("用法: python color_mesh_by_height.py your_mesh.ply [output_mesh.ply]")
        sys.exit(0)

    in_path = sys.argv[1]
    if not os.path.isfile(in_path):
        print(f"[Error] 找不到文件: {in_path}")
        sys.exit(1)

    # 默认输出名：xxx_zcolor.ply
    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
    else:
        base, ext = os.path.splitext(in_path)
        out_path = base + "_zcolor" + ext

    print(f"[Info] 读取网格: {in_path}")
    mesh = o3d.io.read_triangle_mesh(in_path)
    if mesh.is_empty():
        print("[Error] 网格为空或读取失败")
        sys.exit(1)

    verts = np.asarray(mesh.vertices)
    if verts.shape[0] == 0:
        print("[Error] 没有顶点")
        sys.exit(1)

    # ===== 按 z 高度上色 =====
    z = verts[:, 2]
    z_min, z_max = z.min(), z.max()
    print(f"[Info] z range: [{z_min:.6f}, {z_max:.6f}]")

    # 归一化到 [0, 1]
    z_norm = (z - z_min) / (z_max - z_min + 1e-9)

    # 简单做一个 蓝 -> 红 渐变：R 通道随高度增加，B 通道随高度减小
    colors = np.zeros((verts.shape[0], 3), dtype=np.float64)
    colors[:, 0] = z_norm          # R
    colors[:, 2] = 1.0 - z_norm    # B
    # 也可以给一点绿色，看起来更炫:
    # colors[:, 1] = 0.3

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    print(f"[Info] 保存带高度伪彩色的网格到: {out_path}")
    ok = o3d.io.write_triangle_mesh(out_path, mesh)
    if not ok:
        print("[Error] 保存失败")
        sys.exit(1)

    print("[Info] 打开可视化窗口（鼠标左键旋转，中键平移，滚轮缩放）")
    o3d.visualization.draw_geometries([mesh])


if __name__ == "__main__":
    main()
