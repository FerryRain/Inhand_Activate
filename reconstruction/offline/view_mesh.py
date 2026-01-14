#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import open3d as o3d

def main():


    path = "/home/ferry/data/Code2/Research/Inhand_Activate/reconstruction/offline/result/green_cube_02_000/mesh/recon_0_000122_xyz_ascii_mesh.ply"
    if not os.path.isfile(path):
        print(f"[Error] 找不到文件: {path}")
        return

    mesh = o3d.io.read_triangle_mesh(path)
    if mesh.is_empty():
        print("[Error] 网格为空或读取失败")
        return

    # 清理 + 法线（显示干净的关键）
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    # 不要颜色：统一灰色（避免顶点色导致“乱”）
    mesh.paint_uniform_color([0.7, 0.7, 0.7])

    o3d.visualization.draw_geometries(
        [mesh],
        window_name="Mesh Viewer",
        width=1280,
        height=720,
        mesh_show_back_face=True
    )

if __name__ == "__main__":
    main()
