import open3d as o3d
import numpy as np
from pathlib import Path
import argparse, sys

def sample_with_mesh_normals(mesh, n_points, method, normal_from):
    if normal_from == "triangle":
        mesh.compute_triangle_normals()
    else:
        mesh.compute_vertex_normals()
    if method == "poisson":
        pcd = mesh.sample_points_poisson_disk(n_points)
    else:  # uniform
        pcd = mesh.sample_points_uniformly(
            number_of_points=n_points,
            use_triangle_normal=(normal_from == "triangle")
        )

    if not pcd.has_normals():
        if normal_from == "triangle":
            tris = np.asarray(mesh.triangles)
            tri_n = np.asarray(mesh.triangle_normals)
            verts = np.asarray(mesh.vertices)
            centroids = verts[tris].mean(axis=1)
            cent_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centroids))
            kdt = o3d.geometry.KDTreeFlann(cent_pcd)
            normals = []
            for p in np.asarray(pcd.points):
                _, idx, _ = kdt.search_knn_vector_3d(p, 1)
                normals.append(tri_n[idx[0]])
            pcd.normals = o3d.utility.Vector3dVector(np.array(normals))
        else:  # vertex
            verts = np.asarray(mesh.vertices)
            vnorm = np.asarray(mesh.vertex_normals)
            vpcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(verts))
            kdt = o3d.geometry.KDTreeFlann(vpcd)
            normals = []
            for p in np.asarray(pcd.points):
                _, idx, _ = kdt.search_knn_vector_3d(p, 1)
                normals.append(vnorm[idx[0]])
            pcd.normals = o3d.utility.Vector3dVector(np.array(normals))

    return pcd

def save_xyz6(pcd, path):
    P = np.asarray(pcd.points)
    N = np.asarray(pcd.normals)
    N = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)
    np.savetxt(path, np.hstack([P, N]), fmt="%.6f")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, help="fandisk.obj")
    ap.add_argument("--output", required=True)
    ap.add_argument("--points", type=int, default=2000)
    ap.add_argument("--method", choices=["poisson","uniform"], default="poisson")
    ap.add_argument("--normal_from", choices=["vertex","triangle"], default="triangle",)
    args = ap.parse_args()

    mesh = o3d.io.read_triangle_mesh(args.input)
    if mesh.is_empty():
        sys.exit("Failed to read OBJ")

    pcd = sample_with_mesh_normals(mesh, args.points, args.method, args.normal_from)
    save_xyz6(pcd, args.output)
    print(f"Saved: {args.output}  points={len(pcd.points)}  normals=mesh-{args.normal_from}")

if __name__ == "__main__":
    main()