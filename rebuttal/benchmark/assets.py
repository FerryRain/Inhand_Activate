from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import open3d as o3d
import trimesh

from .config import ROOT


SCANNED_OBJECTS = {
    "Cube": "Active/pcd/GT_data/Cube/GT/ply/GT_cube_down.ply",
    "Cylinder": "Active/pcd/GT_data/Cylinder/GT/ply/GT_down.ply",
    "green_Pepper": "Active/pcd/GT_data/green_Pepper/GT/ply/GT_down.ply",
    "L_Shaped": "Active/pcd/GT_data/L_Shaped/GT/ply/GT_down.ply",
    "Cross": "Active/pcd/GT_data/Cross/GT/ply/GT_down.ply",
    "Corner": "Active/pcd/GT_data/Corner/GT/ply/GT_down.ply",
}

IRREGULAR_SOURCE = (
    "LeapHand_rotation/assets/urdf/objects/meshes/set2/"
    "set_obj10_thin_block_corner_0decompose.obj"
)


def _clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def _normalize_mesh(mesh: o3d.geometry.TriangleMesh, target_diagonal: float) -> o3d.geometry.TriangleMesh:
    vertices = np.asarray(mesh.vertices)
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    mesh.translate(-center)
    extent = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent())
    diagonal = float(np.linalg.norm(extent))
    mesh.scale(float(target_diagonal) / max(diagonal, 1e-9), center=np.zeros(3))
    return mesh


def _voxel_watertight_repair(
    mesh: o3d.geometry.TriangleMesh, target_diagonal: float
) -> o3d.geometry.TriangleMesh:
    """Remove Poisson self-intersections via a closed solid voxel isosurface."""
    tri = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        process=True,
    )
    voxel_grid = tri.voxelized(float(target_diagonal) / 100.0).fill()
    repaired = voxel_grid.marching_cubes
    repaired.apply_transform(voxel_grid.transform)
    out = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(repaired.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(repaired.faces, dtype=np.int32)),
    )
    out.compute_vertex_normals()
    return _normalize_mesh(out, target_diagonal)


def _poisson_from_scan(path: Path, target_diagonal: float, depth: int) -> o3d.geometry.TriangleMesh:
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise RuntimeError("Empty scan: %s" % path)
    points = np.asarray(cloud.points)
    center = points.mean(axis=0)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=12.0, max_nn=40))
    normals = np.asarray(cloud.normals)
    radial = points - center[None]
    flip = np.sum(normals * radial, axis=1) < 0.0
    normals[flip] *= -1.0
    cloud.normals = o3d.utility.Vector3dVector(normals)
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=int(depth), scale=1.08, linear_fit=False
    )
    # Poisson produces closed components. Density trimming or AABB cropping would
    # open their boundary, so retain only the largest closed connected component.
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    counts = np.asarray(counts)
    if len(counts) > 1:
        mesh.remove_triangles_by_mask(labels != int(np.argmax(counts)))
        mesh.remove_unreferenced_vertices()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    if len(mesh.triangles) > 50000:
        mesh = mesh.simplify_quadric_decimation(50000)
    mesh = _normalize_mesh(mesh, target_diagonal)
    if not mesh.is_watertight():
        mesh = _voxel_watertight_repair(mesh, target_diagonal)
    return mesh


def _make_bowl(target_diagonal: float, radial_segments: int = 96, vertical_segments: int = 24):
    """Create a closed double-wall bowl with a genuinely concave visible interior."""
    vertices = []
    triangles = []
    outer_radius = 0.050
    inner_radius = 0.041
    height = 0.060
    bottom = -0.030
    inner_bottom = -0.018
    for layer in range(vertical_segments + 1):
        t = layer / float(vertical_segments)
        z = bottom + height * t
        r_out = outer_radius * (0.62 + 0.38 * t)
        r_in = inner_radius * (0.48 + 0.52 * t)
        z_in = inner_bottom + (height + bottom - inner_bottom) * t
        for j in range(radial_segments):
            angle = 2.0 * np.pi * j / radial_segments
            vertices.append([r_out * np.cos(angle), r_out * np.sin(angle), z])
            vertices.append([r_in * np.cos(angle), r_in * np.sin(angle), z_in])
    def vid(layer, j, inner):
        return 2 * (layer * radial_segments + (j % radial_segments)) + int(inner)
    for layer in range(vertical_segments):
        for j in range(radial_segments):
            n = (j + 1) % radial_segments
            triangles.extend([
                [vid(layer, j, 0), vid(layer, n, 0), vid(layer + 1, n, 0)],
                [vid(layer, j, 0), vid(layer + 1, n, 0), vid(layer + 1, j, 0)],
                [vid(layer, j, 1), vid(layer + 1, n, 1), vid(layer, n, 1)],
                [vid(layer, j, 1), vid(layer + 1, j, 1), vid(layer + 1, n, 1)],
            ])
    for j in range(radial_segments):
        n = (j + 1) % radial_segments
        triangles.extend([
            [vid(vertical_segments, j, 0), vid(vertical_segments, n, 1), vid(vertical_segments, n, 0)],
            [vid(vertical_segments, j, 0), vid(vertical_segments, j, 1), vid(vertical_segments, n, 1)],
            [vid(0, j, 0), vid(0, n, 0), vid(0, n, 1)],
            [vid(0, j, 0), vid(0, n, 1), vid(0, j, 1)],
        ])
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)),
    )
    return _normalize_mesh(_clean_mesh(mesh), target_diagonal)


def _load_irregular(path: Path, target_diagonal: float) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if mesh.is_empty():
        raise RuntimeError("Empty irregular mesh: %s" % path)
    return _normalize_mesh(_clean_mesh(mesh), target_diagonal)


def prepare_assets(
    output_dir: Path,
    target_diagonal: float = 0.12,
    poisson_depth: int = 7,
    force: bool = False,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, relative in SCANNED_OBJECTS.items():
        out = output_dir / (name + ".ply")
        if force or not out.exists():
            mesh = _poisson_from_scan(ROOT / relative, target_diagonal, poisson_depth)
            if not o3d.io.write_triangle_mesh(str(out), mesh, write_ascii=False):
                raise RuntimeError("Failed to write %s" % out)
        manifest[name] = str(out.resolve())
    bowl_path = output_dir / "Bowl.ply"
    if force or not bowl_path.exists():
        o3d.io.write_triangle_mesh(str(bowl_path), _make_bowl(target_diagonal))
    manifest["Bowl"] = str(bowl_path.resolve())
    irregular_path = output_dir / "Thin_Irregular.ply"
    if force or not irregular_path.exists():
        mesh = _load_irregular(ROOT / IRREGULAR_SOURCE, target_diagonal)
        o3d.io.write_triangle_mesh(str(irregular_path), mesh)
    manifest["Thin_Irregular"] = str(irregular_path.resolve())
    return manifest


def load_mesh(path: str) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if mesh.is_empty():
        raise RuntimeError("Could not load mesh: %s" % path)
    return _clean_mesh(mesh)
