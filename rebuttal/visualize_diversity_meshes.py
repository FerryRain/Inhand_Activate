"""Create representative qualitative meshes for the two added objects.

The displayed successful examples are the Full Ray-GPIS runs with the lowest
final Chamfer distance for each object. Aggregate results over all 15 poses are reported in
the rebuttal text. The online benchmark evaluates the fused point cloud;
support-trimmed Poisson meshing is qualitative post-processing only.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "results" / "diversity_continuous_realistic"
OUTPUT_ROOT = ROOT / "figures" / "diversity_meshes_continuous_realistic"
OBJECTS = ("Bowl", "Thin_Irregular")


def successful_episodes() -> dict[str, tuple[int, float]]:
    selected: dict[str, tuple[int, float]] = {}
    for object_name in OBJECTS:
        object_rows = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(
                (RESULT_ROOT / "ray_gpis" / object_name).glob(
                    "pose_*/episode_summary.json"
                )
            )
        ]
        if len(object_rows) != 15:
            raise RuntimeError(
                f"Expected 15 completed {object_name} episodes, found {len(object_rows)}"
            )
        representative = min(
            object_rows, key=lambda row: float(row["final_chamfer_m"])
        )
        selected[object_name] = (
            int(representative["initial_pose_seed"]),
            float(representative["final_f@5"]),
        )
    return selected


def supported_poisson_mesh(
    episode: Path, cloud_path: Path
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]:
    cloud = o3d.io.read_point_cloud(str(cloud_path))
    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
    extent = float(np.max(cloud.get_axis_aligned_bounding_box().get_extent()))
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.12 * extent, max_nn=40)
    )

    # Each saved GT pose maps object coordinates to the fixed camera. Orient
    # every estimated normal toward its nearest acquired camera center. This
    # preserves opposing inner/outer normals for concave objects such as Bowl.
    camera_centers = []
    for step in range(6):
        object_to_camera = np.load(episode / f"step_{step:03d}" / "gt_pose.npy")
        camera_centers.append(np.linalg.inv(object_to_camera)[:3, 3])
    camera_centers = np.asarray(camera_centers)
    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)
    nearest = np.argmin(
        np.linalg.norm(points[:, None, :] - camera_centers[None, :, :], axis=2),
        axis=1,
    )
    to_camera = camera_centers[nearest] - points
    flip = np.sum(normals * to_camera, axis=1) < 0
    normals[flip] *= -1.0
    cloud.normals = o3d.utility.Vector3dVector(normals)

    nearest_distance = np.asarray(cloud.compute_nearest_neighbor_distance())
    support_radius = 2.75 * float(np.median(nearest_distance))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=8, scale=1.08, linear_fit=True
    )
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.03))

    # Poisson closes open boundaries. Remove vertices without nearby measured
    # support so the qualitative mesh cannot invent large unobserved sheets.
    mesh_vertices = o3d.geometry.PointCloud()
    mesh_vertices.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    support_distance = np.asarray(mesh_vertices.compute_point_cloud_distance(cloud))
    mesh.remove_vertices_by_mask(support_distance > support_radius)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    unsmoothed = copy.deepcopy(mesh)
    smoothed = mesh.filter_smooth_taubin(number_of_iterations=10)
    mesh = (
        smoothed
        if np.all(np.isfinite(np.asarray(smoothed.vertices)))
        else unsmoothed
    )
    mesh.compute_vertex_normals()
    return cloud, mesh


def shaded_faces(mesh: o3d.geometry.TriangleMesh, base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    faces = vertices[triangles]
    normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(lengths, 1e-12)
    light = np.asarray([0.35, -0.45, 0.82])
    light /= np.linalg.norm(light)
    intensity = 0.48 + 0.52 * np.abs(normals @ light)
    colors = np.clip(base[None, :] * intensity[:, None], 0.0, 1.0)
    return faces, colors


def draw_mesh(
    axis: plt.Axes,
    mesh: o3d.geometry.TriangleMesh,
    base_color: tuple[float, float, float],
    elevation: float,
    azimuth: float,
) -> None:
    faces, colors = shaded_faces(mesh, np.asarray(base_color))
    collection = Poly3DCollection(
        faces, facecolors=colors, edgecolors="none", linewidths=0.0
    )
    axis.add_collection3d(collection)

    vertices = np.asarray(mesh.vertices)
    center = vertices.mean(axis=0)
    radius = 0.54 * float(np.max(np.ptp(vertices, axis=0)))
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.set_proj_type("ortho")
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_axis_off()


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    selected = successful_episodes()
    meshes: dict[str, tuple[o3d.geometry.TriangleMesh, o3d.geometry.TriangleMesh]] = {}

    for object_name, (pose_seed, _) in selected.items():
        episode = RESULT_ROOT / "ray_gpis" / object_name / f"pose_{pose_seed:03d}"
        cloud_path = episode / "step_005" / "fused_cloud.ply"
        gt_path = episode / "object_gt_mesh.ply"
        _, reconstruction = supported_poisson_mesh(episode, cloud_path)
        gt_mesh = o3d.io.read_triangle_mesh(str(gt_path))
        gt_mesh.compute_vertex_normals()

        output_mesh = OUTPUT_ROOT / f"{object_name}_ray_gpis_reconstruction.ply"
        o3d.io.write_triangle_mesh(str(output_mesh), reconstruction)
        meshes[object_name] = (gt_mesh, reconstruction)

    figure = plt.figure(figsize=(7.2, 6.3), facecolor="white")
    view = {"Bowl": (48, -55), "Thin_Irregular": (24, -48)}
    labels = {"Bowl": "Bowl (concave)", "Thin_Irregular": "Thin-Irregular"}

    for row, object_name in enumerate(OBJECTS):
        pose_seed, final_f = selected[object_name]
        gt_mesh, reconstruction = meshes[object_name]
        elevation, azimuth = view[object_name]

        gt_axis = figure.add_subplot(2, 2, 2 * row + 1, projection="3d")
        rec_axis = figure.add_subplot(2, 2, 2 * row + 2, projection="3d")
        draw_mesh(gt_axis, gt_mesh, (0.62, 0.65, 0.70), elevation, azimuth)
        draw_mesh(rec_axis, reconstruction, (0.18, 0.48, 0.82), elevation, azimuth)
        gt_axis.set_title(f"{labels[object_name]} — GT", fontsize=10, pad=0)
        rec_axis.set_title(
            f"Ray-GPIS reconstruction\nF@5={final_f:.3f}, pose {pose_seed:02d}",
            fontsize=10,
            pad=0,
        )

    figure.suptitle(
        "Continuous realistic-stress reconstruction examples",
        fontsize=12,
        fontweight="semibold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.012,
        "Shown: lowest-Chamfer episode per object; aggregate 15-pose results are "
        "reported in the text.\n15 FPS/6 s with hand occlusion and contact mismatch. "
        "Mesh: support-trimmed Poisson + Taubin smoothing.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#444444",
    )
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.06, top=0.88, wspace=0.0, hspace=0.08)
    figure.savefig(OUTPUT_ROOT / "diversity_reconstructed_meshes.png", dpi=300)
    figure.savefig(OUTPUT_ROOT / "diversity_reconstructed_meshes.pdf", dpi=300)
    plt.close(figure)

    for object_name, (pose_seed, final_f) in selected.items():
        mesh = meshes[object_name][1]
        print(
            f"{object_name}: pose={pose_seed:03d}, final_F@5={final_f:.6f}, "
            f"vertices={len(mesh.vertices)}, triangles={len(mesh.triangles)}"
        )


if __name__ == "__main__":
    main()
