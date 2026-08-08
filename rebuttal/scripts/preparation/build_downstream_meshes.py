#!/usr/bin/env python3
"""Extract a common Poisson mesh backend from saved downstream episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[3]


def backproject_saved_view(step_dir: Path) -> np.ndarray:
    depth = np.load(step_dir / "depth.npy")
    mask = np.asarray(o3d.io.read_image(str(step_dir / "mask.png"))) > 0
    pose = np.load(step_dir / "executed_pose.npy")
    config = __import__("json").loads(
        (step_dir.parent / "config.json").read_text(encoding="utf-8")
    )
    intrinsics = np.asarray(config["camera"]["intrinsics"], dtype=np.float64)
    rows, cols = np.nonzero(mask & (depth > 0.0))
    z = depth[rows, cols].astype(np.float64)
    camera = np.stack(
        [
            (cols + 0.5 - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (rows + 0.5 - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        ],
        axis=1,
    )
    points = (camera - pose[:3, 3][None]) @ pose[:3, :3]
    return points.astype(np.float64)


def camera_centers(episode: Path):
    centers = []
    for step in sorted(episode.glob("step_*")):
        pose_path = step / "executed_pose.npy"
        if pose_path.exists():
            centers.append(np.linalg.inv(np.load(pose_path))[:3, 3])
    return np.asarray(centers, dtype=np.float64)


def prepare_oriented_cloud(points: np.ndarray, cameras: np.ndarray):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if len(cloud.points) > 1000:
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=24, std_ratio=2.8)
    diagonal = float(np.linalg.norm(cloud.get_axis_aligned_bounding_box().get_extent()))
    voxel = max(0.0012, diagonal / 160.0)
    cloud = cloud.voxel_down_sample(voxel)
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=max(0.0045, diagonal * 0.055), max_nn=48
        )
    )
    vertices = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)
    if len(cameras):
        nearest = np.argmin(
            np.linalg.norm(vertices[:, None] - cameras[None, :], axis=2), axis=1
        )
        toward_camera = cameras[nearest] - vertices
        flip = np.sum(normals * toward_camera, axis=1) < 0.0
        normals[flip] *= -1.0
        cloud.normals = o3d.utility.Vector3dVector(normals)
    else:
        cloud.orient_normals_consistent_tangent_plane(min(30, len(vertices) - 1))
    return cloud, diagonal


def common_poisson_mesh(points: np.ndarray, cameras: np.ndarray, depth: int = 7):
    cloud, _ = prepare_oriented_cloud(points, cameras)

    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=int(depth), scale=1.06, linear_fit=True
    )
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    counts = np.asarray(counts)
    if len(counts) > 1:
        mesh.remove_triangles_by_mask(labels != int(np.argmax(counts)))
        mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > 40000:
        mesh = mesh.simplify_quadric_decimation(40000)
    mesh.compute_vertex_normals()
    return cloud, mesh


def common_nksr_mesh(points: np.ndarray, cameras: np.ndarray, reconstructor, device):
    import torch

    cloud, diagonal = prepare_oriented_cloud(points, cameras)
    xyz = torch.from_numpy(np.asarray(cloud.points)).float().to(device)
    normals = torch.from_numpy(np.asarray(cloud.normals)).float().to(device)
    # Scale-adaptive resolution keeps thin YCB objects sufficiently resolved
    # while using the same meshing rule for every planner.
    voxel_size = max(0.0015, float(diagonal) / 55.0)
    field = reconstructor.reconstruct(xyz, normals, voxel_size=voxel_size)
    result = field.extract_dual_mesh(mise_iter=1)
    vertices = result.v.detach().float().cpu().numpy().astype(np.float64)
    faces = result.f.detach().long().cpu().numpy().astype(np.int32)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces)
    )
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    if len(counts) > 1:
        mesh.remove_triangles_by_mask(labels != int(np.argmax(counts)))
        mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > 60000:
        mesh = mesh.simplify_quadric_decimation(60000)
    mesh.compute_vertex_normals()
    return cloud, mesh, voxel_size


def build_episode_mesh(
    episode: Path,
    output: Path,
    single_view: bool = False,
    backend: str = "nksr",
    reconstructor=None,
    device=None,
):
    if single_view:
        points = backproject_saved_view(episode / "step_000")
        cameras = camera_centers(episode)[:1]
    else:
        cloud = o3d.io.read_point_cloud(str(episode / "step_005" / "fused_cloud.ply"))
        points = np.asarray(cloud.points)
        cameras = camera_centers(episode)
    if len(points) < 100:
        raise RuntimeError(f"Too few points in {episode}: {len(points)}")
    if backend == "nksr":
        _, mesh, resolution = common_nksr_mesh(
            points, cameras, reconstructor, device
        )
    else:
        _, mesh = common_poisson_mesh(points, cameras)
        resolution = 7
    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(output), mesh, write_ascii=False):
        raise RuntimeError(f"Could not write {output}")
    output.with_suffix(".backend.json").write_text(
        json.dumps(
            {
                "backend": backend,
                "resolution": resolution,
                "input_points": len(points),
                "vertices": len(mesh.vertices),
                "triangles": len(mesh.triangles),
                "watertight": bool(mesh.is_watertight()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return len(points), len(mesh.vertices), len(mesh.triangles), mesh.is_watertight()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root", default="rebuttal/results/downstream_ycb_reconstruction"
    )
    parser.add_argument(
        "--output-root", default="rebuttal/results/downstream_ycb_meshes"
    )
    parser.add_argument(
        "--planners", default="fixed,pose_novelty,pb_nbv,ray_gpis,actnerf"
    )
    parser.add_argument("--backend", choices=("nksr", "poisson"), default="nksr")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-single-view", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result_root = ROOT / args.result_root
    output_root = ROOT / args.output_root
    planners = [value.strip() for value in args.planners.split(",") if value.strip()]
    reconstructor = None
    device = None
    if args.backend == "nksr":
        import nksr
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("NKSR downstream meshing requires CUDA")
        device = torch.device("cuda:0")
        reconstructor = nksr.Reconstructor(device)
    built = 0
    for planner in planners:
        summaries = sorted((result_root / planner).glob("*/pose_*/episode_summary.json"))
        if args.limit > 0:
            summaries = summaries[: args.limit]
        for summary in summaries:
            episode = summary.parent
            object_name = episode.parent.name
            pose_name = episode.name.split("_seed_")[0]
            output = output_root / planner / object_name / f"{pose_name}.ply"
            if output.exists() and not args.overwrite:
                continue
            stats = build_episode_mesh(
                episode,
                output,
                backend=args.backend,
                reconstructor=reconstructor,
                device=device,
            )
            print(f"[mesh] {planner}/{object_name}/{pose_name}: {stats}", flush=True)
            built += 1

    # A true one-view baseline is extracted from the first observation of the
    # paired Ray-GPIS episodes; it does not receive the two unbudgeted bootstrap
    # views used to initialize the learned planners.
    single_summaries = [] if args.skip_single_view else sorted(
        (result_root / "ray_gpis").glob("*/pose_*/episode_summary.json")
    )
    if args.limit > 0:
        single_summaries = single_summaries[: args.limit]
    for summary in single_summaries:
        episode = summary.parent
        object_name = episode.parent.name
        pose_name = episode.name
        output = output_root / "single_view_depth" / object_name / f"{pose_name}.ply"
        if output.exists() and not args.overwrite:
            continue
        stats = build_episode_mesh(
            episode,
            output,
            single_view=True,
            backend=args.backend,
            reconstructor=reconstructor,
            device=device,
        )
        print(f"[mesh] single_view_depth/{object_name}/{pose_name}: {stats}", flush=True)
        built += 1
    print(f"Built {built} meshes", flush=True)


if __name__ == "__main__":
    main()
