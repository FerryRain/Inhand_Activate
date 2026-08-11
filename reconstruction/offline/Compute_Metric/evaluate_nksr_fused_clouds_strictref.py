#!/usr/bin/env python3
"""Re-evaluate the four FaCE+NKSR meshes with strictref-v1 alignment.

The alignment and mesh metric functions are imported directly from
``5_computer_metric_nskr.py`` so this evaluator cannot silently drift from the
protocol used by the original offline evaluation.  Each supplied reconstruction
is treated as an independent episode and receives one reconstruction-to-GT
rigid transform, exactly as a one-state invocation of the original script
would.  No scale fitting beyond that script's unit heuristic is introduced.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import open3d as o3d


CASES = {
    "pitcher": {
        "gt": "/home/ferry/Code/Research/InHand/neuralfeels/data/assets/gt_models/ycb/019_pitcher_base/google_16k/nontextured.ply",
        "urdf_source_max_m": 3.3333,
        "urdf_scale": 0.03,
    },
    "mustard": {
        "gt": "/home/ferry/Code/Research/InHand/neuralfeels/data/assets/gt_models/ycb/006_mustard_bottle/google_16k/nontextured.ply",
        "urdf_source_max_m": 3.9363,
        "urdf_scale": 0.03,
    },
}

TAUS_M = (0.002, 0.005, 0.010)


def load_strictref_module():
    module_path = Path(__file__).with_name("5_computer_metric_nskr.py")
    spec = importlib.util.spec_from_file_location("strictref_v1", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import strictref evaluator: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize_repeats(reports):
    values = {
        "accuracy_mm": [1000.0 * report["accuracy_mean"] for report in reports],
        "completeness_mm": [
            1000.0 * report["completeness_mean"] for report in reports
        ],
        "chamfer_l1_mm": [1000.0 * report["chamfer_l1"] for report in reports],
        "chamfer_l2_mm2": [1.0e6 * report["chamfer_l2"] for report in reports],
    }
    for tau in TAUS_M:
        tag = f"{int(round(1000.0 * tau))}mm"
        values[f"precision_at_{tag}"] = [
            report["fscore"][tau]["precision"] for report in reports
        ]
        values[f"recall_at_{tag}"] = [
            report["fscore"][tau]["recall"] for report in reports
        ]
        values[f"fscore_at_{tag}"] = [
            report["fscore"][tau]["fscore"] for report in reports
        ]
    output = {}
    for key, samples in values.items():
        output[key] = float(np.mean(samples))
        output[f"{key}_std"] = (
            float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default="rebuttal/results/nksr_fused_cloud_eval_20260811"
    )
    parser.add_argument("--sampling-repeats", type=int, default=5)
    parser.add_argument("--mesh-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument(
        "--discover",
        action="store_true",
        help=(
            "Discover pitcher_*/mustard_* variants from face_normals instead "
            "of evaluating only the default/light pair."
        ),
    )
    parser.add_argument(
        "--known-urdf-scale",
        action="store_true",
        help=(
            "Scale each metre-unit YCB GT to the known physical maximum "
            "dimension implied by its URDF source extent and scale=0.03."
        ),
    )
    args = parser.parse_args()

    strictref = load_strictref_module()
    root = Path(args.root).resolve()
    output_root = root / (
        "strictref_known_urdf_scale" if args.known_urdf_scale else "strictref_v1"
    )
    mesh_output_root = output_root / "meshes_aligned_per_variant"
    transform_root = output_root / "transforms"
    mesh_output_root.mkdir(parents=True, exist_ok=True)
    transform_root.mkdir(parents=True, exist_ok=True)

    rows = []
    metadata = {
        "protocol_source": str(
            Path(strictref.__file__).resolve()
        ),
        "alignment_scope": "one strictref transform per independent reconstruction",
        "alignment": {
            "auto_unit": True,
            "auto_voxel": False,
            "voxel_fallback_m": 0.001,
            "icp_iterations": [300, 300, 300],
            "selection_eval_voxel_m": 0.005,
            "selection_precision_threshold_m": 0.005,
            "scale_fitting": False,
            "known_urdf_scale_applied": bool(args.known_urdf_scale),
        },
        "metric": {
            "thresholds_mm": [2, 5, 10],
            "mesh_sample_method": "uniform",
            "surface_samples_per_direction": int(args.mesh_samples),
            "sampling_repeats": int(args.sampling_repeats),
        },
        "objects": {},
    }

    if args.discover:
        variants_by_object = {name: [] for name in CASES}
        for path in sorted((root / "face_normals").glob("*_face.ply")):
            stem = path.stem[:-5] if path.stem.endswith("_face") else path.stem
            lower = stem.lower()
            for object_name in CASES:
                prefix = f"{object_name}_"
                if lower.startswith(prefix):
                    variants_by_object[object_name].append(stem[len(prefix):])
                    break
        variants_by_object = {
            name: variants
            for name, variants in variants_by_object.items()
            if variants
        }
        if not variants_by_object:
            raise RuntimeError(
                f"No discoverable FaCE clouds under {root / 'face_normals'}"
            )
    else:
        variants_by_object = {name: ["default", "light"] for name in CASES}

    case_index = 0
    for object_name, case in CASES.items():
        if object_name not in variants_by_object:
            continue
        gt_path = Path(case["gt"])
        gt_pcd_base = strictref.read_point_cloud(str(gt_path))
        gt_mesh_base = strictref.read_triangle_mesh(str(gt_path))
        gt_extent_m = np.asarray(
            gt_mesh_base.get_axis_aligned_bounding_box().get_extent(),
            dtype=np.float64,
        )
        target_max_m = float(case["urdf_source_max_m"] * case["urdf_scale"])
        manual_gt_scale = (
            target_max_m / float(np.max(gt_extent_m))
            if args.known_urdf_scale
            else 1.0
        )
        metadata["objects"][object_name] = {
            "gt_path": str(gt_path),
            "gt_alignment_points": int(len(gt_pcd_base.points)),
            "gt_vertices": int(len(gt_mesh_base.vertices)),
            "gt_triangles": int(len(gt_mesh_base.triangles)),
            "unscaled_gt_extent_m": gt_extent_m.tolist(),
            "urdf_source_max_m": float(case["urdf_source_max_m"]),
            "urdf_scale": float(case["urdf_scale"]),
            "target_physical_max_m": target_max_m,
            "manual_gt_scale": manual_gt_scale,
            "scaled_gt_extent_m": (manual_gt_scale * gt_extent_m).tolist(),
        }

        for variant in variants_by_object[object_name]:
            seed = int(args.seed) + case_index
            np.random.seed(seed)
            cloud_path = root / "face_normals" / f"{object_name}_{variant}_face.ply"
            mesh_path = (
                root
                / "mesh_nksr"
                / f"{object_name}_{variant}_face_nksr_detail0.4.ply"
            )
            alignment = strictref.strict_align_one_candidate(
                rec_pcd_path=str(cloud_path),
                gt_pcd_path=str(gt_path),
                gt_mesh_path=str(gt_path),
                rec_scale=1.0,
                gt_scale=manual_gt_scale,
                auto_unit=True,
                auto_voxel=False,
                voxel_fallback=0.001,
                icp_iters=(300, 300, 300),
                eval_voxel=0.005,
                pick_tau_m=0.005,
                verbose_icp=False,
            )

            reconstructed = strictref.read_triangle_mesh(str(mesh_path))
            strictref.apply_scale(reconstructed, float(alignment["s_rec"]))
            reconstructed.transform(alignment["T"])
            aligned_path = (
                mesh_output_root / f"{object_name}_{variant}_nksr_strictref.ply"
            )
            if not o3d.io.write_triangle_mesh(str(aligned_path), reconstructed):
                raise RuntimeError(f"Failed to write aligned mesh: {aligned_path}")

            transform_path = (
                transform_root / f"{object_name}_{variant}_T_gt_from_recon.npy"
            )
            np.save(transform_path, alignment["T"])

            reports = []
            gt_pcd_metric = o3d.geometry.PointCloud(gt_pcd_base)
            gt_mesh_metric = o3d.geometry.TriangleMesh(gt_mesh_base)
            strictref.apply_scale(gt_pcd_metric, manual_gt_scale)
            strictref.apply_scale(gt_mesh_metric, manual_gt_scale)
            for _ in range(int(args.sampling_repeats)):
                gt_entry = strictref.build_gt_cache_entry(
                    gt_pcd_base=gt_pcd_metric,
                    gt_mesh_base=gt_mesh_metric,
                    s_gt=float(alignment["s_gt"]),
                    mesh_metric_samples=int(args.mesh_samples),
                    mesh_sample_method="uniform",
                )
                reports.append(
                    strictref.eval_mesh_mesh_against_fixed_ref(
                        rec_mesh=reconstructed,
                        scene_ref=gt_entry["scene_gt"],
                        ref_samples_xyz=gt_entry["gt_samples_xyz"],
                        mesh_samples=int(args.mesh_samples),
                        sample_method="uniform",
                        taus_m=list(TAUS_M),
                    )
                )

            cloud = strictref.read_point_cloud(str(cloud_path))
            row = {
                "object": object_name,
                "variant": variant,
                "face_points": int(len(cloud.points)),
                "nksr_vertices": int(len(reconstructed.vertices)),
                "nksr_triangles": int(len(reconstructed.triangles)),
                "alignment_precision_at_5mm": float(alignment["score"]),
                "alignment_time_s": float(alignment["time_align_s"]),
                "alignment_voxel_m": float(alignment["voxel"]),
                "reconstruction_unit_scale": float(alignment["s_rec"]),
                "manual_gt_scale": manual_gt_scale,
                "gt_unit_scale": float(alignment["s_gt"]),
                **summarize_repeats(reports),
            }
            rows.append(row)
            print(
                f"{object_name}/{variant}: align-P@5="
                f"{row['alignment_precision_at_5mm']:.4f} "
                f"P@5={row['precision_at_5mm']:.4f} "
                f"R@5={row['recall_at_5mm']:.4f} "
                f"F@5={row['fscore_at_5mm']:.4f}"
            )
            case_index += 1

    csv_path = output_root / "metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata["results"] = rows
    (output_root / "evaluation.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
