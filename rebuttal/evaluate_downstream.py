#!/usr/bin/env python3
"""Evaluate reconstructed YCB meshes on sequential placement and regrasp.

Each method receives exactly the mesh it reconstructed.  A task plan is made
once from that mesh and then executed against the metric YCB ground-truth mesh
under ten deterministic placement perturbations.  Three paired initial poses
therefore give 30 trials per object and method.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from downstream.place_regrasp import (
    build_ray_scene,
    evaluate_trial,
    load_mesh,
    plan_sequential_task,
)


ROOT = Path(__file__).resolve().parents[1]


def bootstrap_ci(values, groups, seed=20260807, draws=5000):
    """Object-cluster bootstrap CI, avoiding trial-level pseudo-replication."""
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if not len(unique):
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    estimates = []
    for _ in range(int(draws)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        estimates.append(np.mean([values[groups == group].mean() for group in sampled]))
    return tuple(float(x) for x in np.percentile(estimates, [2.5, 97.5]))


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    output = []
    fields = (
        "placement_success",
        "conditional_regrasp_success",
        "sequential_success",
    )
    for method, method_rows in sorted(grouped.items()):
        item = {
            "method": method,
            "objects": len({row["object_name"] for row in method_rows}),
            "reconstructed_meshes": len(
                {(row["object_name"], row["reconstruction_seed"]) for row in method_rows}
            ),
            "trials": len(method_rows),
        }
        groups = [row["object_name"] for row in method_rows]
        for field in fields:
            values = [float(row[field]) for row in method_rows]
            lo, hi = bootstrap_ci(values, groups)
            item[field] = float(np.mean(values))
            item[f"{field}_ci95"] = [lo, hi]
        output.append(item)
    return output


def aggregate_objects(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["object_name"])].append(row)
    output = []
    for (method, object_name), object_rows in sorted(grouped.items()):
        reasons = defaultdict(int)
        for row in object_rows:
            reasons[row["failure_reason"]] += 1
        output.append(
            {
                "method": method,
                "object": object_name,
                "trials": len(object_rows),
                "placement_success": float(
                    np.mean([row["placement_success"] for row in object_rows])
                ),
                "conditional_regrasp_success": float(
                    np.mean(
                        [row["conditional_regrasp_success"] for row in object_rows]
                    )
                ),
                "sequential_success": float(
                    np.mean([row["sequential_success"] for row in object_rows])
                ),
                "failure_reasons": dict(sorted(reasons.items())),
            }
        )
    return output


def paired_comparisons(rows, reference="ray_gpis"):
    def numeric(value):
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return 1.0 if value.lower() == "true" else 0.0
        return float(value)

    fields = (
        "placement_success",
        "conditional_regrasp_success",
        "sequential_success",
    )
    methods = sorted({row["method"] for row in rows if row["method"] != reference})
    indexed = {
        (
            row["method"],
            row["object_name"],
            row["reconstruction_seed"],
            row["trial_seed"],
        ): row
        for row in rows
    }
    output = []
    rng = np.random.RandomState(20260807)
    for method in methods:
        for field in fields:
            object_differences = defaultdict(list)
            for key, reference_row in indexed.items():
                if key[0] != reference:
                    continue
                paired_key = (method, *key[1:])
                if paired_key not in indexed:
                    continue
                object_differences[key[1]].append(
                    numeric(reference_row[field]) - numeric(indexed[paired_key][field])
                )
            if not object_differences:
                continue
            object_values = np.asarray(
                [np.mean(value) for value in object_differences.values()],
                dtype=np.float64,
            )
            bootstrap = []
            for _ in range(5000):
                bootstrap.append(
                    np.mean(
                        rng.choice(object_values, size=len(object_values), replace=True)
                    )
                )
            try:
                p_value = float(
                    wilcoxon(
                        object_values,
                        alternative="two-sided",
                        zero_method="wilcox",
                    ).pvalue
                )
            except ValueError:
                p_value = 1.0
            output.append(
                {
                    "reference": reference,
                    "comparator": method,
                    "metric": field,
                    "paired_objects": len(object_values),
                    "difference": float(object_values.mean()),
                    "difference_ci95": [
                        float(x) for x in np.percentile(bootstrap, [2.5, 97.5])
                    ],
                    "wilcoxon_p": p_value,
                }
            )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-root", default="rebuttal/results/downstream_ycb_meshes")
    parser.add_argument("--gt-root", default="rebuttal/assets/ycb/models")
    parser.add_argument("--dataset", choices=("ycb", "real"), default="ycb")
    parser.add_argument(
        "--methods",
        default="single_view_depth,trellis2,spar3d,fixed,pose_novelty,pb_nbv,actnerf,ray_gpis",
    )
    parser.add_argument("--trials-per-mesh", type=int, default=10)
    parser.add_argument("--placement-tilt-std-deg", type=float, default=2.5)
    parser.add_argument("--placement-tilt-clip-deg", type=float, default=7.0)
    parser.add_argument("--grasp-translation-std-mm", type=float, default=1.5)
    parser.add_argument("--grasp-yaw-std-deg", type=float, default=1.5)
    parser.add_argument("--output", default="rebuttal/results/downstream_ycb_task")
    args = parser.parse_args()

    mesh_root = ROOT / args.mesh_root
    gt_root = ROOT / args.gt_root
    output_root = ROOT / args.output
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    output_root.mkdir(parents=True, exist_ok=True)
    config_payload = dict(vars(args))
    config_payload.update(
        {
            "mesh_root_resolved": str(mesh_root),
            "gt_root_resolved": str(gt_root),
            "methods_resolved": methods,
            "randomization": "deterministic trial_seed=100000*reconstruction_seed+local_trial",
        }
    )
    (output_root / "config.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8"
    )
    rows = []
    ground_truth_cache = {}
    ground_truth_scene_cache = {}
    for method in methods:
        for predicted_path in sorted((mesh_root / method).glob("*/pose_*.ply")):
            object_name = predicted_path.parent.name
            if args.dataset == "ycb":
                gt_path = gt_root / object_name / "google_16k/nontextured.ply"
            else:
                gt_path = gt_root / object_name / "GT/mesh/GT_mesh.stl"
            if not gt_path.exists():
                raise FileNotFoundError(gt_path)
            reconstruction_seed = int(predicted_path.stem.split("_")[-1])
            predicted = load_mesh(predicted_path)
            if object_name not in ground_truth_cache:
                ground_truth_cache[object_name] = load_mesh(gt_path)
                if (
                    args.dataset == "real"
                    and np.linalg.norm(ground_truth_cache[object_name].extents) > 5.0
                ):
                    ground_truth_cache[object_name].apply_scale(0.001)
                ground_truth_scene_cache[object_name] = build_ray_scene(
                    ground_truth_cache[object_name]
                )
            ground_truth = ground_truth_cache[object_name]
            task_plan = plan_sequential_task(predicted)
            for local_trial in range(int(args.trials_per_mesh)):
                trial_seed = 100000 * reconstruction_seed + local_trial
                result = evaluate_trial(
                    predicted,
                    ground_truth,
                    object_name,
                    method,
                    reconstruction_seed,
                    trial_seed,
                    task_plan=task_plan,
                    ground_truth_scene=ground_truth_scene_cache[object_name],
                    placement_tilt_std_deg=float(args.placement_tilt_std_deg),
                    placement_tilt_clip_deg=float(args.placement_tilt_clip_deg),
                    grasp_translation_std_m=0.001
                    * float(args.grasp_translation_std_mm),
                    grasp_yaw_std_deg=float(args.grasp_yaw_std_deg),
                )
                rows.append(result.to_dict())
            print(
                f"[task] {method}/{object_name}/{predicted_path.stem}: "
                f"{args.trials_per_mesh} trials",
                flush=True,
            )

    csv_path = output_root / "trials.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = aggregate(rows)
    per_object = aggregate_objects(rows)
    comparisons = paired_comparisons(rows)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        if summary:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    (output_root / "per_object.json").write_text(
        json.dumps(per_object, indent=2), encoding="utf-8"
    )
    (output_root / "paired_comparisons.json").write_text(
        json.dumps(comparisons, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
