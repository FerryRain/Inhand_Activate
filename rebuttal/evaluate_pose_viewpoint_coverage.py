#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmark.config import load_config, resolved_path
from benchmark.geometry import current_view_direction_object, fibonacci_sphere


def _trapezoid(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(0.5 * (values[:-1] + values[1:])))


def _episode_metrics(episode_dir: Path, actions, candidates, cap_deg):
    directions = []
    for step in range(len(actions) + 1):
        pose = np.load(str(episode_dir / ("step_%03d" % step) / "executed_pose.npy"))
        directions.append(current_view_direction_object(pose))
    directions = np.asarray(directions, dtype=np.float64)
    threshold = np.cos(np.radians(float(cap_deg)))
    coverage_curve = []
    incremental_novelty = []
    for index in range(len(directions)):
        visited = directions[: index + 1]
        coverage_curve.append(float(np.mean(np.max(candidates @ visited.T, axis=1) >= threshold)))
        if index:
            cosine = np.clip(np.max(directions[index] @ directions[:index].T), -1.0, 1.0)
            incremental_novelty.append(float(np.degrees(np.arccos(cosine))))
    pairwise = np.clip(directions @ directions.T, -1.0, 1.0)
    upper = np.triu_indices(len(directions), k=1)
    pairwise_deg = np.degrees(np.arccos(pairwise[upper]))
    return {
        "final_pose_coverage": float(coverage_curve[-1]),
        "pose_coverage_auc": _trapezoid(coverage_curve),
        "mean_incremental_pose_novelty_deg": float(np.mean(incremental_novelty)),
        "mean_pairwise_view_distance_deg": float(np.mean(pairwise_deg)),
    }


def _write(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate pose-space viewpoint coverage.")
    parser.add_argument("--config", default="rebuttal/configs/formal_120_sixview_gpu.yaml")
    parser.add_argument("--cap-deg", type=float, default=30.0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = resolved_path(cfg["experiment"]["output_dir"])
    candidates = fibonacci_sphere(int(cfg["candidates"]["count"])).astype(np.float64)
    rows = []
    for summary_path in sorted(root.glob("*/*/pose_*/episode_summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = _episode_metrics(
            summary_path.parent, summary["actions"], candidates, args.cap_deg
        )
        rows.append({
            "planner": summary["planner"],
            "object": summary["object"],
            "initial_pose_seed": int(summary["initial_pose_seed"]),
            "planner_seed": int(summary.get("planner_seed", 0)),
            **metrics,
        })

    fields = tuple(key for key in rows[0] if key not in {
        "planner", "object", "initial_pose_seed", "planner_seed"
    })
    paired_groups = defaultdict(list)
    for row in rows:
        paired_groups[(row["planner"], row["object"], row["initial_pose_seed"])].append(row)
    paired = []
    for (planner, object_name, pose_seed), items in sorted(paired_groups.items()):
        paired.append({
            "planner": planner,
            "object": object_name,
            "initial_pose_seed": pose_seed,
            **{field: float(np.mean([item[field] for item in items])) for field in fields},
        })

    planner_groups = defaultdict(list)
    for row in paired:
        planner_groups[row["planner"]].append(row)
    aggregate = []
    for planner, items in sorted(planner_groups.items()):
        result = {"planner": planner, "paired_scenes": len(items)}
        for field in fields:
            values = np.asarray([item[field] for item in items], dtype=np.float64)
            result[field] = float(np.mean(values))
            result[field + "_ci95"] = float(
                1.96 * np.std(values, ddof=1) / np.sqrt(len(values))
            )
        aggregate.append(result)

    _write(root / "pose_viewpoint_metrics_episodes.csv", rows)
    _write(root / "pose_viewpoint_metrics_paired.csv", paired)
    _write(root / "pose_viewpoint_metrics_aggregate.csv", aggregate)
    lines = [
        "# Pose-space viewpoint coverage",
        "",
        "Coverage is the fraction of the shared candidate sphere within %.1f degrees of at least one acquired tracked-pose viewing direction." % args.cap_deg,
        "",
        "| Planner | Coverage AUC | Final coverage | Incremental novelty (deg) | Mean pairwise distance (deg) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| %s | %.4f | %.4f | %.2f | %.2f |" % (
                row["planner"],
                row["pose_coverage_auc"],
                row["final_pose_coverage"],
                row["mean_incremental_pose_novelty_deg"],
                row["mean_pairwise_view_distance_deg"],
            )
        )
    (root / "pose_viewpoint_metrics_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(root / "pose_viewpoint_metrics_summary.md")


if __name__ == "__main__":
    main()
