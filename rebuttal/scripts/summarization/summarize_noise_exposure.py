#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rebuttal.benchmark.config import load_config, resolved_path
from rebuttal.benchmark.geometry import action_rotation, rotation_error_deg


FIELDS = (
    "mean_visibility_ratio",
    "mean_pose_error_deg",
    "mean_translation_error_mm",
    "mean_action_execution_error_deg",
)


def mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    mean = float(finite.mean()) if len(finite) else float("nan")
    ci = (
        float(1.96 * finite.std(ddof=1) / np.sqrt(len(finite)))
        if len(finite) > 1
        else float("nan")
    )
    return mean, ci


def main():
    parser = argparse.ArgumentParser(
        description="Summarize realized observation, tracking, and action noise by planner."
    )
    parser.add_argument("--config", default="rebuttal/configs/ablation_realistic.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = resolved_path(cfg["experiment"]["output_dir"])
    episodes = []
    for summary_path in sorted(root.glob("*/*/pose_*/episode_summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        decisions = json.loads(
            (summary_path.parent / "keyframe_decisions.json").read_text(encoding="utf-8")
        )["decisions"]
        execution_errors = []
        for step, action in enumerate(summary["actions"]):
            before = np.load(str(summary_path.parent / ("step_%03d" % step) / "gt_pose.npy"))
            after = np.load(str(summary_path.parent / ("step_%03d" % (step + 1)) / "gt_pose.npy"))
            realized = after[:3, :3] @ before[:3, :3].T
            nominal = action_rotation(action, cfg["actions"]["angles_deg"])
            execution_errors.append(rotation_error_deg(nominal, realized))
        episodes.append({
            "planner": summary["planner"],
            "object": summary["object"],
            "initial_pose_seed": int(summary["initial_pose_seed"]),
            "mean_visibility_ratio": float(np.mean([row["visibility_ratio"] for row in decisions])),
            "mean_pose_error_deg": float(np.mean([row["pose_error_deg"] for row in decisions])),
            "mean_translation_error_mm": 1000.0 * float(
                np.mean([row["translation_error_m"] for row in decisions])
            ),
            "mean_action_execution_error_deg": float(np.mean(execution_errors)),
        })
    aggregates = []
    for planner in cfg["planners"]["enabled"]:
        items = [row for row in episodes if row["planner"] == planner]
        if not items:
            continue
        result = {"planner": planner, "episodes": len(items)}
        for field in FIELDS:
            result[field], result[field + "_ci95"] = mean_ci(
                [row[field] for row in items]
            )
        aggregates.append(result)
    if episodes:
        with (root / "noise_exposure_episodes.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(episodes[0]))
            writer.writeheader()
            writer.writerows(episodes)
    if aggregates:
        with (root / "noise_exposure.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
            writer.writeheader()
            writer.writerows(aggregates)
    print(root / "noise_exposure.csv")


if __name__ == "__main__":
    main()
