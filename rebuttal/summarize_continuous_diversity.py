#!/usr/bin/env python3
"""Audit and summarize the continuous AURORA-style diversity experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from benchmark.config import load_config, resolved_path


METRICS = (
    "f@5_auc",
    "final_f@5",
    "final_recall@5",
    "final_chamfer_mm",
    "accepted_keyframes",
    "accepted_frame_ratio",
    "mean_visibility_ratio",
    "underrotation_rate",
    "stall_rate",
    "slip_rate",
    "persistent_axis_rate",
)


def mean_ci(values) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    mean = float(array.mean())
    ci = (
        float(1.96 * array.std(ddof=1) / math.sqrt(len(array)))
        if len(array) > 1
        else float("nan")
    )
    return mean, ci


def write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def derive_episode(summary_path: Path, cfg) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    decisions = json.loads(
        (summary_path.parent / "keyframe_decisions.json").read_text(encoding="utf-8")
    )["decisions"]
    bootstrap_count = len(cfg["environment"].get("bootstrap_offsets_deg", [0.0]))
    action_decisions = decisions[bootstrap_count:]
    trajectories = summary.get("action_trajectory_events", [])

    expected_actions = int(cfg["environment"]["action_steps"])
    expected_frames = int(cfg["environment"]["frames_per_action"])
    expected_total_frames = expected_actions * expected_frames
    if len(trajectories) != expected_actions:
        raise ValueError(f"{summary_path}: expected {expected_actions} trajectory events")
    if len(action_decisions) != expected_total_frames:
        raise ValueError(
            f"{summary_path}: expected {expected_total_frames} action-frame decisions, "
            f"found {len(action_decisions)}"
        )
    for event in trajectories:
        if int(event["replay_frames"]) != expected_frames:
            raise ValueError(f"{summary_path}: incorrect frames per trajectory")
        if not np.isclose(event["duration_s"], cfg["environment"]["action_duration_s"]):
            raise ValueError(f"{summary_path}: incorrect action duration")
        if not np.isclose(event["replay_rate_hz"], cfg["environment"]["sensor_rate_hz"]):
            raise ValueError(f"{summary_path}: incorrect sensor rate")
        if not bool(event.get("continuous_rendering", False)):
            raise ValueError(f"{summary_path}: trajectory is not marked continuous")

    tags = [set(event.get("fault_tags", [])) for event in trajectories]
    accepted = sum(bool(item["accepted"]) for item in action_decisions)
    max_pose_error = max(float(item["pose_error_deg"]) for item in action_decisions)
    max_translation_error = max(
        float(item["translation_error_m"]) for item in action_decisions
    )
    if max_pose_error > 1e-8 or max_translation_error > 1e-10:
        raise ValueError(f"{summary_path}: tracking is not exact as declared")

    return {
        "planner": summary["planner"],
        "object": summary["object"],
        "initial_pose_seed": int(summary["initial_pose_seed"]),
        "actions": expected_actions,
        "frames_per_action": expected_frames,
        "total_action_frames": expected_total_frames,
        "f@5_auc": float(summary["f@5_auc"]),
        "final_f@5": float(summary["final_f@5"]),
        "final_recall@5": float(summary["final_recall@5"]),
        "final_chamfer_mm": 1000.0 * float(summary["final_chamfer_m"]),
        "accepted_keyframes": int(accepted),
        "accepted_frame_ratio": float(accepted / expected_total_frames),
        "mean_visibility_ratio": float(
            np.mean([float(item["visibility_ratio"]) for item in action_decisions])
        ),
        "underrotation_rate": float(
            np.mean(["action_underrotation" in item for item in tags])
        ),
        "stall_rate": float(np.mean(["action_stall" in item for item in tags])),
        "slip_rate": float(np.mean(["grip_slip" in item for item in tags])),
        "persistent_axis_rate": float(
            np.mean(["persistent_axis_misalignment" in item for item in tags])
        ),
        "max_pose_error_deg": max_pose_error,
        "max_translation_error_mm": 1000.0 * max_translation_error,
    }


def aggregate(rows, label: str, value: str) -> dict:
    result = {label: value, "episodes": len(rows)}
    for metric in METRICS:
        mean, ci = mean_ci(row[metric] for row in rows)
        result[metric] = mean
        result[metric + "_ci95"] = ci
    return result


def fmt(row, metric: str, digits: int = 4) -> str:
    return f"{row[metric]:.{digits}f}+-{row[metric + '_ci95']:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="rebuttal/configs/diversity_continuous_realistic.yaml"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = resolved_path(cfg["experiment"]["output_dir"])
    paths = sorted(root.glob("ray_gpis/*/pose_*/episode_summary.json"))
    if not paths:
        raise RuntimeError(f"No completed episodes under {root}")
    rows = [derive_episode(path, cfg) for path in paths]

    object_rows = []
    for object_name in sorted({row["object"] for row in rows}):
        subset = [row for row in rows if row["object"] == object_name]
        object_rows.append(aggregate(subset, "object", object_name))
    overall = aggregate(rows, "object", "All added objects")
    output_rows = object_rows + [overall]

    write_csv(root / "continuous_diversity_episodes.csv", rows)
    write_csv(root / "continuous_diversity_summary.csv", output_rows)

    lines = [
        "# Continuous AURORA-style challenging-object evaluation",
        "",
        (
            "Each episode uses one initial observation and five complete 6 s "
            "active primitives. The fixed RGB-D camera renders at 15 FPS (90 "
            "frames/action); full AURORA keyframe filtering and fusion run "
            "throughout, and Ray-GPIS replans only at primitive boundaries. "
            "Tracking inference is omitted and exact executed pose is supplied."
        ),
        "",
        (
            "Stressors: dynamic palm/two-finger occlusion, depth/mask corruption, "
            "55--85% non-stalled progress, 25% configured stall probability, and "
            "22% configured grip-slip probability with persistent axis drift."
        ),
        "",
        "| Object | Episodes | F-AUC | Final F@5 | Final Recall@5 | Chamfer (mm) | Accepted keyframes | Visibility | Stall | Slip |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in output_rows:
        lines.append(
            "| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                row["object"],
                row["episodes"],
                fmt(row, "f@5_auc"),
                fmt(row, "final_f@5"),
                fmt(row, "final_recall@5"),
                fmt(row, "final_chamfer_mm", 3),
                fmt(row, "accepted_keyframes", 1),
                fmt(row, "mean_visibility_ratio", 3),
                fmt(row, "stall_rate", 3),
                fmt(row, "slip_rate", 3),
            )
        )
    lines += [
        "",
        f"Audit: {len(rows)} episodes; every episode contains 450 continuously rendered action frames and zero simulated tracking error.",
    ]
    output = root / "continuous_diversity_summary.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
