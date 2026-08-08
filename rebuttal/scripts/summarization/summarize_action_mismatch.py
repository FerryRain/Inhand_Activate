#!/usr/bin/env python3
"""Summarize persistent under-rotation, stall, and grip-slip experiments.

The simulator records action-fault tags on the observation produced by each
primitive.  This script reports both the realized mismatch and the policy's
immediate/next-decision behavior.  All comparisons remain paired by object and
initial orientation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rebuttal.benchmark.config import load_config, resolved_path
from rebuttal.benchmark.geometry import (
    action_rotation,
    current_view_direction_object,
    next_view_direction,
    rotation_error_deg,
)


METRICS = (
    "mean_action_so3_error_deg",
    "mean_nominal_target_view_error_deg",
    "stall_action_rate",
    "slip_action_rate",
    "persistent_misalignment_action_rate",
    "episode_stall_or_slip_rate",
    "fault_action_gain_f@5",
    "fault_action_regret_f@5",
    "post_fault_next_gain_f@5",
    "post_fault_next_regret_f@5",
)


def _angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _mean(values) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    return float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")


def _mean_ci(values):
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), 0
    mean = float(values.mean())
    ci = float(1.96 * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")
    return mean, ci, int(len(values))


def _write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _derive(path: Path, cfg):
    item = json.loads(path.read_text(encoding="utf-8"))
    action_count = len(item["actions"])
    tags_by_action = [set() for _ in range(action_count)]
    for event in item.get("fault_events", []):
        # Result observations are numbered 1..T; their faults belong to the
        # primitive selected at decision indices 0..T-1.
        index = int(event["step"]) - 1
        if 0 <= index < action_count:
            tags_by_action[index].update(event.get("fault_tags", []))

    so3_errors = []
    view_errors = []
    for index, action in enumerate(item["actions"]):
        before = np.load(str(path.parent / ("step_%03d" % index) / "gt_pose.npy"))
        after = np.load(str(path.parent / ("step_%03d" % (index + 1)) / "gt_pose.npy"))
        realized = after[:3, :3] @ before[:3, :3].T
        nominal = action_rotation(action, cfg["actions"]["angles_deg"])
        so3_errors.append(rotation_error_deg(nominal, realized))
        nominal_view = next_view_direction(before, action, cfg["actions"]["angles_deg"])
        actual_view = current_view_direction_object(after)
        view_errors.append(_angle_deg(nominal_view, actual_view))

    fault_indices = [
        index for index, tags in enumerate(tags_by_action)
        if "action_stall" in tags or "grip_slip" in tags
    ]
    next_indices = [index + 1 for index in fault_indices if index + 1 < action_count]
    policy = item["planner_metrics"]
    return {
        "planner": item["planner"],
        "object": item["object"],
        "initial_pose_seed": int(item["initial_pose_seed"]),
        "mean_action_so3_error_deg": _mean(so3_errors),
        "mean_nominal_target_view_error_deg": _mean(view_errors),
        "stall_action_rate": _mean("action_stall" in tags for tags in tags_by_action),
        "slip_action_rate": _mean("grip_slip" in tags for tags in tags_by_action),
        "persistent_misalignment_action_rate": _mean(
            "persistent_axis_misalignment" in tags for tags in tags_by_action
        ),
        "episode_stall_or_slip_rate": float(bool(fault_indices)),
        "fault_action_gain_f@5": _mean(policy[index]["selected_gain_f@5"] for index in fault_indices),
        "fault_action_regret_f@5": _mean(policy[index]["oracle_regret_f@5"] for index in fault_indices),
        "post_fault_next_gain_f@5": _mean(policy[index]["selected_gain_f@5"] for index in next_indices),
        "post_fault_next_regret_f@5": _mean(policy[index]["oracle_regret_f@5"] for index in next_indices),
        "fault_actions": int(len(fault_indices)),
        "post_fault_decisions": int(len(next_indices)),
    }


def _aggregate(rows, planners):
    results = []
    for planner in planners:
        items = [row for row in rows if row["planner"] == planner]
        if not items:
            continue
        result = {"planner": planner, "paired_scenes": len(items)}
        for metric in METRICS:
            mean, ci, count = _mean_ci(row[metric] for row in items)
            result[metric] = mean
            result[metric + "_ci95"] = ci
            result[metric + "_n"] = count
        result["fault_actions"] = int(sum(row["fault_actions"] for row in items))
        result["post_fault_decisions"] = int(sum(row["post_fault_decisions"] for row in items))
        results.append(result)
    return results


def _paired_differences(rows, planners):
    lookup = {
        (row["planner"], row["object"], row["initial_pose_seed"]): row for row in rows
    }
    results = []
    for first_index, first in enumerate(planners):
        for second in planners[first_index + 1:]:
            keys = sorted(
                (row["object"], row["initial_pose_seed"])
                for row in rows if row["planner"] == first
                and (second, row["object"], row["initial_pose_seed"]) in lookup
            )
            for metric in METRICS:
                deltas = [lookup[(second, *key)][metric] - lookup[(first, *key)][metric] for key in keys]
                mean, ci, count = _mean_ci(deltas)
                results.append({
                    "first_planner": first,
                    "second_planner": second,
                    "metric": metric,
                    "second_minus_first": mean,
                    "ci95": ci,
                    "valid_pairs": count,
                })
    return results


def _fmt(row, metric, digits=4):
    value = row[metric]
    ci = row[metric + "_ci95"]
    if not np.isfinite(value):
        return "--"
    if not np.isfinite(ci):
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f}+-{ci:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = resolved_path(cfg["experiment"]["output_dir"])
    paths = sorted(root.glob("*/*/pose_*/episode_summary.json"))
    rows = [_derive(path, cfg) for path in paths]
    if not rows:
        raise RuntimeError("No episode summaries under %s" % root)
    aggregate = _aggregate(rows, cfg["planners"]["enabled"])
    paired = _paired_differences(rows, cfg["planners"]["enabled"])
    _write_csv(root / "action_mismatch_episodes.csv", rows)
    _write_csv(root / "action_mismatch_aggregate.csv", aggregate)
    _write_csv(root / "action_mismatch_paired_differences.csv", paired)

    lines = [
        "# Persistent action-mismatch summary",
        "",
        "Declared simulation stress: primitive under-rotation/stall plus grip-slip-induced persistent axis drift; it is not claimed as a fit to measured hardware errors.",
        "",
        "| Planner | SO(3) action error (deg) | Target-view error (deg) | Stall rate | Slip rate | Persistent-axis rate | Post-fault next gain | Post-fault next regret |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                row["planner"],
                _fmt(row, "mean_action_so3_error_deg", 2),
                _fmt(row, "mean_nominal_target_view_error_deg", 2),
                _fmt(row, "stall_action_rate", 3),
                _fmt(row, "slip_action_rate", 3),
                _fmt(row, "persistent_misalignment_action_rate", 3),
                _fmt(row, "post_fault_next_gain_f@5"),
                _fmt(row, "post_fault_next_regret_f@5"),
            )
        )
    (root / "action_mismatch_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(root / "action_mismatch_summary.md")


if __name__ == "__main__":
    main()
