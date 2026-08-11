#!/usr/bin/env python3
"""Summarize the joint pose-error and transient depth-gap planner test."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from rebuttal.benchmark.config import load_config, resolved_path


METRICS = {
    "missing_patch_recall@5": ("selected_recovery_recall@5", True),
    "selected_f@5_gain": ("selected_gain_f@5", True),
    "score_gain_correlation": ("score_gain_spearman", True),
    "oracle_regret_f@5": ("oracle_regret_f@5", False),
    "oracle_action_accuracy": ("oracle_action_accuracy", True),
    "planning_time_s": ("planning_s", False),
}
TARGET_METRICS = (
    "gap_score_recovery_correlation",
    "gap_oracle_regret_recall@5",
    "gap_oracle_action_accuracy",
)
ALL_AGGREGATE_METRICS = tuple(METRICS) + TARGET_METRICS


def _write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    mean = float(finite.mean()) if len(finite) else float("nan")
    ci = (
        float(1.96 * finite.std(ddof=1) / np.sqrt(len(finite)))
        if len(finite) > 1
        else float("nan")
    )
    return mean, ci, int(len(finite))


def _bootstrap_ci(values, seed=2026, draws=10000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return np.percentile(samples, [2.5, 97.5])


def _planner_value(row, field):
    if field == "planning_s":
        return float(row["runtime"]["planning_s"])
    if field in row:
        return float(row[field])
    value = row["quality"][field]
    return float(value) if value is not None else float("nan")


def _gap_quality(scene, planner_row):
    actions = ("minus_x", "minus_y", "plus_z")
    scores = np.asarray(
        [
            np.nan
            if planner_row["action_scores"][action] is None
            else float(planner_row["action_scores"][action])
            for action in actions
        ],
        dtype=np.float64,
    )
    recoveries = np.asarray(
        [
            float(scene["action_branches"][action]["recovery_recall@5"])
            for action in actions
        ],
        dtype=np.float64,
    )
    valid = (
        np.all(np.isfinite(scores))
        and np.ptp(scores) > 1e-12
        and np.ptp(recoveries) > 1e-12
    )
    if valid:
        from scipy.stats import spearmanr

        correlation = float(spearmanr(scores, recoveries).correlation)
    else:
        correlation = float("nan")
    selected = actions.index(planner_row["selected_action"])
    oracle = int(np.argmax(recoveries))
    return {
        "gap_score_recovery_correlation": correlation,
        "gap_oracle_regret_recall@5": float(
            recoveries[oracle] - recoveries[selected]
        ),
        "gap_oracle_action_accuracy": float(selected == oracle),
    }


def _holm(rows):
    order = sorted(range(len(rows)), key=lambda index: rows[index]["p_two_sided"])
    running = 0.0
    total = len(order)
    for rank, index in enumerate(order):
        running = max(
            running,
            min(1.0, (total - rank) * rows[index]["p_two_sided"]),
        )
        rows[index]["p_holm"] = float(running)


def _validate_scene(scene, planners):
    key = (scene["object"], int(scene["initial_pose_seed"]))
    if int(scene["recovery_target_point_count"]) <= 0:
        raise RuntimeError("Empty missing-patch target for %r" % (key,))
    if set(scene["action_branches"]) != {"minus_x", "minus_y", "plus_z"}:
        raise RuntimeError("Three shared action branches required for %r" % (key,))
    if any(planner not in scene["planners"] for planner in planners):
        raise RuntimeError("Missing planner output for %r" % (key,))

    prefix_faults = scene["protocol"].get("shared_prefix_observation_faults", [])
    if len(prefix_faults) != 1:
        raise RuntimeError("Exactly one faulty prefix observation required for %r" % (key,))
    prefix = prefix_faults[0]
    tags = set(prefix["fault_tags"])
    required = {"contiguous_depth_registration_failure", "scheduled_pose_outlier"}
    if tags != required:
        raise RuntimeError("Unexpected joint prefix faults for %r: %r" % (key, tags))
    if not np.isclose(float(prefix["pose_error_deg"]), 6.0, atol=1e-6):
        raise RuntimeError("Unexpected prefix rotation error for %r" % (key,))
    if not np.isclose(float(prefix["translation_error_mm"]), 3.0, atol=1e-6):
        raise RuntimeError("Unexpected prefix translation error for %r" % (key,))

    for action, branch in scene["action_branches"].items():
        faults = branch.get("observation_faults", [])
        if len(faults) != 1:
            raise RuntimeError("One recovery observation required for %r/%s" % (key, action))
        row = faults[0]
        tags = set(row["fault_tags"])
        if tags != {"scheduled_pose_outlier"}:
            raise RuntimeError("Unexpected recovery faults for %r/%s: %r" % (
                key, action, tags
            ))
        if not np.isclose(float(row["pose_error_deg"]), 6.0, atol=1e-6):
            raise RuntimeError("Unexpected branch rotation error for %r/%s" % (key, action))
        if not np.isclose(float(row["translation_error_mm"]), 3.0, atol=1e-6):
            raise RuntimeError("Unexpected branch translation error for %r/%s" % (key, action))


def _fmt(row, metric, digits=3):
    return ("%.*f$\\pm$%.*f" % (
        digits,
        float(row[metric]),
        digits,
        float(row[metric + "_ci95"]),
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="rebuttal/configs/baseline_joint_pose_depth_gap.yaml",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    robustness = cfg["environment"]["robustness"]
    if str(robustness.get("action_noise_mode", "none")) != "none":
        raise RuntimeError("Joint pose/depth test must not include action noise")
    if float(robustness.get("action_noise_scale", 0.0)) != 0.0:
        raise RuntimeError("Joint pose/depth test must have zero action-noise scale")
    root = resolved_path(cfg["experiment"]["output_dir"])
    planners = list(cfg["planners"]["enabled"])
    scenes = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("*/pose_*/summary.json"))
    ]
    expected = len(cfg["assets"]["objects"]) * len(
        cfg["experiment"]["initial_pose_seeds"]
    )
    if len(scenes) != expected:
        raise RuntimeError("Expected %d scenes, found %d" % (expected, len(scenes)))
    for scene in scenes:
        _validate_scene(scene, planners)

    episode_rows = []
    for scene in scenes:
        for planner in planners:
            planner_row = scene["planners"][planner]
            row = {
                "object": scene["object"],
                "initial_pose_seed": int(scene["initial_pose_seed"]),
                "planner": planner,
                "selected_action": planner_row["selected_action"],
                "target_point_count": int(scene["recovery_target_point_count"]),
            }
            for metric, (field, _) in METRICS.items():
                row[metric] = _planner_value(planner_row, field)
            row.update(_gap_quality(scene, planner_row))
            episode_rows.append(row)

    aggregate = []
    for planner in planners:
        selected = [row for row in episode_rows if row["planner"] == planner]
        out = {"planner": planner, "paired_scenes": len(selected)}
        for metric in ALL_AGGREGATE_METRICS:
            out[metric], out[metric + "_ci95"], out[metric + "_n"] = _mean_ci(
                [row[metric] for row in selected]
            )
        weights = np.asarray(
            [row["target_point_count"] for row in selected], dtype=np.float64
        )
        recoveries = np.asarray(
            [row["missing_patch_recall@5"] for row in selected], dtype=np.float64
        )
        out["point_weighted_missing_patch_recall@5"] = float(
            np.sum(weights * recoveries) / np.sum(weights)
        )
        aggregate.append(out)

    lookup = {
        (row["planner"], row["object"], row["initial_pose_seed"]): row
        for row in episode_rows
    }
    primary_tests = []
    for comparator in planners:
        if comparator == "ray_gpis":
            continue
        differences = []
        for scene in scenes:
            key = (scene["object"], int(scene["initial_pose_seed"]))
            full = lookup[("ray_gpis", *key)]["missing_patch_recall@5"]
            other = lookup[(comparator, *key)]["missing_patch_recall@5"]
            differences.append(full - other)
        differences = np.asarray(differences, dtype=np.float64)
        ci = _bootstrap_ci(differences)
        nonzero = differences[np.abs(differences) > 1e-12]
        p_value = (
            float(wilcoxon(nonzero, alternative="two-sided").pvalue)
            if len(nonzero)
            else 1.0
        )
        primary_tests.append({
            "comparator": comparator,
            "pairs": int(len(differences)),
            "mean_ray_improvement": float(differences.mean()),
            "ci95_low": float(ci[0]),
            "ci95_high": float(ci[1]),
            "wins": int(np.sum(differences > 1e-12)),
            "ties": int(np.sum(np.abs(differences) <= 1e-12)),
            "losses": int(np.sum(differences < -1e-12)),
            "p_two_sided": p_value,
        })
    _holm(primary_tests)

    _write_csv(root / "joint_gap_episodes.csv", episode_rows)
    _write_csv(root / "joint_gap_aggregate.csv", aggregate)
    _write_csv(root / "joint_gap_paired_tests.csv", primary_tests)

    labels = {
        "pose_novelty": "Pose-Novelty",
        "actnerf": "Adapted ActNeRF",
        "pb_nbv": "Adapted PB-NBV",
        "ray_gpis": "Ray-GPIS",
    }
    lines = [
        "# Joint pose/depth-gap active-planner robustness",
        "",
        "A shared tracked prefix view loses one contiguous 70% depth region. "
        "The prefix and each recovery branch have exactly 6 deg / 3 mm pose "
        "error. The depth gap is transient and no method is prevented from "
        "revisiting the missing region.",
        "",
        "| Method | Missing-patch R@5 | Gap corr. | Gap regret | Gap accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append("| %s | %s | %s | %s | %s |" % (
            labels.get(row["planner"], row["planner"]),
            _fmt(row, "missing_patch_recall@5"),
            _fmt(row, "gap_score_recovery_correlation"),
            _fmt(row, "gap_oracle_regret_recall@5"),
            _fmt(row, "gap_oracle_action_accuracy"),
        ))
    lines += [
        "",
        "Gap correlation and regret compare action scores with recovery of the "
        "specific missing prefix region. The next table instead evaluates "
        "global F@5 gain over the complete reconstruction.",
        "",
        "| Method | Global F gain | Global corr. | Global regret | Time (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append("| %s | %s | %s | %s | %s |" % (
            labels.get(row["planner"], row["planner"]),
            _fmt(row, "selected_f@5_gain"),
            _fmt(row, "score_gain_correlation"),
            _fmt(row, "oracle_regret_f@5"),
            _fmt(row, "planning_time_s"),
        ))
    lines += [
        "",
        "Paired differences below are Ray-GPIS minus each comparator on "
        "missing-patch R@5; Holm correction covers the three comparisons.",
        "",
        "| Comparator | Difference [95% bootstrap CI] | W/T/L | p (Holm) |",
        "|---|---:|---:|---:|",
    ]
    for row in primary_tests:
        lines.append(
            "| %s | %.4f [%.4f, %.4f] | %d/%d/%d | %.3g |"
            % (
                labels.get(row["comparator"], row["comparator"]),
                row["mean_ray_improvement"],
                row["ci95_low"],
                row["ci95_high"],
                row["wins"],
                row["ties"],
                row["losses"],
                row["p_holm"],
            )
        )
    (root / "joint_gap_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(root / "joint_gap_summary.md")


if __name__ == "__main__":
    main()
