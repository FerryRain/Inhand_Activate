#!/usr/bin/env python3
"""Summarize shared-branch six-second one-decision experiments."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

from rebuttal.benchmark.config import load_config, resolved_path


PLANNERS = ("pose_novelty", "ray_gpis")
FIELDS = (
    "selected_gain_f@5",
    "selected_gain_recall@5",
    "selected_chamfer_reduction_mm",
    "selected_recovery_recall@5",
    "score_gain_correlation",
    "oracle_regret_f@5",
    "oracle_action_accuracy",
    "selected_first_second_error_deg",
    "selected_accepted_frames",
    "planning_time_s",
)


def _number(value):
    return float(value) if value is not None else float("nan")


def _mean_ci(values):
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), 0
    mean = float(values.mean())
    ci = float(1.96 * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")
    return mean, ci, int(len(values))


def _bootstrap_ci(values, seed=2026, draws=10000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(seed)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load(root):
    scenes = []
    planner_rows = []
    for path in sorted(root.glob("*/pose_*/summary.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        branch_early = {
            action: float(row["trajectory_metadata"]["first_second_error_deg"])
            for action, row in item["action_branches"].items()
        }
        scene = {
            "object": item["object"],
            "initial_pose_seed": int(item["initial_pose_seed"]),
            "mean_branch_first_second_error_deg": float(np.mean(list(branch_early.values()))),
            "max_branch_first_second_error_deg": float(np.max(list(branch_early.values()))),
        }
        scenes.append(scene)
        for planner in PLANNERS:
            row = item["planners"][planner]
            quality = row["quality"]
            planner_rows.append({
                **scene,
                "planner": planner,
                "selected_action": row["selected_action"],
                "selected_gain_f@5": float(quality["selected_gain_f@5"]),
                "selected_gain_recall@5": float(row["selected_gain_recall@5"]),
                "selected_chamfer_reduction_mm": float(row["selected_chamfer_reduction_mm"]),
                "selected_recovery_recall@5": float(row["selected_recovery_recall@5"]),
                "score_gain_correlation": _number(quality["score_gain_spearman"]),
                "oracle_regret_f@5": float(quality["oracle_regret_f@5"]),
                "oracle_action_accuracy": float(quality["oracle_action_accuracy"]),
                "selected_first_second_error_deg": float(
                    row["selected_trajectory_metadata"]["first_second_error_deg"]
                ),
                "selected_accepted_frames": int(row["selected_accepted_frames"]),
                "planning_time_s": float(row["runtime"]["planning_s"]),
            })
    return scenes, planner_rows


def _aggregate(rows):
    output = []
    for planner in PLANNERS:
        items = [row for row in rows if row["planner"] == planner]
        result = {"planner": planner, "paired_scenes": len(items)}
        for field in FIELDS:
            result[field], result[field + "_ci95"], result[field + "_n"] = _mean_ci(
                row[field] for row in items
            )
        output.append(result)
    return output


def _paired_tests(scenes, rows):
    lookup = {
        (row["planner"], row["object"], row["initial_pose_seed"]): row for row in rows
    }
    difficulty = np.asarray(
        [scene["mean_branch_first_second_error_deg"] for scene in scenes], dtype=np.float64
    )
    hard_threshold = float(np.percentile(difficulty, 75.0))
    subsets = {
        "all": scenes,
        "hard_first_second_quartile": [
            scene for scene in scenes
            if scene["mean_branch_first_second_error_deg"] >= hard_threshold
        ],
        "planner_action_disagreement": [
            scene for scene in scenes
            if lookup[("pose_novelty", scene["object"], scene["initial_pose_seed"])]["selected_action"]
            != lookup[("ray_gpis", scene["object"], scene["initial_pose_seed"])]["selected_action"]
        ],
    }
    results = []
    for subset_name, subset in subsets.items():
        for field in FIELDS[:7]:
            deltas = []
            for scene in subset:
                key = (scene["object"], scene["initial_pose_seed"])
                pose = lookup[("pose_novelty", *key)][field]
                ray = lookup[("ray_gpis", *key)][field]
                # Positive always means Ray is better.
                delta = pose - ray if field == "oracle_regret_f@5" else ray - pose
                if np.isfinite(delta):
                    deltas.append(float(delta))
            values = np.asarray(deltas, dtype=np.float64)
            ci = _bootstrap_ci(values) if len(values) else [float("nan"), float("nan")]
            nonzero = values[np.abs(values) > 1e-12]
            p_value = (
                float(wilcoxon(nonzero, alternative="two-sided").pvalue)
                if len(nonzero) else 1.0
            )
            results.append({
                "subset": subset_name,
                "metric": field,
                "pairs": int(len(values)),
                "ray_improvement": float(np.mean(values)) if len(values) else float("nan"),
                "ci95_low": float(ci[0]),
                "ci95_high": float(ci[1]),
                "wins": int(np.sum(values > 1e-12)),
                "ties": int(np.sum(np.abs(values) <= 1e-12)),
                "losses": int(np.sum(values < -1e-12)),
                "p_two_sided": p_value,
                "hard_threshold_deg": hard_threshold,
            })
    return results, hard_threshold


def _fmt(row, field, digits=4):
    value = float(row[field])
    ci = float(row[field + "_ci95"])
    return f"{value:.{digits}f}+-{ci:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = resolved_path(cfg["experiment"]["output_dir"])
    scenes, rows = _load(root)
    expected = len(cfg["assets"]["objects"]) * len(cfg["experiment"]["initial_pose_seeds"])
    if len(scenes) != expected:
        raise RuntimeError("Expected %d scenes, found %d" % (expected, len(scenes)))
    aggregate = _aggregate(rows)
    tests, hard_threshold = _paired_tests(scenes, rows)
    _write_csv(root / "one_step_scenes.csv", scenes)
    _write_csv(root / "one_step_planner_episodes.csv", rows)
    _write_csv(root / "one_step_aggregate.csv", aggregate)
    _write_csv(root / "one_step_paired_tests.csv", tests)

    lines = [
        "# Six-second continuous one-decision summary",
        "",
        "Primary analysis: selected F@5 gain over the complete six-second empirical trajectory. Positive paired improvement means Ray-GPIS is better.",
        "",
        "| Planner | Selected F gain | Recall gain | Missing-surface recovery | Chamfer reduction (mm) | Corr. | Regret | Accuracy | Accepted frames | Time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            row["planner"],
            _fmt(row, "selected_gain_f@5"),
            _fmt(row, "selected_gain_recall@5"),
            _fmt(row, "selected_recovery_recall@5"),
            _fmt(row, "selected_chamfer_reduction_mm", 3),
            _fmt(row, "score_gain_correlation"),
            _fmt(row, "oracle_regret_f@5"),
            _fmt(row, "oracle_action_accuracy"),
            _fmt(row, "selected_accepted_frames", 2),
            _fmt(row, "planning_time_s", 3),
        ))
    lines += [
        "",
        "The hard-motion stratum is the predeclared top quartile of scene-level mean first-second transition error (threshold %.2f deg); membership uses trajectory metadata only, not reconstruction outcomes." % hard_threshold,
        "",
        "| Subset | Metric | Pairs | Ray improvement [95% CI] | W/T/L | p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in tests:
        if row["metric"] not in ("selected_gain_f@5", "selected_recovery_recall@5", "score_gain_correlation", "oracle_regret_f@5"):
            continue
        lines.append("| %s | %s | %d | %.5f [%.5f, %.5f] | %d/%d/%d | %.3g |" % (
            row["subset"], row["metric"], row["pairs"], row["ray_improvement"],
            row["ci95_low"], row["ci95_high"], row["wins"], row["ties"], row["losses"],
            row["p_two_sided"],
        ))
    (root / "one_step_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(root / "one_step_summary.md")


if __name__ == "__main__":
    main()
