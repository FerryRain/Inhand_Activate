#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from benchmark.config import load_config, resolved_path


ACTIONS = ("minus_x", "minus_y", "plus_z")

FINAL_FIELDS = (
    "final_recall@5",
    "final_f@5",
    "final_chamfer_mm",
)
EXPLORATION_FIELDS = (
    "final_visibility_coverage",
    "visibility_coverage_auc",
    "recall@5_auc",
    "f@5_auc",
    "cumulative_active_gain_f@5",
)
POLICY_FIELDS = (
    "selected_gain_f@5_per_action",
    "oracle_gain_f@5_per_action",
    "score_gain_correlation",
    "correlation_valid_rate",
    "oracle_regret_f@5",
    "oracle_action_accuracy",
)
RUNTIME_FIELDS = (
    "representation_update_s",
    "candidate_scoring_s",
    "planning_time_s",
)
ALL_FIELDS = FINAL_FIELDS + EXPLORATION_FIELDS + POLICY_FIELDS + RUNTIME_FIELDS


def _finite_mean(values: Iterable[float]) -> float:
    array = np.asarray([
        float(value) if value is not None else float("nan") for value in values
    ], dtype=np.float64)
    return float(np.nanmean(array)) if np.any(np.isfinite(array)) else float("nan")


def _as_float(value) -> float:
    return float(value) if value is not None else float("nan")


def _load_summaries(root: Path):
    by_key = {}
    for path in root.glob("*/*/pose_*/episode_summary.json"):
        with path.open("r", encoding="utf-8") as handle:
            item = json.load(handle)
        item.setdefault("planner_seed", 0)
        item["_episode_dir"] = str(path.parent)
        key = (item["planner"], item["object"], item["initial_pose_seed"], item["planner_seed"])
        if key not in by_key or "_seed_" in path.parent.name:
            by_key[key] = item
    return list(by_key.values())


def _counterfactual_gains(episode_dir: Path, step: int) -> Dict[str, float]:
    gains = {}
    counterfactual_dir = episode_dir / ("step_%03d" % step) / "counterfactual"
    for action in ACTIONS:
        path = counterfactual_dir / (action + "_metrics.json")
        with path.open("r", encoding="utf-8") as handle:
            gains[action] = float(json.load(handle)["gain_f@5"])
    return gains


def _derive_episode(item: Dict, policy_step_rows: List[Dict], curve_rows: List[Dict]) -> Dict:
    reconstruction = item["reconstruction_metrics"]
    policy = item["planner_metrics"]
    runtime = item["runtime_metrics"]
    episode_dir = Path(item["_episode_dir"])
    planner = item["planner"]
    object_name = item["object"]
    pose_seed = int(item["initial_pose_seed"])
    planner_seed = int(item.get("planner_seed", 0))
    with (episode_dir / "visibility_coverage.json").open("r", encoding="utf-8") as handle:
        visibility = json.load(handle)
    visibility_curve = [float(value) for value in visibility["coverage_curve"]]
    if len(visibility_curve) != len(reconstruction):
        raise ValueError("Visibility/reconstruction state count mismatch in %s" % episode_dir)

    for step, metrics in enumerate(reconstruction):
        curve_rows.append({
            "planner": planner,
            "object": object_name,
            "initial_pose_seed": pose_seed,
            "planner_seed": planner_seed,
            "step": step,
            "recall@5": float(metrics["recall@5"]),
            "f@5": float(metrics["f@5"]),
            "surface_coverage": float(metrics["surface_coverage"]),
            "visibility_surface_coverage": visibility_curve[step],
            "chamfer_mm": float(metrics["chamfer_m"]) * 1000.0,
        })

    selected_gains = []
    oracle_gains = []
    correlations = []
    regrets = []
    accuracies = []
    consistency_errors = []
    for step, quality in enumerate(policy):
        gains = _counterfactual_gains(episode_dir, step)
        selected_action = item["actions"][step]
        selected_gain = float(gains[selected_action])
        oracle_action = max(ACTIONS, key=lambda action: gains[action])
        oracle_gain = float(gains[oracle_action])
        actual_gain = float(reconstruction[step + 1]["f@5"] - reconstruction[step]["f@5"])
        correlation = _as_float(quality["score_gain_spearman"])
        regret = float(oracle_gain - selected_gain)
        accuracy = float(selected_action == oracle_action)
        scores = quality.get("action_scores", {})
        selected_gains.append(selected_gain)
        oracle_gains.append(oracle_gain)
        correlations.append(correlation)
        regrets.append(regret)
        accuracies.append(accuracy)
        consistency_errors.append(abs(selected_gain - actual_gain))
        policy_step_rows.append({
            "planner": planner,
            "object": object_name,
            "initial_pose_seed": pose_seed,
            "planner_seed": planner_seed,
            "step": step,
            "selected_action": selected_action,
            "oracle_action": oracle_action,
            "selected_gain_f@5": selected_gain,
            "oracle_gain_f@5": oracle_gain,
            "oracle_regret_f@5": regret,
            "oracle_action_accuracy": accuracy,
            "score_gain_spearman": correlation,
            "minus_x_gain_f@5": gains["minus_x"],
            "minus_y_gain_f@5": gains["minus_y"],
            "plus_z_gain_f@5": gains["plus_z"],
            "minus_x_score": _as_float(scores.get("minus_x", float("nan"))),
            "minus_y_score": _as_float(scores.get("minus_y", float("nan"))),
            "plus_z_score": _as_float(scores.get("plus_z", float("nan"))),
        })

    last = reconstruction[-1]
    result = {
        "planner": planner,
        "object": object_name,
        "initial_pose_seed": pose_seed,
        "planner_seed": planner_seed,
        "final_recall@5": float(last["recall@5"]),
        "final_f@5": float(last["f@5"]),
        "final_chamfer_mm": float(last["chamfer_m"]) * 1000.0,
        "final_visibility_coverage": float(visibility["final_coverage"]),
        "visibility_coverage_auc": float(visibility["coverage_auc"]),
        "recall@5_auc": float(item["recall@5_auc"]),
        "f@5_auc": float(item["f@5_auc"]),
        "cumulative_active_gain_f@5": float(last["f@5"] - reconstruction[0]["f@5"]),
        "selected_gain_f@5_per_action": _finite_mean(selected_gains),
        "oracle_gain_f@5_per_action": _finite_mean(oracle_gains),
        "score_gain_correlation": _finite_mean(correlations),
        "correlation_valid_rate": (
            float(np.mean(np.isfinite(correlations))) if planner != "fixed" else float("nan")
        ),
        "oracle_regret_f@5": _finite_mean(regrets),
        # Oracle accuracy only needs the selected action and the counterfactual
        # gains. It is therefore defined for the open-loop fixed schedule too,
        # even though score--gain correlation is not.
        "oracle_action_accuracy": _finite_mean(accuracies),
        "representation_update_s": _finite_mean(row["representation_update_s"] for row in runtime),
        "candidate_scoring_s": _finite_mean(row["candidate_scoring_s"] for row in runtime),
        "planning_time_s": _finite_mean(row["planning_s"] for row in runtime),
        "gain_consistency_max_abs_error": max(consistency_errors) if consistency_errors else 0.0,
        "coverage_recall_max_abs_error": max(
            abs(float(row["surface_coverage"]) - float(row["recall@5"])) for row in reconstruction
        ),
        "visibility_monotonic_min_delta": float(np.min(np.diff(visibility_curve))),
    }
    return result


def _collapse_pairs(episodes: List[Dict]) -> List[Dict]:
    groups = defaultdict(list)
    for item in episodes:
        groups[(item["planner"], item["object"], item["initial_pose_seed"])].append(item)
    paired = []
    for (planner, object_name, pose_seed), items in sorted(groups.items()):
        row = {
            "planner": planner,
            "object": object_name,
            "initial_pose_seed": pose_seed,
            "planner_seeds": len(items),
        }
        for field in ALL_FIELDS:
            row[field] = _finite_mean(item[field] for item in items)
        paired.append(row)
    return paired


def _aggregate(paired: List[Dict]) -> List[Dict]:
    groups = defaultdict(list)
    for item in paired:
        groups[item["planner"]].append(item)
    rows = []
    for planner, items in sorted(groups.items()):
        row = {"planner": planner, "paired_scenes": len(items)}
        for field in ALL_FIELDS:
            values = np.asarray([item[field] for item in items], dtype=np.float64)
            finite = values[np.isfinite(values)]
            row[field] = float(np.mean(finite)) if len(finite) else float("nan")
            row[field + "_ci95"] = (
                float(1.96 * np.std(finite, ddof=1) / np.sqrt(len(finite))) if len(finite) > 1 else float("nan")
            )
            row[field + "_n"] = int(len(finite))
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: List[Dict], fieldnames=None):
    if not rows:
        return
    names = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _write_category_csv(root: Path, filename: str, rows: List[Dict], fields):
    fieldnames = ["planner", "paired_scenes"]
    for field in fields:
        fieldnames.extend((field, field + "_ci95", field + "_n"))
    projected = [{name: row[name] for name in fieldnames} for row in rows]
    _write_csv(root / filename, projected, fieldnames)


def _fmt(row: Dict, field: str, digits: int = 4, scale: float = 1.0) -> str:
    mean = float(row[field]) * scale
    ci = float(row[field + "_ci95"]) * scale
    if not np.isfinite(mean):
        return "—"
    if not np.isfinite(ci):
        return ("%%.%df" % digits) % mean
    return (("%%.%df±%%.%df" % (digits, digits)) % (mean, ci))


def _markdown(rows: List[Dict], validation: Dict, cfg: Dict) -> str:
    by_name = {row["planner"]: row for row in rows}
    order = [name for name in ("fixed", "pb_nbv", "actnerf", "ray_gpis") if name in by_name]
    paired_counts = {int(row["paired_scenes"]) for row in rows}
    paired_text = str(next(iter(paired_counts))) if len(paired_counts) == 1 else "/".join(map(str, sorted(paired_counts)))
    bootstrap_offsets = cfg["environment"].get("bootstrap_offsets_deg")
    bootstrap_count = len(bootstrap_offsets) if bootstrap_offsets is not None else 3
    lines = [
        "# Complete baseline metric summary",
        "",
        f"All uncertainty intervals are 95% normal-approximation confidence intervals over {paired_text} paired object/pose scenes. "
        "ActNeRF's initialization seeds are averaged within each scene first.",
        "",
        "## A. Final geometry",
        "",
        "| Method | Recall@5 ↑ | F-score@5 ↑ | Chamfer (mm) ↓ |",
        "|---|---:|---:|---:|",
    ]
    for name in order:
        row = by_name[name]
        lines.append("| %s | %s | %s | %s |" % (
            name,
            _fmt(row, "final_recall@5"),
            _fmt(row, "final_f@5"),
            _fmt(row, "final_chamfer_mm", digits=3),
        ))
    lines += [
        "",
        "## B. Active exploration efficiency",
        "",
        "| Method | Final visible surface ↑ | Visible-area AUC ↑ | Recall@5 AUC ↑ | F-score@5 AUC ↑ | Cumulative active F@5 gain ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in order:
        row = by_name[name]
        lines.append("| %s | %s | %s | %s | %s | %s |" % (
            name,
            _fmt(row, "final_visibility_coverage"),
            _fmt(row, "visibility_coverage_auc"),
            _fmt(row, "recall@5_auc"),
            _fmt(row, "f@5_auc"),
            _fmt(row, "cumulative_active_gain_f@5"),
        ))
    lines += [
        "",
        "Visible surface is the cumulative area of GT mesh triangles hit by the actually acquired camera rays, divided by total GT mesh area. "
        "AUC is the trapezoidal mean over six cumulative reconstruction states: one initial state and five active-action states. "
        "The initial state contains %d shared observation(s)." % bootstrap_count,
        "",
        "## C. Active decision policy",
        "",
        "| Method | Selected gain/action ↑ | Oracle gain/action ↑ | Score–gain Spearman ↑ | Valid corr. steps ↑ | Oracle regret ↓ | Oracle accuracy ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in order:
        row = by_name[name]
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            name,
            _fmt(row, "selected_gain_f@5_per_action"),
            _fmt(row, "oracle_gain_f@5_per_action"),
            _fmt(row, "score_gain_correlation"),
            _fmt(row, "correlation_valid_rate", digits=1, scale=100.0) + "%" if np.isfinite(row["correlation_valid_rate"]) else "—",
            _fmt(row, "oracle_regret_f@5"),
            _fmt(row, "oracle_action_accuracy", digits=1, scale=100.0) + "%" if np.isfinite(row["oracle_action_accuracy"]) else "—",
        ))
    lines += [
        "",
        "At each state, all three executable actions {-x, -y, +z} are rolled out counterfactually. "
        "Spearman correlation compares their action scores with their realized F@5 gains. "
        "Fixed has no planner scores, so only its correlation is undefined; oracle accuracy remains measurable from its selected action.",
        "",
        "## D. Runtime per planning step",
        "",
        "| Method | Representation update (s) ↓ | Candidate scoring (s) ↓ | Total planning (s) ↓ |",
        "|---|---:|---:|---:|",
    ]
    for name in order:
        row = by_name[name]
        lines.append("| %s | %s | %s | %s |" % (
            name,
            _fmt(row, "representation_update_s", digits=3),
            _fmt(row, "candidate_scoring_s", digits=6),
            _fmt(row, "planning_time_s", digits=3),
        ))
    lines += [
        "",
        "Total planning time is representation update plus candidate scoring. For adapted ActNeRF, representation update is five-model ensemble training and candidate scoring is ensemble rendering/variance evaluation.",
        "",
        "## Consistency checks",
        "",
        "- Episodes checked: %d" % validation["episodes"],
        "- Maximum |saved selected gain - actual next-state F@5 difference|: %.3e" % validation["gain_max_abs_error"],
        "- Maximum |legacy evaluator surface_coverage - Recall@5| (backward-compatibility check): %.3e" % validation["coverage_recall_max_abs_error"],
        "- Minimum visibility-coverage step increment: %.3e" % validation["visibility_monotonic_min_delta"],
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rebuttal/configs/formal_120_sixview_gpu.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = resolved_path(cfg["experiment"]["output_dir"])
    summaries = _load_summaries(root)
    if not summaries:
        raise RuntimeError("No episode summaries under %s" % root)

    policy_steps = []
    curves = []
    episode_rows = [_derive_episode(item, policy_steps, curves) for item in summaries]
    paired = _collapse_pairs(episode_rows)
    aggregate = _aggregate(paired)

    aggregate_fields = ["planner", "paired_scenes"]
    for field in ALL_FIELDS:
        aggregate_fields.extend((field, field + "_ci95", field + "_n"))
    _write_csv(root / "all_metrics_episodes.csv", episode_rows)
    _write_csv(root / "all_metrics_paired.csv", paired)
    _write_csv(root / "all_metrics_aggregate.csv", aggregate, aggregate_fields)
    _write_csv(root / "policy_step_metrics.csv", policy_steps)
    _write_csv(root / "reconstruction_step_metrics.csv", curves)
    _write_category_csv(root, "final_geometry_metrics.csv", aggregate, FINAL_FIELDS)
    _write_category_csv(root, "exploration_efficiency_metrics.csv", aggregate, EXPLORATION_FIELDS)
    _write_category_csv(root, "decision_policy_metrics.csv", aggregate, POLICY_FIELDS)
    _write_category_csv(root, "runtime_metrics.csv", aggregate, RUNTIME_FIELDS)

    validation = {
        "episodes": len(episode_rows),
        "gain_max_abs_error": max(row["gain_consistency_max_abs_error"] for row in episode_rows),
        "coverage_recall_max_abs_error": max(row["coverage_recall_max_abs_error"] for row in episode_rows),
        "visibility_monotonic_min_delta": min(row["visibility_monotonic_min_delta"] for row in episode_rows),
    }
    with (root / "all_metrics_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2)
    (root / "all_metrics_summary.md").write_text(_markdown(aggregate, validation, cfg), encoding="utf-8")
    print(root / "all_metrics_summary.md")


if __name__ == "__main__":
    main()
