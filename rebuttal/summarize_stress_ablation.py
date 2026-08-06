#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from benchmark.config import load_config, resolved_path
from benchmark.evaluation import trapezoid_auc


SCENARIOS = {
    "sparse": {
        "config": "rebuttal/configs/stress_sparse.yaml",
        "ablation": "ray_gpis_novelty_only",
        "primary": "target_recall@5_auc",
        "higher": True,
    },
    "ghost": {
        "config": "rebuttal/configs/stress_ghost.yaml",
        "ablation": "ray_gpis_uncertainty_only",
        "primary": "post_outlier_oracle_regret",
        "higher": False,
    },
    "hole": {
        "config": "rebuttal/configs/stress_hole.yaml",
        "ablation": "ray_gpis_hit_only",
        "primary": "target_recall@5_auc",
        "higher": True,
    },
}


def _mean(values):
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.nanmean(array)) if np.any(np.isfinite(array)) else float("nan")


def _number(value):
    return float(value) if value is not None else float("nan")


def _load(root: Path):
    rows = []
    for path in sorted(root.glob("*/*/pose_*/episode_summary.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        item["_episode_dir"] = str(path.parent)
        rows.append(item)
    return rows


def _derive(item, scenario):
    reconstruction = item["reconstruction_metrics"]
    planner = item["planner_metrics"]
    runtime = item["runtime_metrics"]
    post_planner = planner[1:]
    post_reconstruction = reconstruction[1:]
    target_curve = [
        _number(row.get("target_recall@5", float("nan"))) for row in reconstruction
    ]
    finite_target = np.asarray(target_curve, dtype=np.float64)
    reached = np.flatnonzero(finite_target >= 0.25)
    fault_events = [
        event
        for event in item.get("fault_events", [])
        if "scheduled_pose_outlier" in event.get("fault_tags", [])
    ]
    episode_dir = Path(item["_episode_dir"])
    decisions = json.loads(
        (episode_dir / "keyframe_decisions.json").read_text(encoding="utf-8")
    )["decisions"]
    gain_errors = [
        abs(
            float(planner[index]["selected_gain_f@5"])
            - (float(reconstruction[index + 1]["f@5"]) - float(reconstruction[index]["f@5"]))
        )
        for index in range(len(planner))
    ]
    return {
        "scenario": scenario,
        "planner": item["planner"],
        "object": item["object"],
        "initial_pose_seed": int(item["initial_pose_seed"]),
        "f@5_auc": float(item["f@5_auc"]),
        "recall@5_auc": float(item["recall@5_auc"]),
        "target_recall@5_auc": _number(item.get("target_recall@5_auc", float("nan"))),
        "final_target_recall@5": _number(item.get("final_target_recall@5", float("nan"))),
        "target_surface_point_count": int(item.get("target_surface_point_count", 0)),
        "target_first_25_step": int(reached[0]) if len(reached) else 6,
        "post_outlier_oracle_regret": _mean(
            row["oracle_regret_f@5"] for row in post_planner
        ),
        "post_outlier_f@5_auc": trapezoid_auc(
            [float(row["f@5"]) for row in post_reconstruction]
        ),
        "post_outlier_unique_gain": _mean(
            row.get("selected_gain_surface_coverage", float("nan"))
            for row in post_planner
        ),
        "ghost_ratio": _mean(event.get("ghost_ratio", float("nan")) for event in fault_events),
        "miss_direction_selection_rate": _mean(
            not bool(row.get("selected_direction_hit", True)) for row in runtime
        ),
        "planning_time_s": _mean(row["planning_s"] for row in runtime),
        "mean_visibility_ratio": _mean(row["visibility_ratio"] for row in decisions),
        "mean_pose_error_deg": _mean(row["pose_error_deg"] for row in decisions),
        "mean_translation_error_mm": 1000.0
        * _mean(row.get("translation_error_m", 0.0) for row in decisions),
        "cuda_steps": int(sum(row.get("device") == "cuda" for row in runtime)),
        "states": int(len(reconstruction)),
        "actions": int(len(planner)),
        "gain_consistency_error": float(max(gain_errors, default=0.0)),
        "scheduled_outlier_events": int(len(fault_events)),
    }


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap(values, seed=2026, draws=10000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(seed)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def _holm(rows):
    ordered = sorted(range(len(rows)), key=lambda index: float(rows[index]["p_two_sided"]))
    running = 0.0
    total = len(ordered)
    for rank, index in enumerate(ordered):
        running = max(
            running,
            min(1.0, (total - rank) * float(rows[index]["p_two_sided"])),
        )
        rows[index]["p_holm"] = float(running)
    for row in rows:
        row["supported"] = bool(
            float(row["mean_improvement"]) > 0.0
            and float(row["ci95_low"]) > 0.0
            and float(row["p_holm"]) < 0.05
        )


def summarize_scenario(name, spec):
    cfg = load_config(spec["config"])
    root = resolved_path(cfg["experiment"]["output_dir"])
    enabled = set(cfg["planners"]["enabled"])
    items = [item for item in _load(root) if item["planner"] in enabled]
    if not items:
        raise RuntimeError("No summaries under %s" % root)
    rows = [_derive(item, name) for item in items]
    expected_pairs = len(cfg["assets"]["objects"]) * len(cfg["experiment"]["initial_pose_seeds"])
    lookup = {
        (row["planner"], row["object"], row["initial_pose_seed"]): row for row in rows
    }
    full_pairs = sorted(
        (row["object"], row["initial_pose_seed"])
        for row in rows
        if row["planner"] == "ray_gpis"
    )
    if len(full_pairs) != expected_pairs:
        raise RuntimeError("%s: expected %d Full pairs, found %d" % (name, expected_pairs, len(full_pairs)))
    if any(row["states"] != 6 or row["actions"] != 5 for row in rows):
        raise RuntimeError("%s: strict six-state budget violated" % name)
    if any(row["cuda_steps"] != 5 for row in rows):
        raise RuntimeError("%s: non-CUDA planner step detected" % name)
    if max(row["gain_consistency_error"] for row in rows) > 1e-9:
        raise RuntimeError("%s: selected gain consistency failed" % name)
    if name in ("sparse", "hole") and any(
        row["target_surface_point_count"] <= 0 for row in rows
    ):
        raise RuntimeError("%s: empty target surface" % name)
    if name == "ghost" and any(row["scheduled_outlier_events"] != 1 for row in rows):
        raise RuntimeError("ghost: expected exactly one scheduled outlier per episode")

    metric = spec["primary"]
    baseline = spec["ablation"]
    differences = []
    for pair in full_pairs:
        full = float(lookup[("ray_gpis",) + pair][metric])
        other = float(lookup[(baseline,) + pair][metric])
        differences.append((full - other) if spec["higher"] else (other - full))
    differences = np.asarray(differences, dtype=np.float64)
    ci = _bootstrap(differences)
    nonzero = differences[np.abs(differences) > 1e-12]
    p_value = float(wilcoxon(nonzero, alternative="two-sided").pvalue) if len(nonzero) else 1.0
    paired = {
        "scenario": name,
        "metric": metric,
        "full": _mean(lookup[("ray_gpis",) + pair][metric] for pair in full_pairs),
        "ablation": baseline,
        "ablated": _mean(lookup[(baseline,) + pair][metric] for pair in full_pairs),
        "positive_means_full_better": True,
        "mean_improvement": float(differences.mean()),
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
        "p_two_sided": p_value,
        "pairs": int(len(differences)),
        "wins": int(np.sum(differences > 1e-12)),
        "ties": int(np.sum(np.abs(differences) <= 1e-12)),
        "losses": int(np.sum(differences < -1e-12)),
    }
    aggregate = []
    numeric = [
        "f@5_auc",
        "recall@5_auc",
        "target_recall@5_auc",
        "final_target_recall@5",
        "target_first_25_step",
        "post_outlier_oracle_regret",
        "post_outlier_f@5_auc",
        "post_outlier_unique_gain",
        "ghost_ratio",
        "miss_direction_selection_rate",
        "planning_time_s",
        "mean_visibility_ratio",
        "mean_pose_error_deg",
        "mean_translation_error_mm",
    ]
    for planner_name in cfg["planners"]["enabled"]:
        selected = [row for row in rows if row["planner"] == planner_name]
        out = {"scenario": name, "planner": planner_name, "paired_scenes": len(selected)}
        for field in numeric:
            values = np.asarray([row[field] for row in selected], dtype=np.float64)
            finite = values[np.isfinite(values)]
            out[field] = float(finite.mean()) if len(finite) else float("nan")
            out[field + "_ci95"] = (
                float(1.96 * finite.std(ddof=1) / np.sqrt(len(finite)))
                if len(finite) > 1
                else float("nan")
            )
        aggregate.append(out)
    _write_csv(root / "stress_metrics_episodes.csv", rows)
    _write_csv(root / "stress_metrics_aggregate.csv", aggregate)
    _write_csv(root / "stress_primary_paired.csv", [paired])
    return paired, aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="sparse,ghost,hole")
    parser.add_argument(
        "--output", default="rebuttal/results/stress_ablation_summary"
    )
    args = parser.parse_args()
    requested = [value.strip() for value in args.scenarios.split(",") if value.strip()]
    paired_rows = []
    aggregate_rows = []
    for name in requested:
        paired, aggregate = summarize_scenario(name, SCENARIOS[name])
        paired_rows.append(paired)
        aggregate_rows.extend(aggregate)
    output = resolved_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "active_stress_primary_paired.csv", paired_rows)
    _write_csv(output / "active_stress_aggregate.csv", aggregate_rows)
    lines = [
        "# Targeted active stress ablations",
        "",
        "Positive paired improvements mean Full is better. Values are pre-Holm; pose stability is added before final correction.",
        "",
        "| Stress | Primary metric | Full | Ablated | Improvement (95% CI) | p | W/T/L |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in paired_rows:
        lines.append(
            "| {scenario} | {metric} | {full:.6f} | {ablated:.6f} | "
            "{mean_improvement:.6f} [{ci95_low:.6f}, {ci95_high:.6f}] | "
            "{p_two_sided:.3g} | {wins}/{ties}/{losses} |".format(**row)
        )
    (output / "active_stress_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    pose_path = resolved_path(
        "rebuttal/results/stress_pose_stability/pose_stability_primary_paired.csv"
    )
    final_rows = list(paired_rows)
    if pose_path.exists():
        with pose_path.open("r", encoding="utf-8") as handle:
            final_rows.extend(csv.DictReader(handle))
    _holm(final_rows)
    _write_csv(output / "stress_primary_all.csv", final_rows)
    final_lines = [
        "# Pre-registered special-case stress ablation",
        "",
        "Positive improvements mean Full is better. Holm correction is applied jointly to the four pre-registered primary hypotheses.",
        "",
        "| Stress | Primary metric | Full | Ablated | Improvement (95% CI) | p (Holm) | Supported |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in final_rows:
        final_lines.append(
            "| {scenario} | {metric} | {full:.6f} | {ablated:.6f} | "
            "{mean_improvement:.6f} [{ci95_low:.6f}, {ci95_high:.6f}] | "
            "{p_holm:.3g} | {supported} |".format(
                **{
                    **row,
                    "full": float(row["full"]),
                    "ablated": float(row["ablated"]),
                    "mean_improvement": float(row["mean_improvement"]),
                    "ci95_low": float(row["ci95_low"]),
                    "ci95_high": float(row["ci95_high"]),
                    "p_holm": float(row["p_holm"]),
                }
            )
        )
    (output / "stress_summary.md").write_text(
        "\n".join(final_lines) + "\n", encoding="utf-8"
    )
    print(output / "stress_summary.md")


if __name__ == "__main__":
    main()
