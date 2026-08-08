#!/usr/bin/env python3
"""Summarize the visited-view local depth-registration-gap experiment."""
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
    "selected_recall@5_gain": ("selected_gain_recall@5", True),
    "chamfer_reduction_mm": ("selected_chamfer_reduction_mm", True),
    "score_gain_correlation": ("score_gain_spearman", True),
    "oracle_regret_f@5": ("oracle_regret_f@5", False),
    "oracle_action_accuracy": ("oracle_action_accuracy", True),
}


def _load(root: Path):
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("*/pose_*/summary.json"))
    ]


def _value(scene, planner, field):
    row = scene["planners"][planner]
    value = row[field] if field in row else row["quality"][field]
    return float(value) if value is not None else float("nan")


def _mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    mean = float(finite.mean()) if len(finite) else float("nan")
    ci = (
        float(1.96 * finite.std(ddof=1) / np.sqrt(len(finite)))
        if len(finite) > 1 else float("nan")
    )
    return mean, ci, int(len(finite))


def _bootstrap(values, seed=2026, draws=10000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(seed)
    sampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return np.percentile(sampled, [2.5, 97.5])


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _holm(rows):
    order = sorted(range(len(rows)), key=lambda index: rows[index]["p_two_sided"])
    running = 0.0
    total = len(order)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * rows[index]["p_two_sided"]))
        rows[index]["p_holm"] = float(running)
    for row in rows:
        row["supported"] = bool(
            row["mean_full_improvement"] > 0.0
            and row["ci95_low"] > 0.0
            and row["p_holm"] < 0.05
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="rebuttal/configs/visited_registration_gap.yaml"
    )
    parser.add_argument(
        "--ablation-config",
        default="rebuttal/configs/visited_registration_gap_ablation.yaml",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    ablation_cfg = load_config(args.ablation_config)
    root = resolved_path(cfg["experiment"]["output_dir"])
    ablation_root = resolved_path(ablation_cfg["experiment"]["output_dir"])
    main_scenes = _load(root)
    ablation_scenes = _load(ablation_root)
    expected = len(cfg["assets"]["objects"]) * len(cfg["experiment"]["initial_pose_seeds"])
    if len(main_scenes) != expected or len(ablation_scenes) != expected:
        raise RuntimeError(
            "Expected %d paired scenes, found main=%d ablation=%d"
            % (expected, len(main_scenes), len(ablation_scenes))
        )
    main_lookup = {
        (row["object"], int(row["initial_pose_seed"])): row for row in main_scenes
    }
    ablation_lookup = {
        (row["object"], int(row["initial_pose_seed"])): row for row in ablation_scenes
    }
    if set(main_lookup) != set(ablation_lookup):
        raise RuntimeError("Main and ablation scene keys differ")
    for key in sorted(main_lookup):
        left, right = main_lookup[key], ablation_lookup[key]
        if int(left["recovery_target_point_count"]) <= 0:
            raise RuntimeError("Empty missing-patch target for %r" % (key,))
        if len(left["action_branches"]) != 3 or len(right["action_branches"]) != 3:
            raise RuntimeError("Three shared action branches required for %r" % (key,))
        for action in left["action_branches"]:
            for field in ("gain_f@5", "gain_recall@5", "recovery_recall@5"):
                if not np.isclose(
                    float(left["action_branches"][action][field]),
                    float(right["action_branches"][action][field]),
                    atol=1e-12,
                ):
                    raise RuntimeError("Unpaired branch %s/%s/%s" % (key, action, field))
        if left["planners"]["ray_gpis"]["selected_action"] != right["planners"]["ray_gpis"]["selected_action"]:
            raise RuntimeError("Full Ray decision changed between outputs for %r" % (key,))

    combined = {}
    for key, scene in main_lookup.items():
        combined[key] = {
            "pose_novelty": scene["planners"]["pose_novelty"],
            **ablation_lookup[key]["planners"],
        }
    planner_names = [
        "pose_novelty",
        "ray_gpis_novelty_only",
        "ray_gpis_uncertainty_only",
        "ray_gpis_pointwise",
        "ray_gpis_hit_only",
        "ray_gpis",
    ]
    aggregate = []
    episode_rows = []
    for key in sorted(combined):
        target_count = int(main_lookup[key]["recovery_target_point_count"])
        for planner in planner_names:
            row = combined[key][planner]
            episode = {
                "object": key[0],
                "initial_pose_seed": key[1],
                "target_point_count": target_count,
                "planner": planner,
                "selected_action": row["selected_action"],
            }
            for metric, (field, _) in METRICS.items():
                episode[metric] = _value({"planners": {planner: row}}, planner, field)
            episode_rows.append(episode)
    for planner in planner_names:
        selected = [row for row in episode_rows if row["planner"] == planner]
        out = {"planner": planner, "paired_scenes": len(selected)}
        for metric in METRICS:
            out[metric], out[metric + "_ci95"], _ = _mean_ci(
                [row[metric] for row in selected]
            )
        weights = np.asarray([row["target_point_count"] for row in selected], dtype=np.float64)
        recoveries = np.asarray(
            [row["missing_patch_recall@5"] for row in selected], dtype=np.float64
        )
        out["point_weighted_missing_patch_recall@5"] = float(
            np.sum(weights * recoveries) / np.sum(weights)
        )
        aggregate.append(out)

    comparisons = [
        ("pose_novelty", "Pose-Novelty"),
        ("ray_gpis_novelty_only", "Novelty only"),
        ("ray_gpis_uncertainty_only", "Uncertainty only"),
        ("ray_gpis_pointwise", "w/o receptive-field integration"),
        ("ray_gpis_hit_only", "w/o miss-ray interpolation"),
    ]
    primary_rows = []
    all_tests = []
    for comparator, label in comparisons:
        for metric, (field, higher) in METRICS.items():
            differences = []
            for key in sorted(combined):
                full = _value({"planners": {"ray_gpis": combined[key]["ray_gpis"]}}, "ray_gpis", field)
                other = _value({"planners": {comparator: combined[key][comparator]}}, comparator, field)
                differences.append((full - other) if higher else (other - full))
            differences = np.asarray(differences, dtype=np.float64)
            differences = differences[np.isfinite(differences)]
            ci = _bootstrap(differences)
            nonzero = differences[np.abs(differences) > 1e-12]
            p_value = (
                float(wilcoxon(nonzero, alternative="two-sided").pvalue)
                if len(nonzero) else 1.0
            )
            row = {
                "comparator": comparator,
                "comparator_label": label,
                "metric": metric,
                "pairs": int(len(differences)),
                "mean_full_improvement": float(differences.mean()),
                "ci95_low": float(ci[0]),
                "ci95_high": float(ci[1]),
                "wins": int(np.sum(differences > 1e-12)),
                "ties": int(np.sum(np.abs(differences) <= 1e-12)),
                "losses": int(np.sum(differences < -1e-12)),
                "p_two_sided": p_value,
            }
            all_tests.append(row)
            if metric == "missing_patch_recall@5":
                primary_rows.append(row)
    _holm(primary_rows)
    primary_by_comparator = {row["comparator"]: row for row in primary_rows}
    for row in all_tests:
        if row["metric"] == "missing_patch_recall@5":
            row.update({
                "p_holm": primary_by_comparator[row["comparator"]]["p_holm"],
                "supported": primary_by_comparator[row["comparator"]]["supported"],
            })
        else:
            row.update({"p_holm": float("nan"), "supported": False})

    _write_csv(root / "visited_gap_episodes.csv", episode_rows)
    _write_csv(root / "visited_gap_aggregate.csv", aggregate)
    _write_csv(root / "visited_gap_paired_tests.csv", all_tests)
    lines = [
        "# Visited-view local registration-gap robustness",
        "",
        "A tracked prefix view loses one contiguous 70% depth region before fusion. "
        "All subsequent branches are fault-free and shared. The primary metric is "
        "one-action Recall@5 on points that remained missing after the faulty prefix.",
        "",
        "| Planner | Missing-patch Recall@5 | Point-weighted Recall@5 | F@5 gain | Regret |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| {planner} | {missing_patch_recall@5:.4f}+-{missing_patch_recall@5_ci95:.4f} | "
            "{point_weighted_missing_patch_recall@5:.4f} | "
            "{selected_f@5_gain:.4f}+-{selected_f@5_gain_ci95:.4f} | "
            "{oracle_regret_f@5:.4f}+-{oracle_regret_f@5_ci95:.4f} |".format(**row)
        )
    lines += [
        "",
        "Positive improvement means Full Ray-GPIS is better. Holm correction is "
        "applied jointly to the five missing-patch recovery comparisons.",
        "",
        "| Comparator | Full improvement [95% bootstrap CI] | W/T/L | p (Holm) | Supported |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in primary_rows:
        lines.append(
            "| {comparator_label} | {mean_full_improvement:.4f} "
            "[{ci95_low:.4f}, {ci95_high:.4f}] | {wins}/{ties}/{losses} | "
            "{p_holm:.3g} | {supported} |".format(**row)
        )
    (root / "visited_gap_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(root / "visited_gap_summary.md")


if __name__ == "__main__":
    main()
