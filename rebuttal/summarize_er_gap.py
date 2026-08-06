#!/usr/bin/env python3
"""Summarize ER-GPIS versus Ray-GPIS on the visited-view depth gap."""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np
from scipy.stats import wilcoxon

from benchmark.config import load_config, resolved_path


FIELDS = {
    "missing_patch_recall@5": ("selected_recovery_recall@5", True),
    "selected_f@5_gain": ("selected_gain_f@5", True),
    "selected_recall@5_gain": ("selected_gain_recall@5", True),
    "chamfer_reduction_mm": ("selected_chamfer_reduction_mm", True),
    "score_gain_correlation": ("score_gain_spearman", True),
    "oracle_regret_f@5": ("oracle_regret_f@5", False),
    "oracle_action_accuracy": ("oracle_action_accuracy", True),
}


def _value(scene, planner, field):
    row = scene["planners"][planner]
    value = row[field] if field in row else row["quality"].get(field)
    return float(value) if value is not None else float("nan")


def _bootstrap(values, seed=2026, draws=10000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(seed)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def _mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    mean = float(values.mean()) if len(values) else float("nan")
    ci = (
        float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))
        if len(values) > 1 else float("nan")
    )
    return mean, ci


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="rebuttal/configs/visited_registration_gap_er_ray.yaml"
    )
    parser.add_argument(
        "--reference-config", default="rebuttal/configs/visited_registration_gap.yaml"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = resolved_path(cfg["experiment"]["output_dir"])
    scenes = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("*/pose_*/summary.json"))
    ]
    expected = len(cfg["assets"]["objects"]) * len(cfg["experiment"]["initial_pose_seeds"])
    if len(scenes) != expected:
        raise RuntimeError("Expected %d scenes, found %d" % (expected, len(scenes)))
    reference_cfg = load_config(args.reference_config)
    reference_root = resolved_path(reference_cfg["experiment"]["output_dir"])
    reference = {
        (row["object"], int(row["initial_pose_seed"])): row
        for row in (
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(reference_root.glob("*/pose_*/summary.json"))
        )
    }
    for scene in scenes:
        key = (scene["object"], int(scene["initial_pose_seed"]))
        if int(scene["recovery_target_point_count"]) <= 0:
            raise RuntimeError("Empty missing-patch target for %r" % (key,))
        if set(scene["planners"]) != {"er_gpis", "ray_gpis"}:
            raise RuntimeError("Unexpected planners for %r" % (key,))
        if len(scene["action_branches"]) != 3:
            raise RuntimeError("Expected three action branches for %r" % (key,))
        if key not in reference:
            raise RuntimeError("Missing reference scene %r" % (key,))
        for action, branch in scene["action_branches"].items():
            for field in ("gain_f@5", "gain_recall@5", "recovery_recall@5"):
                if not np.isclose(
                    float(branch[field]),
                    float(reference[key]["action_branches"][action][field]),
                    atol=1e-12,
                ):
                    raise RuntimeError("Unpaired reference branch %r/%s/%s" % (key, action, field))

    aggregate = []
    for planner in ("er_gpis", "ray_gpis"):
        row = {"planner": planner, "paired_scenes": len(scenes)}
        for metric, (field, _) in FIELDS.items():
            row[metric], row[metric + "_ci95"] = _mean_ci(
                [_value(scene, planner, field) for scene in scenes]
            )
        aggregate.append(row)

    paired = []
    for metric, (field, higher) in FIELDS.items():
        differences = []
        for scene in scenes:
            ray = _value(scene, "ray_gpis", field)
            er = _value(scene, "er_gpis", field)
            delta = (ray - er) if higher else (er - ray)
            if np.isfinite(delta):
                differences.append(float(delta))
        differences = np.asarray(differences, dtype=np.float64)
        ci = _bootstrap(differences)
        nonzero = differences[np.abs(differences) > 1e-12]
        p_value = (
            float(wilcoxon(nonzero, alternative="two-sided").pvalue)
            if len(nonzero) else 1.0
        )
        paired.append({
            "metric": metric,
            "pairs": int(len(differences)),
            "ray_improvement": float(differences.mean()),
            "ci95_low": float(ci[0]),
            "ci95_high": float(ci[1]),
            "wins": int(np.sum(differences > 1e-12)),
            "ties": int(np.sum(np.abs(differences) <= 1e-12)),
            "losses": int(np.sum(differences < -1e-12)),
            "p_two_sided": p_value,
            "supported_ray_better": bool(
                differences.mean() > 0.0 and ci[0] > 0.0 and p_value < 0.05
            ),
        })
    _write_csv(root / "er_ray_gap_aggregate.csv", aggregate)
    _write_csv(root / "er_ray_gap_paired.csv", paired)
    primary = next(row for row in paired if row["metric"] == "missing_patch_recall@5")
    lookup = {row["planner"]: row for row in aggregate}
    lines = [
        "# ER-GPIS versus Ray-GPIS: visited-view registration gap",
        "",
        "The primary metric is one-action Recall@5 on the missing prefix patch. "
        "Positive paired improvement means Ray-GPIS is better.",
        "",
        "| Planner | Missing-patch Recall@5 | F@5 gain | Corr. | Regret | Time (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for planner in ("er_gpis", "ray_gpis"):
        row = lookup[planner]
        lines.append(
            "| {planner} | {missing_patch_recall@5:.4f}+-{missing_patch_recall@5_ci95:.4f} | "
            "{selected_f@5_gain:.4f}+-{selected_f@5_gain_ci95:.4f} | "
            "{score_gain_correlation:.4f}+-{score_gain_correlation_ci95:.4f} | "
            "{oracle_regret_f@5:.4f}+-{oracle_regret_f@5_ci95:.4f} | -- |".format(**row)
        )
    lines += [
        "",
        "Primary Ray improvement: {ray_improvement:.5f} "
        "[{ci95_low:.5f}, {ci95_high:.5f}], W/T/L "
        "{wins}/{ties}/{losses}, p={p_two_sided:.3g}, supported={supported_ray_better}.".format(**primary),
        "",
        "Conclusion: this condition does not support a Ray-over-ER claim; the "
        "ER comparison is therefore omitted from the rebuttal.",
    ]
    (root / "er_ray_gap_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(root / "er_ray_gap_summary.md")


if __name__ == "__main__":
    main()
