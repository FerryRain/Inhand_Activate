#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

from benchmark.config import load_config, resolved_path


METRICS = {
    "final_recall@5": 1.0,
    "final_f@5": 1.0,
    "final_chamfer_mm": -1.0,
    "final_visibility_coverage": 1.0,
    "visibility_coverage_auc": 1.0,
    "recall@5_auc": 1.0,
    "f@5_auc": 1.0,
    "selected_gain_f@5_per_action": 1.0,
    "score_gain_correlation": 1.0,
    "oracle_regret_f@5": -1.0,
    "oracle_action_accuracy": 1.0,
    "planning_time_s": -1.0,
}


def bootstrap_ci(values, rng, samples=10000):
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return np.percentile(draws, [2.5, 97.5]).tolist()


def holm_adjust(rows):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["metric"]].append((index, row["p_two_sided"]))
    for items in groups.values():
        ordered = sorted(items, key=lambda item: item[1])
        running = 0.0
        count = len(ordered)
        for rank, (index, value) in enumerate(ordered):
            running = max(running, min(1.0, (count - rank) * value))
            rows[index]["p_holm"] = running


def main():
    parser = argparse.ArgumentParser(description="Paired statistics for all formal baseline metrics.")
    parser.add_argument("--config", default="rebuttal/configs/formal_120_sixview_gpu.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = resolved_path(cfg["experiment"]["output_dir"])
    path = root / "all_metrics_paired.csv"
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    lookup = {
        (row["planner"], row["object"], int(row["initial_pose_seed"])): row
        for row in rows
    }
    pairs = sorted((row["object"], int(row["initial_pose_seed"])) for row in rows if row["planner"] == "ray_gpis")
    baselines = sorted({row["planner"] for row in rows if row["planner"] != "ray_gpis"})
    rng = np.random.RandomState(2026)
    results = []
    for metric, direction in METRICS.items():
        for baseline in baselines:
            improvements = []
            for pair in pairs:
                ray = float(lookup[("ray_gpis",) + pair][metric])
                other = float(lookup[(baseline,) + pair][metric])
                if np.isfinite(ray) and np.isfinite(other):
                    improvements.append(direction * (ray - other))
            if not improvements:
                continue
            values = np.asarray(improvements, dtype=np.float64)
            low, high = bootstrap_ci(values, rng)
            nonzero = values[np.abs(values) > 1e-12]
            p_value = float(wilcoxon(nonzero, alternative="two-sided").pvalue) if len(nonzero) else 1.0
            results.append({
                "metric": metric,
                "baseline": baseline,
                "pairs": len(values),
                "positive_means_ray_better": True,
                "mean_improvement": float(values.mean()),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "wins": int(np.sum(values > 1e-12)),
                "ties": int(np.sum(np.abs(values) <= 1e-12)),
                "losses": int(np.sum(values < -1e-12)),
                "p_two_sided": p_value,
                "p_holm": float("nan"),
            })
    holm_adjust(results)
    output = root / "paired_statistics_all.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    lines = [
        "| Metric | Baseline | Pairs | Ray improvement (95% CI) | W/T/L | p (two-sided) | p (Holm) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            "| {metric} | {baseline} | {pairs} | {mean_improvement:.6f} [{ci95_low:.6f}, {ci95_high:.6f}] "
            "| {wins}/{ties}/{losses} | {p_two_sided:.3g} | {p_holm:.3g} |".format(**row)
        )
    (root / "paired_statistics_all.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
