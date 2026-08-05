#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmark.config import load_config, resolved_path


COLORS = {
    "fixed": "#777777",
    "pb_nbv": "#E69F00",
    "actnerf": "#56B4E9",
    "ray_gpis": "#009E73",
}


def load_summaries(root: Path):
    by_seed = {}
    for path in root.glob("*/*/pose_*/episode_summary.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("planner_seed", 0)
        key = (data["planner"], data["object"], data["initial_pose_seed"], data["planner_seed"])
        if key not in by_seed or "_seed_" in path.parent.name:
            by_seed[key] = data
    return list(by_seed.values())


def paired_curves(items, metric):
    groups = defaultdict(list)
    for item in items:
        key = (item["planner"], item["object"], item["initial_pose_seed"])
        groups[key].append([step[metric] for step in item["reconstruction_metrics"]])
    method_curves = defaultdict(list)
    for (planner, _, _), curves in groups.items():
        method_curves[planner].append(np.mean(np.asarray(curves, dtype=float), axis=0))
    return method_curves


def plot_curve(root: Path, items, metric: str, ylabel: str, filename: str):
    curves = paired_curves(items, metric)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for planner, values in sorted(curves.items()):
        values = np.asarray(values, dtype=float)
        mean = values.mean(axis=0)
        ci = 1.96 * values.std(axis=0, ddof=1) / np.sqrt(max(len(values), 1)) if len(values) > 1 else np.zeros_like(mean)
        steps = np.arange(len(mean))
        ax.plot(steps, mean, marker="o", label=planner, color=COLORS.get(planner))
        ax.fill_between(steps, mean - ci, mean + ci, alpha=0.18, color=COLORS.get(planner))
    ax.set_xlabel("Active actions")
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(6))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(root / filename, dpi=220)
    plt.close(fig)


def plot_per_object(root: Path):
    path = root / "per_object_table.csv"
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    objects = sorted({row["object"] for row in rows})
    planners = sorted({row["planner"] for row in rows})
    x = np.arange(len(objects))
    width = 0.8 / max(len(planners), 1)
    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    for index, planner in enumerate(planners):
        lookup = {row["object"]: float(row["f@5_auc"]) for row in rows if row["planner"] == planner}
        values = [lookup.get(object_name, np.nan) for object_name in objects]
        ax.bar(x + (index - (len(planners) - 1) / 2.0) * width, values, width, label=planner, color=COLORS.get(planner))
    ax.set_xticks(x)
    ax.set_xticklabels(objects, rotation=25, ha="right")
    ax.set_ylabel("F@5 AUC")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=len(planners))
    fig.tight_layout()
    fig.savefig(root / "per_object_f5_auc.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rebuttal/configs/formal_120_sixview_gpu.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = resolved_path(cfg["experiment"]["output_dir"])
    items = load_summaries(root)
    if not items:
        raise RuntimeError("No summaries under %s" % root)
    plot_curve(root, items, "f@5", "F@5", "f5_curve.png")
    plot_curve(root, items, "recall@5", "Recall@5", "recall_curve.png")
    if (root / "per_object_table.csv").exists():
        plot_per_object(root)
    print(str(root))


if __name__ == "__main__":
    main()
