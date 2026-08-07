#!/usr/bin/env python3
"""Generate a compact Markdown/LaTeX report for downstream results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ORDER = (
    "single_view_depth",
    "spar3d",
    "trellis2",
    "fixed",
    "pb_nbv",
    "actnerf",
    "pose_novelty",
    "ray_gpis",
)
LABELS = {
    "single_view_depth": "Single RGB-D",
    "spar3d": "SPAR3D (oracle-align)",
    "trellis2": "TRELLIS.2 (oracle-align)",
    "fixed": "Fixed schedule",
    "pb_nbv": "Adapted PB-NBV",
    "actnerf": "Adapted ActNeRF",
    "pose_novelty": "Pose-Novelty",
    "ray_gpis": "Full Ray-GPIS",
}


def percent(value):
    return 100.0 * float(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", default="rebuttal/results/downstream_ycb_task")
    parser.add_argument("--real-root", default="rebuttal/results/downstream_real_task")
    parser.add_argument(
        "--reconstruction-root",
        default="rebuttal/results/downstream_ycb_reconstruction",
    )
    parser.add_argument(
        "--output", default="rebuttal/results/downstream_ycb_task/RESULTS.md"
    )
    args = parser.parse_args()
    task_root = ROOT / args.task_root
    summary = {
        row["method"]: row
        for row in json.loads((task_root / "summary.json").read_text())
    }
    lines = [
        "# Sequential place-and-regrasp results",
        "",
        "Ten YCB objects, three paired reconstructions/object, ten execution "
        "perturbations/reconstruction (30 trials/object/method).",
        "",
        "| Reconstruction source | Placement (%) | Stage-2 regrasp (%) | Sequential (%) |",
        "|---|---:|---:|---:|",
    ]
    tex = []
    for method in ORDER:
        if method not in summary:
            continue
        row = summary[method]
        values = (
            percent(row["placement_success"]),
            percent(row["conditional_regrasp_success"]),
            percent(row["sequential_success"]),
        )
        lines.append(
            f"| {LABELS[method]} | {values[0]:.1f} | {values[1]:.1f} | {values[2]:.1f} |"
        )
        tex.append(
            f"{LABELS[method]} & {values[0]:.1f} & {values[1]:.1f} & {values[2]:.1f} \\\\"
        )

    reconstruction_root = ROOT / args.reconstruction_root
    acquisition_rows = defaultdict(list)
    for path in reconstruction_root.glob("*/*/pose_*/episode_summary.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        acquisition_rows[row["planner"]].append(row)
    acquisition_summary = []
    for method in ORDER:
        method_rows = acquisition_rows.get(method, [])
        if not method_rows:
            continue
        item = {"method": method, "episodes": len(method_rows)}
        for field in ("f@5_auc", "final_f@5", "planning_time_s", "accepted_frames"):
            values = np.asarray([row[field] for row in method_rows], dtype=np.float64)
            item[field] = float(values.mean())
            item[field + "_ci95"] = float(
                1.96 * values.std(ddof=1) / np.sqrt(len(values))
            ) if len(values) > 1 else float("nan")
        acquisition_summary.append(item)
    if acquisition_summary:
        lines += [
            "",
            "## Realistic continuous-acquisition reconstruction",
            "",
            "Thirty paired episodes/method (10 YCB objects x 3 initial poses); "
            "five 6 s primitives at 15 FPS with dynamic occlusion and manipulation errors.",
            "",
            "| Planner | F@5 AUC | Final F@5 | Accepted frames | Planning (s/step) |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in acquisition_summary:
            lines.append(
                f"| {LABELS[row['method']]} | {row['f@5_auc']:.4f} | "
                f"{row['final_f@5']:.4f} | {row['accepted_frames']:.1f} | "
                f"{row['planning_time_s']:.3f} |"
            )
        (task_root / "reconstruction_summary.json").write_text(
            json.dumps(acquisition_summary, indent=2), encoding="utf-8"
        )
    comparison_path = task_root / "paired_comparisons.json"
    if comparison_path.exists():
        comparisons = json.loads(comparison_path.read_text())
        lines += ["", "## Paired object-level comparisons", ""]
        for row in comparisons:
            if row["comparator"] not in ("single_view_depth", "fixed"):
                continue
            lines.append(
                f"- Ray-GPIS vs {LABELS[row['comparator']]} on "
                f"`{row['metric']}`: {percent(row['difference']):+.1f} pp "
                f"(object-level Wilcoxon p={row['wilcoxon_p']:.4f})."
            )
    real_path = ROOT / args.real_root / "summary.json"
    if real_path.exists():
        real = json.loads(real_path.read_text())[0]
        lines += [
            "",
            "## Existing real-mesh transfer subset",
            "",
            f"Six meshes x 30 perturbations: placement {percent(real['placement_success']):.1f}%, "
            f"stage-2 regrasp {percent(real['conditional_regrasp_success']):.1f}%, "
            f"sequential {percent(real['sequential_success']):.1f}%.",
        ]
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output.parent / "table_rows.tex").write_text(
        "\n".join(tex) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
