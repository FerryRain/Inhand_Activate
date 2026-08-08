#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rebuttal.benchmark.config import load_config, resolved_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rebuttal/configs/robustness.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = resolved_path(cfg["experiment"]["output_dir"])
    rows = []
    for condition in cfg["robustness_conditions"]:
        for filter_mode in cfg["filter_modes"]:
            summaries = []
            for path in (root / condition / filter_mode).glob("*/*/pose_*/episode_summary.json"):
                summaries.append(json.loads(path.read_text(encoding="utf-8")))
            if not summaries:
                continue
            final = [item["reconstruction_metrics"][-1] for item in summaries]
            rows.append({
                "condition": condition,
                "filter": filter_mode,
                "episodes": len(summaries),
                "f@5": float(np.mean([item["final_f@5"] for item in summaries])),
                "chamfer_m": float(np.mean([item["final_chamfer_m"] for item in summaries])),
                "surface_thickness_m": float(np.mean([item["surface_thickness_m"] for item in final])),
                "outlier_ratio": float(np.mean([item["outlier_ratio"] for item in final])),
                "action_switching_frequency": float(np.mean([item["action_switching_frequency"] for item in summaries])),
                "accepted_frames": float(np.mean([item["accepted_frames"] for item in summaries])),
            })
    output = root / "robustness_table.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["condition", "filter"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(str(output))


if __name__ == "__main__":
    main()
