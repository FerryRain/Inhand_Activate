#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rebuttal.benchmark.config import load_config, resolved_path


FIELDS = (
    "actions",
    "f@5_auc",
    "recall@5_auc",
    "surface_coverage_auc",
    "final_f@5",
    "final_recall@5",
    "final_chamfer_m",
    "score_gain_correlation",
    "oracle_regret",
    "oracle_action_accuracy",
    "reconstruction_metrics",
    "planner_metrics",
)


def main():
    parser = argparse.ArgumentParser(
        description="Verify that the full ablation arm reproduces the formal Ray-GPIS baseline."
    )
    parser.add_argument("--config", default="rebuttal/configs/ablation.yaml")
    parser.add_argument(
        "--reference-config", default="rebuttal/configs/formal_120_sixview_gpu.yaml"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    reference_cfg = load_config(args.reference_config)
    root = resolved_path(cfg["experiment"]["output_dir"])
    reference_root = resolved_path(reference_cfg["experiment"]["output_dir"])
    mismatches = []
    compared = 0
    for path in sorted((root / "ray_gpis").glob("*/pose_*/episode_summary.json")):
        reference = reference_root / "ray_gpis" / path.relative_to(root / "ray_gpis")
        if not reference.exists():
            mismatches.append({"episode": str(path.parent), "reason": "missing reference"})
            continue
        current_payload = json.loads(path.read_text(encoding="utf-8"))
        reference_payload = json.loads(reference.read_text(encoding="utf-8"))
        differing = [field for field in FIELDS if current_payload[field] != reference_payload[field]]
        if differing:
            mismatches.append({"episode": str(path.parent), "fields": differing})
        compared += 1
    report = {
        "episodes_compared": compared,
        "mismatches": mismatches,
        "valid": compared == 120 and not mismatches,
    }
    output = root / "full_reference_consistency.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
