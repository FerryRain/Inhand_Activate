#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from rebuttal.benchmark.config import load_config
from rebuttal.benchmark.runner import run_suite


def main():
    parser = argparse.ArgumentParser(description="Run Level-B keyframe filtering robustness experiments.")
    parser.add_argument("--config", default="rebuttal/configs/robustness.yaml")
    parser.add_argument("--conditions", default="")
    parser.add_argument("--filters", default="")
    parser.add_argument("--objects", default="")
    parser.add_argument("--pose-seeds", default="")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    base = load_config(args.config)
    conditions = [item for item in args.conditions.split(",") if item] or list(base["robustness_conditions"])
    filters = [item for item in args.filters.split(",") if item] or list(base["filter_modes"])
    objects = [item for item in args.objects.split(",") if item] or None
    pose_seeds = [int(item) for item in args.pose_seeds.split(",") if item] or None
    root = Path(base["experiment"]["output_dir"])
    for condition in conditions:
        for filter_mode in filters:
            cfg = copy.deepcopy(base)
            cfg["environment"]["robustness"] = copy.deepcopy(base["robustness_conditions"][condition])
            cfg["fusion"]["filter_mode"] = filter_mode
            cfg["experiment"]["output_dir"] = str(root / condition / filter_mode)
            run_suite(
                cfg,
                objects=objects,
                pose_seeds=pose_seeds,
                action_steps=args.steps,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
