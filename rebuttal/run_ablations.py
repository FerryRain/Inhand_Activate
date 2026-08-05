#!/usr/bin/env python3
from __future__ import annotations

import argparse

from benchmark.config import load_config
from benchmark.runner import run_suite


def main():
    parser = argparse.ArgumentParser(description="Run clean Ray-GPIS component ablations.")
    parser.add_argument("--config", default="rebuttal/configs/ablation.yaml")
    parser.add_argument("--objects", default="")
    parser.add_argument("--pose-seeds", default="")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    objects = [item.strip() for item in args.objects.split(",") if item.strip()] or None
    pose_seeds = [int(item) for item in args.pose_seeds.split(",") if item.strip()] or None
    run_suite(
        cfg,
        objects=objects,
        pose_seeds=pose_seeds,
        action_steps=args.steps,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
