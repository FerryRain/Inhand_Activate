#!/usr/bin/env python3
from __future__ import annotations

import argparse

from rebuttal.benchmark.config import load_config
from rebuttal.benchmark.runner import run_suite


def _csv(value, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run the paired unified NBV benchmark.")
    parser.add_argument("--config", default="rebuttal/configs/formal_120_sixview_gpu.yaml")
    parser.add_argument("--planners", default="")
    parser.add_argument("--objects", default="")
    parser.add_argument("--pose-seeds", default="")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if bool(cfg.get("runtime", {}).get("require_cuda", False)):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("This benchmark config requires CUDA, but torch.cuda.is_available() is False")
        print("[runtime] CUDA device: %s" % torch.cuda.get_device_name(0), flush=True)
    run_suite(
        cfg,
        planners=_csv(args.planners) if args.planners else None,
        objects=_csv(args.objects) if args.objects else None,
        pose_seeds=_csv(args.pose_seeds, int) if args.pose_seeds else None,
        action_steps=args.steps,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
