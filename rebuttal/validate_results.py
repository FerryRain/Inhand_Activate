#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmark.config import load_config, resolved_path
from benchmark.geometry import ACTIONS
from benchmark.io import write_json


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Audit completeness and pairing of baseline outputs.")
    parser.add_argument("--config", default="rebuttal/configs/formal_120_sixview_gpu.yaml")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = resolved_path(cfg["experiment"]["output_dir"])
    methods = list(cfg["planners"]["enabled"])
    objects = list(cfg["assets"]["objects"])
    poses = [int(value) for value in cfg["experiment"]["initial_pose_seeds"]]
    total_steps = int(cfg["environment"]["action_steps"])
    errors = []
    episode_count = 0
    reference_hashes = {}
    for method in methods:
        seeds = cfg["experiment"].get("actnerf_seeds", [0, 1, 2]) if method == "actnerf" else [0]
        for object_name in objects:
            for pose in poses:
                for planner_seed in seeds:
                    suffix = "pose_%03d" % pose
                    if method == "actnerf":
                        suffix += "_seed_%03d" % int(planner_seed)
                    episode = root / method / object_name / suffix
                    summary_path = episode / "episode_summary.json"
                    if not summary_path.exists():
                        errors.append("missing summary: %s" % summary_path)
                        continue
                    episode_count += 1
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    if len(summary.get("actions", [])) != total_steps:
                        errors.append("wrong action count: %s" % episode)
                    initial_step = episode / "step_000"
                    key = (object_name, pose)
                    hashes = tuple(digest(initial_step / name) for name in ("depth.npy", "mask.png", "gt_pose.npy", "candidate_directions.npy"))
                    if key not in reference_hashes:
                        reference_hashes[key] = hashes
                    elif reference_hashes[key] != hashes:
                        errors.append("unpaired initial observation: %s" % episode)
                    for step in range(total_steps + 1):
                        step_dir = episode / ("step_%03d" % step)
                        required = [
                            "depth.npy", "gt_pose.npy", "executed_pose.npy", "fused_cloud.ply",
                            "candidate_directions.npy", "planner_scores.npy", "selected_nbv.npy",
                            "selected_action.json", "metrics.json",
                        ]
                        for name in required:
                            if not (step_dir / name).exists():
                                errors.append("missing %s" % (step_dir / name))
                        if step == total_steps:
                            continue
                        action = json.loads((step_dir / "selected_action.json").read_text(encoding="utf-8")).get("action")
                        if action not in ACTIONS:
                            errors.append("invalid action in %s" % step_dir)
                        for counterfactual_action in ACTIONS:
                            path = step_dir / "counterfactual" / (counterfactual_action + "_metrics.json")
                            if not path.exists():
                                errors.append("missing counterfactual: %s" % path)
                        scores = np.load(str(step_dir / "planner_scores.npy"))
                        if len(scores) != int(cfg["candidates"]["count"]):
                            errors.append("wrong score count: %s" % step_dir)
                        if method != "fixed" and not np.any(np.isfinite(scores)):
                            errors.append("no finite scores: %s" % step_dir)
    runs_per_scene = sum(
        len(cfg["experiment"].get("actnerf_seeds", [0, 1, 2])) if method == "actnerf" else 1
        for method in methods
    )
    expected = len(objects) * len(poses) * runs_per_scene
    report = {
        "root": str(root),
        "expected_episodes": expected,
        "complete_episodes": episode_count,
        "paired_initial_states": len(reference_hashes),
        "errors": errors,
        "valid": not errors and episode_count == expected,
    }
    write_json(root / "validation_report.json", report)
    print(json.dumps(report, indent=2))
    if not report["valid"] and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
