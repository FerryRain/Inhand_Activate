#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter

from rebuttal.benchmark.config import load_config, resolved_path


def main():
    parser = argparse.ArgumentParser(description="Audit recorded planner devices in the formal GPU sweep.")
    parser.add_argument("--config", default="rebuttal/configs/formal_120_sixview_gpu.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = resolved_path(cfg["experiment"]["output_dir"])
    report = {}
    errors = []
    for method in cfg["planners"]["enabled"]:
        summaries = list((root / method).glob("*/pose_*/episode_summary.json"))
        runtime_rows = []
        invalid_surface_steps = 0
        for path in summaries:
            payload = json.loads(path.read_text(encoding="utf-8"))
            runtime_rows.extend(payload["runtime_metrics"])
            invalid_surface_steps += sum(not row.get("valid_surface", True) for row in payload["runtime_metrics"])
        devices = Counter(row.get("device", "native_cpu") for row in runtime_rows)
        report[method] = {
            "episodes": len(summaries),
            "planning_steps": len(runtime_rows),
            "devices": dict(devices),
            "invalid_surface_steps": invalid_surface_steps,
        }
        requires_cuda = method in ("actnerf", "er_gpis") or method.startswith("ray_gpis")
        if requires_cuda and set(devices) != {"cuda"}:
            errors.append("%s did not exclusively use CUDA: %s" % (method, dict(devices)))
    output = {
        "methods": report,
        "errors": errors,
        "valid": not errors,
    }
    path = root / "gpu_runtime_validation.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
