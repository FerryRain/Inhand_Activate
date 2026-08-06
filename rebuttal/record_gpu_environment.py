#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import gpytorch
import open3d
import torch

from benchmark.config import load_config, resolved_path


def main():
    parser = argparse.ArgumentParser(description="Record the runtime stack used for GPU timing.")
    parser.add_argument("--config", default="rebuttal/configs/formal_120_sixview_gpu.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; refusing to write a GPU runtime manifest")
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    fields = [field.strip() for field in query.split(",")]
    payload = {
        "conda_environment": "robosyn_gpu",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "gpytorch": gpytorch.__version__,
        "open3d": open3d.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": fields[0],
        "driver": fields[1],
        "memory_mib": int(fields[2]),
        "compute_capability": fields[3],
        "timing_protocol": "single-process wall clock with torch.cuda.synchronize at phase boundaries",
        "native_devices": {
            method: (
                "cuda"
                if method in ("actnerf", "er_gpis") or method.startswith("ray_gpis")
                else "cpu"
            )
            for method in cfg["planners"]["enabled"]
        },
    }
    root = resolved_path(cfg["experiment"]["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    output = root / "gpu_environment.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
