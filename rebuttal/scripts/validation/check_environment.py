#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import platform


def main():
    packages = [
        "numpy",
        "scipy",
        "sklearn",
        "cv2",
        "open3d",
        "torch",
        "gpytorch",
        "trimesh",
        "skimage",
        "yaml",
    ]
    missing = []
    print("Python:", platform.python_version())
    for package in packages:
        available = importlib.util.find_spec(package) is not None
        print("%-12s %s" % (package, "OK" if available else "MISSING"))
        if not available:
            missing.append(package)
    import torch

    print("CUDA:", torch.cuda.is_available())
    print("Isaac Sim:", importlib.util.find_spec("isaacsim") is not None)
    print("Isaac Gym:", importlib.util.find_spec("isaacgym") is not None)
    if missing:
        raise SystemExit("Missing required packages: %s" % ", ".join(missing))


if __name__ == "__main__":
    main()
