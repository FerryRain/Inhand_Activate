#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.assets import prepare_assets
from benchmark.config import load_config, resolved_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rebuttal/configs/base.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    asset_cfg = cfg["assets"]
    output_dir = resolved_path(asset_cfg["output_dir"])
    manifest = prepare_assets(
        output_dir,
        target_diagonal=float(asset_cfg["target_diagonal_m"]),
        poisson_depth=int(asset_cfg["poisson_depth"]),
        force=bool(args.force),
    )
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(str(manifest_path))


if __name__ == "__main__":
    main()
