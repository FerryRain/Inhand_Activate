#!/usr/bin/env python3
"""Batch SPAR3D inference with stable object/pose output names.

The upstream CLI accesses remeshing-only arguments even when optional remesh
packages are absent.  This wrapper calls the released model API directly and
does not alter SPAR3D.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image


SPAR_ROOT = Path("/home/ferry/data/Code2/Research/TRELLIS/SPAR3D")
sys.path.insert(0, str(SPAR_ROOT))
from spar3d.system import SPAR3D  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="rebuttal/assets/ycb/single_view_inputs")
    parser.add_argument(
        "--output", default="rebuttal/results/downstream_single_view_raw/spar3d"
    )
    parser.add_argument(
        "--model", default="stabilityai/stable-point-aware-3d"
    )
    parser.add_argument("--texture-resolution", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SPAR3D downstream inference requires CUDA")
    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    model = SPAR3D.from_pretrained(
        args.model,
        config_name="config.yaml",
        weight_name="model.safetensors",
        low_vram_mode=False,
    ).cuda().eval()
    inputs = sorted(input_root.glob("*/pose_*.png"))
    if not inputs:
        raise RuntimeError(f"No rendered inputs under {input_root}")
    for index, image_path in enumerate(inputs):
        object_name = image_path.parent.name
        output = output_root / object_name / f"{image_path.stem}.glb"
        if output.exists() and not args.overwrite:
            continue
        image = Image.open(image_path).convert("RGBA")
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            mesh, _ = model.run_image(
                [image],
                bake_resolution=int(args.texture_resolution),
                remesh="none",
                vertex_count=-1,
                return_points=True,
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(output), include_normals=True)
        mesh.export(str(output.with_suffix(".ply")))
        print(f"[{index + 1}/{len(inputs)}] {output}", flush=True)


if __name__ == "__main__":
    main()
