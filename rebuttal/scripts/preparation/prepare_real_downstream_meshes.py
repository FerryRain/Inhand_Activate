#!/usr/bin/env python3
"""Prepare the six existing real AURORA meshes for downstream simulation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from rebuttal.scripts.preparation.align_downstream_meshes import align_similarity, load, metric_gt


ROOT = Path(__file__).resolve().parents[3]
RUNS = {
    "Big_Cylinder": "003_ICP",
    "cube_obj_01": "004",
    "cube_obj_02": "002_ICP",
    "cube_purple": "001_ICP",
    "green_Pepper": "002_ICP",
    "tetraprism": "003_ICP",
}


def frame_id(path: Path):
    match = re.search(r"_(\d{6})(?:_|\.)", path.name)
    return int(match.group(1)) if match else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root", default="reconstruction/offline/result/offline_tracking"
    )
    parser.add_argument("--gt-root", default="reconstruction/offline/GT_data")
    parser.add_argument("--output", default="rebuttal/results/downstream_real_meshes")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source_root = ROOT / args.source_root
    gt_root = ROOT / args.gt_root
    output_root = ROOT / args.output
    for object_name, run_name in RUNS.items():
        candidates = list(
            (source_root / object_name / run_name / "mesh_nksr").glob("*.ply")
        )
        if not candidates:
            raise FileNotFoundError(f"No real mesh for {object_name}/{run_name}")
        source_path = max(candidates, key=frame_id)
        gt_path = gt_root / object_name / "GT/mesh/GT_mesh.stl"
        output = output_root / "aurora_real" / object_name / "pose_000.ply"
        metadata = output.with_suffix(".source.json")
        if output.exists() and not args.overwrite:
            continue
        source = load(source_path)
        target = metric_gt(gt_path)
        # Unit scales are known (reconstruction in metres, scanner GT in mm),
        # so only the evaluator's rigid coordinate registration is estimated.
        transform, scale, error = align_similarity(
            source, target, count=8000, allow_scale=False
        )
        if abs(scale - 1.0) > 1e-9:
            raise RuntimeError("Real-mesh alignment unexpectedly changed scale")
        source.apply_transform(transform)
        output.parent.mkdir(parents=True, exist_ok=True)
        source.export(output)
        metadata.write_text(
            json.dumps(
                {
                    "source": str(source_path.relative_to(ROOT)),
                    "gt": str(gt_path.relative_to(ROOT)),
                    "run": run_name,
                    "frame": frame_id(source_path),
                    "alignment": "rigid_evaluation_registration",
                    "symmetric_alignment_error_m": error,
                    "transform": np.asarray(transform).tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[real] {object_name}/{run_name}/frame_{frame_id(source_path):06d}: "
            f"alignment error={error * 1000:.2f} mm",
            flush=True,
        )


if __name__ == "__main__":
    main()
