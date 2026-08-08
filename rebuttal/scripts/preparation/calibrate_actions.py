#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


AXIS_TO_ACTION = {"x": "minus_x", "y": "minus_y", "z": "plus_z"}


def main():
    parser = argparse.ArgumentParser(description="Estimate 6 s action magnitudes from axis-labelled pose logs.")
    parser.add_argument("--root", default="Real_deploy/results/ablation/offline_tracking")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--output", default="rebuttal/assets/action_calibration.json")
    args = parser.parse_args()
    target_frames = int(round(args.fps * args.seconds))
    output = {"fps": args.fps, "seconds": args.seconds, "target_frames": target_frames, "actions": {}}
    for axis, action in AXIS_TO_ACTION.items():
        records = []
        for directory in glob.glob(os.path.join(args.root, "*", axis, "keyframes", "poses")):
            files = glob.glob(os.path.join(directory, "*.txt"))
            files.sort(key=lambda path: int(Path(path).stem))
            if len(files) < 2:
                continue
            frame_ids = np.array([int(Path(path).stem) for path in files])
            target = frame_ids[0] + target_frames
            index = int(np.argmin(np.abs(frame_ids - target)))
            if abs(int(frame_ids[index] - target)) > int(args.fps * 2.0):
                continue
            initial = np.loadtxt(files[0])[:3, :3]
            final = np.loadtxt(files[index])[:3, :3]
            rotvec = Rotation.from_matrix(initial.T @ final).as_rotvec()
            records.append({
                "log": directory,
                "end_frame": int(frame_ids[index]),
                "angle_deg": float(np.degrees(np.linalg.norm(rotvec))),
                "rotvec": rotvec.tolist(),
            })
        angles = [record["angle_deg"] for record in records]
        output["actions"][action] = {
            "axis_label": axis,
            "median_angle_deg": float(np.median(angles)) if angles else None,
            "episodes": records,
        }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(str(path))


if __name__ == "__main__":
    main()
