#!/usr/bin/env python3
"""Extract reproducible six-second primitive trajectories from real pose logs.

Each source sequence is expressed as a camera-frame relative rotation
Q(t)=R(t)R(0)^T.  A single conjugation aligns its endpoint axis with the
labelled simulator primitive while preserving the measured angle evolution,
axis changes, reversals, and abrupt increments along the trajectory.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from rebuttal.benchmark.geometry import ACTION_AXES, action_rotation, rotation_error_deg


AXIS_TO_ACTION = {"x": "minus_x", "y": "minus_y", "z": "plus_z"}


def _align_vector(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source /= np.linalg.norm(source) + 1e-12
    target /= np.linalg.norm(target) + 1e-12
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if cosine > 1.0 - 1e-10:
        return np.eye(3, dtype=np.float64)
    if cosine < -1.0 + 1e-10:
        basis = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(source, basis))) > 0.9:
            basis = np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, basis)
        axis /= np.linalg.norm(axis) + 1e-12
        return Rotation.from_rotvec(np.pi * axis).as_matrix()
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    skew = np.array(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + skew + (skew @ skew) * ((1.0 - cosine) / (sine * sine))


def _trajectory(directory: Path, action: str, fps: float, seconds: float, angles_deg):
    files = sorted(directory.glob("*.txt"), key=lambda path: int(path.stem))
    if len(files) < 2:
        return None
    frame_ids = np.asarray([int(path.stem) for path in files], dtype=np.int64)
    target_frame = int(frame_ids[0] + round(fps * seconds))
    end_index = int(np.argmin(np.abs(frame_ids - target_frame)))
    if abs(int(frame_ids[end_index] - target_frame)) > int(round(2.0 * fps)):
        return None
    files = files[: end_index + 1]
    frame_ids = frame_ids[: end_index + 1]
    rotations = np.stack([np.loadtxt(path)[:3, :3] for path in files], axis=0)
    relative = rotations @ rotations[0].T
    endpoint_rotvec = Rotation.from_matrix(relative[-1]).as_rotvec()
    endpoint_angle = float(np.linalg.norm(endpoint_rotvec))
    if endpoint_angle <= 1e-8:
        return None
    endpoint_axis = endpoint_rotvec / endpoint_angle
    target_axis = np.asarray(ACTION_AXES[action], dtype=np.float64)
    target_axis /= np.linalg.norm(target_axis)
    alignment = _align_vector(endpoint_axis, target_axis)
    aligned = alignment[None] @ relative @ alignment.T[None]

    relative_frames = frame_ids - frame_ids[0]
    requested_anchor_frames = np.arange(int(round(seconds)) + 1) * float(fps)
    anchor_indices = np.unique(
        [int(np.argmin(np.abs(relative_frames - target))) for target in requested_anchor_frames]
    )
    if anchor_indices[0] != 0:
        anchor_indices = np.concatenate([[0], anchor_indices])
    if anchor_indices[-1] != len(aligned) - 1:
        anchor_indices = np.concatenate([anchor_indices, [len(aligned) - 1]])
    anchor_frames = relative_frames[anchor_indices].astype(np.float64)
    anchor_times = anchor_frames / float(fps)
    # The endpoint is the tracked pose closest to 6 s.  Normalize its small
    # timestamp discrepancy so every replay has the same decision horizon.
    anchor_times *= float(seconds) / float(anchor_times[-1])
    anchor_rotations = aligned[anchor_indices]
    interpolator = Slerp(anchor_times, Rotation.from_matrix(anchor_rotations))
    one_second_rotation = interpolator([min(1.0, float(anchor_times[-1]))]).as_matrix()[0]
    expected_one_second = Rotation.from_rotvec(
        target_axis * np.radians(float(angles_deg[action])) / float(seconds)
    ).as_matrix()
    early_error_deg = rotation_error_deg(expected_one_second, one_second_rotation)
    endpoint_error_deg = rotation_error_deg(action_rotation(action, angles_deg), aligned[-1])
    raw_increments = aligned[1:] @ np.transpose(aligned[:-1], (0, 2, 1))
    raw_increment_angles = Rotation.from_matrix(raw_increments).magnitude()
    anchor_increments = anchor_rotations[1:] @ np.transpose(anchor_rotations[:-1], (0, 2, 1))
    anchor_increment_angles = Rotation.from_matrix(anchor_increments).magnitude()
    endpoint_aligned_rotvec = Rotation.from_matrix(aligned[-1]).as_rotvec()
    endpoint_aligned_axis = endpoint_aligned_rotvec / (
        np.linalg.norm(endpoint_aligned_rotvec) + 1e-12
    )

    repo_root = Path(__file__).resolve().parents[3]
    return {
        "log": str(directory.relative_to(repo_root)),
        "source_object": directory.parents[2].name,
        "keyframes": int(len(aligned)),
        "frame_ids": relative_frames.astype(int).tolist(),
        "times_s": (relative_frames.astype(np.float64) / float(fps)).tolist(),
        "relative_rotations": aligned.tolist(),
        "anchor_frame_ids": anchor_frames.astype(int).tolist(),
        "anchor_times_s": anchor_times.tolist(),
        "anchor_relative_rotations": anchor_rotations.tolist(),
        "endpoint_angle_deg": float(np.degrees(np.linalg.norm(endpoint_aligned_rotvec))),
        "endpoint_axis_alignment": float(np.dot(endpoint_aligned_axis, target_axis)),
        "first_second_error_deg": float(early_error_deg),
        "endpoint_nominal_error_deg": float(endpoint_error_deg),
        "anchor_path_length_deg": float(np.degrees(np.sum(anchor_increment_angles))),
        "max_anchor_increment_deg": float(np.degrees(np.max(anchor_increment_angles))),
        "raw_rotation_path_length_deg": float(np.degrees(np.sum(raw_increment_angles))),
        "raw_max_keyframe_increment_deg": float(np.degrees(np.max(raw_increment_angles))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="Real_deploy/results/ablation/offline_tracking")
    parser.add_argument("--base-config", default="rebuttal/configs/base.yaml")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--output", default="rebuttal/assets/action_trajectories_6s.json")
    args = parser.parse_args()

    from benchmark.config import load_config, resolved_path

    cfg = load_config(args.base_config)
    root = resolved_path(args.root)
    output = {
        "fps": float(args.fps),
        "seconds": float(args.seconds),
        "coordinate_model": "aligned camera-frame relative rotation Q(t)=R(t)R(0)^T",
        "replay_model": "SO(3) geodesic interpolation through nearest 0..6 s tracked-pose anchors",
        "actions": {},
    }
    for axis, action in AXIS_TO_ACTION.items():
        trajectories = []
        pattern = str(root / "*" / axis / "keyframes" / "poses")
        for directory_text in sorted(glob.glob(pattern)):
            item = _trajectory(
                Path(directory_text), action, args.fps, args.seconds, cfg["actions"]["angles_deg"]
            )
            if item is not None:
                trajectories.append(item)
        if not trajectories:
            raise RuntimeError("No valid trajectories for %s" % action)
        output["actions"][action] = {"trajectories": trajectories}

    path = resolved_path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
