#!/usr/bin/env python3
"""Evaluate one planner decision over shared executable action branches.

For each object/initial-pose scene the three executable action branches are
rendered exactly once and shared by every enabled planner. Planners score only
the state after the optional shared prefix and cannot switch actions during the
evaluated trajectory.
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from rebuttal.benchmark.assets import load_mesh
from rebuttal.benchmark.config import load_config, resolved_path
from rebuttal.benchmark.environment import KinematicRGBDEnv
from rebuttal.benchmark.evaluation import ReconstructionEvaluator
from rebuttal.benchmark.fusion import PointCloudFusion
from rebuttal.benchmark.geometry import (
    ACTIONS,
    candidate_action_assignments,
    fibonacci_sphere,
    map_nbv_to_action,
)
from rebuttal.benchmark.io import write_json
from rebuttal.benchmark.runner import (
    _action_scores,
    _planner_quality,
    make_planner,
    prepare_manifest,
)


DEFAULT_PLANNERS = ("pose_novelty", "ray_gpis")


def _metrics(evaluator, points):
    item = evaluator.evaluate(points)
    return {
        "recall@5": float(item["recall@5"]),
        "f@5": float(item["f@5"]),
        "chamfer_mm": 1000.0 * float(item["chamfer_m"]),
    }


def _observation_fault_row(observation):
    return {
        "step": int(observation.step),
        "pose_error_deg": float(observation.pose_error_deg),
        "translation_error_mm": 1000.0 * float(observation.translation_error_m),
        "visibility_ratio": float(observation.visibility_ratio),
        "fault_tags": list(observation.fault_tags),
    }


def run_scene(cfg, manifest, object_name, pose_seed, overwrite=False):
    root = resolved_path(cfg["experiment"]["output_dir"])
    scene_dir = root / object_name / ("pose_%03d" % int(pose_seed))
    summary_path = scene_dir / "summary.json"
    if summary_path.exists() and not overwrite:
        print("[skip] %s" % scene_dir, flush=True)
        return
    scene_dir.mkdir(parents=True, exist_ok=True)
    scene_cfg = copy.deepcopy(cfg)
    scene_cfg["episode"] = {
        "object": object_name,
        "initial_pose_seed": int(pose_seed),
        "planner": "shared_one_step",
    }
    mesh_path = manifest[object_name]
    mesh = load_mesh(mesh_path)
    env = KinematicRGBDEnv(scene_cfg, mesh_path, seed=int(cfg["seed"]) + int(pose_seed))
    env.reset(int(pose_seed))
    bootstrap = env.bootstrap_observations(
        float(scene_cfg["environment"]["bootstrap_degrees"])
    )
    fusion = PointCloudFusion(scene_cfg)
    for observation in bootstrap:
        fusion.update(observation)
    initial_observation = bootstrap[0]
    evaluator = ReconstructionEvaluator(mesh, scene_cfg, seed=int(pose_seed))
    prefix_options = list(scene_cfg["environment"].get("shared_prefix_actions", []))
    prefix_sequences = list(scene_cfg["environment"].get("shared_prefix_sequences", []))
    prefix_actions = []
    if prefix_sequences:
        prefix_actions = list(prefix_sequences[int(pose_seed) % len(prefix_sequences)])
    elif prefix_options:
        prefix_actions = [prefix_options[int(pose_seed) % len(prefix_options)]]
    prefix_observations = []
    prefix_removed_targets = []
    prefix_metadata = []
    prefix_accepted_frames = 0
    for prefix_action in prefix_actions:
        action_observations = env.step_sequence(
            prefix_action, int(scene_cfg["environment"]["frames_per_action"])
        )
        decisions_before = len(fusion.decisions)
        for observation in action_observations:
            fusion.update(observation)
        prefix_decisions = fusion.decisions[decisions_before:]
        prefix_accepted_frames += int(
            sum(bool(row["accepted"]) for row in prefix_decisions)
        )
        prefix_observations.extend(action_observations)
        prefix_removed_targets.extend(
            observation.target_points_object
            for observation in action_observations
            if observation.target_points_object is not None
            and len(observation.target_points_object)
        )
        prefix_metadata.append(dict(env.last_trajectory_metadata))
    current_observation = prefix_observations[-1] if prefix_observations else initial_observation
    boundary_tracking_recovered = bool(
        prefix_observations
        and scene_cfg["environment"].get("recover_tracking_at_decision_boundary", False)
    )
    if boundary_tracking_recovered:
        current_observation = copy.deepcopy(current_observation)
        current_observation.executed_pose = current_observation.gt_pose.copy()
        current_observation.pose_error_deg = 0.0
        current_observation.translation_error_m = 0.0
        current_observation.fault_tags = tuple(current_observation.fault_tags) + (
            "tracking_recovered_at_boundary",
        )
    initial_metrics = _metrics(evaluator, fusion.points)
    recovery_target_mode = str(
        scene_cfg["evaluation"].get("recovery_target_mode", "initial_unseen")
    )
    if recovery_target_mode == "removed_prefix":
        recovery_target_points = (
            np.concatenate(prefix_removed_targets, axis=0).astype(np.float32)
            if prefix_removed_targets
            else np.zeros((0, 3), dtype=np.float32)
        )
        if len(recovery_target_points) and len(fusion.points):
            distances = cKDTree(fusion.points).query(
                recovery_target_points, k=1, workers=-1
            )[0]
            recovery_target_points = recovery_target_points[distances > 0.005]
    elif recovery_target_mode == "initial_unseen":
        initial_gt_distances = evaluator.distances_from_gt(fusion.points)
        recovery_target_points = evaluator.gt_points[initial_gt_distances > 0.005].copy()
    else:
        raise ValueError("Unknown evaluation.recovery_target_mode: %s" % recovery_target_mode)
    candidates = fibonacci_sphere(int(scene_cfg["candidates"]["count"]))
    assignments = candidate_action_assignments(
        candidates, current_observation.executed_pose, env.angles_deg
    )

    planner_rows = {}
    score_arrays = {}
    planner_names = tuple(scene_cfg["planners"].get("enabled", DEFAULT_PLANNERS))
    for planner_name in planner_names:
        planner = make_planner(planner_name, scene_cfg)
        if planner_name == "pose_novelty" and prefix_observations:
            planner.reset(initial_observation, fusion)
            # Pose coverage receives the full tracked-pose stream, including
            # frames rejected by geometric keyframe filtering.
            for observation in prefix_observations:
                planner.update(observation, fusion)
            if boundary_tracking_recovered:
                planner.update(current_observation, fusion)
        else:
            planner.reset(current_observation, fusion)
        start = time.perf_counter()
        desired_view = planner.select_nbv(candidates)
        selection_wall_s = time.perf_counter() - start
        selected_action, mapper = map_nbv_to_action(
            desired_view, current_observation.executed_pose, env.angles_deg
        )
        action_scores = _action_scores(planner.last_scores, assignments)
        diagnostics = planner.diagnostics()
        diagnostics["selection_wall_s"] = float(selection_wall_s)
        planner_rows[planner_name] = {
            "selected_action": selected_action,
            "selected_nbv": desired_view,
            "action_scores": action_scores,
            "mapper_cosine_similarities": mapper,
            "runtime": diagnostics,
        }
        score_arrays[planner_name] = np.asarray(planner.last_scores, dtype=np.float32)

    action_rows = {}
    frames_per_action = int(scene_cfg["environment"]["frames_per_action"])
    replay_rate_hz = float(scene_cfg["environment"].get("trajectory_replay_rate_hz", 1.0))
    evaluation_stride = max(1, int(round(replay_rate_hz)))
    for action in ACTIONS:
        branch_env = env.clone()
        branch_fusion = fusion.clone()
        decisions_before = len(branch_fusion.decisions)
        observations = branch_env.step_sequence(action, frames_per_action)
        curve = [{"time_s": 0.0, **initial_metrics}]
        for frame_index, observation in enumerate(observations, start=1):
            branch_fusion.update(observation)
            if frame_index % evaluation_stride == 0 or frame_index == len(observations):
                curve.append({
                    "time_s": float(frame_index) / float(
                        replay_rate_hz
                    ),
                    **_metrics(evaluator, branch_fusion.points),
                })
        final_metrics = curve[-1]
        recovery_recall = evaluator.recall_subset(
            branch_fusion.points, recovery_target_points, 0.005
        ) if len(recovery_target_points) else 1.0
        decisions = branch_fusion.decisions[decisions_before:]
        action_rows[action] = {
            "final_metrics": final_metrics,
            "gain_f@5": float(final_metrics["f@5"] - initial_metrics["f@5"]),
            "gain_recall@5": float(final_metrics["recall@5"] - initial_metrics["recall@5"]),
            "chamfer_reduction_mm": float(
                initial_metrics["chamfer_mm"] - final_metrics["chamfer_mm"]
            ),
            "recovery_recall@5": float(recovery_recall),
            "metric_curve": curve,
            "trajectory_metadata": dict(branch_env.last_trajectory_metadata),
            "observation_faults": [
                _observation_fault_row(observation)
                for observation in observations
            ],
            "acquired_frames": int(len(observations)),
            "accepted_frames": int(sum(bool(row["accepted"]) for row in decisions)),
            "rejected_frames": int(sum(not bool(row["accepted"]) for row in decisions)),
        }

    counterfactual = {
        action: {
            "gain_f@5": row["gain_f@5"],
            "gain_surface_coverage": row["gain_recall@5"],
        }
        for action, row in action_rows.items()
    }
    for planner_name, row in planner_rows.items():
        quality = _planner_quality(
            row["action_scores"], counterfactual, row["selected_action"]
        )
        selected = action_rows[row["selected_action"]]
        row["quality"] = quality
        row["selected_gain_recall@5"] = selected["gain_recall@5"]
        row["selected_chamfer_reduction_mm"] = selected["chamfer_reduction_mm"]
        row["selected_recovery_recall@5"] = selected["recovery_recall@5"]
        row["selected_trajectory_metadata"] = selected["trajectory_metadata"]
        row["selected_accepted_frames"] = selected["accepted_frames"]

    summary = {
        "object": object_name,
        "initial_pose_seed": int(pose_seed),
        "initial_metrics": initial_metrics,
        "recovery_target_point_count": int(len(recovery_target_points)),
        "action_branches": action_rows,
        "planners": planner_rows,
        "protocol": {
            "decision_horizon_s": 6.0,
            "trajectory_replay_rate_hz": replay_rate_hz,
            "recovery_target_mode": recovery_target_mode,
            "planner_decisions": 1,
            "shared_action_branches": True,
            "shared_prefix_actions": prefix_actions,
            "shared_prefix_frames": int(len(prefix_observations)),
            "shared_prefix_accepted_frames": int(prefix_accepted_frames),
            "shared_prefix_trajectory_metadata": prefix_metadata,
            "shared_prefix_observation_faults": [
                _observation_fault_row(observation)
                for observation in prefix_observations
            ],
            "boundary_tracking_recovered": boundary_tracking_recovered,
        },
    }
    write_json(summary_path, summary)
    write_json(scene_dir / "config.json", scene_cfg)
    np.save(str(scene_dir / "candidate_directions.npy"), candidates)
    for planner_name, scores in score_arrays.items():
        np.save(str(scene_dir / (planner_name + "_scores.npy")), scores)
    gains = " ".join(
        "%s=%.4f" % (name, planner_rows[name]["quality"]["selected_gain_f@5"])
        for name in planner_names
    )
    print("[done] %s %s" % (scene_dir, gains), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="rebuttal/configs/continuous_empirical_one_step_motion.yaml"
    )
    parser.add_argument("--objects", default="")
    parser.add_argument("--pose-seeds", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if bool(cfg.get("runtime", {}).get("require_cuda", False)):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        print("[runtime] CUDA device: %s" % torch.cuda.get_device_name(0), flush=True)
    manifest = prepare_manifest(cfg)
    objects = args.objects.split(",") if args.objects else cfg["assets"]["objects"]
    pose_seeds = (
        [int(value) for value in args.pose_seeds.split(",")]
        if args.pose_seeds else cfg["experiment"]["initial_pose_seeds"]
    )
    for object_name in objects:
        for pose_seed in pose_seeds:
            run_scene(cfg, manifest, object_name, int(pose_seed), args.overwrite)


if __name__ == "__main__":
    main()
