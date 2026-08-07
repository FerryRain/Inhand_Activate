from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from .assets import load_mesh, prepare_assets
from .config import resolved_path
from .environment import KinematicRGBDEnv
from .evaluation import ReconstructionEvaluator, trapezoid_auc
from .fusion import PointCloudFusion
from .geometry import (
    ACTIONS,
    candidate_action_assignments,
    fibonacci_sphere,
    map_nbv_to_action,
    next_view_direction,
)
from .io import save_step, write_json
from .planners import (
    ActNeRFPlanner,
    ERGPISPlanner,
    FixedSchedulePlanner,
    PBNBVPlanner,
    PoseNoveltyPlanner,
    RayGPISPlanner,
)


PLANNER_TYPES = {
    "fixed": FixedSchedulePlanner,
    "pose_novelty": PoseNoveltyPlanner,
    "pb_nbv": PBNBVPlanner,
    "er_gpis": ERGPISPlanner,
    "ray_gpis": RayGPISPlanner,
    "actnerf": ActNeRFPlanner,
}


def make_planner(name: str, cfg: Dict):
    match = re.fullmatch(r"pb_nbv_d(30|50|70)(_no_partition)?", name)
    if match:
        cfg = copy.deepcopy(cfg)
        cfg["planners"]["pb_nbv"]["voxel_divisor"] = float(match.group(1))
        cfg["planners"]["pb_nbv"]["use_partition"] = match.group(2) is None
        return PBNBVPlanner(cfg)
    if name == "pb_nbv_no_partition":
        cfg = copy.deepcopy(cfg)
        cfg["planners"]["pb_nbv"]["use_partition"] = False
        return PBNBVPlanner(cfg)
    if name.startswith("ray_gpis_"):
        variant = name[len("ray_gpis_"):]
        cfg = copy.deepcopy(cfg)
        cfg["planners"]["ray_gpis"]["variant"] = variant
        return RayGPISPlanner(cfg)
    if name not in PLANNER_TYPES:
        raise KeyError("Unknown planner: %s" % name)
    return PLANNER_TYPES[name](cfg)


def _action_scores(scores, assignments):
    out = {}
    for index, action in enumerate(ACTIONS):
        values = np.asarray(scores)[assignments == index]
        finite = values[np.isfinite(values)]
        out[action] = float(np.max(finite)) if len(finite) else float("nan")
    return out


def _counterfactual(env, fusion, evaluator, current_metrics, target_points=None):
    outcomes = {}
    for action in ACTIONS:
        cloned_env = env.clone()
        cloned_fusion = fusion.clone()
        decisions_before = len(cloned_fusion.decisions)
        frames_per_action = int(cloned_env.cfg["environment"].get("frames_per_action", 1))
        observations = cloned_env.step_sequence(action, frames_per_action)
        for observation in observations:
            cloned_fusion.update(observation)
        metrics = evaluator.evaluate(cloned_fusion.points)
        if target_points is not None:
            metrics["target_recall@5"] = evaluator.recall_subset(
                cloned_fusion.points, target_points, 0.005
            )
            metrics["gain_target_recall@5"] = float(
                metrics["target_recall@5"] - current_metrics["target_recall@5"]
            )
        outcomes[action] = dict(metrics)
        outcomes[action]["gain_f@5"] = float(metrics["f@5"] - current_metrics["f@5"])
        outcomes[action]["gain_surface_coverage"] = float(
            metrics["surface_coverage"] - current_metrics["surface_coverage"]
        )
        new_decisions = cloned_fusion.decisions[decisions_before:]
        outcomes[action]["trajectory_metadata"] = dict(
            cloned_env.last_trajectory_metadata
        )
        outcomes[action]["acquired_frames"] = int(len(observations))
        outcomes[action]["accepted_frames"] = int(
            sum(bool(item["accepted"]) for item in new_decisions)
        )
    return outcomes


def _planner_quality(action_scores, counterfactual, selected_action):
    gains = np.array([counterfactual[action]["gain_f@5"] for action in ACTIONS], dtype=np.float64)
    scores = np.array([action_scores[action] for action in ACTIONS], dtype=np.float64)
    valid = np.all(np.isfinite(scores)) and np.ptp(scores) > 1e-12 and np.ptp(gains) > 1e-12
    correlation = float(spearmanr(scores, gains).correlation) if valid else float("nan")
    oracle_index = int(np.argmax(gains))
    selected_index = ACTIONS.index(selected_action)
    return {
        "score_gain_spearman": correlation,
        "oracle_regret_f@5": float(gains[oracle_index] - gains[selected_index]),
        "oracle_action_accuracy": float(selected_index == oracle_index),
        "oracle_action": ACTIONS[oracle_index],
        "selected_gain_f@5": float(gains[selected_index]),
        "selected_gain_surface_coverage": float(
            counterfactual[selected_action]["gain_surface_coverage"]
        ),
        "oracle_regret_surface_coverage": float(
            max(counterfactual[action]["gain_surface_coverage"] for action in ACTIONS)
            - counterfactual[selected_action]["gain_surface_coverage"]
        ),
        "action_scores": action_scores,
    }


def _evaluate_state(evaluator, fused_points, target_points=None):
    metrics = evaluator.evaluate(fused_points)
    if target_points is not None:
        metrics["target_recall@5"] = evaluator.recall_subset(
            fused_points, target_points, 0.005
        )
    return metrics


def run_episode(
    cfg: Dict,
    planner_name: str,
    object_name: str,
    mesh_path: str,
    initial_pose_seed: int,
    planner_seed: int = 0,
    action_steps: Optional[int] = None,
    overwrite: bool = False,
) -> Path:
    output_root = resolved_path(cfg["experiment"]["output_dir"])
    suffix = "pose_%03d" % int(initial_pose_seed)
    if planner_name == "actnerf":
        suffix += "_seed_%03d" % int(planner_seed)
    episode_dir = output_root / planner_name / object_name / suffix
    summary_path = episode_dir / "episode_summary.json"
    if summary_path.exists() and not overwrite:
        print("[skip] %s" % episode_dir)
        return episode_dir
    episode_dir.mkdir(parents=True, exist_ok=True)

    episode_cfg = copy.deepcopy(cfg)
    episode_cfg.setdefault("runtime", {})["planner_seed"] = int(planner_seed)
    episode_cfg["episode"] = {
        "planner": planner_name,
        "object": object_name,
        "initial_pose_seed": int(initial_pose_seed),
        "planner_seed": int(planner_seed),
        "mesh_path": str(mesh_path),
    }
    write_json(episode_dir / "config.json", episode_cfg)
    mesh = load_mesh(mesh_path)
    o3d.io.write_triangle_mesh(str(episode_dir / "object_gt_mesh.ply"), mesh)

    env = KinematicRGBDEnv(episode_cfg, mesh_path, seed=int(cfg["seed"]) + int(initial_pose_seed))
    env.reset(initial_pose_seed)
    bootstrap = env.bootstrap_observations(float(episode_cfg["environment"]["bootstrap_degrees"]))
    fusion = PointCloudFusion(episode_cfg)
    for observation in bootstrap:
        fusion.update(observation)
    current_observation = bootstrap[0]
    evaluator = ReconstructionEvaluator(mesh, episode_cfg, seed=initial_pose_seed)
    target_mode = str(episode_cfg["evaluation"].get("target_surface_mode", "none"))
    target_points = None
    if target_mode == "removed_observation":
        removed = [
            observation.target_points_object
            for observation in bootstrap
            if observation.target_points_object is not None
            and len(observation.target_points_object)
        ]
        target_points = (
            np.concatenate(removed, axis=0).astype(np.float32)
            if removed
            else np.zeros((0, 3), dtype=np.float32)
        )
        if len(target_points) and len(fusion.points):
            initial_distances = cKDTree(fusion.points).query(
                target_points, k=1, workers=-1
            )[0]
            target_points = target_points[
                (initial_distances > 0.005) & (initial_distances <= 0.010)
            ]
    elif target_mode == "initial_unseen":
        distances = evaluator.distances_from_gt(fusion.points)
        target_points = evaluator.gt_points[distances > 0.005].copy()
    elif target_mode != "none":
        raise ValueError("Unknown evaluation.target_surface_mode: %s" % target_mode)
    if target_points is not None:
        np.save(str(episode_dir / "target_surface_points.npy"), target_points)
    planner = make_planner(planner_name, episode_cfg)
    planner.reset(current_observation, fusion)

    candidates = fibonacci_sphere(int(cfg["candidates"]["count"]))
    total_steps = int(cfg["environment"]["action_steps"] if action_steps is None else action_steps)
    reconstruction_metrics = []
    planner_metrics = []
    runtime_metrics = []
    action_history = []
    fault_events = []
    action_trajectory_events = []

    for step in range(total_steps):
        metrics = _evaluate_state(evaluator, fusion.points, target_points)
        reconstruction_metrics.append(metrics)
        planning_start = time.perf_counter()
        if planner_name == "fixed":
            selected_action = planner.next_action()
            selected_nbv = next_view_direction(current_observation.executed_pose, selected_action, env.angles_deg)
            scores = np.full((len(candidates),), np.nan, dtype=np.float32)
            mapper_similarities = {}
        else:
            selected_nbv = planner.select_nbv(candidates)
            selected_action, mapper_similarities = map_nbv_to_action(
                selected_nbv, current_observation.executed_pose, env.angles_deg
            )
            scores = planner.last_scores.copy()
        planning_wall = time.perf_counter() - planning_start
        assignments = candidate_action_assignments(candidates, current_observation.executed_pose, env.angles_deg)
        action_scores = _action_scores(scores, assignments)
        counterfactual = (
            _counterfactual(env, fusion, evaluator, metrics, target_points)
            if cfg["experiment"].get("counterfactual", True)
            else {}
        )
        quality = _planner_quality(action_scores, counterfactual, selected_action) if counterfactual else {
            "score_gain_spearman": float("nan"),
            "oracle_regret_f@5": float("nan"),
            "oracle_action_accuracy": float("nan"),
            "oracle_action": None,
            "selected_gain_f@5": float("nan"),
            "action_scores": action_scores,
        }
        diagnostics = planner.diagnostics()
        diagnostics["selection_wall_s"] = float(planning_wall)
        runtime_metrics.append(diagnostics)
        planner_metrics.append(quality)
        action_history.append(selected_action)
        step_metrics = dict(metrics)
        step_metrics["planner_quality"] = quality
        action_payload = {
            "action": selected_action,
            "desired_view": selected_nbv,
            "mapper_cosine_similarities": mapper_similarities,
            "diagnostics": diagnostics,
        }
        save_step(
            episode_dir / ("step_%03d" % step),
            current_observation,
            fusion.points,
            candidates,
            scores,
            step_metrics,
            selected_nbv,
            action_payload,
            counterfactual,
            save_images=bool(cfg["experiment"].get("save_images", True)),
        )

        frames_per_action = int(episode_cfg["environment"].get("frames_per_action", 1))
        decisions_before = len(fusion.decisions)
        generated_observations = env.step_sequence(selected_action, frames_per_action)
        for generated in generated_observations:
            inserted_points = fusion.update(generated)
            if generated.fault_tags:
                event = {
                    "step": int(generated.step),
                    "fault_tags": list(generated.fault_tags),
                    "inserted_points": int(len(inserted_points)),
                    "pose_error_deg": float(generated.pose_error_deg),
                    "translation_error_m": float(generated.translation_error_m),
                    "visibility_ratio": float(generated.visibility_ratio),
                }
                if "scheduled_pose_outlier" in generated.fault_tags and len(inserted_points):
                    distances = evaluator.distances_to_gt(inserted_points)
                    event["ghost_points"] = int(np.sum(distances > 0.005))
                    event["ghost_ratio"] = float(np.mean(distances > 0.005))
                fault_events.append(event)
        if env.last_trajectory_metadata:
            trajectory_event = dict(env.last_trajectory_metadata)
            new_decisions = fusion.decisions[decisions_before:]
            trajectory_event.update({
                "decision_step": int(step),
                "selected_action": selected_action,
                "accepted_frames": int(
                    sum(bool(item["accepted"]) for item in new_decisions)
                ),
                "rejected_frames": int(
                    sum(not bool(item["accepted"]) for item in new_decisions)
                ),
            })
            action_trajectory_events.append(trajectory_event)
        # Filtering decides whether a frame enters reconstruction, not whether
        # its latest tracking pose is available to the action mapper.
        current_observation = generated_observations[-1]
        planner.update(current_observation, fusion)

    final_metrics = _evaluate_state(evaluator, fusion.points, target_points)
    reconstruction_metrics.append(final_metrics)
    save_step(
        episode_dir / ("step_%03d" % total_steps),
        current_observation,
        fusion.points,
        candidates,
        np.full((len(candidates),), np.nan, dtype=np.float32),
        final_metrics,
        None,
        {"terminal": True},
        None,
        save_images=bool(cfg["experiment"].get("save_images", True)),
    )
    f_values = [item["f@5"] for item in reconstruction_metrics]
    recall_values = [item["recall@5"] for item in reconstruction_metrics]
    coverage_values = [item["surface_coverage"] for item in reconstruction_metrics]
    target_values = (
        [item["target_recall@5"] for item in reconstruction_metrics]
        if target_points is not None
        else []
    )
    correlations = [item["score_gain_spearman"] for item in planner_metrics]
    regrets = [item["oracle_regret_f@5"] for item in planner_metrics]
    accuracies = [item["oracle_action_accuracy"] for item in planner_metrics]
    accepted_frames = sum(int(item["accepted"]) for item in fusion.decisions)
    action_switches = sum(action_history[index] != action_history[index - 1] for index in range(1, len(action_history)))
    summary = {
        "planner": planner_name,
        "object": object_name,
        "initial_pose_seed": int(initial_pose_seed),
        "planner_seed": int(planner_seed),
        "actions": action_history,
        "f@5_auc": trapezoid_auc(f_values),
        "recall@5_auc": trapezoid_auc(recall_values),
        "surface_coverage_auc": trapezoid_auc(coverage_values),
        "target_surface_mode": target_mode,
        "target_surface_point_count": int(len(target_points)) if target_points is not None else 0,
        "target_recall@5_auc": trapezoid_auc(target_values) if target_values else float("nan"),
        "final_target_recall@5": float(target_values[-1]) if target_values else float("nan"),
        "final_f@5": float(f_values[-1]),
        "final_recall@5": float(recall_values[-1]),
        "final_chamfer_m": float(final_metrics["chamfer_m"]),
        "score_gain_correlation": float(np.nanmean(correlations)) if np.any(np.isfinite(correlations)) else float("nan"),
        "oracle_regret": float(np.nanmean(regrets)) if np.any(np.isfinite(regrets)) else float("nan"),
        "oracle_action_accuracy": float(np.nanmean(accuracies)) if np.any(np.isfinite(accuracies)) else float("nan"),
        "planning_time_s": float(np.mean([item["planning_s"] for item in runtime_metrics])) if runtime_metrics else 0.0,
        "candidate_scoring_time_s": float(np.mean([item["candidate_scoring_s"] for item in runtime_metrics])) if runtime_metrics else 0.0,
        "accepted_frames": int(accepted_frames),
        "rejected_frames": int(len(fusion.decisions) - accepted_frames),
        "action_switching_frequency": float(action_switches / max(len(action_history) - 1, 1)),
        "action_repetition_rate": float(1.0 - action_switches / max(len(action_history) - 1, 1)),
        "reconstruction_metrics": reconstruction_metrics,
        "planner_metrics": planner_metrics,
        "runtime_metrics": runtime_metrics,
        "fault_events": fault_events,
        "action_trajectory_events": action_trajectory_events,
    }
    write_json(episode_dir / "keyframe_decisions.json", {"decisions": fusion.decisions})
    write_json(summary_path, summary)
    print("[done] %s F@5-AUC=%.4f" % (episode_dir, summary["f@5_auc"]))
    return episode_dir


def prepare_manifest(cfg: Dict) -> Dict[str, str]:
    asset_cfg = cfg["assets"]
    output_dir = resolved_path(asset_cfg["output_dir"])
    manifest_path = output_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    elif not bool(asset_cfg.get("skip_default_prepare", False)):
        manifest = prepare_assets(
            output_dir,
            float(asset_cfg["target_diagonal_m"]),
            int(asset_cfg["poisson_depth"]),
        )

    # External metric meshes (e.g. official YCB models) can participate in the
    # identical renderer/fusion/planner loop without being normalized or
    # copied into the eight-object rebuttal asset set.
    for object_name, value in asset_cfg.get("mesh_paths", {}).items():
        mesh_path = resolved_path(value)
        if not mesh_path.exists():
            raise FileNotFoundError("External object mesh not found: %s" % mesh_path)
        manifest[str(object_name)] = str(mesh_path)

    missing = [name for name in asset_cfg.get("objects", []) if name not in manifest]
    if missing:
        raise KeyError("Objects missing from asset manifest: %s" % ", ".join(missing))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    return manifest


def run_suite(
    cfg: Dict,
    planners: Optional[Iterable[str]] = None,
    objects: Optional[Iterable[str]] = None,
    pose_seeds: Optional[Iterable[int]] = None,
    action_steps: Optional[int] = None,
    overwrite: bool = False,
):
    manifest = prepare_manifest(cfg)
    planners = list(planners or cfg["planners"]["enabled"])
    objects = list(objects or cfg["assets"]["objects"])
    pose_seeds = list(pose_seeds or cfg["experiment"]["initial_pose_seeds"])
    for planner in planners:
        for object_name in objects:
            for pose_seed in pose_seeds:
                planner_seeds = cfg["experiment"].get("actnerf_seeds", [0, 1, 2]) if planner == "actnerf" else [0]
                for planner_seed in planner_seeds:
                    run_episode(
                        cfg,
                        planner,
                        object_name,
                        manifest[object_name],
                        int(pose_seed),
                        planner_seed=int(planner_seed),
                        action_steps=action_steps,
                        overwrite=overwrite,
                    )
