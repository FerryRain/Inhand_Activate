from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import open3d as o3d
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
from .planners import ActNeRFPlanner, FixedSchedulePlanner, PBNBVPlanner, RayGPISPlanner


PLANNER_TYPES = {
    "fixed": FixedSchedulePlanner,
    "pb_nbv": PBNBVPlanner,
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


def _counterfactual(env, fusion, evaluator, current_metrics):
    outcomes = {}
    for action in ACTIONS:
        cloned_env = env.clone()
        cloned_fusion = fusion.clone()
        observation = cloned_env.step(action)
        cloned_fusion.update(observation)
        metrics = evaluator.evaluate(cloned_fusion.points)
        outcomes[action] = dict(metrics)
        outcomes[action]["gain_f@5"] = float(metrics["f@5"] - current_metrics["f@5"])
        outcomes[action]["gain_surface_coverage"] = float(
            metrics["surface_coverage"] - current_metrics["surface_coverage"]
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
        "action_scores": action_scores,
    }


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
    planner = make_planner(planner_name, episode_cfg)
    planner.reset(current_observation, fusion)

    candidates = fibonacci_sphere(int(cfg["candidates"]["count"]))
    total_steps = int(cfg["environment"]["action_steps"] if action_steps is None else action_steps)
    reconstruction_metrics = []
    planner_metrics = []
    runtime_metrics = []
    action_history = []

    for step in range(total_steps):
        metrics = evaluator.evaluate(fusion.points)
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
        counterfactual = _counterfactual(env, fusion, evaluator, metrics) if cfg["experiment"].get("counterfactual", True) else {}
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

        generated_observations = [env.step(selected_action)]
        frames_per_action = int(episode_cfg["environment"].get("frames_per_action", 1))
        for _ in range(1, frames_per_action):
            generated_observations.append(env.render())
        for generated in generated_observations:
            fusion.update(generated)
        current_observation = fusion.observations[-1] if fusion.observations else generated_observations[-1]
        planner.update(current_observation, fusion)

    final_metrics = evaluator.evaluate(fusion.points)
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
    }
    write_json(episode_dir / "keyframe_decisions.json", {"decisions": fusion.decisions})
    write_json(summary_path, summary)
    print("[done] %s F@5-AUC=%.4f" % (episode_dir, summary["f@5_auc"]))
    return episode_dir


def prepare_manifest(cfg: Dict) -> Dict[str, str]:
    asset_cfg = cfg["assets"]
    output_dir = resolved_path(asset_cfg["output_dir"])
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    manifest = prepare_assets(
        output_dir,
        float(asset_cfg["target_diagonal_m"]),
        int(asset_cfg["poisson_depth"]),
    )
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
