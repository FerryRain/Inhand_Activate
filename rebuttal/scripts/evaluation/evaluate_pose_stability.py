#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import zlib
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr, wilcoxon

from rebuttal.benchmark.config import load_config, resolved_path
from rebuttal.benchmark.environment import Observation
from rebuttal.benchmark.fusion import PointCloudFusion
from rebuttal.benchmark.geometry import map_nbv_to_action
from rebuttal.benchmark.planners.ray_gpis import RayGPISPlanner


LEVELS = {
    "mild": (1.5, 0.75),
    "moderate": (3.0, 1.5),
    "severe": (6.0, 3.0),
}


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _observation(step_dir: Path, cfg, perturbation=None):
    depth = np.load(str(step_dir / "depth.npy"))
    mask = cv2.imread(str(step_dir / "mask.png"), cv2.IMREAD_GRAYSCALE) > 0
    gt_pose = np.load(str(step_dir / "gt_pose.npy"))
    executed_pose = np.load(str(step_dir / "executed_pose.npy"))
    if perturbation is not None:
        rotvec, translation = perturbation
        executed_pose = executed_pose.copy()
        executed_pose[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix() @ executed_pose[:3, :3]
        executed_pose[:3, 3] += translation
    return Observation(
        rgb=np.zeros((*depth.shape, 3), dtype=np.uint8),
        depth=depth,
        mask=mask,
        gt_pose=gt_pose,
        executed_pose=executed_pose,
        camera_intrinsics=np.asarray(cfg["camera"]["intrinsics"], dtype=np.float64),
        step=int(step_dir.name.split("_")[-1]),
    )


def _fusion(episode_dir: Path, cfg, perturbations=None):
    fusion = PointCloudFusion(cfg)
    observations = []
    for step in range(3):
        perturbation = None if perturbations is None else perturbations[step]
        observation = _observation(
            episode_dir / ("step_%03d" % step), cfg, perturbation
        )
        fusion.update(observation)
        observations.append(observation)
    return fusion, observations


def _score_pair(planner, fusion):
    planner._estimate(fusion)
    return (
        np.asarray(planner.estimator.scores, dtype=np.float64).copy(),
        np.asarray(
            planner.estimator._unc_base * planner.estimator._nov_base,
            dtype=np.float64,
        ).copy(),
        np.asarray(planner.estimator.dirs, dtype=np.float64).copy(),
    )


def _selected(scores, directions, pose, angles):
    index = int(np.nanargmax(scores))
    direction = directions[index]
    action, _ = map_nbv_to_action(direction, pose, angles)
    return index, direction, action


def _correlation(reference, perturbed):
    valid = np.isfinite(reference) & np.isfinite(perturbed)
    if valid.sum() < 3 or np.ptp(reference[valid]) <= 1e-12 or np.ptp(perturbed[valid]) <= 1e-12:
        return float("nan")
    return float(spearmanr(reference[valid], perturbed[valid]).correlation)


def _seed(object_name, pose_seed, level, repeat, frame):
    text = "%s|%d|%s|%d|%d" % (object_name, pose_seed, level, repeat, frame)
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def _bootstrap(values, seed=2026, draws=10000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(seed)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rebuttal/configs/stress_pose_reference.yaml")
    parser.add_argument("--objects", default="")
    parser.add_argument("--pose-seeds", default="")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", default="rebuttal/results/stress_pose_stability")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = resolved_path(cfg["experiment"]["output_dir"])
    output = resolved_path(args.output)
    objects = [value for value in args.objects.split(",") if value] or list(cfg["assets"]["objects"])
    poses = [int(value) for value in args.pose_seeds.split(",") if value] or list(
        cfg["experiment"]["initial_pose_seeds"]
    )
    rows = []
    planner = RayGPISPlanner(cfg)
    if planner.estimator.device != "cuda":
        raise RuntimeError("Pose-stability diagnostic requires CUDA")
    for object_name in objects:
        for pose_seed in poses:
            episode_dir = root / "ray_gpis" / object_name / ("pose_%03d" % pose_seed)
            if not (episode_dir / "episode_summary.json").exists():
                raise FileNotFoundError(episode_dir / "episode_summary.json")
            reference_fusion, reference_observations = _fusion(episode_dir, cfg)
            ref_full, ref_point, directions = _score_pair(planner, reference_fusion)
            variants = {"ray_gpis": ref_full, "ray_gpis_pointwise": ref_point}
            reference_selection = {
                name: _selected(
                    scores,
                    directions,
                    reference_observations[-1].executed_pose,
                    cfg["actions"]["angles_deg"],
                )
                for name, scores in variants.items()
            }
            counterfactual_dir = episode_dir / "step_002" / "counterfactual"
            gains = {
                action: float(
                    json.loads(
                        (counterfactual_dir / (action + "_metrics.json")).read_text(
                            encoding="utf-8"
                        )
                    )["gain_f@5"]
                )
                for action in ("minus_x", "minus_y", "plus_z")
            }
            oracle_gain = max(gains.values())
            for level, (rotation_deg, translation_mm) in LEVELS.items():
                for repeat in range(args.repeats):
                    perturbations = []
                    for frame in range(3):
                        rng = np.random.RandomState(
                            _seed(object_name, pose_seed, level, repeat, frame)
                        )
                        perturbations.append(
                            (
                                rng.normal(0.0, np.radians(rotation_deg), size=3),
                                rng.normal(0.0, 0.001 * translation_mm, size=3),
                            )
                        )
                    perturbed_fusion, perturbed_observations = _fusion(
                        episode_dir, cfg, perturbations
                    )
                    full, point, perturbed_directions = _score_pair(
                        planner, perturbed_fusion
                    )
                    for variant, reference_scores, perturbed_scores in (
                        ("ray_gpis", ref_full, full),
                        ("ray_gpis_pointwise", ref_point, point),
                    ):
                        _, selected_direction, selected_action = _selected(
                            perturbed_scores,
                            perturbed_directions,
                            perturbed_observations[-1].executed_pose,
                            cfg["actions"]["angles_deg"],
                        )
                        _, reference_direction, reference_action = reference_selection[variant]
                        angular = float(
                            np.degrees(
                                np.arccos(
                                    np.clip(
                                        np.dot(selected_direction, reference_direction),
                                        -1.0,
                                        1.0,
                                    )
                                )
                            )
                        )
                        rows.append({
                            "variant": variant,
                            "object": object_name,
                            "initial_pose_seed": int(pose_seed),
                            "level": level,
                            "repeat": int(repeat),
                            "rotation_std_deg": float(rotation_deg),
                            "translation_std_mm": float(translation_mm),
                            "score_map_correlation": _correlation(
                                reference_scores, perturbed_scores
                            ),
                            "action_flip": float(selected_action != reference_action),
                            "top_direction_deviation_deg": angular,
                            "oracle_regret": float(oracle_gain - gains[selected_action]),
                            "selected_action": selected_action,
                            "reference_action": reference_action,
                            "device": planner.estimator.device,
                        })
            print("[pose-stability] %s pose=%d" % (object_name, pose_seed), flush=True)
    _write_csv(output / "pose_stability_samples.csv", rows)

    aggregate = []
    scene_rows = []
    fields = (
        "score_map_correlation",
        "action_flip",
        "top_direction_deviation_deg",
        "oracle_regret",
    )
    for level in LEVELS:
        for variant in ("ray_gpis_pointwise", "ray_gpis"):
            selected = [
                row for row in rows if row["level"] == level and row["variant"] == variant
            ]
            out = {"level": level, "variant": variant, "samples": len(selected)}
            for field in fields:
                values = np.asarray([row[field] for row in selected], dtype=np.float64)
                finite = values[np.isfinite(values)]
                out[field] = float(finite.mean())
                out[field + "_ci95"] = float(
                    1.96 * finite.std(ddof=1) / np.sqrt(len(finite))
                )
            aggregate.append(out)
            for object_name in objects:
                for pose_seed in poses:
                    scene = [
                        row
                        for row in selected
                        if row["object"] == object_name
                        and row["initial_pose_seed"] == pose_seed
                    ]
                    scene_rows.append({
                        "level": level,
                        "variant": variant,
                        "object": object_name,
                        "initial_pose_seed": int(pose_seed),
                        **{
                            field: float(np.nanmean([row[field] for row in scene]))
                            for field in fields
                        },
                    })
    _write_csv(output / "pose_stability_aggregate.csv", aggregate)
    _write_csv(output / "pose_stability_scenes.csv", scene_rows)

    severe = [row for row in scene_rows if row["level"] == "severe"]
    lookup = {
        (row["variant"], row["object"], row["initial_pose_seed"]): row for row in severe
    }
    pairs = [(object_name, pose_seed) for object_name in objects for pose_seed in poses]
    differences = np.asarray([
        lookup[("ray_gpis",) + pair]["score_map_correlation"]
        - lookup[("ray_gpis_pointwise",) + pair]["score_map_correlation"]
        for pair in pairs
    ])
    ci = _bootstrap(differences)
    nonzero = differences[np.abs(differences) > 1e-12]
    primary = [{
        "scenario": "pose_stability",
        "metric": "severe_score_map_correlation",
        "full": float(np.mean([lookup[("ray_gpis",) + pair]["score_map_correlation"] for pair in pairs])),
        "ablation": "ray_gpis_pointwise",
        "ablated": float(np.mean([lookup[("ray_gpis_pointwise",) + pair]["score_map_correlation"] for pair in pairs])),
        "positive_means_full_better": True,
        "mean_improvement": float(differences.mean()),
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
        "p_two_sided": float(wilcoxon(nonzero, alternative="two-sided").pvalue) if len(nonzero) else 1.0,
        "pairs": len(pairs),
        "wins": int(np.sum(differences > 1e-12)),
        "ties": int(np.sum(np.abs(differences) <= 1e-12)),
        "losses": int(np.sum(differences < -1e-12)),
    }]
    _write_csv(output / "pose_stability_primary_paired.csv", primary)
    print(output / "pose_stability_primary_paired.csv")


if __name__ == "__main__":
    main()
