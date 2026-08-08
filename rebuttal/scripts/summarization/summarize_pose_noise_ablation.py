#!/usr/bin/env python3
"""Summarize the paired 6D pose-noise planner ablation.

The report compares the existing clean ablation with a separate run in which
all planners receive the same deterministic SE(3) tracking error at every
post-action state.  Positive ``full_improvement`` always means Full Ray-GPIS
is better; positive ``noise_degradation`` always means performance worsened
relative to the clean condition.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import wilcoxon

from rebuttal.benchmark.config import load_config, resolved_path
from rebuttal.benchmark.io import write_json


PLANNERS = (
    "ray_gpis_novelty_only",
    "ray_gpis_uncertainty_only",
    "ray_gpis_pointwise",
    "ray_gpis_hit_only",
    "ray_gpis",
)
DISPLAY_NAMES = {
    "ray_gpis_novelty_only": "Novelty only",
    "ray_gpis_uncertainty_only": "Uncertainty only",
    "ray_gpis_pointwise": "w/o receptive-field integration",
    "ray_gpis_hit_only": "w/o miss-ray interpolation",
    "ray_gpis": "Full Ray-GPIS",
}
METRICS = {
    "f@5_auc": 1.0,
    "final_f@5": 1.0,
    "score_gain_correlation": 1.0,
    "oracle_regret_f@5": -1.0,
    "consecutive_score_map_correlation": 1.0,
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_lookup(rows: Iterable[Dict[str, str]]) -> Dict[Tuple[str, str, int], Dict[str, str]]:
    return {
        (row["planner"], row["object"], int(row["initial_pose_seed"])): row
        for row in rows
        if row["planner"] in PLANNERS
    }


def bootstrap_ci(values: np.ndarray, rng: np.random.RandomState, samples: int = 10000):
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return np.percentile(draws, [2.5, 97.5]).tolist()


def holm_adjust(rows: List[Dict[str, object]]) -> None:
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["metric"])].append((index, float(row["p_two_sided"])))
    for entries in groups.values():
        ordered = sorted(entries, key=lambda item: item[1])
        running = 0.0
        for rank, (index, value) in enumerate(ordered):
            running = max(running, min(1.0, (len(ordered) - rank) * value))
            rows[index]["p_holm"] = running


def summarize_metrics(clean_rows, noisy_rows, rng):
    clean = paired_lookup(clean_rows)
    noisy = paired_lookup(noisy_rows)
    expected_keys = {
        (planner, object_name, pose_seed)
        for planner in PLANNERS
        for object_name in sorted({key[1] for key in noisy})
        for pose_seed in sorted({key[2] for key in noisy if key[1] == object_name})
    }
    if set(clean) != expected_keys or set(noisy) != expected_keys:
        raise ValueError(
            "Clean/noisy planner-scene keys are incomplete or do not match: "
            f"clean={len(clean)}, noisy={len(noisy)}, expected={len(expected_keys)}"
        )

    aggregate = []
    for planner in PLANNERS:
        keys = sorted(key for key in expected_keys if key[0] == planner)
        row = {
            "planner": planner,
            "display_name": DISPLAY_NAMES[planner],
            "paired_scenes": len(keys),
        }
        for metric, direction in METRICS.items():
            clean_values = np.asarray([float(clean[key][metric]) for key in keys])
            noisy_values = np.asarray([float(noisy[key][metric]) for key in keys])
            degradation = direction * (clean_values - noisy_values)
            low, high = bootstrap_ci(degradation, rng)
            row[f"clean_{metric}"] = float(clean_values.mean())
            row[f"noise_{metric}"] = float(noisy_values.mean())
            row[f"noise_degradation_{metric}"] = float(degradation.mean())
            row[f"noise_degradation_{metric}_ci95_low"] = float(low)
            row[f"noise_degradation_{metric}_ci95_high"] = float(high)
        aggregate.append(row)

    comparisons = []
    scene_keys = sorted((key[1], key[2]) for key in expected_keys if key[0] == "ray_gpis")
    for metric, direction in METRICS.items():
        full_values = np.asarray(
            [float(noisy[("ray_gpis",) + scene][metric]) for scene in scene_keys]
        )
        for planner in PLANNERS:
            if planner == "ray_gpis":
                continue
            ablated_values = np.asarray(
                [float(noisy[(planner,) + scene][metric]) for scene in scene_keys]
            )
            improvements = direction * (full_values - ablated_values)
            low, high = bootstrap_ci(improvements, rng)
            nonzero = improvements[np.abs(improvements) > 1e-12]
            p_value = (
                float(wilcoxon(nonzero, alternative="two-sided").pvalue)
                if len(nonzero)
                else 1.0
            )
            comparisons.append(
                {
                    "metric": metric,
                    "ablation": planner,
                    "display_name": DISPLAY_NAMES[planner],
                    "pairs": len(improvements),
                    "full_improvement": float(improvements.mean()),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "wins": int(np.sum(improvements > 1e-12)),
                    "ties": int(np.sum(np.abs(improvements) <= 1e-12)),
                    "losses": int(np.sum(improvements < -1e-12)),
                    "p_two_sided": p_value,
                    "p_holm": float("nan"),
                }
            )
    holm_adjust(comparisons)
    return aggregate, comparisons


def validate_identical_pose_noise(noise_cfg: Dict, noise_root: Path) -> Dict[str, object]:
    planners = list(noise_cfg["planners"]["enabled"])
    objects = list(noise_cfg["assets"]["objects"])
    pose_seeds = [int(value) for value in noise_cfg["experiment"]["initial_pose_seeds"]]
    steps = int(noise_cfg["environment"]["action_steps"])
    stress = noise_cfg["environment"]["stress"]
    target_rotation = float(stress["pose_outlier_deg"])
    target_translation = float(stress["translation_outlier_mm"])
    scheduled_steps = {int(value) for value in stress["pose_outlier_steps"]}
    errors = []
    devices = set()
    reference_errors = {}
    checked_states = 0

    for planner in planners:
        for object_name in objects:
            for pose_seed in pose_seeds:
                episode = noise_root / planner / object_name / f"pose_{pose_seed:03d}"
                summary_path = episode / "episode_summary.json"
                if not summary_path.is_file():
                    errors.append(f"missing summary: {summary_path}")
                    continue
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                devices.update(str(item.get("device", "unknown")) for item in summary["runtime_metrics"])
                for step in range(steps + 1):
                    step_dir = episode / f"step_{step:03d}"
                    gt_path = step_dir / "gt_pose.npy"
                    executed_path = step_dir / "executed_pose.npy"
                    if not gt_path.is_file() or not executed_path.is_file():
                        errors.append(f"missing pose pair: {step_dir}")
                        continue
                    gt_pose = np.load(gt_path)
                    executed_pose = np.load(executed_path)
                    rotation_error_matrix = executed_pose[:3, :3] @ gt_pose[:3, :3].T
                    translation_error = executed_pose[:3, 3] - gt_pose[:3, 3]
                    rotation_deg = float(np.degrees(Rotation.from_matrix(rotation_error_matrix).magnitude()))
                    translation_mm = float(1000.0 * np.linalg.norm(translation_error))
                    expected_rotation = target_rotation if step in scheduled_steps else 0.0
                    expected_translation = target_translation if step in scheduled_steps else 0.0
                    if abs(rotation_deg - expected_rotation) > 1e-6:
                        errors.append(
                            f"rotation error mismatch {planner}/{object_name}/{pose_seed}/{step}: "
                            f"{rotation_deg} vs {expected_rotation}"
                        )
                    if abs(translation_mm - expected_translation) > 1e-6:
                        errors.append(
                            f"translation error mismatch {planner}/{object_name}/{pose_seed}/{step}: "
                            f"{translation_mm} vs {expected_translation}"
                        )
                    key = (object_name, pose_seed, step)
                    pair = (rotation_error_matrix, translation_error)
                    if key not in reference_errors:
                        reference_errors[key] = pair
                    else:
                        reference_rotation, reference_translation = reference_errors[key]
                        if not np.allclose(rotation_error_matrix, reference_rotation, atol=1e-9):
                            errors.append(f"unpaired rotation noise: {planner}/{key}")
                        if not np.allclose(translation_error, reference_translation, atol=1e-9):
                            errors.append(f"unpaired translation noise: {planner}/{key}")
                    checked_states += 1

    expected_episodes = len(planners) * len(objects) * len(pose_seeds)
    return {
        "valid": not errors and devices == {"cuda"},
        "expected_episodes": expected_episodes,
        "checked_states": checked_states,
        "paired_noise_states": len(reference_errors),
        "target_rotation_deg": target_rotation,
        "target_translation_mm": target_translation,
        "devices": sorted(devices),
        "errors": errors,
    }


def write_markdown(root: Path, aggregate, comparisons, validation):
    lines = [
        "# Paired 6D pose-noise planner ablation",
        "",
        (
            "All variants use the same deterministic %.1f deg / %.1f mm tracking "
            "perturbation at each post-action state. Action execution and RGB-D are unchanged."
            % (validation["target_rotation_deg"], validation["target_translation_mm"])
        ),
        "",
        "| Variant | Clean F-AUC | Pose-noise F-AUC | F-AUC degradation | Noise Corr. | Noise Regret |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| {display_name} | {clean_f@5_auc:.4f} | {noise_f@5_auc:.4f} | "
            "{noise_degradation_f@5_auc:+.4f} | {noise_score_gain_correlation:.4f} | "
            "{noise_oracle_regret_f@5:.4f} |".format(**row)
        )
    lines += [
        "",
        "## Full Ray-GPIS paired improvements under pose noise",
        "",
        "| Metric | Ablation | Full improvement (95% CI) | W/T/L | p (Holm) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            "| {metric} | {display_name} | {full_improvement:+.6f} "
            "[{ci95_low:+.6f}, {ci95_high:+.6f}] | {wins}/{ties}/{losses} | "
            "{p_holm:.3g} |".format(**row)
        )
    lines += [
        "",
        "## Validation",
        "",
        f"- Valid: `{validation['valid']}`",
        f"- Episodes: `{validation['expected_episodes']}`",
        f"- Checked states: `{validation['checked_states']}`",
        f"- Devices: `{', '.join(validation['devices'])}`",
    ]
    (root / "pose_noise_ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-config", default="rebuttal/configs/ablation.yaml")
    parser.add_argument("--noise-config", default="rebuttal/configs/ablation_pose_noise.yaml")
    args = parser.parse_args()
    clean_cfg = load_config(args.clean_config)
    noise_cfg = load_config(args.noise_config)
    clean_root = resolved_path(clean_cfg["experiment"]["output_dir"])
    noise_root = resolved_path(noise_cfg["experiment"]["output_dir"])
    clean_rows = read_csv(clean_root / "all_metrics_paired.csv")
    noisy_rows = read_csv(noise_root / "all_metrics_paired.csv")
    aggregate, comparisons = summarize_metrics(
        clean_rows,
        noisy_rows,
        np.random.RandomState(int(noise_cfg["seed"])),
    )
    validation = validate_identical_pose_noise(noise_cfg, noise_root)
    write_csv(noise_root / "pose_noise_ablation_aggregate.csv", aggregate)
    write_csv(noise_root / "pose_noise_full_comparisons.csv", comparisons)
    write_json(noise_root / "pose_noise_pairing_validation.json", validation)
    write_markdown(noise_root, aggregate, comparisons, validation)
    print(noise_root / "pose_noise_ablation_summary.md")
    if not validation["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
