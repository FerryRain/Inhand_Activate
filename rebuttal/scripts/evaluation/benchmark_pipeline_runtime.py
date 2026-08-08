#!/usr/bin/env python3
"""Audit real-pipeline latency and benchmark SAM2 segmentation on saved frames."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "rebuttal/results/pipeline_runtime"
DEFAULT_SAM2_SOURCE = ROOT.parent / "PoseEstimation/sam2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_TRACKING_GLOB = (
    "reconstruction/offline/result/online_tracking/**/timing_stats.json"
)
DEFAULT_FRAME_ROOTS = (
    "Tracking/BundleTrack/results/xyz",
    "Tracking/BundleTrack/results/001",
    "Tracking/BundleTrack/results/002",
)


def normal_summary(values: Sequence[float]) -> Dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        raise ValueError("Cannot summarize an empty sample")
    ci95 = 0.0
    if data.size > 1:
        ci95 = 1.96 * float(data.std(ddof=1)) / np.sqrt(float(data.size))
    return {
        "mean": float(data.mean()),
        "ci95": float(ci95),
        "std": float(data.std(ddof=1)) if data.size > 1 else 0.0,
        "median": float(np.median(data)),
        "min": float(data.min()),
        "max": float(data.max()),
        "n": int(data.size),
    }


def load_real_tracking_logs(pattern: str) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    paths = sorted(ROOT.glob(pattern))
    if not paths:
        raise RuntimeError("No real-deployment timing logs match %s" % pattern)

    rows: List[Dict[str, object]] = []
    for path in paths:
        payload = json.loads(path.read_text())
        sums = payload.get("sums_sec", {})
        counts = payload.get("counts", {})
        if not counts.get("stage/tracking"):
            continue
        row: Dict[str, object] = {
            "path": str(path.relative_to(ROOT)),
            "tracking_calls": int(counts["stage/tracking"]),
            "tracking_ms": 1000.0
            * float(sums["stage/tracking"])
            / int(counts["stage/tracking"]),
        }
        for key, label in (
            ("active/reconstruct", "fusion_ms"),
            ("active/pick_axis", "action_mapping_ms"),
        ):
            if counts.get(key):
                row[label] = 1000.0 * float(sums[key]) / int(counts[key])
                row[label.replace("_ms", "_calls")] = int(counts[key])
        rows.append(row)

    if not rows:
        raise RuntimeError("Logs exist, but none contains stage/tracking")

    summary: Dict[str, object] = {
        "sequences": len(rows),
        "tracking_calls": int(sum(int(r["tracking_calls"]) for r in rows)),
        # Equal weight per real sequence avoids long runs dominating the interval.
        "tracking_ms_per_frame": normal_summary(
            [float(r["tracking_ms"]) for r in rows]
        ),
        "source_note": (
            "stage/tracking is the recorded end-to-end online tracker call and "
            "includes image acquisition, segmentation, BundleTrack request, and response"
        ),
    }
    for field, out_name in (
        ("fusion_ms", "fusion_ms_per_update"),
        ("action_mapping_ms", "action_mapping_ms_per_update"),
    ):
        values = [float(r[field]) for r in rows if field in r]
        if values:
            summary[out_name] = normal_summary(values)
            call_field = field.replace("_ms", "_calls")
            summary[out_name]["total_calls"] = int(
                sum(int(r.get(call_field, 0)) for r in rows)
            )
    return rows, summary


def mask_box(mask: np.ndarray, pad: int = 14) -> List[int]:
    rows, cols = np.nonzero(mask > 0)
    if rows.size == 0:
        raise ValueError("Empty prompt mask")
    height, width = mask.shape[:2]
    return [
        max(0, int(cols.min()) - pad),
        max(0, int(rows.min()) - pad),
        min(width - 1, int(cols.max()) + pad),
        min(height - 1, int(rows.max()) + pad),
    ]


def discover_frame_pairs(frame_roots: Iterable[str]) -> List[Tuple[Path, Path, str]]:
    pairs: List[Tuple[Path, Path, str]] = []
    for root_value in frame_roots:
        root = ROOT / root_value
        rgb_dir = root / "keyframes/rgb_full"
        mask_dir = root / "keyframes/mask"
        for rgb_path in sorted(list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png"))):
            candidates = [mask_dir / (rgb_path.stem + suffix) for suffix in (".png", ".jpg")]
            mask_path = next((p for p in candidates if p.exists()), None)
            if mask_path is not None:
                pairs.append((rgb_path, mask_path, str(root.relative_to(ROOT))))
    if not pairs:
        raise RuntimeError("No matched RGB/mask keyframes were found")
    return pairs


def evenly_sample(items: Sequence[Tuple[Path, Path, str]], count: int) -> List[Tuple[Path, Path, str]]:
    if count <= 0 or count >= len(items):
        return list(items)
    indices = np.linspace(0, len(items) - 1, num=count, dtype=np.int64)
    return [items[int(i)] for i in indices]


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_segmentation(
    frame_roots: Sequence[str],
    checkpoint: Path,
    config: str,
    samples: int,
    warmup: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the segmentation runtime benchmark")
    if DEFAULT_SAM2_SOURCE.exists() and str(DEFAULT_SAM2_SOURCE) not in sys.path:
        sys.path.insert(0, str(DEFAULT_SAM2_SOURCE))
    from Tracking.sam2_class import SamSegmenter

    pairs = evenly_sample(discover_frame_pairs(frame_roots), samples)
    segmenter = SamSegmenter(
        checkpoint=str(checkpoint),
        model_cfg=config,
        device="cuda",
        use_amp=True,
    )

    first_rgb = cv2.imread(str(pairs[0][0]), cv2.IMREAD_COLOR)
    first_mask = cv2.imread(str(pairs[0][1]), cv2.IMREAD_GRAYSCALE)
    if first_rgb is None or first_mask is None:
        raise RuntimeError("Failed to read warm-up input")
    first_box = mask_box(first_mask)
    for _ in range(max(0, warmup)):
        segmenter.segment_from_box(first_rgb, first_box)
    synchronize_cuda()

    rows: List[Dict[str, object]] = []
    for rgb_path, mask_path, dataset in pairs:
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if rgb is None or mask is None:
            raise RuntimeError("Failed to read %s or %s" % (rgb_path, mask_path))
        box = mask_box(mask)
        synchronize_cuda()
        start = time.perf_counter()
        prediction = segmenter.segment_from_box(rgb, box)
        synchronize_cuda()
        elapsed_ms = 1000.0 * (time.perf_counter() - start)
        pred = prediction.astype(bool)
        target = mask > 0
        union = int(np.logical_or(pred, target).sum())
        iou = float(np.logical_and(pred, target).sum() / union) if union else 0.0
        rows.append(
            {
                "dataset": dataset,
                "frame": rgb_path.stem,
                "height": int(rgb.shape[0]),
                "width": int(rgb.shape[1]),
                "latency_ms": elapsed_ms,
                "predicted_pixels": int(pred.sum()),
                "prompt_mask_iou": iou,
            }
        )

    summary = {
        "model": "SAM2.1 Hiera Tiny",
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "config": config,
        "device": torch.cuda.get_device_name(0),
        "amp": True,
        "warmup": int(warmup),
        "latency_ms_per_frame": normal_summary([float(r["latency_ms"]) for r in rows]),
        "nonempty_predictions": int(sum(int(r["predicted_pixels"]) > 0 for r in rows)),
        "prompt_mask_iou": normal_summary([float(r["prompt_mask_iou"]) for r in rows]),
        "frame_roots": list(frame_roots),
    }
    return rows, summary


def load_ray_runtime(path: Path) -> Dict[str, object]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    row = next((r for r in rows if r["planner"] == "ray_gpis"), None)
    if row is None:
        raise RuntimeError("ray_gpis row missing from %s" % path)
    return {
        "representation_update_ms": {
            "mean": 1000.0 * float(row["representation_update_s"]),
            "ci95": 1000.0 * float(row["representation_update_s_ci95"]),
            "n": int(row["representation_update_s_n"]),
        },
        "candidate_scoring_ms": {
            "mean": 1000.0 * float(row["candidate_scoring_s"]),
            "ci95": 1000.0 * float(row["candidate_scoring_s_ci95"]),
            "n": int(row["candidate_scoring_s_n"]),
        },
        "total_planning_ms": {
            "mean": 1000.0 * float(row["planning_time_s"]),
            "ci95": 1000.0 * float(row["planning_time_s_ci95"]),
            "n": int(row["planning_time_s_n"]),
        },
        "source": str(path.relative_to(ROOT)),
    }


def metric_text(metric: Dict[str, object], digits: int = 1) -> str:
    return ("%.*f $\\pm$ %.*f" % (digits, metric["mean"], digits, metric["ci95"]))


def write_markdown(payload: Dict[str, object], path: Path) -> None:
    tracking = payload["real_online_logs"]
    ray = payload["ray_gpis"]
    segmentation = payload.get("segmentation")
    lines = [
        "# Online pipeline runtime audit",
        "",
        "All values are milliseconds. Intervals are 95% normal intervals; real-pipeline "
        "logs are summarized with equal weight per sequence.",
        "",
        "| Module | Runtime | Unit | Evidence |",
        "|---|---:|---|---|",
    ]
    if segmentation is not None:
        metric = segmentation["latency_ms_per_frame"]
        lines.append(
            "| SAM2-tiny segmentation | %s | per frame | %d synchronized CUDA frames |"
            % (metric_text(metric), metric["n"])
        )
    metric = tracking["tracking_ms_per_frame"]
    lines.append(
        "| End-to-end perception/tracking | %s | per frame | %d real sequences, %d calls |"
        % (metric_text(metric), tracking["sequences"], tracking["tracking_calls"])
    )
    metric = tracking["fusion_ms_per_update"]
    lines.append(
        "| Point-cloud fusion | %s | per update | %d real sequences, %d calls |"
        % (metric_text(metric), metric["n"], metric["total_calls"])
    )
    metric = ray["representation_update_ms"]
    lines.append(
        "| GP/representation update | %s | per planning step | %d paired scenes |"
        % (metric_text(metric), metric["n"])
    )
    metric = ray["candidate_scoring_ms"]
    lines.append(
        "| Candidate scoring | %s | per planning step | %d paired scenes |"
        % (metric_text(metric, digits=3), metric["n"])
    )
    metric = tracking["action_mapping_ms_per_update"]
    lines.append(
        "| NBV-to-action mapping | %s | per update | %d real sequences, %d calls |"
        % (metric_text(metric, digits=3), metric["n"], metric["total_calls"])
    )
    lines.extend(
        [
            "",
            "The end-to-end perception/tracking row already includes segmentation and is "
            "therefore not additive with the standalone segmentation row. Fusion is invoked "
            "at planning boundaries. Model loading, visualization, disk export, and the 6 s "
            "manipulation primitive are excluded.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tracking-glob", default=DEFAULT_TRACKING_GLOB)
    parser.add_argument("--frame-roots", nargs="+", default=list(DEFAULT_FRAME_ROOTS))
    parser.add_argument("--segmentation-samples", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "sam_model/sam2.1_hiera_tiny.pt"
    )
    parser.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--skip-segmentation", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    tracking_rows, tracking_summary = load_real_tracking_logs(args.tracking_glob)
    ray_runtime = load_ray_runtime(
        ROOT / "rebuttal/results/formal_120_sixview_gpu/runtime_metrics.csv"
    )
    payload: Dict[str, object] = {
        "protocol": {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "tracking_log_glob": args.tracking_glob,
        },
        "real_online_logs": tracking_summary,
        "ray_gpis": ray_runtime,
    }

    segmentation_rows: List[Dict[str, object]] = []
    if not args.skip_segmentation:
        segmentation_rows, segmentation_summary = benchmark_segmentation(
            frame_roots=args.frame_roots,
            checkpoint=args.checkpoint,
            config=args.sam_config,
            samples=args.segmentation_samples,
            warmup=args.warmup,
        )
        payload["segmentation"] = segmentation_summary

    (args.output / "pipeline_runtime.json").write_text(json.dumps(payload, indent=2))
    (args.output / "real_sequence_runtime.json").write_text(
        json.dumps(tracking_rows, indent=2)
    )
    if segmentation_rows:
        with (args.output / "segmentation_samples.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(segmentation_rows[0]))
            writer.writeheader()
            writer.writerows(segmentation_rows)
    write_markdown(payload, args.output / "pipeline_runtime_summary.md")
    print(args.output / "pipeline_runtime_summary.md")


if __name__ == "__main__":
    main()
