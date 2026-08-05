# Unified Ray-GPIS Rebuttal Benchmark

This folder implements the fixed-camera, kinematic planner benchmark specified for the rebuttal. Every method receives the same initial RGB-D observation, object mask, object pose, 256 Fibonacci candidate views, `{-x, -y, +z}` action primitives, point-cloud fusion, evaluator, and five-action budget. Each formal episode therefore contains exactly six acquired images: one initial image and five planner-selected images. Only the planner changes.

## Current renderer backend

The minimal kinematic environment uses deterministic Open3D CPU ray casting. It has the fixed-camera and direct-pose-update semantics required by the protocol. The renderer is isolated in `benchmark/environment.py`; planner, fusion, action mapping, evaluator, counterfactual rollout, and output formats do not depend on the renderer.

Final timing uses the isolated `robosyn_gpu` environment on an RTX 4090 D. Ray-GPIS's GPyTorch exact GP and ActNeRF's radiance-field ensemble run on CUDA; Fixed and PB-NBV are native CPU planners measured on the same machine. Timers call `torch.cuda.synchronize()` at representation-update and candidate-scoring boundaries. The exact stack is stored in `results/formal_120_sixview_gpu/gpu_environment.json`.

The ActNeRF adapter uses an object-centric NeRF ensemble with volume rendering, different Xavier initializations, warm starts, segmented RGB supervision, mean-opacity ROI filtering, and ensemble RGB variance. The formal config uses five compact PyTorch radiance fields. This preserves the core ActNeRF planner criterion but is not the original Instant-NGP implementation.

## Assets and action calibration

The six existing scanned GT objects are reconstructed, normalized to a 0.12 m bounding-box diagonal, repaired with a solid voxel isosurface when required, and validated as watertight. A closed concave bowl and a thin irregular object are added, giving eight objects. Generated meshes are under `assets/objects/` and the original assets are never modified.

The configured action magnitudes come from the median rotation at frame 180 (about 6 s at 30 Hz) over six axis-labelled real logs per action:

- `minus_x`: 106.48 degrees
- `minus_y`: 55.76 degrees
- `plus_z`: 82.38 degrees

The full evidence is saved in `assets/action_calibration.json` and can be regenerated with:

```bash
conda run -n robosyn python rebuttal/calibrate_actions.py
```

## Commands

Setup and unit tests can run in `robosyn`; final GPU baselines use `robosyn_gpu`:

```bash
conda run -n robosyn python rebuttal/prepare_assets.py --config rebuttal/configs/base.yaml
conda run -n robosyn python -m unittest discover -s rebuttal/tests -v

# Final strict-six-image suite: 120 paired scenes, 720 stored runs.
# Fixed/PB-NBV/Ray-GPIS use one run per scene; ActNeRF uses three seeds.
conda run --no-capture-output -n robosyn_gpu python rebuttal/run_baselines.py \
  --config rebuttal/configs/formal_120_sixview_gpu.yaml

conda run -n robosyn_gpu python rebuttal/evaluate_visibility_coverage.py \
  --config rebuttal/configs/formal_120_sixview_gpu.yaml --overwrite
conda run -n robosyn_gpu python rebuttal/summarize_all_metrics.py \
  --config rebuttal/configs/formal_120_sixview_gpu.yaml
conda run -n robosyn_gpu python rebuttal/paired_statistics_all.py \
  --config rebuttal/configs/formal_120_sixview_gpu.yaml
conda run -n robosyn_gpu python rebuttal/validate_results.py \
  --config rebuttal/configs/formal_120_sixview_gpu.yaml
conda run -n robosyn_gpu python rebuttal/validate_gpu_runtime.py \
  --config rebuttal/configs/formal_120_sixview_gpu.yaml

# Clean component ablations
conda run --no-capture-output -n robosyn_gpu python rebuttal/run_ablations.py \
  --config rebuttal/configs/ablation.yaml

# PB-NBV scale/partition sanity check
conda run --no-capture-output -n robosyn_gpu python rebuttal/run_baselines.py \
  --config rebuttal/configs/pb_sanity_120_gpu.yaml

# Level-B filtering robustness
conda run --no-capture-output -n robosyn_gpu python rebuttal/run_robustness.py \
  --config rebuttal/configs/robustness.yaml
conda run -n robosyn_gpu python rebuttal/summarize_robustness.py \
  --config rebuttal/configs/robustness.yaml
```

## Completed formal baseline results

The final 120-pair strict-six-image GPU suite is complete and validated. The main table, complete
categorized metric summary, paired statistics, per-object table, and plots are
under `results/formal_120_sixview_gpu/`; see `RESULTS.md` and
`results/formal_120_sixview_gpu/all_metrics_summary.md`. The retained
`results/pb_sanity_120_gpu/` directory contains the reportable PB-NBV
scale/partition sweep. Preliminary, debug, CPU, and incomplete result trees
have been removed. Formal ablation and Level-B robustness sweeps have not yet
been generated.

## Saved episode data

Each episode stores its resolved config and normalized GT mesh. Each `step_###/` contains `rgb.png`, `depth.npy`, `mask.png`, GT and executed poses, fused cloud, shared candidate directions, raw planner scores, selected NBV/action, metrics, and all three counterfactual action gains. `episode_summary.json` contains F@5/Recall/Coverage AUC, final metrics, score-gain Spearman correlation, oracle regret/accuracy, and timing.

`summarize_all_metrics.py` produces raw per-seed data, paired-episode CSVs, and
categorized Markdown/CSV tables with 95% confidence intervals. ActNeRF's three
initialization seeds are averaged inside each `(object, initial pose)` pair
before method aggregation, so all methods have the same 120 paired units.

The Level-B runner renders five observations per action and supports pose jitter/drift/outliers, action-axis and magnitude errors, mask morphology, and dynamic partial occlusion. The three fusion modes are `all`, `motion_only`, and `full`; the full mode enforces minimum view change, visibility, and injected pose-error thresholds. It additionally reports surface thickness, outlier ratio, accepted/rejected frames, and action switching frequency.
