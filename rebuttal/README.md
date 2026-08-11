# Unified Ray-GPIS Rebuttal Benchmark

This folder implements the fixed-camera, kinematic planner benchmark specified for the rebuttal. Every method receives the same initial RGB-D observation, object mask, object pose, 256 Fibonacci candidate views, `{-x, -y, +z}` action primitives, point-cloud fusion, evaluator, and five-action budget. Each formal episode therefore contains exactly six acquired images: one initial image and five planner-selected images. Only the planner changes.

## Code layout

Reusable benchmark logic is under `benchmark/`, downstream task logic is under
`downstream/`, and command-line entry points are grouped by stage under
`scripts/`. Run those entry points as modules from the repository root; see
[`scripts/README.md`](scripts/README.md) for the category map.

## Current renderer backend

The minimal kinematic environment uses deterministic Open3D CPU ray casting. It has the fixed-camera and direct-pose-update semantics required by the protocol. The renderer is isolated in `benchmark/environment.py`; planner, fusion, action mapping, evaluator, counterfactual rollout, and output formats do not depend on the renderer.

Final timing uses the isolated `robosyn_gpu` environment on an RTX 4090 D. Ray-GPIS's GPyTorch exact GP, adapted ER-GPIS, and ActNeRF's radiance-field ensemble run on CUDA; Fixed, Pose-Novelty, and PB-NBV are native CPU planners measured on the same machine. Timers call `torch.cuda.synchronize()` at representation-update and candidate-scoring boundaries. The exact stack is stored in each formal result root's `gpu_environment.json`.

The ActNeRF adapter uses an object-centric NeRF ensemble with volume rendering, different Xavier initializations, warm starts, segmented RGB supervision, mean-opacity ROI filtering, and ensemble RGB variance. The formal config uses five compact PyTorch radiance fields. This preserves the core ActNeRF planner criterion but is not the original Instant-NGP implementation.

## Assets and action calibration

The six existing scanned GT objects are reconstructed, normalized to a 0.12 m bounding-box diagonal, repaired with a solid voxel isosurface when required, and validated as watertight. A closed concave bowl and a thin irregular object are added, giving eight objects. Generated meshes are under `assets/objects/` and the original assets are never modified.

The configured action magnitudes come from the median rotation at frame 180 (about 6 s at 30 Hz) over six axis-labelled real logs per action:

- `minus_x`: 106.48 degrees
- `minus_y`: 55.76 degrees
- `plus_z`: 82.38 degrees

The full evidence is saved in `assets/action_calibration.json` and can be regenerated with:

```bash
conda run -n robosyn python -m rebuttal.scripts.preparation.calibrate_actions
```

## Commands

Setup and unit tests can run in `robosyn`; final GPU baselines use `robosyn_gpu`:

```bash
conda run -n robosyn python -m rebuttal.scripts.preparation.prepare_assets --config rebuttal/configs/base.yaml
conda run -n robosyn python -m unittest discover -s rebuttal/tests -v

# Final strict-six-image suite: 120 paired scenes, 960 stored runs.
# Fixed/Pose/PB/ER/Ray use one run per scene; ActNeRF uses three seeds.
conda run --no-capture-output -n robosyn_gpu python -m rebuttal.scripts.runners.run_baselines \
  --config rebuttal/configs/formal_120_sixview_gpu.yaml

conda run -n robosyn_gpu python -m rebuttal.scripts.evaluation.evaluate_visibility_coverage \
  --config rebuttal/configs/formal_120_sixview_gpu.yaml --overwrite
conda run -n robosyn_gpu python -m rebuttal.scripts.evaluation.evaluate_pose_viewpoint_coverage \
  --config rebuttal/configs/formal_120_sixview_gpu.yaml
conda run -n robosyn_gpu python -m rebuttal.scripts.summarization.summarize_all_metrics \
  --config rebuttal/configs/formal_120_sixview_gpu.yaml
conda run -n robosyn_gpu python -m rebuttal.scripts.evaluation.paired_statistics_all \
  --config rebuttal/configs/formal_120_sixview_gpu.yaml
conda run -n robosyn_gpu python -m rebuttal.scripts.validation.validate_results \
  --config rebuttal/configs/formal_120_sixview_gpu.yaml
conda run -n robosyn_gpu python -m rebuttal.scripts.validation.validate_gpu_runtime \
  --config rebuttal/configs/formal_120_sixview_gpu.yaml

# Module-level online runtime audit. This combines synchronized SAM2-tiny
# inference on 120 real frames with eight saved real-deployment timing logs.
conda run --no-capture-output -n robosyn_gpu python -m rebuttal.scripts.evaluation.benchmark_pipeline_runtime \
  --segmentation-samples 120 --warmup 10

# Clean component ablations
conda run --no-capture-output -n robosyn_gpu python -m rebuttal.scripts.runners.run_ablations \
  --config rebuttal/configs/ablation.yaml
conda run -n robosyn_gpu python -m rebuttal.scripts.evaluation.evaluate_visibility_coverage \
  --config rebuttal/configs/ablation.yaml --overwrite
conda run -n robosyn_gpu python -m rebuttal.scripts.summarization.summarize_all_metrics \
  --config rebuttal/configs/ablation.yaml
conda run -n robosyn_gpu python -m rebuttal.scripts.evaluation.paired_statistics_all \
  --config rebuttal/configs/ablation.yaml
conda run -n robosyn_gpu python -m rebuttal.scripts.validation.validate_results \
  --config rebuttal/configs/ablation.yaml
conda run -n robosyn_gpu python -m rebuttal.scripts.validation.validate_gpu_runtime \
  --config rebuttal/configs/ablation.yaml
conda run -n robosyn_gpu python -m rebuttal.scripts.validation.validate_ablation_consistency

# Strict-six-image component ablations under moderate empirical/noise corruption
conda run --no-capture-output -n robosyn_gpu python -m rebuttal.scripts.runners.run_ablations \
  --config rebuttal/configs/ablation_realistic.yaml
conda run -n robosyn_gpu python -m rebuttal.scripts.evaluation.evaluate_visibility_coverage \
  --config rebuttal/configs/ablation_realistic.yaml --overwrite
conda run -n robosyn_gpu python -m rebuttal.scripts.summarization.summarize_all_metrics \
  --config rebuttal/configs/ablation_realistic.yaml
conda run -n robosyn_gpu python -m rebuttal.scripts.evaluation.paired_statistics_all \
  --config rebuttal/configs/ablation_realistic.yaml
conda run -n robosyn_gpu python -m rebuttal.scripts.summarization.summarize_noise_exposure \
  --config rebuttal/configs/ablation_realistic.yaml
conda run -n robosyn_gpu python -m rebuttal.scripts.validation.validate_results \
  --config rebuttal/configs/ablation_realistic.yaml
conda run -n robosyn_gpu python -m rebuttal.scripts.validation.validate_gpu_runtime \
  --config rebuttal/configs/ablation_realistic.yaml

# Persistent action-mismatch stress: under-rotation/stall and grip slip.
# The first sweep directly compares Pose-Novelty and Full Ray-GPIS; the
# second uses the identical stress schedule for all five Ray-GPIS variants.
conda run --no-capture-output -n robosyn_gpu python -m rebuttal.scripts.runners.run_baselines \
  --config rebuttal/configs/baseline_slip_pose_ray.yaml
conda run --no-capture-output -n robosyn_gpu python -m rebuttal.scripts.runners.run_ablations \
  --config rebuttal/configs/ablation_slip_robustness.yaml

for config in baseline_slip_pose_ray ablation_slip_robustness; do
  conda run -n robosyn_gpu python -m rebuttal.scripts.evaluation.evaluate_visibility_coverage \
    --config rebuttal/configs/${config}.yaml --overwrite
  conda run -n robosyn_gpu python -m rebuttal.scripts.summarization.summarize_all_metrics \
    --config rebuttal/configs/${config}.yaml
  conda run -n robosyn_gpu python -m rebuttal.scripts.evaluation.paired_statistics_all \
    --config rebuttal/configs/${config}.yaml
  conda run -n robosyn_gpu python -m rebuttal.scripts.summarization.summarize_action_mismatch \
    --config rebuttal/configs/${config}.yaml
done

# Fresh-seed special-case active stress ablations
for name in sparse ghost hole; do
  conda run --no-capture-output -n robosyn_gpu python -m rebuttal.scripts.runners.run_ablations \
    --config rebuttal/configs/stress_${name}.yaml
  conda run -n robosyn_gpu python -m rebuttal.scripts.evaluation.evaluate_visibility_coverage \
    --config rebuttal/configs/stress_${name}.yaml
  conda run -n robosyn_gpu python -m rebuttal.scripts.summarization.summarize_all_metrics \
    --config rebuttal/configs/stress_${name}.yaml
done

# Shared reference trajectories and Full/Pointwise pose-stability diagnostic
conda run --no-capture-output -n robosyn_gpu python -m rebuttal.scripts.runners.run_ablations \
  --config rebuttal/configs/stress_pose_reference.yaml
conda run --no-capture-output -n robosyn_gpu python -m rebuttal.scripts.evaluation.evaluate_pose_stability --repeats 20
conda run -n robosyn_gpu python -m rebuttal.scripts.summarization.summarize_stress_ablation

# Visited-view local depth-registration gap: Pose/Hit/Full and exhaustive
# five-variant component rerun under identical shared action branches.
conda run --no-capture-output -n robosyn_gpu python -m rebuttal.scripts.runners.run_continuous_one_step \
  --config rebuttal/configs/visited_registration_gap.yaml
conda run --no-capture-output -n robosyn_gpu python -m rebuttal.scripts.runners.run_continuous_one_step \
  --config rebuttal/configs/visited_registration_gap_ablation.yaml
conda run -n robosyn_gpu python -m rebuttal.scripts.summarization.summarize_visited_gap

# ER-GPIS versus Ray-GPIS on the identical visited-view gap scenes.
conda run --no-capture-output -n robosyn_gpu python -m rebuttal.scripts.runners.run_continuous_one_step \
  --config rebuttal/configs/visited_registration_gap_er_ray.yaml
conda run -n robosyn_gpu python -m rebuttal.scripts.summarization.summarize_er_gap

# All Table-1 planners under a transient 70% depth gap plus paired
# 6 deg / 3 mm pose error; later views remain available for revisiting.
conda run --no-capture-output -n robosyn_gpu python -m rebuttal.scripts.runners.run_continuous_one_step \
  --config rebuttal/configs/baseline_joint_pose_depth_gap.yaml
conda run -n robosyn_gpu python -m rebuttal.scripts.summarization.summarize_joint_pose_depth_gap \
  --config rebuttal/configs/baseline_joint_pose_depth_gap.yaml

# PB-NBV scale/partition sanity check
conda run --no-capture-output -n robosyn_gpu python -m rebuttal.scripts.runners.run_baselines \
  --config rebuttal/configs/pb_sanity_120_gpu.yaml

```

## Completed formal results

The final 120-pair strict-six-image GPU suite is complete and validated for
Fixed, Pose-Novelty, adapted PB-NBV, adapted ER-GPIS, Ray-GPIS, and adapted
ActNeRF. The main table, complete
categorized metric summary, paired statistics, per-object table, and plots are
under `results/formal_120_sixview_gpu/`; see `RESULTS.md` and
`results/formal_120_sixview_gpu/all_metrics_summary.md`. The retained
`results/pb_sanity_120_gpu/` directory contains the reportable PB-NBV
scale/partition sweep. Preliminary, debug, CPU, and incomplete result trees
have been removed. The 600-run clean component study is complete under
`results/ablations/`, including visibility coverage, paired statistics, plots,
GPU/runtime audits, and an exact consistency check against the formal full
Ray-GPIS arm. The fixed moderate-error component study is also complete under
`results/ablations_realistic/`. The fresh-seed special-case suite is complete
under `results/stress_sparse/`, `stress_ghost/`, `stress_hole/`, and
`stress_pose_stability/`; its joint corrected table is in
`results/stress_ablation_summary/`.
The persistent execution-mismatch Pose/Ray comparison is complete under
`results/baseline_slip_pose_ray/`, and its five-variant component sweep is
complete under `results/ablation_slip_robustness/`. Both retain the same
strict-six-image, 120-paired-scene protocol; their complete statistical and
CUDA audits are stored in the respective result roots.
The 64-scene visited-view registration-gap comparison and its exhaustive
same-scene component rerun are complete under
`results/visited_registration_gap/` and
`results/visited_registration_gap_ablation/`; paired intervals and audits are
in `results/visited_registration_gap/visited_gap_summary.md`.
The same-scene ER-GPIS/Ray-GPIS check is complete under
`results/visited_registration_gap_er_ray/`; it does not support a Ray-over-ER
claim and is retained as a negative result rather than used in the rebuttal.
The four-planner joint pose/depth test is complete under
`results/baseline_joint_pose_depth_gap/`; its audited global planner metrics,
missing-region recovery, and corrected paired tests are in
`joint_gap_summary.md`.
The online runtime audit is stored under `results/pipeline_runtime/`. It
contains all 120 synchronized segmentation samples, the eight source-sequence
summaries, hardware/protocol metadata, and the compact module-level table used
in the rebuttal.
The sequential downstream evaluation is complete under
`results/downstream_ycb_task/`: eight reconstruction sources, 10 YCB objects,
240 reconstructed meshes, and 2,400 perturbed place-and-regrasp trials. Its
GT-oracle sanity check is under `results/downstream_ycb_task_oracle/`. The six
existing real AURORA meshes are sourced exclusively from
`reconstruction/offline/result/offline_tracking` and evaluated separately
under `results/downstream_real_task/`. Protocol and method-specific environment
details are documented in `downstream/README.md`.

## Saved episode data

Each episode stores its resolved config and normalized GT mesh. Each `step_###/` contains `rgb.png`, `depth.npy`, `mask.png`, GT and executed poses, fused cloud, shared candidate directions, raw planner scores, selected NBV/action, metrics, and all three counterfactual action gains. `episode_summary.json` contains F@5/Recall/Coverage AUC, final metrics, score-gain Spearman correlation, oracle regret/accuracy, and timing.

The `rebuttal.scripts.summarization.summarize_all_metrics` module produces raw
per-seed data, paired-episode CSVs, and
categorized Markdown/CSV tables with 95% confidence intervals. ActNeRF's three
initialization seeds are averaged inside each `(object, initial pose)` pair
before method aggregation, so all methods have the same 120 paired units.

The realistic environment supports pose jitter/drift/outliers, empirical
action-axis and magnitude residuals, depth/mask corruption, and dynamic
palm/finger occlusion. The reportable noisy component study still uses exactly
one initial image plus five active images; no intermediate video frames enter
its reconstruction.

The persistent action-mismatch profile additionally models an object that is
difficult to rotate and can slip inside the grasp. A non-stalled primitive
reaches 55--85% of its requested angle; a 25% stall event reaches 8--30%.
Grip slip occurs with 22% probability, adds a 12--30 degree uncommanded
rotation, and rotates the effective primitive-axis frame by 15--30 degrees for
all later actions (capped at 50 degrees). These are declared simulation-stress
parameters, not an empirical fit. Event occurrence is deterministic for each
object/initial-pose/step and shared by paired planners, while action-dependent
trajectories are allowed to diverge naturally.

Stress configurations can additionally schedule deterministic step-specific
pose outliers, correlated depth dropout, and per-step occlusion fractions. The
reconstruction evaluator supports exact PyTorch3D CUDA nearest-neighbor
distances, and Ray-GPIS exposes its already-computed hit mask so Hit-only does
not repeat CPU ray tests.

The visited-view registration-gap stress records a shared prefix pose in pose
history while removing one deterministic contiguous depth region before
fusion. The corruption is disabled for the next decision, whose three action
branches are generated once and shared by every planner. This isolates the
case in which nominal viewpoint coverage and acquired reconstruction coverage
disagree.
