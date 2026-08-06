# Final experiment summary

## 1. Shared controlled setting

- 8 objects and 15 shared initial orientations: 120 paired scenes.
- Exactly 6 reconstruction states per episode: 1 shared initial RGB-D image
  followed by 5 images selected by the planner.
- Fixed 400 x 225 RGB-D camera with scaled Azure Kinect intrinsics.
- 256 shared Fibonacci-sphere candidate directions.
- Three executable primitives: `minus_x`, `minus_y`, and `plus_z`, using
  calibrated 6-second median rotations of 106.48, 55.76, and 82.38 degrees.
- All methods share observations, poses, NBV-to-action mapping, action
  execution model, point-cloud fusion, evaluator, and three-action
  counterfactual evaluation protocol. Only the planner changes.
- Final geometry: Recall@5 mm, F-score@5 mm, and Chamfer distance.
- Exploration: Recall/F-score AUC and independently ray-traced visible-area
  AUC over the six states.
- Decision policy: selected F-score gain, three-action score--gain Spearman
  correlation, oracle regret, and oracle action accuracy.
- Runtime is representation update plus candidate scoring. Ray-GPIS,
  ER-GPIS, and ActNeRF are synchronized CUDA measurements on an RTX 4090 D;
  Ray-GPIS uses PyTorch 2.4.1+cu121 and GPyTorch 1.11.

## 2. Planner implementations

| Method | Planner flow used in the shared benchmark |
|---|---|
| Fixed | Open-loop `+z,-x,-y,+z,-x`; no reconstruction feedback. |
| Pose-Novelty | Predicts each primitive's next object-frame viewing direction and maximizes its minimum spherical distance to pose history; never reads the cloud. |
| Adapted ActNeRF | Five compact NeRFs with different initializations; candidate score is ensemble RGB rendering variance. Ensemble updating and rendering are included in time. |
| Adapted PB-NBV | Voxel occupied/frontier classification, ellipsoid projection score, and global partition; best predeclared scale is bbox diagonal / 70. |
| Adapted ER-GPIS | Global E-GPIS uncertainty core with the common visual action mapper; tactile local-motion/recovery controllers are excluded. |
| Full Ray-GPIS | Hit/miss rays, interpolated miss anchors, angular--radial receptive-field uncertainty, GP posterior variance, and novelty weighting. |

## 3. Formal 120-scene baselines

### Final geometry and exploration

| Method | Recall@5 | Final F@5 | Chamfer (mm) | Recall AUC | F-AUC | Visible-area AUC |
|---|---:|---:|---:|---:|---:|---:|
| Fixed | 0.8881±0.0082 | 0.9401±0.0046 | 3.027±0.079 | 0.6780±0.0078 | 0.7948±0.0061 | 0.1868±0.0366 |
| Pose-Novelty | **0.9891±0.0027** | **0.9945±0.0014** | **2.176±0.031** | **0.8553±0.0066** | **0.9090±0.0043** | **0.2201±0.0476** |
| Adapted PB-NBV | 0.7597±0.0338 | 0.8496±0.0234 | 5.344±0.603 | 0.6188±0.0258 | 0.7475±0.0193 | 0.1749±0.0359 |
| Adapted ER-GPIS | 0.9807±0.0065 | 0.9899±0.0037 | 2.272±0.053 | 0.8459±0.0086 | 0.9033±0.0055 | 0.2173±0.0473 |
| Adapted ActNeRF | 0.9357±0.0089 | 0.9651±0.0050 | 2.677±0.090 | 0.7847±0.0095 | 0.8665±0.0062 | 0.2059±0.0433 |
| Full Ray-GPIS | 0.9682±0.0157 | 0.9814±0.0100 | 2.347±0.120 | 0.8397±0.0136 | 0.8987±0.0091 | 0.2135±0.0473 |

### Active decision policy and runtime

| Method | Selected gain/action | Corr. | Regret | Oracle accuracy | Time/step (s) |
|---|---:|---:|---:|---:|---:|
| Fixed | 0.0709±0.0018 | -- | 0.0806±0.0025 | 29.0±1.8% | 0.000 |
| Pose-Novelty | **0.0817±0.0020** | **0.6186±0.0294** | **0.0025±0.0006** | **71.0±2.0%** | 0.000272 |
| Adapted PB-NBV | 0.0528±0.0043 | -0.3563±0.0731 | 0.1076±0.0124 | 19.2±3.8% | 0.391±0.024 |
| Adapted ER-GPIS | 0.0808±0.0020 | 0.3439±0.0587 | 0.0060±0.0022 | 63.2±4.0% | 0.097±0.002 |
| Adapted ActNeRF | 0.0758±0.0019 | 0.1619±0.0337 | 0.0317±0.0029 | 42.3±2.4% | 2.238±0.018 |
| Full Ray-GPIS | 0.0791±0.0026 | 0.4722±0.0667 | 0.0088±0.0051 | 68.0±4.1% | 0.261±0.024 |

Ray-GPIS versus ActNeRF: F-AUC +0.03227, 95% CI [0.02260, 0.04051],
106/1/13 wins/ties/losses, Holm p=2.62e-14; correlation +0.31031; regret
-0.02294; runtime -88.3%. Ray-GPIS versus ER-GPIS: F-AUC is tied, while
correlation improves by 0.12835, CI [0.04496, 0.20898], Holm p=9.92e-4.
Pose-Novelty is the strongest clean-setting planner, so the clean experiment
does not support universal Ray-over-Pose superiority.

PB-NBV's best tested scale is reported: F-AUC is 0.6939 for d/30, 0.7212 for
d/50, 0.7475 for d/70, and 0.6863 for d/50 without global partition. Its
projection criterion assumes much freer candidate-view execution; collapsing
256 scored views to three coarse in-hand primitives produces negative
score--gain correlation and explains why it can underperform Fixed.

## 4. Ray-GPIS component ablations

### Clean, 120 paired scenes

| Variant | F-AUC | Final F@5 | Corr. | Regret | Score-map stability |
|---|---:|---:|---:|---:|---:|
| Novelty only | 0.8949±0.0096 | 0.9809±0.0100 | 0.3990±0.0661 | 0.0112±0.0053 | 0.2490±0.0383 |
| Uncertainty only | 0.8953±0.0096 | 0.9799±0.0101 | 0.4342±0.0698 | 0.0105±0.0051 | 0.6084±0.0233 |
| w/o receptive-field integration | **0.8991±0.0090** | **0.9822±0.0098** | 0.4623±0.0669 | **0.0086±0.0051** | 0.2135±0.0319 |
| Hit rays only | 0.8791±0.0099 | 0.9849±0.0069 | **0.4950±0.0536** | 0.0185±0.0047 | 0.3295±0.0319 |
| Full | 0.8987±0.0091 | 0.9814±0.0100 | 0.4722±0.0667 | 0.0088±0.0051 | 0.3346±0.0380 |

Supported clean effects: Full versus Novelty-only F-AUC +0.00386 (Holm
p=0.00784); Full versus Hit-only F-AUC +0.01969 (Holm p=1.41e-6); and
receptive-field integration versus Pointwise score-map stability +0.1211
(Holm p=2.12e-20). Full and Pointwise F-AUC are indistinguishable.

### Frozen moderate-noise profile, 120 paired scenes

| Variant | F-AUC | Final F@5 | Corr. | Regret | Score-map stability |
|---|---:|---:|---:|---:|---:|
| Novelty only | 0.6884±0.0139 | 0.8602±0.0135 | 0.3992±0.0668 | 0.0220±0.0042 | 0.3935±0.0347 |
| Uncertainty only | 0.7137±0.0122 | 0.8769±0.0125 | 0.4443±0.0638 | 0.0126±0.0032 | 0.5819±0.0293 |
| w/o receptive-field integration | 0.7158±0.0122 | **0.8789±0.0124** | 0.4283±0.0661 | **0.0118±0.0030** | 0.3302±0.0293 |
| Hit rays only | 0.7082±0.0111 | 0.8836±0.0099 | **0.4789±0.0514** | 0.0154±0.0029 | 0.4060±0.0260 |
| Full | **0.7159±0.0122** | 0.8784±0.0124 | 0.4308±0.0653 | 0.0119±0.0031 | 0.4895±0.0324 |

Supported moderate-noise effects: Full versus Novelty-only F-AUC +0.02748
(Holm p=9.14e-7); Full versus Hit-only +0.00764 (Holm p=0.0326); and
receptive-field integration score-map stability +0.15931 (Holm p=8.29e-21).

## 5. Targeted robustness tests used in the rebuttal

| Failure mode / comparator | Primary metric | Comparator | Full | Full benefit (95% CI) | Holm p |
|---|---|---:|---:|---:|---:|
| Contiguous palm hole / Hit-only | unseen Recall AUC | 0.5498 | 0.5715 | +0.02172 [0.00594, 0.03726] | 0.00487 |
| 6 deg / 3 mm pose jitter / Pointwise | score-map correlation | 0.2724 | 0.5028 | +0.23041 [0.20586, 0.25351] | 1.55e-11 |
| Visited-view 70% depth gap / Pose-Novelty | missing-patch Recall@5 after one action | 0.2827 | 0.5066 | +0.22389 [0.13979, 0.31363] | 7.15e-6 |

The visited-view test is the supported Pose-versus-Ray special case. A shared
prefix pose is recorded as visited, but a contiguous 70% depth block is lost
before fusion; the next three action branches are fault-free and shared. The
episode-mean relative recovery improvement is 79.2%; the point-weighted values
are 0.2763 for Pose and 0.5775 for Ray. Overall one-action F-gain improves by
only 0.00686 and is not significant, so this result supports local missing-
geometry recovery, not universal overall reconstruction superiority.

The exhaustive five-variant rerun under the same depth-gap stress is negative
for component claims: Novelty-only 0.5066, Uncertainty-only 0.5030, Pointwise
0.5404, Hit-only 0.4833, and Full 0.5066 missing-patch Recall@5. These values
are retained in the record but not used as an ablation advantage.

An additional 64-scene ER-GPIS comparison under the identical depth gap is
also negative for a Ray-over-ER claim. ER-GPIS obtains 0.5321 missing-patch
Recall@5 and Ray-GPIS obtains 0.5066; Ray-minus-ER is -0.02545 with 95% CI
[-0.09830, 0.04551] and p=0.426. Ray's one-action F-gain is only 0.00215
higher and is not significant. ER-GPIS is therefore omitted from the rebuttal
comparison, while the negative result remains archived.

## 6. Pose-versus-Ray robustness ledger

| Experiment | Units | Pose | Ray | Supported interpretation |
|---|---:|---:|---:|---|
| Clean six-image benchmark, F-AUC | 120 | **0.9090** | 0.8987 | Pose is numerically stronger; no universal Ray claim. |
| Broad moderate noise, F-AUC | 120 | **0.7193** | 0.7159 | Essentially tied; no Ray robustness claim. |
| Persistent stall/slip diagnostic, F-AUC | 120 | **0.6434** | 0.6358 | Archived and excluded from rebuttal; it favors Pose. |
| Real six-second trajectory plus axis registration outage, selected F gain | 120 | **0.2991** | 0.2965 | Ray-Pose -0.00253, CI crosses zero; inconclusive. |
| Existing palm-hole stress, target Recall AUC | 64 | **0.5798** | 0.5715 | Does not support Ray over Pose. |
| Visited-view local depth gap, missing-patch Recall@5 | 64 | 0.2827 | **0.5066** | Ray +0.22389, corrected p=7.15e-6; supported local case. |

Additional development diagnostics were non-discriminative and are not used
as evidence: a single-view empirical replay made both planners select the same
action in all 43 completed scenes; one shared-prefix Cube pilot agreed in
14/15 scenes; a two-prefix Cube pilot gave only +0.00074 mean Ray F gain.

## 7. Audit status and result locations

- Formal baselines: 960/960 stored runs complete; all 120 initial states paired.
- Clean ablations: 600/600 episodes and 3,000 CUDA planning steps complete.
- Moderate-noise ablations: 600/600 episodes and 3,000 CUDA steps complete.
- Targeted stress suite: 448 episode runs plus 7,680 pose-stability samples.
- Visited-view gap: 64 shared scenes for Pose/Hit/Full and a separate complete
  64-scene five-variant rerun; all targets are nonempty and all three action
  branches match exactly between outputs.
- ER-GPIS depth-gap check: 64/64 scenes complete, with action branches exactly
  matching the Pose/Ray reference output.
- Selected saved counterfactual gain equals the realized next-state F@5 change.

Primary files:

- `results/formal_120_sixview_gpu/all_metrics_summary.md`
- `results/formal_120_sixview_gpu/paired_statistics_all.md`
- `results/ablations/all_metrics_summary.md`
- `results/ablations_realistic/all_metrics_summary.md`
- `results/stress_ablation_summary/stress_summary.md`
- `results/visited_registration_gap/visited_gap_summary.md`
- `results/visited_registration_gap_er_ray/er_ray_gap_summary.md`
- `RESULTS.md`
