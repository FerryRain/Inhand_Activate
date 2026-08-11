# Experiment protocol and implementation decisions

## Fairness boundary

The following components are instantiated once per episode and shared by every planner:

- fixed camera calibration and renderer;
- normalized watertight object mesh and initial orientation;
- exactly one shared initial observation;
- 256 Fibonacci-sphere directions and virtual-camera geometry;
- `minus_x`, `minus_y`, and `plus_z` rotations calibrated from real pose logs;
- desired-view-to-action mapper based on the three reachable next viewing directions;
- masked depth back-projection, 1.5 mm voxel fusion, and GT mesh evaluator;
- five active actions and the same three counterfactual action rollouts at every state.

No planner receives a future mask. GT pose and GT mask are used by all methods in Level A. Planner-private representations are never used for reconstruction metrics.

## Coordinate convention

`T_CO` maps object-frame points into the fixed camera frame. The physical camera uses +z forward, +x right, and +y down. A candidate direction points from the object center toward a virtual camera and is expressed in the object frame. For each discrete primitive, the mapper predicts the next physical camera direction in the object frame and chooses the action with maximum cosine similarity to the planner's desired direction.

## Planner adaptations

### Fixed schedule

The schedule is `plus_z, minus_x, minus_y, plus_z, minus_x`. It never accesses the reconstruction or scores.

### Pose-Novelty

This baseline never accesses RGB-D, the fused point cloud, uncertainty, or a
mesh. It converts every acquired tracked pose into the fixed camera's viewing
direction in the object frame. For each of the three actions it applies the
same calibrated nominal rotation used by the shared mapper, predicts the next
view direction, and scores its minimum geodesic distance to all historical
directions. The action with maximum minimum distance is selected. This is the
strict pose-conditioned viewpoint-coverage baseline; no learned transition
model or reconstruction feedback is used.

### PB-NBV

The implementation classifies occupied, empty, unknown, and frontier voxels from the shared point cloud and historical depth rays. Occupied and frontier centers are independently fit with full-covariance GMMs; component count 1–10 is selected by BIC. A 95% Gaussian equiprobability ellipsoid approximates each cluster's enclosing ellipsoid. Candidate scoring projects covariance ellipses through the common camera, applies the paper's depth weight `0.5**rank`, and computes frontier projected area minus occupied projected area. A 120-pair sanity sweep tests normalized voxel sizes `d/30`, `d/50`, and `d/70`, plus `d/50` without global partition. The strongest setting, `d/70` with four-way longitude partitioning, is used in the main table.

### ActNeRF

The adapter trains an ensemble of five object-centric radiance fields from segmented RGB and object-relative camera rays only. Depth is not passed to the planner. Models use different Xavier initializations and are warm-started after each new observation. Candidate views are rendered with volumetric alpha compositing. Mean predicted opacity defines a non-privileged ROI, and summed pixelwise RGB ensemble variance is the view score. Three planner initialization seeds are nested inside each object/pose pair.

The formal adapter runs on CUDA in `robosyn_gpu` but uses compact PyTorch radiance fields rather than the original Instant-NGP backend. It preserves the planner criterion and data flow; absolute runtime should therefore be labelled as adapted ActNeRF rather than official ActNeRF runtime.

### ER-GPIS

The original ER-GPIS system performs contact exploration, so its tactile local
sliding and contact-recovery controller cannot be transferred fairly to the
shared visual action space. The adapter retains its global E-GPIS principle:
fused surface samples and inside/outside normal-offset dummy samples train an
exact GP with the inverse-multiquadric kernel. Each common candidate direction
is sampled radially and scored by its maximum posterior variance. The model is
trained and evaluated with GPyTorch/CUDA; all actions still pass through the
same three-primitive mapper.

### Ray-GPIS

The planner wraps the repository's `GPISNBVv3` implementation. It retains surface/miss ray anchors, spherical miss-depth interpolation, exact-GP posterior variance, angular receptive-field aggregation, novelty, and multiplicative full score. Its Fibonacci directions are numerically identical to the common candidate set. Ablation aliases replace exactly one final component while using the same GP training and candidates.

## Metrics

The reconstruction evaluator samples a deterministic 30,000-point GT surface
for every paired episode. It reports bidirectional precision, recall, and
F-score at 2/5/10 mm and symmetric mean Chamfer distance. F@5 and Recall@5 AUC
use trapezoidal averaging over the six reconstruction states.

The final report uses a separate, threshold-free visibility coverage metric.
For every actually acquired camera pose, the renderer records the GT mesh
triangle IDs hit by camera rays. Cumulative surface coverage is the summed area
of all uniquely observed triangles divided by total GT mesh area; its AUC is
computed over the same six states. The evaluator's legacy
`surface_coverage` field remains an exact alias of Recall@5 and is retained only
for backward compatibility, not reported as an independent metric.

At each of the first five states, the environment and fusion are cloned for all three actions. Action score is the maximum candidate score among candidates mapped to that action. These three score/gain pairs produce Spearman correlation, oracle regret, and oracle action accuracy.

## Fixed realistic-noise ablation

The reportable realistic component ablation retains the strict six-image
budget (`frames_per_action: 1`) and uses one frozen moderate profile. The true
6-second rotation is sampled from action-specific residuals measured in 18
real executions. Pose jitter, accumulated rotational/translational drift, and
occasional outliers corrupt the pose supplied to fusion. Depth noise/dropout,
mask erosion, and a moving palm-plus-two-finger occluder corrupt the
observation; the occluded fraction varies by image and includes a heavy tail.
This directly tests the same five planner-component variants under realistic
sensing/execution errors without introducing a second action/image protocol.

## Persistent action-mismatch stress

The persistent-mismatch profile inherits every corruption above and adds two
contact-level execution failures. If `R_a` is the action-specific empirical
rotation, its rotation-vector angle is scaled by `alpha`: ordinary actions use
`alpha` uniformly in [0.55, 0.85], while a 25% stall event uses [0.08, 0.30].
A 22% grip-slip event left-multiplies an uncommanded 12--30 degree rotation.
It also rotates a persistent control-frame state by 15--30 degrees, capped at
50 degrees; every later primitive axis is transformed by that state. Thus a
slip changes subsequent action execution instead of acting as independent
per-step Gaussian noise.

These values are declared simulation-stress parameters rather than a fit to
hardware logs. Event occurrence is a deterministic function of object,
initial-pose seed, and step. Paired planners therefore share the same
stall/slip schedule, while selecting different action labels may still produce
different realized trajectories. Counterfactual clones copy the persistent
control-frame state and share the next event draw.

## Pre-registered special-case stress ablation

Four mechanism-specific tests use all eight objects and eight fresh initial
poses (seeds 15--22), for 64 paired scenes per comparison. They retain one
initial plus five active images and half-scale empirical action residuals.
Fault realizations are deterministic functions of object, pose seed, and step,
so paired planners receive the same scheduled corruption process.

- Observed-sparse: 35% correlated dropout in the initial depth image; Full is
  compared with Novelty-only using sparse-target Recall@5 AUC.
- Transient ghost: exactly one 15-degree/8-mm pose outlier after the first
  action, followed by tracking recovery; Full is compared with
  Uncertainty-only using post-outlier oracle regret.
- Contiguous hole: 45% palm/finger occlusion for the first two images and 30%
  thereafter; Full is compared with Hit-only using initially-unseen Recall@5
  AUC.
- Pose stability: the same three-frame reconstruction is re-fused under 20
  perturbations at 1.5/0.75, 3/1.5, and 6 degrees/mm. Full and Pointwise scores
  come from the same GP posterior; severe-noise score-map correlation is the
  primary metric.

The four primary paired tests are corrected together with Holm's method.
Secondary metrics cannot rescue a failed primary hypothesis. GPU reconstruction
evaluation uses exact PyTorch3D KNN; a numerical equivalence check against the
SciPy evaluator gives zero Recall@5/F@5 difference and a Chamfer difference of
approximately 2e-11 m.

## Visited-view local registration gap

This additional Pose-versus-Ray mechanism test uses eight objects and eight
fresh pose seeds (64 paired scenes). It starts from one shared image and one
shared prefix rotation. The prefix tracked pose is visible to Pose-Novelty,
but a deterministic spatially contiguous 70% region of its depth is removed
before fusion. Points already within 5 mm of the fused cloud are excluded from
the target, leaving only geometry that the failed frame did not acquire.

The failure is transient. At the next decision the environment renders each
of the three executable action branches exactly once with normal depth, and
the same branches are used for Pose-Novelty, all Ray-GPIS variants, and the
oracle. The predeclared primary metric is one-action Recall@5 on the missing
prefix patch. Episode-mean and missing-point-weighted results are both saved.
The five Full-versus-comparator primary tests use a joint Holm correction;
overall F-score gain is secondary and cannot replace the local primary metric.

## Canonical configurations

- `configs/base.yaml`: shared camera, object, action, fusion, evaluator, and planner parameters; it is not a reportable experiment by itself.
- `configs/formal_120_common.yaml`: shared 120-pair, one-initial-plus-five-active protocol inherited by formal experiments.
- `configs/formal_120_sixview_gpu.yaml`: final baseline suite with synchronized GPU timing.
- `configs/pb_sanity_120_gpu.yaml`: reportable 120-pair PB-NBV scale/partition sweep.
- `configs/ablation.yaml`: formal Level-A component ablations, inheriting the same 120-pair GPU protocol.
- `configs/ablation_realistic.yaml`: the same five component variants under a fixed moderate noise profile and the strict six-image budget.
- `configs/action_slip_realistic.yaml`: shared persistent under-rotation,
  stall, grip-slip, and later-axis-misalignment profile.
- `configs/baseline_slip_pose_ray.yaml`: 120-pair Pose-Novelty/Full comparison
  under persistent action mismatch.
- `configs/ablation_slip_robustness.yaml`: all five Ray-GPIS component variants
  under the identical persistent action-mismatch profile.
- `configs/stress_sparse.yaml`, `stress_ghost.yaml`, and `stress_hole.yaml`:
  the three active special-case paired tests.
- `configs/stress_pose_reference.yaml`: shared trajectories used by the
  Full/Pointwise pose-stability diagnostic.
- `configs/visited_registration_gap.yaml`: Pose-Novelty, Hit-only, and Full on
  the shared one-action visited-view depth-gap test.
- `configs/visited_registration_gap_ablation.yaml`: exhaustive five-variant
  rerun under the identical 64 scenes and action branches.
- `configs/visited_registration_gap_er_ray.yaml`: adapted ER-GPIS and Full
  Ray-GPIS on the same depth-gap scenes and shared action branches.
- `configs/baseline_joint_pose_depth_gap.yaml`: Pose-Novelty, adapted ActNeRF,
  adapted PB-NBV, and Ray-GPIS on the transient 70% depth gap with paired
  6 deg / 3 mm pose error and no action-noise confound.
