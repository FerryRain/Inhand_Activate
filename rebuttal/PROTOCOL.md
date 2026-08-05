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

### PB-NBV

The implementation classifies occupied, empty, unknown, and frontier voxels from the shared point cloud and historical depth rays. Occupied and frontier centers are independently fit with full-covariance GMMs; component count 1–10 is selected by BIC. A 95% Gaussian equiprobability ellipsoid approximates each cluster's enclosing ellipsoid. Candidate scoring projects covariance ellipses through the common camera, applies the paper's depth weight `0.5**rank`, and computes frontier projected area minus occupied projected area. A 120-pair sanity sweep tests normalized voxel sizes `d/30`, `d/50`, and `d/70`, plus `d/50` without global partition. The strongest setting, `d/70` with four-way longitude partitioning, is used in the main table.

### ActNeRF

The adapter trains an ensemble of five object-centric radiance fields from segmented RGB and object-relative camera rays only. Depth is not passed to the planner. Models use different Xavier initializations and are warm-started after each new observation. Candidate views are rendered with volumetric alpha compositing. Mean predicted opacity defines a non-privileged ROI, and summed pixelwise RGB ensemble variance is the view score. Three planner initialization seeds are nested inside each object/pose pair.

The formal adapter runs on CUDA in `robosyn_gpu` but uses compact PyTorch radiance fields rather than the original Instant-NGP backend. It preserves the planner criterion and data flow; absolute runtime should therefore be labelled as adapted ActNeRF rather than official ActNeRF runtime.

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

## Level-B filtering

The realistic runner supports five RGB-D frames per action. Action-axis and magnitude errors change the true rendered pose; pose jitter, accumulated drift, and outliers change the pose supplied to fusion; mask morphology and moving partial occlusion change visible object pixels. `all` accepts every frame, `motion_only` applies minimum view change, and `full` additionally applies visibility and injected pose-error thresholds. The robustness evaluator adds 90th-percentile surface thickness, outlier ratio, accepted/rejected frames, and action switching frequency.

## Canonical configurations

- `configs/base.yaml`: shared camera, object, action, fusion, evaluator, and planner parameters; it is not a reportable experiment by itself.
- `configs/formal_120_common.yaml`: shared 120-pair, one-initial-plus-five-active protocol inherited by formal experiments.
- `configs/formal_120_sixview_gpu.yaml`: final baseline suite with synchronized GPU timing.
- `configs/pb_sanity_120_gpu.yaml`: reportable 120-pair PB-NBV scale/partition sweep.
- `configs/ablation.yaml`: formal Level-A component ablations, inheriting the same 120-pair GPU protocol.
- `configs/robustness.yaml`: formal Level-B filtering experiment, inheriting the same object/pose suite and GPU requirement.
