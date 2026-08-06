# Pre-registered special-case stress ablation

Status: completed on fresh seeds 15--22. The frozen confirmatory results and
joint Holm correction are under `results/stress_ablation_summary/`. Two of the
four primary hypotheses are supported; the sparse and ghost hypotheses are
not supported and must not be presented as positive findings.

## Original hypotheses

The experiment was designed around the hypotheses that each component might
protect the planner from a different realistic failure mode:

- GP uncertainty handles observed-but-sparse geometry;
- novelty suppresses redundant acquisition around noisy or ghost surfaces;
- miss-ray interpolation exposes directions containing no current surface hit;
- angular--radial receptive-field aggregation stabilizes the score map under
  tracking perturbations.

This is a targeted mechanism ablation, not a noise-severity sweep. The fault,
time of injection, primary comparison, and primary metric are fixed before the
formal runs. Existing moderate-noise episodes and their post-hoc tails are not
part of the confirmatory sample.

## Shared protocol

- Environment: `robosyn_gpu`, with CUDA required for every GP planning step.
- Image budget: exactly one initial RGB-D image and five planner-selected
  images. No additional bootstrap or intermediate frames enter fusion.
- View/action interface: 256 shared Fibonacci candidates, mapped to the same
  `minus_x`, `minus_y`, and `plus_z` primitives.
- Geometry backend: shared point-cloud fusion and evaluator; GT is used only
  for evaluation and counterfactual rollouts.
- GP settings: identical kernel, training data, training iterations, receptive
  field, novelty scale, and candidate set for all variants.
- Action execution: the same half-scale empirical action residuals calibrated
  from the 18 real 6-second executions are used in all active stress cases.
- Pairing: the injected fault realization is generated from `(object,
  initial_pose_seed, step)` and is therefore identical across planners even
  after their action histories diverge.
- Formal sample: all eight objects and eight new initial-pose seeds, giving 64
  paired scenes per active comparison. Seeds used by the existing clean and
  moderate-noise tables must not be reused.
- Pilot: at most two objects and two disjoint seeds. Pilot results only check
  that the fault and metrics are implemented correctly and are never pooled
  with formal results.

## Stress U: observed-but-sparse surface

Purpose: isolate the contribution of GP uncertainty.

At the initial observation only, remove 35% of the object depth samples in
four to eight seeded connected patches, erode the mask by two pixels, and add
1.5 mm depth noise. The retained and removed pixels must both occur in every
patch so that the affected region is sparse rather than a completely unseen
hole. Later images use 1.0 mm depth noise without structured dropout.

Define the sparse-surface target set before planning as GT surface samples that
are visible in the clean initial rendering, removed by the correlated dropout,
and within 10 mm of a retained initial sample. This definition uses neither a
planner score nor a future action outcome.

- Primary comparison: Full vs. Novelty-only.
- Primary metric: sparse-surface Recall@5 AUC.
- Secondary metrics: overall Recall@5 AUC, F@5 AUC, realized F@5 gain, and
  oracle regret.
- Supported conclusion if successful: uncertainty helps revisit geometry that
  has been observed but remains insufficiently constrained.

## Stress N: transient tracking outlier and ghost surface

Purpose: isolate the contribution of novelty.

The first active action is executed and rendered normally. For the resulting
second image only, corrupt the pose supplied to fusion and the planner by a
seeded rotation of exactly 15 degrees and translation of exactly 8 mm. The GT
rendering and counterfactual evaluator remain correct, and tracking recovers
from the next image onward. The corrupted cloud is deliberately retained, as
it would be when an outlier passes the online keyframe filter.

Define ghost points as samples inserted by the corrupted image whose distance
to the GT surface exceeds 5 mm. Define redundant acquisition before each
action as the fraction of the action-visible GT surface already covered within
5 mm.

- Primary comparison: Full vs. Uncertainty-only.
- Primary metric: mean oracle regret over the actions after the outlier.
- Secondary metrics: redundant-acquisition ratio, unique-surface gain,
  post-outlier F@5 AUC, and recovery after two correct observations.
- Supported conclusion if successful: novelty prevents high GP variance near
  a noisy known/ghost surface from repeatedly attracting the planner.

The injected outlier must be deterministic in step and magnitude. A random
outlier probability is not acceptable here because unequal outlier exposure
would invalidate the paired comparison.

## Stress M: contiguous unseen region under palm occlusion

Purpose: isolate miss-ray anchor interpolation.

Use the seeded palm-plus-two-finger occluder with a fixed 45% target occlusion
for the initial image and first active image, followed by 30% for the remaining
images. Disable the random occlusion heavy tail. Because the hand is fixed in
the camera frame while the object rotates, an initially hidden object region
can become observable from another action. This creates a physically plausible
large gap without using a GT mask to guide the planner.

Define the initially unseen target set once, before planning, as GT surface
samples farther than 5 mm from the initial fused cloud. Report its coverage at
every acquisition step.

- Primary comparison: Full vs. Hit-rays-only.
- Primary metric: initially-unseen-surface Recall@5 AUC.
- Secondary metrics: first step reaching 25% target-set recall, overall F@5
  AUC, miss-direction selection rate, realized gain, and oracle regret.
- Supported conclusion if successful: miss interpolation lets the planner
  discover directions for which the current point cloud provides no direct
  surface hit.

## Stress R: tracking-perturbation decision stability

Purpose: isolate angular--radial receptive-field aggregation without allowing
different trajectories to confound the comparison.

Collect one shared reference reconstruction state after active step two for
each of the 64 fresh scenes. Re-fuse that same observation history under 20
independent, paired pose perturbations at each of three fixed levels:

| Level | Rotation std. | Translation std. |
|---|---:|---:|
| Mild | 1.5 degrees | 0.75 mm |
| Moderate | 3.0 degrees | 1.5 mm |
| Severe | 6.0 degrees | 3.0 mm |

Full and Pointwise scores must be extracted from the same trained GP posterior
for every perturbed cloud; this removes training randomness and halves the
runtime. No new observation is acquired in this diagnostic.

- Primary comparison: Full vs. Pointwise GP.
- Primary metric: Spearman correlation between the perturbed and unperturbed
  256-direction score maps, averaged per scene at the severe level.
- Secondary metrics: executable-action flip rate, top-direction angular
  deviation, and oracle regret of the perturbed selected action in the clean
  reference state.
- Supported conclusion if successful: receptive-field aggregation stabilizes
  active decisions; it need not improve terminal F-score in a three-action
  environment.

## Optional compound check

After the four hypotheses above are frozen, run all five variants on 32
additional paired scenes under a single compound profile:

- half-scale empirical action residuals;
- 3-degree / 1.5-mm continuous pose jitter;
- one deterministic 15-degree / 8-mm tracking outlier after action one;
- 35% palm/finger occlusion;
- 1-mm depth noise and 10% correlated depth dropout.

This check is not used to attribute a gain to an individual component. It only
tests whether Full provides the best overall trade-off when failure modes
co-occur. Report F@5 AUC, Recall@5 AUC, oracle regret, and each method's
performance retention relative to its own clean result. Do not claim uniform
dominance if another variant wins an individual metric.

## Statistical analysis and acceptance criteria

For each active stress case, compute the per-scene Full-minus-ablation paired
difference. Use 10,000 scene-level bootstrap resamples for the 95% confidence
interval and a paired two-sided test. Correct the four pre-registered primary
hypotheses with Holm's method.

A component is supported only when:

1. the primary paired difference has the hypothesized sign;
2. its corrected p-value is below 0.05 and its paired 95% interval excludes
   zero;
3. all methods have identical fault exposure and the exact six-image budget;
4. the selected counterfactual gain equals the measured next-state gain; and
5. all formal planning steps record CUDA execution and finite valid scores.

Secondary metrics explain the mechanism but cannot rescue a failed primary
hypothesis. For Stress R, score-map stability is intentionally the primary
metric because the existing data already show that three executable actions
can hide large directional-score differences in final reconstruction AUC.

## Rebuttal-sized result table

The final rebuttal should contain one compact mechanism table, not every
available metric:

| Stress / removed component | Primary metric | Full | Ablated | Paired gain (95% CI) |
|---|---|---:|---:|---:|
| Sparse surface / uncertainty | sparse Recall@5 AUC | | | |
| Ghost surface / novelty | post-outlier regret (lower is better) | | | |
| Contiguous hole / miss interpolation | unseen Recall@5 AUC | | | |
| Pose jitter / receptive field | score-map correlation | | | |

Claim permitted by the completed confirmation:

> The average clean benchmark compresses some component effects because 256
> candidate directions are mapped to only three executable actions. Controlled
> failure-mode ablations show that miss-ray interpolation improves discovery
> of contiguous unseen regions and receptive-field aggregation stabilizes the
> directional score under pose perturbations. The targeted sparse-dropout and
> transient-ghost tests do not establish additional robustness from uncertainty
> and novelty, respectively; those components should only be described using
> the clean and frozen moderate-noise evidence.
