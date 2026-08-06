# Final strict-six-image baseline result

The final Level-A benchmark contains 8 objects and 15 shared initial poses, for
120 paired scenes. Every episode contains exactly one initial image and five
planner-selected images. Fixed, Pose-Novelty, PB-NBV, ER-GPIS, and Ray-GPIS run
once per scene; ActNeRF runs three initialization seeds which are averaged
inside each scene. This produces 960 stored runs and 120 method-level paired
units.

| Method | F@5 AUC | Recall AUC | Final F@5 | Corr. | Regret | Accuracy | Time/step (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed schedule | 0.7948±0.0061 | 0.6780±0.0078 | 0.9401±0.0046 | N/A | 0.0806±0.0025 | 29.0±1.8% | 0.000 |
| **Pose-Novelty** | **0.9090±0.0043** | **0.8553±0.0066** | **0.9945±0.0014** | **0.6186±0.0294** | **0.0025±0.0006** | **71.0±2.0%** | **0.000272** |
| Adapted PB-NBV (d/70) | 0.7475±0.0193 | 0.6188±0.0258 | 0.8496±0.0234 | -0.3563±0.0731 | 0.1076±0.0124 | 19.2±3.8% | 0.391±0.024 |
| Adapted ER-GPIS | 0.9033±0.0055 | 0.8459±0.0086 | 0.9899±0.0037 | 0.3439±0.0587 | 0.0060±0.0022 | 63.2±4.0% | 0.097±0.002 |
| Adapted ActNeRF | 0.8665±0.0062 | 0.7847±0.0095 | 0.9651±0.0050 | 0.1619±0.0337 | 0.0317±0.0029 | 42.3±2.4% | 2.238±0.018 |
| Ray-GPIS | 0.8987±0.0091 | 0.8397±0.0136 | 0.9814±0.0100 | 0.4722±0.0667 | 0.0088±0.0051 | 68.0±4.1% | 0.261±0.024 |

Against adapted ActNeRF, Ray-GPIS improves paired F@5 AUC by 0.03227
(95% bootstrap CI 0.02258 to 0.04038), wins/ties/loses 106/1/13 scenes, and
has a two-sided Wilcoxon p-value of 8.73e-15 after Holm correction. It improves
score-gain Spearman correlation by 0.31031 and reduces oracle regret by
0.02294. Final F@5 improves by 0.01634 and Chamfer decreases by 0.33064 mm.

Pose-Novelty is the strongest method in this clean three-action benchmark: its
final 30-degree pose-space coverage is 0.4010 versus 0.3352 for Ray-GPIS, and
its F@5 AUC is numerically higher by 0.01028. The paired Holm-corrected
Wilcoxon test does not establish an F-AUC difference (p=0.463), although the
Pose-Novelty score has significantly higher score-gain correlation (p=0.00191).
This clean result does not support a claim that reconstruction conditioning
universally outperforms pose coverage.

Adapted ER-GPIS is also tied with Ray-GPIS in clean F@5 AUC (paired difference
-0.00454 for Ray minus ER; corrected p=0.463). Ray-GPIS has higher score-gain
correlation by 0.12835 (95% bootstrap CI 0.04496 to 0.20898; corrected
p=9.92e-4).

Timing uses PyTorch 2.4.1+cu121 and GPyTorch 1.11 on an RTX 4090 D. CUDA timers
are synchronized at phase boundaries. All 600 Ray-GPIS, 600 ER-GPIS, and
1,800 ActNeRF planning steps record `device=cuda`; PB-NBV, Pose-Novelty, and
Fixed are native CPU planners measured on the same machine. Ray-GPIS uses GPyTorch ExactGP and
averages 0.261 s per planning step (0.261 s representation update and 0.000072
s candidate lookup), versus 2.238 s for adapted ActNeRF.

PB-NBV sensitivity confirms that the main result uses its strongest tested
configuration: F@5 AUC is 0.6939 for d/30, 0.7212 for d/50, 0.7475 for d/70,
and 0.6863 for d/50 without partition.

Validation status:

- 960/960 formal runs and 480/480 PB sanity runs are complete;
- all 120 paired initial states are identical across methods;
- every active state contains 256 candidate scores and three counterfactual actions;
- saved selected gains equal next-state F@5 changes exactly;
- independent visible-triangle coverage is monotonic;
- all ten core unit tests pass.

The complete categorized report is under
`results/formal_120_sixview_gpu/all_metrics_summary.md`; paired confidence
intervals and corrected tests are in `paired_statistics_all.md`. The ActNeRF
baseline preserves ensemble RGB-variance planning but uses compact PyTorch
radiance fields, not the original Instant-NGP implementation.

## Clean Ray-GPIS component ablations

Each variant uses the same 8 objects, 15 initial poses, one initial image, five
active images, 256 candidates, three executable actions, fusion/evaluation
backend, GP training settings, and synchronized CUDA timing. The sweep contains
600 episodes and 3,000 GPU planning steps.

| Variant | F@5 AUC | Final F@5 | Corr. | Regret | Accuracy | Score-map stability | Time/step (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Novelty only | 0.8949±0.0096 | 0.9809±0.0100 | 0.3990±0.0661 | 0.0112±0.0053 | 65.2±4.1% | 0.2490±0.0383 | 0.274±0.025 |
| Uncertainty only | 0.8953±0.0096 | 0.9799±0.0101 | 0.4342±0.0698 | 0.0105±0.0051 | 64.8±4.5% | 0.6084±0.0233 | 0.260±0.025 |
| Pointwise GP | 0.8991±0.0090 | 0.9822±0.0098 | 0.4623±0.0669 | 0.0086±0.0051 | 67.8±4.5% | 0.2135±0.0319 | 0.263±0.025 |
| Hit rays only | 0.8791±0.0099 | 0.9849±0.0069 | 0.4950±0.0536 | 0.0185±0.0047 | 60.7±3.8% | 0.3295±0.0319 | 0.291±0.026 |
| **Full Ray-GPIS** | 0.8987±0.0091 | 0.9814±0.0100 | 0.4722±0.0667 | 0.0088±0.0051 | 68.0±4.1% | 0.3346±0.0380 | 0.261±0.025 |

Full improves F@5 AUC over novelty-only by 0.00386 (paired 95% bootstrap
CI 0.00099 to 0.00777; Holm-corrected p=0.00784) and improves its score-gain
correlation by 0.0732. Uncertainty-only has the same smaller trend and repeats
actions more often (54.8% versus 52.3%). Pointwise reconstruction is
statistically indistinguishable from full, but its consecutive score-map
correlation is lower by 0.1211 (CI 0.1084 to 0.1344; Holm-corrected
p=2.12e-20), isolating the stabilization provided by receptive-field
aggregation. Hit-only has the largest efficiency loss: full improves F@5 AUC
by 0.01969 (CI 0.01212 to 0.02718; Holm-corrected p=1.41e-6), reduces regret
by 0.00973, and improves visible-area AUC from 0.2080 to 0.2135. Hit-only can
catch up at the final state, so the primary effect of miss interpolation is
faster discovery rather than a guaranteed terminal-score increase.

Validation status:

- 600/600 episodes and all 120 paired initial states pass the strict audit;
- all 3,000 planner steps record `device=cuda`, and every score map has finite candidates;
- Full Ray-GPIS exactly matches all non-runtime fields of the independently
  generated formal baseline in every one of the 120 scenes;
- selected counterfactual gain is exactly the next-state F@5 change, and
  visibility coverage is monotonic in every episode.

Complete metrics are in `results/ablations/all_metrics_summary.md`; paired
tests are in `results/ablations/paired_statistics_all.md`.

## Moderate-noise strict-six-image component ablations

This second 600-episode sweep keeps the same one-initial-plus-five-active image
budget and adds the frozen moderate profile: half-scale action-specific
residuals from 18 real 6-second executions, dynamic palm/two-finger occlusion,
depth noise/dropout, mask erosion, and rotational/translational pose jitter,
drift, and outliers. The observed exposure is closely matched across variants;
Full Ray-GPIS averages visibility 0.576, pose error 6.37 degrees / 3.30 mm, and
action execution error 34.42 degrees.

| Variant | F@5 AUC | Final F@5 | Corr. | Regret | Accuracy | Score-map stability | Time/step (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Novelty only | 0.6884±0.0139 | 0.8602±0.0135 | 0.3992±0.0668 | 0.0220±0.0042 | 54.8±4.4% | 0.3935±0.0347 | 0.139±0.013 |
| Uncertainty only | 0.7137±0.0122 | 0.8769±0.0125 | 0.4443±0.0638 | 0.0126±0.0032 | 62.0±4.3% | 0.5819±0.0293 | 0.137±0.013 |
| Pointwise GP | 0.7158±0.0122 | 0.8789±0.0124 | 0.4283±0.0661 | 0.0118±0.0030 | 62.5±4.5% | 0.3302±0.0293 | 0.135±0.013 |
| Hit rays only | 0.7082±0.0111 | 0.8836±0.0099 | 0.4789±0.0514 | 0.0154±0.0029 | 61.8±4.0% | 0.4060±0.0260 | 0.175±0.013 |
| **Full Ray-GPIS** | **0.7159±0.0122** | 0.8784±0.0124 | 0.4308±0.0653 | **0.0119±0.0031** | **63.5±4.5%** | **0.4895±0.0324** | 0.136±0.013 |

Full improves paired F@5 AUC over novelty-only by 0.02748 (95% bootstrap CI
0.01706 to 0.03856; Holm-corrected p=9.14e-7) and over hit-only by 0.00764
(CI 0.00065 to 0.01430; corrected p=0.0326). It lowers oracle regret by
0.01012 and 0.00355, respectively. Full and pointwise have indistinguishable
F@5 AUC (difference 0.000015), while Full improves consecutive score-map
correlation by 0.15931 (CI 0.14908 to 0.17000; corrected p=8.29e-21).

Validation status:

- 600/600 episodes, 120/120 paired initial states, and all expected files pass;
- all 3,000 planner steps use CUDA and report valid GP surfaces;
- all 9,000 counterfactual action files are present, with zero selected-gain consistency error;
- executed-pose visibility coverage is monotonic for all episodes.

Complete metrics, noise exposure, paired tests, and plots are under
`results/ablations_realistic/`.

## Persistent under-rotation, stall, and grip-slip stress

This declared simulation stress extends the moderate-noise profile with two
contact-execution failures. Every non-stalled primitive reaches only 55--85%
of its requested angle; a 25% stall event reaches 8--30%. A grip slip occurs
with 22% probability, adds a 12--30 degree uncommanded rotation, and biases the
effective axes of all later primitives by 15--30 degrees (capped at 50
degrees). The parameters are not claimed as an empirical fit. Event draws are
shared by object, initial pose, and step, so paired planners encounter the same
stall/slip schedule even when their selected actions and resulting trajectories
differ.

The realized 120-scene exposure was 26.7% stalled actions, 18.8% slipped
actions, and 33.5% later actions with persistent axis misalignment. Mean
nominal-versus-realized SO(3) action error was 54.1--56.8 degrees across the
five variants. Every episode still contains exactly one initial image and five
active images.

| Variant | F@5 AUC | Final F@5 | Corr. | Regret | Accuracy | Score-map stability | Time/step (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Novelty only | 0.6141±0.0174 | 0.8081±0.0181 | 0.4039±0.0596 | 0.0180±0.0032 | 57.3±4.5% | 0.5222±0.0387 | 0.135±0.015 |
| Uncertainty only | 0.6323±0.0156 | 0.8272±0.0139 | 0.3953±0.0525 | 0.0116±0.0023 | 63.0±4.1% | **0.7882±0.0192** | 0.139±0.015 |
| Pointwise GP | **0.6365±0.0152** | **0.8338±0.0124** | 0.4092±0.0499 | **0.0107±0.0021** | 64.0±3.8% | 0.5353±0.0275 | 0.138±0.015 |
| Hit rays only | 0.6199±0.0157 | 0.8091±0.0147 | 0.3511±0.0548 | 0.0174±0.0031 | 55.7±4.4% | 0.5477±0.0237 | 0.136±0.014 |
| **Full Ray-GPIS** | 0.6358±0.0152 | 0.8320±0.0127 | **0.4161±0.0497** | 0.0108±0.0021 | **64.3±4.0%** | 0.7029±0.0245 | 0.138±0.015 |

Full improves paired F@5 AUC over Novelty-only by 0.02171 (95% bootstrap CI
0.01236 to 0.03149; Holm-corrected p=3.22e-6), reduces regret by 0.00721,
and improves final F@5 by 0.02384. Against Hit-only, Full improves F@5 AUC
by 0.01587 (CI 0.00961 to 0.02272; corrected p=6.14e-6), reduces regret by
0.00667, and improves final F@5 by 0.02290. Full and Pointwise remain tied in
F@5 AUC (Full minus Pointwise -0.00067), while receptive-field integration
increases consecutive score-map correlation by 0.16765. The smaller Full
minus Uncertainty-only F@5 AUC difference of 0.00349 is not significant after
correction.

A separate 120-pair comparison does not support the anticipated claim that
reconstruction conditioning is more robust than pose coverage under this
stress: Pose-Novelty obtains 0.6434±0.0143 F@5 AUC and Full obtains
0.6358±0.0152. The paired Ray-minus-Pose difference is -0.00765 (Holm-corrected
p=0.00485). Their post-fault next-decision regrets are 0.0056 and 0.0066,
respectively. Consequently, this test supports uncertainty and miss-ray
interpolation relative to their Ray-GPIS ablations, but not a Ray-over-Pose
robustness claim.

Validation status:

- 240/240 Pose/Ray episodes and 600/600 component-ablation episodes pass the
  strict completeness and paired-initial-state audits;
- all 600 baseline Ray steps and all 3,000 component-ablation steps record
  `device=cuda`, with no invalid GP surface;
- all stall, slip, and persistent-axis event rates are exactly matched across
  paired planners by construction;
- selected counterfactual gains equal actual next-state F@5 changes exactly,
  and executed-view visibility coverage is monotonic.

Complete results are under `results/baseline_slip_pose_ray/` and
`results/ablation_slip_robustness/`.

## Fresh-seed special-case stress ablations

The three active tests each contain 128 episodes (64 paired scenes), and the
pose-stability test contains 64 shared reference scenes with 7,680 paired
variant/perturbation samples. All use new pose seeds 15--22. Positive paired
improvements below mean Full is better; Holm correction is joint over the four
pre-registered primary hypotheses.

| Stress / removed component | Primary metric | Full | Ablated | Paired improvement (95% CI) | Holm p | Supported |
|---|---|---:|---:|---:|---:|---:|
| Sparse / uncertainty | sparse Recall@5 AUC | 0.5899 | 0.5920 | -0.00215 [-0.01944, 0.01107] | 0.821 | no |
| Ghost / novelty | post-outlier regret | 0.01304 | 0.01128 | -0.00176 [-0.00346, -0.00018] | 0.0762 | no |
| Contiguous hole / miss interpolation | unseen Recall@5 AUC | 0.5715 | 0.5498 | +0.02172 [0.00594, 0.03726] | 0.00487 | yes |
| Severe pose jitter / receptive field | score-map correlation | 0.5028 | 0.2724 | +0.23041 [0.20586, 0.25351] | 1.55e-11 | yes |

The contiguous-hole result also improves overall Recall@5 AUC by 0.01654,
F@5 AUC by 0.01252, and lowers oracle regret by 0.00598. Full selects a
miss-interpolated direction on 87.5% of its actions, while Hit-only cannot do
so by construction. Under severe pose perturbations, receptive-field
aggregation improves score-map stability in 63/64 scenes. The executable
action-flip rates remain close (26.1% Full versus 27.4% Pointwise), consistent
with 256 directional scores collapsing to three actions.

The two failures matter for interpretation. Correlated initial dropout does
not produce a significant sparse-target recall advantage over Novelty-only
(48/64 exact trajectory ties). A transient ghost surface also does not show a
novelty benefit; its negative regret trend is not significant after joint Holm
correction. Consequently, the stress suite supports special-case robustness
for miss interpolation and receptive-field aggregation, not a claim that every
Full component dominates every failure mode. Uncertainty remains supported by
the independent frozen moderate-noise Full-versus-Novelty result above.

Validation status:

- all 448 episode runs are complete and use exactly six reconstruction states;
- all 2,240 active planning steps record CUDA and valid GP surfaces;
- all paired initial states, counterfactual files, and selected-gain checks pass;
- all 7,680 pose-stability samples are finite, unique, and CUDA-generated.

The evaluator now supports exact PyTorch3D CUDA KNN. On the same episode it
reduces end-to-end time from 3.668 s to 3.211 s (1.14x) with identical
Recall@5/F@5 and a 2.0e-11 m Chamfer difference. Reusing the GP's existing hit
mask removes 256 redundant CPU ray tests from Hit-only and reduces its measured
time from 0.175 s/step in the earlier moderate sweep to 0.130 s/step, equal to
Full in the new hole test.

Complete results are in `results/stress_ablation_summary/`,
`results/stress_sparse/`, `results/stress_ghost/`, `results/stress_hole/`, and
`results/stress_pose_stability/`.

## Visited-view local depth-registration gap: Pose versus Ray

This mechanism-specific test uses 64 fresh paired scenes (8 objects x 8 pose
seeds). After one shared prefix rotation, the tracked pose is added to the
viewpoint history, but one spatially contiguous 70% region of that frame's
depth is removed before fusion. The failure is transient: all three subsequent
action branches have normal depth and are rendered once, then shared by every
planner. The primary metric is Recall@5 on the prefix points that remain more
than 5 mm from the fused cloud after the faulty observation.

| Planner | Missing-patch Recall@5 | Point-weighted Recall@5 | One-action F@5 gain | Regret |
|---|---:|---:|---:|---:|
| Pose-Novelty | 0.2827±0.0759 | 0.2763 | 0.1816±0.0145 | 0.0243±0.0096 |
| Novelty only | 0.5066±0.0971 | 0.5775 | 0.1885±0.0144 | 0.0174±0.0088 |
| Uncertainty only | 0.5030±0.0963 | 0.5768 | 0.1874±0.0141 | 0.0185±0.0092 |
| w/o receptive-field integration | **0.5404±0.0980** | **0.6161** | **0.1908±0.0130** | **0.0151±0.0077** |
| Hit rays only | 0.4833±0.0956 | 0.4962 | 0.1827±0.0151 | 0.0232±0.0106 |
| **Full Ray-GPIS** | 0.5066±0.0971 | 0.5775 | 0.1885±0.0144 | 0.0174±0.0088 |

Full improves missing-patch Recall@5 over Pose-Novelty by 0.22389 (95%
bootstrap CI 0.13979 to 0.31363), with 21/42/1 wins/ties/losses and
Holm-corrected p=7.15e-6 across the five declared comparator tests. This is a
79.2% relative improvement in the episode mean; the point-weighted sensitivity
analysis gives 0.2763 versus 0.5775. The overall one-action F@5-gain difference
is only +0.00686 and is not significant, so the supported claim is narrowly
that reconstruction-aware planning better recovers a region that pose history
marks as visited but geometry never acquired.

The same stress does not support a Full-versus-component claim. Full ties
Novelty-only exactly, is statistically indistinguishable from
Uncertainty-only and Hit-only, and is lower than Pointwise on this local metric.
These negative component results are retained rather than folded into the
representative robustness table. Complete outputs are in
`results/visited_registration_gap/` and
`results/visited_registration_gap_ablation/`.

We also reran adapted ER-GPIS on the identical 64 shared scenes. Its
missing-patch Recall@5 is 0.5321 versus 0.5066 for Ray-GPIS; the paired
Ray-minus-ER difference is -0.02545 (95% bootstrap CI [-0.09830, 0.04551],
p=0.426). The secondary one-action F-gain difference is +0.00215 for Ray and
is also not significant. This experiment does not support a Ray-over-ER claim,
so ER-GPIS is omitted from the rebuttal comparison. The complete negative
result is retained under `results/visited_registration_gap_er_ray/`.

## Pose-versus-Ray robustness interpretation

The experiments support a conditional, not universal, conclusion. Pose is
stronger in the clean benchmark (0.9090 versus 0.8987 F-AUC), essentially tied
under the broad moderate-noise profile (0.7193 versus 0.7159), and the archived
stall/slip stress favors Pose (0.6434 versus 0.6358). A 120-scene, one-decision
replay of measured six-second trajectories with an axis-specific registration
outage is also inconclusive: Ray-minus-Pose selected F gain is -0.00253 with
95% CI [-0.01412, 0.00975]. Ray's supported advantage appears in the precise
case where viewpoint history and acquired geometry disagree: the visited-view
local depth gap above. This is the claim used in the rebuttal.
