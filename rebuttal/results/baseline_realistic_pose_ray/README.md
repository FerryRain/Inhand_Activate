# Pose-Novelty versus Ray-GPIS under the frozen realistic-noise profile

This is a dedicated planner comparison, not a component ablation. It uses the
same eight objects, 15 initial orientations, 120 paired scenes, 256 candidate
directions, three executable actions, and strict one-initial-plus-five-active
image budget as the clean baseline. Both planners receive the same deterministic
object/pose/step noise process.

The frozen profile combines half-scale action-specific residuals measured from
real 6 s executions, pose jitter/drift/outliers, translation jitter/drift,
depth noise/dropout, mask morphology, and dynamic palm/finger occlusion.

| Method | F@5 AUC | Final F@5 | Corr. | Regret | Time/step (s) |
|---|---:|---:|---:|---:|---:|
| Pose-Novelty | 0.7193+/-0.0115 | 0.8794+/-0.0110 | 0.5017+/-0.0454 | 0.0090+/-0.0018 | 0.000240 |
| Full Ray-GPIS | 0.7159+/-0.0122 | 0.8784+/-0.0124 | 0.4308+/-0.0653 | 0.0119+/-0.0031 | 0.130 |

The paired Ray-minus-Pose F@5-AUC difference is -0.00345 (95% bootstrap CI
[-0.00863, 0.00103]; two-sided Wilcoxon p=0.290). Final F@5, Chamfer,
score-gain correlation, and oracle regret also do not establish a Ray-GPIS
advantage. Consequently, this general random-corruption profile cannot be used
as evidence that reconstruction-conditioned planning is more robust than pose
novelty.

Realized exposure is comparable: mean visibility is 0.579 versus 0.576, pose
error is 6.59 versus 6.37 degrees, and translation error is 3.41 versus 3.30 mm
for Pose-Novelty and Ray-GPIS. Action error differs slightly (31.6 versus 34.4
degrees) because the planners choose different action-conditioned empirical
residual distributions.

All 240 episodes, 120 paired initial observations, six reconstruction states,
and counterfactual action files pass validation. All 600 Ray-GPIS planning
steps use CUDA; Pose-Novelty is CPU-native by design.
