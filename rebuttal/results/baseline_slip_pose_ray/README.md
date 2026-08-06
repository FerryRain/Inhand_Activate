# Pose-Novelty versus Ray-GPIS under persistent action mismatch

This directory contains 120 paired scenes per planner: eight objects, 15
initial orientations, one initial RGB-D image, and five active images. It uses
the existing realistic sensing corruption plus a declared execution stress:
55--85% ordinary primitive progress, 25% stalls with 8--30% progress, 22%
grip-slip probability, and persistent action-axis drift after each slip.

| Planner | F@5 AUC | Final F@5 | Corr. | Regret | Post-fault next regret |
|---|---:|---:|---:|---:|---:|
| Pose-Novelty | 0.6434±0.0143 | 0.8411±0.0114 | 0.4308±0.0486 | 0.0082±0.0014 | 0.0056±0.0018 |
| Full Ray-GPIS | 0.6358±0.0152 | 0.8320±0.0127 | 0.4161±0.0497 | 0.0108±0.0021 | 0.0066±0.0022 |

The paired Ray-minus-Pose F@5-AUC difference is -0.00765 (95% bootstrap
interval -0.01255 to -0.00364; Holm-corrected Wilcoxon p=0.00485). Thus this
stress does **not** support a Ray-over-Pose robustness claim. It is retained as
a negative result and as the shared Full arm of the component study.

Audits: 240/240 episodes are complete, the 120 initial states are paired,
all 600 Ray-GPIS steps use CUDA, and every saved selected gain matches the
actual next-state F@5 change.

See `all_metrics_summary.md`, `paired_statistics_all.md`,
`action_mismatch_summary.md`, and `gpu_runtime_validation.json` for complete
results and audits.
