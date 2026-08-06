# Ray-GPIS component ablation under persistent action mismatch

This directory contains 600 strict-six-image episodes: 120 paired scenes for
Novelty-only, Uncertainty-only, Pointwise GP, Hit-only, and Full Ray-GPIS.
The stall/slip event schedule is identical across variants for each
object/initial-pose/step.

| Variant | F@5 AUC | Final F@5 | Corr. | Regret | Score-map stability |
|---|---:|---:|---:|---:|---:|
| Novelty only | 0.6141±0.0174 | 0.8081±0.0181 | 0.4039±0.0596 | 0.0180±0.0032 | 0.5222±0.0387 |
| Uncertainty only | 0.6323±0.0156 | 0.8272±0.0139 | 0.3953±0.0525 | 0.0116±0.0023 | 0.7882±0.0192 |
| Pointwise GP | 0.6365±0.0152 | 0.8338±0.0124 | 0.4092±0.0499 | 0.0107±0.0021 | 0.5353±0.0275 |
| Hit rays only | 0.6199±0.0157 | 0.8091±0.0147 | 0.3511±0.0548 | 0.0174±0.0031 | 0.5477±0.0237 |
| Full Ray-GPIS | 0.6358±0.0152 | 0.8320±0.0127 | 0.4161±0.0497 | 0.0108±0.0021 | 0.7029±0.0245 |

Supported paired findings:

- Full minus Novelty-only F@5 AUC: +0.02171, 95% bootstrap interval
  [0.01236, 0.03149], Holm p=3.22e-6.
- Full minus Hit-only F@5 AUC: +0.01587, interval [0.00961, 0.02272],
  Holm p=6.14e-6.
- Full minus Pointwise score-map stability: +0.16765, interval
  [0.15912, 0.17569], Holm p=7.89e-21; their F@5 AUC is tied.
- Full minus Uncertainty-only F@5 AUC: +0.00349, not significant after
  correction.

Audits: 600/600 episodes and 120/120 paired initial states pass; all 3,000
planning steps use CUDA and no GP surface is invalid.

See `all_metrics_summary.md`, `paired_statistics_all.md`,
`action_mismatch_summary.md`, and `gpu_runtime_validation.json` for complete
results and audits.
