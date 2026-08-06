# Complete experiment metric summary

All uncertainty intervals are 95% normal-approximation confidence intervals over 120 paired object/pose scenes.

## A. Final geometry

| Method | Recall@5 ↑ | F-score@5 ↑ | Chamfer (mm) ↓ |
|---|---:|---:|---:|
| Pose-Novelty | 0.7624±0.0151 | 0.8411±0.0114 | 6.164±0.227 |
| Full Ray-GPIS | 0.7509±0.0170 | 0.8320±0.0127 | 6.301±0.234 |

## B. Active exploration efficiency

| Method | Final visible surface ↑ | Visible-area AUC ↑ | Recall@5 AUC ↑ | F-score@5 AUC ↑ | Cumulative active F@5 gain ↑ |
|---|---:|---:|---:|---:|---:|
| Pose-Novelty | 0.2911±0.0504 | 0.2021±0.0418 | 0.5155±0.0144 | 0.6434±0.0143 | 0.4919±0.0170 |
| Full Ray-GPIS | 0.2846±0.0503 | 0.1986±0.0416 | 0.5066±0.0157 | 0.6358±0.0152 | 0.4828±0.0171 |

Visible surface is the cumulative area of GT mesh triangles hit by the actually acquired camera rays, divided by total GT mesh area. AUC is the trapezoidal mean over six cumulative reconstruction states: one initial state and five active-action states. The initial state contains 1 shared observation(s).

## C. Active decision policy

| Method | Selected gain/action ↑ | Oracle gain/action ↑ | Score–gain Spearman ↑ | Valid corr. steps ↑ | Oracle regret ↓ | Oracle accuracy ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Pose-Novelty | 0.0984±0.0034 | 0.1066±0.0037 | 0.4308±0.0486 | 100.0±0.0% | 0.0082±0.0014 | 67.8±3.5% |
| Full Ray-GPIS | 0.0966±0.0034 | 0.1073±0.0038 | 0.4161±0.0497 | 100.0±0.0% | 0.0108±0.0021 | 64.3±4.0% |

At each state, all three executable actions {-x, -y, +z} are rolled out counterfactually. Spearman correlation compares their action scores with their realized F@5 gains. Correlation is averaged over steps with non-constant finite action scores and realized gains.

## D. Runtime per planning step

| Method | Representation update (s) ↓ | Candidate scoring (s) ↓ | Total planning (s) ↓ |
|---|---:|---:|---:|
| Pose-Novelty | 0.000±0.000 | 0.000249±0.000010 | 0.000±0.000 |
| Full Ray-GPIS | 0.141±0.016 | 0.000067±0.000001 | 0.141±0.016 |

Total planning time is representation update plus candidate scoring, with CUDA synchronization at both phase boundaries.

## Consistency checks

- Episodes checked: 240
- Maximum |saved selected gain - actual next-state F@5 difference|: 0.000e+00
- Maximum |legacy evaluator surface_coverage - Recall@5| (backward-compatibility check): 0.000e+00
- Minimum visibility-coverage step increment: 0.000e+00
