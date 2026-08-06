# Complete experiment metric summary

All uncertainty intervals are 95% normal-approximation confidence intervals over 120 paired object/pose scenes.

## A. Final geometry

| Method | Recall@5 ↑ | F-score@5 ↑ | Chamfer (mm) ↓ |
|---|---:|---:|---:|
| Pose-Novelty | 0.8174±0.0135 | 0.8794±0.0110 | 5.572±0.191 |
| Full Ray-GPIS | 0.8132±0.0159 | 0.8784±0.0124 | 5.530±0.194 |

## B. Active exploration efficiency

| Method | Final visible surface ↑ | Visible-area AUC ↑ | Recall@5 AUC ↑ | F-score@5 AUC ↑ | Cumulative active F@5 gain ↑ |
|---|---:|---:|---:|---:|---:|
| Pose-Novelty | 0.2984±0.0520 | 0.2195±0.0473 | 0.6015±0.0123 | 0.7193±0.0115 | 0.5141±0.0142 |
| Full Ray-GPIS | 0.2890±0.0511 | 0.2141±0.0468 | 0.5964±0.0135 | 0.7159±0.0122 | 0.5131±0.0157 |

Visible surface is the cumulative area of GT mesh triangles hit by the actually acquired camera rays, divided by total GT mesh area. AUC is the trapezoidal mean over six cumulative reconstruction states: one initial state and five active-action states. The initial state contains 1 shared observation(s).

## C. Active decision policy

| Method | Selected gain/action ↑ | Oracle gain/action ↑ | Score–gain Spearman ↑ | Valid corr. steps ↑ | Oracle regret ↓ | Oracle accuracy ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Pose-Novelty | 0.1028±0.0028 | 0.1118±0.0031 | 0.5017±0.0454 | 100.0±0.0% | 0.0090±0.0018 | 66.2±3.6% |
| Full Ray-GPIS | 0.1026±0.0031 | 0.1145±0.0032 | 0.4308±0.0653 | 100.0±0.0% | 0.0119±0.0031 | 63.5±4.5% |

At each state, all three executable actions {-x, -y, +z} are rolled out counterfactually. Spearman correlation compares their action scores with their realized F@5 gains. Correlation is averaged over steps with non-constant finite action scores and realized gains.

## D. Runtime per planning step

| Method | Representation update (s) ↓ | Candidate scoring (s) ↓ | Total planning (s) ↓ |
|---|---:|---:|---:|
| Pose-Novelty | 0.000±0.000 | 0.000240±0.000008 | 0.000±0.000 |
| Full Ray-GPIS | 0.130±0.013 | 0.000067±0.000001 | 0.130±0.013 |

Total planning time is representation update plus candidate scoring, with CUDA synchronization at both phase boundaries.

## Consistency checks

- Episodes checked: 240
- Maximum |saved selected gain - actual next-state F@5 difference|: 0.000e+00
- Maximum |legacy evaluator surface_coverage - Recall@5| (backward-compatibility check): 0.000e+00
- Minimum visibility-coverage step increment: 0.000e+00
