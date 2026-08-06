# Complete experiment metric summary

All uncertainty intervals are 95% normal-approximation confidence intervals over 64 paired object/pose scenes.

## A. Final geometry

| Method | Recall@5 ↑ | F-score@5 ↑ | Chamfer (mm) ↓ |
|---|---:|---:|---:|
| Novelty only | 0.9623±0.0214 | 0.9785±0.0129 | 3.146±0.136 |
| Full Ray-GPIS | 0.9632±0.0215 | 0.9789±0.0129 | 3.134±0.137 |

## B. Active exploration efficiency

| Method | Final visible surface ↑ | Visible-area AUC ↑ | Recall@5 AUC ↑ | F-score@5 AUC ↑ | Cumulative active F@5 gain ↑ |
|---|---:|---:|---:|---:|---:|
| Novelty only | 0.2818±0.0701 | 0.2103±0.0641 | 0.7928±0.0187 | 0.8666±0.0123 | 0.4747±0.0162 |
| Full Ray-GPIS | 0.2820±0.0700 | 0.2106±0.0641 | 0.7951±0.0190 | 0.8680±0.0125 | 0.4752±0.0162 |

Visible surface is the cumulative area of GT mesh triangles hit by the actually acquired camera rays, divided by total GT mesh area. AUC is the trapezoidal mean over six cumulative reconstruction states: one initial state and five active-action states. The initial state contains 1 shared observation(s).

## C. Active decision policy

| Method | Selected gain/action ↑ | Oracle gain/action ↑ | Score–gain Spearman ↑ | Valid corr. steps ↑ | Oracle regret ↓ | Oracle accuracy ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Novelty only | 0.0949±0.0032 | 0.1064±0.0051 | 0.4428±0.1063 | 97.2±1.9% | 0.0115±0.0065 | 65.0±6.4% |
| Full Ray-GPIS | 0.0950±0.0032 | 0.1058±0.0051 | 0.5000±0.1056 | 99.1±1.0% | 0.0107±0.0065 | 67.8±6.4% |

At each state, all three executable actions {-x, -y, +z} are rolled out counterfactually. Spearman correlation compares their action scores with their realized F@5 gains. Correlation is averaged over steps with non-constant finite action scores and realized gains.

## D. Runtime per planning step

| Method | Representation update (s) ↓ | Candidate scoring (s) ↓ | Total planning (s) ↓ |
|---|---:|---:|---:|
| Novelty only | 0.239±0.029 | 0.000056±0.000001 | 0.239±0.029 |
| Full Ray-GPIS | 0.244±0.029 | 0.000055±0.000001 | 0.244±0.029 |

Total planning time is representation update plus candidate scoring, with CUDA synchronization at both phase boundaries.

## Consistency checks

- Episodes checked: 128
- Maximum |saved selected gain - actual next-state F@5 difference|: 0.000e+00
- Maximum |legacy evaluator surface_coverage - Recall@5| (backward-compatibility check): 0.000e+00
- Minimum visibility-coverage step increment: 0.000e+00
