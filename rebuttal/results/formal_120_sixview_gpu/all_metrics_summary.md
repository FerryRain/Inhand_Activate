# Complete baseline metric summary

All uncertainty intervals are 95% normal-approximation confidence intervals over 120 paired object/pose scenes. ActNeRF's initialization seeds are averaged within each scene first.

## A. Final geometry

| Method | Recall@5 ↑ | F-score@5 ↑ | Chamfer (mm) ↓ |
|---|---:|---:|---:|
| fixed | 0.8881±0.0082 | 0.9401±0.0046 | 3.027±0.079 |
| pb_nbv | 0.7597±0.0338 | 0.8496±0.0234 | 5.344±0.603 |
| actnerf | 0.9357±0.0089 | 0.9651±0.0050 | 2.677±0.090 |
| ray_gpis | 0.9682±0.0157 | 0.9814±0.0100 | 2.347±0.120 |

## B. Active exploration efficiency

| Method | Final visible surface ↑ | Visible-area AUC ↑ | Recall@5 AUC ↑ | F-score@5 AUC ↑ | Cumulative active F@5 gain ↑ |
|---|---:|---:|---:|---:|---:|
| fixed | 0.2770±0.0469 | 0.1868±0.0366 | 0.6780±0.0078 | 0.7948±0.0061 | 0.3543±0.0090 |
| pb_nbv | 0.2443±0.0422 | 0.1749±0.0359 | 0.6188±0.0258 | 0.7475±0.0193 | 0.2638±0.0216 |
| actnerf | 0.2828±0.0487 | 0.2059±0.0433 | 0.7847±0.0095 | 0.8665±0.0062 | 0.3792±0.0097 |
| ray_gpis | 0.2843±0.0513 | 0.2135±0.0473 | 0.8397±0.0136 | 0.8987±0.0091 | 0.3956±0.0132 |

Visible surface is the cumulative area of GT mesh triangles hit by the actually acquired camera rays, divided by total GT mesh area. AUC is the trapezoidal mean over six cumulative reconstruction states: one initial state and five active-action states. The initial state contains 1 shared observation(s).

## C. Active decision policy

| Method | Selected gain/action ↑ | Oracle gain/action ↑ | Score–gain Spearman ↑ | Valid corr. steps ↑ | Oracle regret ↓ | Oracle accuracy ↑ |
|---|---:|---:|---:|---:|---:|---:|
| fixed | 0.0709±0.0018 | 0.1515±0.0035 | — | — | 0.0806±0.0025 | 29.0±1.8% |
| pb_nbv | 0.0528±0.0043 | 0.1603±0.0094 | -0.3563±0.0731 | 93.8±1.9% | 0.1076±0.0124 | 19.2±3.8% |
| actnerf | 0.0758±0.0019 | 0.1076±0.0032 | 0.1619±0.0337 | 99.0±1.0% | 0.0317±0.0029 | 42.3±2.4% |
| ray_gpis | 0.0791±0.0026 | 0.0879±0.0040 | 0.4722±0.0667 | 97.8±1.5% | 0.0088±0.0051 | 68.0±4.1% |

At each state, all three executable actions {-x, -y, +z} are rolled out counterfactually. Spearman correlation compares their action scores with their realized F@5 gains. Fixed has no planner scores, so only its correlation is undefined; oracle accuracy remains measurable from its selected action.

## D. Runtime per planning step

| Method | Representation update (s) ↓ | Candidate scoring (s) ↓ | Total planning (s) ↓ |
|---|---:|---:|---:|
| fixed | 0.000±0.000 | 0.000000±0.000000 | 0.000±0.000 |
| pb_nbv | 0.335±0.024 | 0.056251±0.000881 | 0.391±0.024 |
| actnerf | 2.009±0.016 | 0.229522±0.001585 | 2.238±0.018 |
| ray_gpis | 0.261±0.024 | 0.000072±0.000001 | 0.261±0.024 |

Total planning time is representation update plus candidate scoring. For adapted ActNeRF, representation update is five-model ensemble training and candidate scoring is ensemble rendering/variance evaluation.

## Consistency checks

- Episodes checked: 720
- Maximum |saved selected gain - actual next-state F@5 difference|: 0.000e+00
- Maximum |legacy evaluator surface_coverage - Recall@5| (backward-compatibility check): 0.000e+00
- Minimum visibility-coverage step increment: 0.000e+00
