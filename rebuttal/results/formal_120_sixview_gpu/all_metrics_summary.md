# Complete experiment metric summary

All uncertainty intervals are 95% normal-approximation confidence intervals over 120 paired object/pose scenes. ActNeRF's initialization seeds are averaged within each scene first.

## A. Final geometry

| Method | Recall@5 ↑ | F-score@5 ↑ | Chamfer (mm) ↓ |
|---|---:|---:|---:|
| Fixed schedule | 0.8881±0.0082 | 0.9401±0.0046 | 3.027±0.079 |
| Pose-Novelty | 0.9891±0.0027 | 0.9945±0.0014 | 2.176±0.031 |
| Adapted PB-NBV | 0.7597±0.0338 | 0.8496±0.0234 | 5.344±0.603 |
| Adapted ER-GPIS | 0.9807±0.0065 | 0.9899±0.0037 | 2.272±0.053 |
| Full Ray-GPIS | 0.9682±0.0157 | 0.9814±0.0100 | 2.347±0.120 |
| Adapted ActNeRF | 0.9357±0.0089 | 0.9651±0.0050 | 2.677±0.090 |

## B. Active exploration efficiency

| Method | Final visible surface ↑ | Visible-area AUC ↑ | Recall@5 AUC ↑ | F-score@5 AUC ↑ | Cumulative active F@5 gain ↑ |
|---|---:|---:|---:|---:|---:|
| Fixed schedule | 0.2770±0.0469 | 0.1868±0.0366 | 0.6780±0.0078 | 0.7948±0.0061 | 0.3543±0.0090 |
| Pose-Novelty | 0.2982±0.0523 | 0.2201±0.0476 | 0.8553±0.0066 | 0.9090±0.0043 | 0.4086±0.0102 |
| Adapted PB-NBV | 0.2443±0.0422 | 0.1749±0.0359 | 0.6188±0.0258 | 0.7475±0.0193 | 0.2638±0.0216 |
| Adapted ER-GPIS | 0.2941±0.0520 | 0.2173±0.0473 | 0.8459±0.0086 | 0.9033±0.0055 | 0.4041±0.0102 |
| Full Ray-GPIS | 0.2843±0.0513 | 0.2135±0.0473 | 0.8397±0.0136 | 0.8987±0.0091 | 0.3956±0.0132 |
| Adapted ActNeRF | 0.2828±0.0487 | 0.2059±0.0433 | 0.7847±0.0095 | 0.8665±0.0062 | 0.3792±0.0097 |

Visible surface is the cumulative area of GT mesh triangles hit by the actually acquired camera rays, divided by total GT mesh area. AUC is the trapezoidal mean over six cumulative reconstruction states: one initial state and five active-action states. The initial state contains 1 shared observation(s).

## C. Active decision policy

| Method | Selected gain/action ↑ | Oracle gain/action ↑ | Score–gain Spearman ↑ | Valid corr. steps ↑ | Oracle regret ↓ | Oracle accuracy ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Fixed schedule | 0.0709±0.0018 | 0.1515±0.0035 | — | — | 0.0806±0.0025 | 29.0±1.8% |
| Pose-Novelty | 0.0817±0.0020 | 0.0842±0.0023 | 0.6186±0.0294 | 98.7±1.3% | 0.0025±0.0006 | 71.0±2.0% |
| Adapted PB-NBV | 0.0528±0.0043 | 0.1603±0.0094 | -0.3563±0.0731 | 93.8±1.9% | 0.1076±0.0124 | 19.2±3.8% |
| Adapted ER-GPIS | 0.0808±0.0020 | 0.0868±0.0027 | 0.3439±0.0587 | 98.3±1.4% | 0.0060±0.0022 | 63.2±4.0% |
| Full Ray-GPIS | 0.0791±0.0026 | 0.0879±0.0040 | 0.4722±0.0667 | 97.8±1.5% | 0.0088±0.0051 | 68.0±4.1% |
| Adapted ActNeRF | 0.0758±0.0019 | 0.1076±0.0032 | 0.1619±0.0337 | 99.0±1.0% | 0.0317±0.0029 | 42.3±2.4% |

At each state, all three executable actions {-x, -y, +z} are rolled out counterfactually. Spearman correlation compares their action scores with their realized F@5 gains. Fixed has no planner scores, so only its correlation is undefined; oracle accuracy remains measurable from its selected action.

## D. Runtime per planning step

| Method | Representation update (s) ↓ | Candidate scoring (s) ↓ | Total planning (s) ↓ |
|---|---:|---:|---:|
| Fixed schedule | 0.000±0.000 | 0.000000±0.000000 | 0.000±0.000 |
| Pose-Novelty | 0.000±0.000 | 0.000249±0.000008 | 0.000±0.000 |
| Adapted PB-NBV | 0.335±0.024 | 0.056251±0.000881 | 0.391±0.024 |
| Adapted ER-GPIS | 0.093±0.002 | 0.003960±0.000070 | 0.097±0.002 |
| Full Ray-GPIS | 0.261±0.024 | 0.000072±0.000001 | 0.261±0.024 |
| Adapted ActNeRF | 2.009±0.016 | 0.229522±0.001585 | 2.238±0.018 |

Total planning time is representation update plus candidate scoring. For adapted ActNeRF, representation update is five-model ensemble training and candidate scoring is ensemble rendering/variance evaluation.

## Consistency checks

- Episodes checked: 960
- Maximum |saved selected gain - actual next-state F@5 difference|: 0.000e+00
- Maximum |legacy evaluator surface_coverage - Recall@5| (backward-compatibility check): 0.000e+00
- Minimum visibility-coverage step increment: 0.000e+00
