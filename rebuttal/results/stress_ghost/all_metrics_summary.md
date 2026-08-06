# Complete experiment metric summary

All uncertainty intervals are 95% normal-approximation confidence intervals over 64 paired object/pose scenes.

## A. Final geometry

| Method | Recall@5 ↑ | F-score@5 ↑ | Chamfer (mm) ↓ |
|---|---:|---:|---:|
| Uncertainty only | 0.9523±0.0193 | 0.9459±0.0108 | 3.529±0.133 |
| Full Ray-GPIS | 0.9457±0.0191 | 0.9423±0.0107 | 3.570±0.132 |

## B. Active exploration efficiency

| Method | Final visible surface ↑ | Visible-area AUC ↑ | Recall@5 AUC ↑ | F-score@5 AUC ↑ | Cumulative active F@5 gain ↑ |
|---|---:|---:|---:|---:|---:|
| Uncertainty only | 0.2794±0.0688 | 0.2096±0.0637 | 0.7705±0.0157 | 0.8189±0.0108 | 0.3680±0.0158 |
| Full Ray-GPIS | 0.2820±0.0701 | 0.2103±0.0641 | 0.7671±0.0150 | 0.8167±0.0102 | 0.3643±0.0165 |

Visible surface is the cumulative area of GT mesh triangles hit by the actually acquired camera rays, divided by total GT mesh area. AUC is the trapezoidal mean over six cumulative reconstruction states: one initial state and five active-action states. The initial state contains 1 shared observation(s).

## C. Active decision policy

| Method | Selected gain/action ↑ | Oracle gain/action ↑ | Score–gain Spearman ↑ | Valid corr. steps ↑ | Oracle regret ↓ | Oracle accuracy ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Uncertainty only | 0.0736±0.0032 | 0.0845±0.0042 | 0.4106±0.0973 | 99.1±1.0% | 0.0109±0.0051 | 62.2±6.4% |
| Full Ray-GPIS | 0.0729±0.0033 | 0.0852±0.0041 | 0.3891±0.0876 | 100.0±0.0% | 0.0124±0.0051 | 57.5±5.5% |

At each state, all three executable actions {-x, -y, +z} are rolled out counterfactually. Spearman correlation compares their action scores with their realized F@5 gains. Correlation is averaged over steps with non-constant finite action scores and realized gains.

## D. Runtime per planning step

| Method | Representation update (s) ↓ | Candidate scoring (s) ↓ | Total planning (s) ↓ |
|---|---:|---:|---:|
| Uncertainty only | 0.294±0.029 | 0.000053±0.000001 | 0.294±0.029 |
| Full Ray-GPIS | 0.295±0.028 | 0.000055±0.000001 | 0.295±0.028 |

Total planning time is representation update plus candidate scoring, with CUDA synchronization at both phase boundaries.

## Consistency checks

- Episodes checked: 128
- Maximum |saved selected gain - actual next-state F@5 difference|: 0.000e+00
- Maximum |legacy evaluator surface_coverage - Recall@5| (backward-compatibility check): 0.000e+00
- Minimum visibility-coverage step increment: 0.000e+00
