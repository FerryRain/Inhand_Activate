# Complete experiment metric summary

All uncertainty intervals are 95% normal-approximation confidence intervals over 120 paired object/pose scenes.

## A. Final geometry

| Method | Recall@5 ↑ | F-score@5 ↑ | Chamfer (mm) ↓ |
|---|---:|---:|---:|
| Novelty only | 0.9672±0.0158 | 0.9809±0.0100 | 2.360±0.120 |
| Uncertainty only | 0.9655±0.0161 | 0.9799±0.0101 | 2.370±0.122 |
| Pointwise GP | 0.9697±0.0153 | 0.9822±0.0098 | 2.339±0.116 |
| Hit rays only | 0.9726±0.0116 | 0.9849±0.0069 | 2.314±0.085 |
| Full Ray-GPIS | 0.9682±0.0157 | 0.9814±0.0100 | 2.347±0.120 |

## B. Active exploration efficiency

| Method | Final visible surface ↑ | Visible-area AUC ↑ | Recall@5 AUC ↑ | F-score@5 AUC ↑ | Cumulative active F@5 gain ↑ |
|---|---:|---:|---:|---:|---:|
| Novelty only | 0.2837±0.0513 | 0.2132±0.0472 | 0.8341±0.0143 | 0.8949±0.0096 | 0.3950±0.0131 |
| Uncertainty only | 0.2838±0.0512 | 0.2125±0.0470 | 0.8344±0.0145 | 0.8953±0.0096 | 0.3941±0.0132 |
| Pointwise GP | 0.2848±0.0513 | 0.2135±0.0472 | 0.8404±0.0135 | 0.8991±0.0090 | 0.3964±0.0137 |
| Hit rays only | 0.2887±0.0516 | 0.2080±0.0452 | 0.8094±0.0149 | 0.8791±0.0099 | 0.3990±0.0110 |
| Full Ray-GPIS | 0.2843±0.0513 | 0.2135±0.0473 | 0.8397±0.0136 | 0.8987±0.0091 | 0.3956±0.0132 |

Visible surface is the cumulative area of GT mesh triangles hit by the actually acquired camera rays, divided by total GT mesh area. AUC is the trapezoidal mean over six cumulative reconstruction states: one initial state and five active-action states. The initial state contains 1 shared observation(s).

## C. Active decision policy

| Method | Selected gain/action ↑ | Oracle gain/action ↑ | Score–gain Spearman ↑ | Valid corr. steps ↑ | Oracle regret ↓ | Oracle accuracy ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Novelty only | 0.0790±0.0026 | 0.0902±0.0043 | 0.3990±0.0661 | 96.2±1.9% | 0.0112±0.0053 | 65.2±4.1% |
| Uncertainty only | 0.0788±0.0026 | 0.0893±0.0040 | 0.4342±0.0698 | 96.8±1.6% | 0.0105±0.0051 | 64.8±4.5% |
| Pointwise GP | 0.0793±0.0027 | 0.0879±0.0040 | 0.4623±0.0669 | 97.7±1.5% | 0.0086±0.0051 | 67.8±4.5% |
| Hit rays only | 0.0798±0.0022 | 0.0983±0.0047 | 0.4950±0.0536 | 97.3±1.6% | 0.0185±0.0047 | 60.7±3.8% |
| Full Ray-GPIS | 0.0791±0.0026 | 0.0879±0.0040 | 0.4722±0.0667 | 97.8±1.5% | 0.0088±0.0051 | 68.0±4.1% |

At each state, all three executable actions {-x, -y, +z} are rolled out counterfactually. Spearman correlation compares their action scores with their realized F@5 gains. Correlation is averaged over steps with non-constant finite action scores and realized gains.

## D. Runtime per planning step

| Method | Representation update (s) ↓ | Candidate scoring (s) ↓ | Total planning (s) ↓ |
|---|---:|---:|---:|
| Novelty only | 0.274±0.025 | 0.000080±0.000002 | 0.274±0.025 |
| Uncertainty only | 0.260±0.025 | 0.000072±0.000001 | 0.260±0.025 |
| Pointwise GP | 0.263±0.025 | 0.000070±0.000001 | 0.263±0.025 |
| Hit rays only | 0.291±0.026 | 0.000082±0.000001 | 0.291±0.026 |
| Full Ray-GPIS | 0.261±0.025 | 0.000070±0.000001 | 0.261±0.025 |

Total planning time is representation update plus candidate scoring, with CUDA synchronization at both phase boundaries.

## Consistency checks

- Episodes checked: 600
- Maximum |saved selected gain - actual next-state F@5 difference|: 0.000e+00
- Maximum |legacy evaluator surface_coverage - Recall@5| (backward-compatibility check): 0.000e+00
- Minimum visibility-coverage step increment: 0.000e+00
