# Complete experiment metric summary

All uncertainty intervals are 95% normal-approximation confidence intervals over 64 paired object/pose scenes.

## A. Final geometry

| Method | Recall@5 ↑ | F-score@5 ↑ | Chamfer (mm) ↓ |
|---|---:|---:|---:|
| Pose-Novelty | 0.9088±0.0103 | 0.9517±0.0058 | 3.373±0.070 |
| Adapted ER-GPIS | 0.9032±0.0123 | 0.9484±0.0071 | 3.430±0.085 |
| Hit rays only | 0.9070±0.0143 | 0.9502±0.0083 | 3.380±0.094 |
| Full Ray-GPIS | 0.9026±0.0204 | 0.9465±0.0132 | 3.430±0.143 |

## B. Active exploration efficiency

| Method | Final visible surface ↑ | Visible-area AUC ↑ | Recall@5 AUC ↑ | F-score@5 AUC ↑ | Cumulative active F@5 gain ↑ |
|---|---:|---:|---:|---:|---:|
| Pose-Novelty | 0.2982±0.0716 | 0.2181±0.0643 | 0.6809±0.0099 | 0.7874±0.0073 | 0.5646±0.0132 |
| Adapted ER-GPIS | 0.2873±0.0696 | 0.2111±0.0626 | 0.6705±0.0120 | 0.7793±0.0092 | 0.5612±0.0135 |
| Hit rays only | 0.2922±0.0709 | 0.2144±0.0645 | 0.6582±0.0152 | 0.7696±0.0114 | 0.5631±0.0134 |
| Full Ray-GPIS | 0.2837±0.0692 | 0.2120±0.0640 | 0.6748±0.0147 | 0.7821±0.0108 | 0.5593±0.0177 |

Visible surface is the cumulative area of GT mesh triangles hit by the actually acquired camera rays, divided by total GT mesh area. AUC is the trapezoidal mean over six cumulative reconstruction states: one initial state and five active-action states. The initial state contains 1 shared observation(s).

## C. Active decision policy

| Method | Selected gain/action ↑ | Oracle gain/action ↑ | Score–gain Spearman ↑ | Valid corr. steps ↑ | Oracle regret ↓ | Oracle accuracy ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Pose-Novelty | 0.1129±0.0026 | 0.1185±0.0026 | 0.4203±0.0490 | 100.0±0.0% | 0.0056±0.0012 | 70.3±3.6% |
| Adapted ER-GPIS | 0.1122±0.0027 | 0.1226±0.0035 | 0.4219±0.0627 | 100.0±0.0% | 0.0104±0.0030 | 64.7±4.5% |
| Hit rays only | 0.1126±0.0027 | 0.1277±0.0040 | 0.4551±0.0641 | 100.0±0.0% | 0.0151±0.0042 | 65.0±5.0% |
| Full Ray-GPIS | 0.1119±0.0035 | 0.1210±0.0035 | 0.4895±0.0917 | 100.0±0.0% | 0.0091±0.0051 | 72.5±5.8% |

At each state, all three executable actions {-x, -y, +z} are rolled out counterfactually. Spearman correlation compares their action scores with their realized F@5 gains. Correlation is averaged over steps with non-constant finite action scores and realized gains.

## D. Runtime per planning step

| Method | Representation update (s) ↓ | Candidate scoring (s) ↓ | Total planning (s) ↓ |
|---|---:|---:|---:|
| Pose-Novelty | 0.000±0.000 | 0.000209±0.000005 | 0.000±0.000 |
| Adapted ER-GPIS | 0.093±0.003 | 0.003363±0.000114 | 0.096±0.003 |
| Hit rays only | 0.130±0.013 | 0.000057±0.000002 | 0.130±0.013 |
| Full Ray-GPIS | 0.130±0.014 | 0.000059±0.000002 | 0.130±0.014 |

Total planning time is representation update plus candidate scoring, with CUDA synchronization at both phase boundaries.

## Consistency checks

- Episodes checked: 256
- Maximum |saved selected gain - actual next-state F@5 difference|: 0.000e+00
- Maximum |legacy evaluator surface_coverage - Recall@5| (backward-compatibility check): 0.000e+00
- Minimum visibility-coverage step increment: 0.000e+00
