# Complete experiment metric summary

All uncertainty intervals are 95% normal-approximation confidence intervals over 120 paired object/pose scenes.

## A. Final geometry

| Method | Recall@5 ↑ | F-score@5 ↑ | Chamfer (mm) ↓ |
|---|---:|---:|---:|
| Novelty only | 0.7164±0.0232 | 0.8081±0.0181 | 6.920±0.450 |
| Uncertainty only | 0.7456±0.0183 | 0.8272±0.0139 | 6.386±0.258 |
| Pointwise GP | 0.7538±0.0168 | 0.8338±0.0124 | 6.247±0.226 |
| Hit rays only | 0.7182±0.0201 | 0.8091±0.0147 | 6.872±0.346 |
| Full Ray-GPIS | 0.7509±0.0170 | 0.8320±0.0127 | 6.301±0.234 |

## B. Active exploration efficiency

| Method | Final visible surface ↑ | Visible-area AUC ↑ | Recall@5 AUC ↑ | F-score@5 AUC ↑ | Cumulative active F@5 gain ↑ |
|---|---:|---:|---:|---:|---:|
| Novelty only | 0.2828±0.0504 | 0.1945±0.0406 | 0.4811±0.0182 | 0.6141±0.0174 | 0.4589±0.0193 |
| Uncertainty only | 0.2822±0.0497 | 0.1979±0.0416 | 0.5032±0.0162 | 0.6323±0.0156 | 0.4780±0.0177 |
| Pointwise GP | 0.2849±0.0503 | 0.1986±0.0416 | 0.5077±0.0157 | 0.6365±0.0152 | 0.4846±0.0175 |
| Hit rays only | 0.2731±0.0468 | 0.1896±0.0385 | 0.4874±0.0164 | 0.6199±0.0157 | 0.4599±0.0188 |
| Full Ray-GPIS | 0.2846±0.0503 | 0.1986±0.0416 | 0.5066±0.0157 | 0.6358±0.0152 | 0.4828±0.0171 |

Visible surface is the cumulative area of GT mesh triangles hit by the actually acquired camera rays, divided by total GT mesh area. AUC is the trapezoidal mean over six cumulative reconstruction states: one initial state and five active-action states. The initial state contains 1 shared observation(s).

## C. Active decision policy

| Method | Selected gain/action ↑ | Oracle gain/action ↑ | Score–gain Spearman ↑ | Valid corr. steps ↑ | Oracle regret ↓ | Oracle accuracy ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Novelty only | 0.0918±0.0039 | 0.1098±0.0043 | 0.4039±0.0596 | 82.3±4.7% | 0.0180±0.0032 | 57.3±4.5% |
| Uncertainty only | 0.0956±0.0035 | 0.1072±0.0039 | 0.3953±0.0525 | 99.2±0.9% | 0.0116±0.0023 | 63.0±4.1% |
| Pointwise GP | 0.0969±0.0035 | 0.1077±0.0039 | 0.4092±0.0499 | 100.0±0.0% | 0.0107±0.0021 | 64.0±3.8% |
| Hit rays only | 0.0920±0.0038 | 0.1094±0.0042 | 0.3511±0.0548 | 100.0±0.0% | 0.0174±0.0031 | 55.7±4.4% |
| Full Ray-GPIS | 0.0966±0.0034 | 0.1073±0.0038 | 0.4161±0.0497 | 100.0±0.0% | 0.0108±0.0021 | 64.3±4.0% |

At each state, all three executable actions {-x, -y, +z} are rolled out counterfactually. Spearman correlation compares their action scores with their realized F@5 gains. Correlation is averaged over steps with non-constant finite action scores and realized gains.

## D. Runtime per planning step

| Method | Representation update (s) ↓ | Candidate scoring (s) ↓ | Total planning (s) ↓ |
|---|---:|---:|---:|
| Novelty only | 0.135±0.015 | 0.000067±0.000001 | 0.135±0.015 |
| Uncertainty only | 0.139±0.015 | 0.000068±0.000001 | 0.139±0.015 |
| Pointwise GP | 0.138±0.015 | 0.000066±0.000001 | 0.138±0.015 |
| Hit rays only | 0.136±0.014 | 0.000073±0.000012 | 0.136±0.014 |
| Full Ray-GPIS | 0.138±0.015 | 0.000066±0.000001 | 0.138±0.015 |

Total planning time is representation update plus candidate scoring, with CUDA synchronization at both phase boundaries.

## Consistency checks

- Episodes checked: 600
- Maximum |saved selected gain - actual next-state F@5 difference|: 0.000e+00
- Maximum |legacy evaluator surface_coverage - Recall@5| (backward-compatibility check): 0.000e+00
- Minimum visibility-coverage step increment: 0.000e+00
