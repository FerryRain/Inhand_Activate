# Complete experiment metric summary

All uncertainty intervals are 95% normal-approximation confidence intervals over 120 paired object/pose scenes.

## A. Final geometry

| Method | Recall@5 ↑ | F-score@5 ↑ | Chamfer (mm) ↓ |
|---|---:|---:|---:|
| Novelty only | 0.9665±0.0133 | 0.9795±0.0084 | 3.844±0.098 |
| Uncertainty only | 0.9650±0.0133 | 0.9790±0.0084 | 3.844±0.097 |
| Pointwise GP | 0.9698±0.0130 | 0.9813±0.0082 | 3.818±0.095 |
| Hit rays only | 0.9698±0.0096 | 0.9821±0.0057 | 3.831±0.081 |
| Full Ray-GPIS | 0.9662±0.0133 | 0.9793±0.0084 | 3.846±0.097 |

## B. Active exploration efficiency

| Method | Final visible surface ↑ | Visible-area AUC ↑ | Recall@5 AUC ↑ | F-score@5 AUC ↑ | Cumulative active F@5 gain ↑ |
|---|---:|---:|---:|---:|---:|
| Novelty only | 0.2826±0.0511 | 0.2131±0.0473 | 0.8265±0.0132 | 0.8895±0.0091 | 0.3937±0.0125 |
| Uncertainty only | 0.2837±0.0512 | 0.2122±0.0470 | 0.8253±0.0137 | 0.8892±0.0093 | 0.3932±0.0120 |
| Pointwise GP | 0.2847±0.0512 | 0.2133±0.0471 | 0.8324±0.0124 | 0.8936±0.0084 | 0.3955±0.0127 |
| Hit rays only | 0.2857±0.0506 | 0.2092±0.0459 | 0.8032±0.0137 | 0.8744±0.0093 | 0.3963±0.0100 |
| Full Ray-GPIS | 0.2837±0.0512 | 0.2133±0.0473 | 0.8318±0.0124 | 0.8932±0.0084 | 0.3935±0.0125 |

Visible surface is the cumulative area of GT mesh triangles hit by the actually acquired camera rays, divided by total GT mesh area. AUC is the trapezoidal mean over six cumulative reconstruction states: one initial state and five active-action states. The initial state contains 1 shared observation(s).

## C. Active decision policy

| Method | Selected gain/action ↑ | Oracle gain/action ↑ | Score–gain Spearman ↑ | Valid corr. steps ↑ | Oracle regret ↓ | Oracle accuracy ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Novelty only | 0.0787±0.0025 | 0.0887±0.0037 | 0.3806±0.0653 | 98.5±1.3% | 0.0099±0.0044 | 63.2±4.2% |
| Uncertainty only | 0.0786±0.0024 | 0.0882±0.0036 | 0.4302±0.0676 | 99.2±0.7% | 0.0096±0.0044 | 65.2±4.5% |
| Pointwise GP | 0.0791±0.0025 | 0.0865±0.0033 | 0.4596±0.0658 | 100.0±0.0% | 0.0074±0.0041 | 66.8±4.3% |
| Hit rays only | 0.0793±0.0020 | 0.0962±0.0042 | 0.4304±0.0571 | 99.8±0.3% | 0.0169±0.0041 | 58.2±3.9% |
| Full Ray-GPIS | 0.0787±0.0025 | 0.0863±0.0033 | 0.4066±0.0673 | 99.8±0.3% | 0.0076±0.0041 | 63.8±4.4% |

At each state, all three executable actions {-x, -y, +z} are rolled out counterfactually. Spearman correlation compares their action scores with their realized F@5 gains. Correlation is averaged over steps with non-constant finite action scores and realized gains.

## D. Runtime per planning step

| Method | Representation update (s) ↓ | Candidate scoring (s) ↓ | Total planning (s) ↓ |
|---|---:|---:|---:|
| Novelty only | 1.088±0.164 | 0.000091±0.000003 | 1.088±0.164 |
| Uncertainty only | 1.172±0.099 | 0.000100±0.000003 | 1.172±0.099 |
| Pointwise GP | 1.064±0.105 | 0.000097±0.000002 | 1.064±0.105 |
| Hit rays only | 1.048±0.089 | 0.000099±0.000004 | 1.048±0.089 |
| Full Ray-GPIS | 0.990±0.063 | 0.000096±0.000002 | 0.990±0.063 |

Total planning time is representation update plus candidate scoring, with CUDA synchronization at both phase boundaries.

## Consistency checks

- Episodes checked: 600
- Maximum |saved selected gain - actual next-state F@5 difference|: 0.000e+00
- Maximum |legacy evaluator surface_coverage - Recall@5| (backward-compatibility check): 0.000e+00
- Minimum visibility-coverage step increment: 0.000e+00
