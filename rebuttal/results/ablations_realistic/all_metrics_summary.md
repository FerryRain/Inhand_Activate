# Complete experiment metric summary

All uncertainty intervals are 95% normal-approximation confidence intervals over 120 paired object/pose scenes.

## A. Final geometry

| Method | Recall@5 ↑ | F-score@5 ↑ | Chamfer (mm) ↓ |
|---|---:|---:|---:|
| Pose-Novelty | 0.8174±0.0135 | 0.8794±0.0110 | 5.572±0.191 |
| Adapted ER-GPIS | 0.8020±0.0139 | 0.8690±0.0111 | 5.716±0.191 |
| Novelty only | 0.7880±0.0172 | 0.8602±0.0135 | 5.818±0.250 |
| Uncertainty only | 0.8112±0.0159 | 0.8769±0.0125 | 5.559±0.204 |
| Pointwise GP | 0.8157±0.0157 | 0.8789±0.0124 | 5.530±0.194 |
| Hit rays only | 0.8209±0.0128 | 0.8836±0.0099 | 5.466±0.169 |
| Full Ray-GPIS | 0.8132±0.0159 | 0.8784±0.0124 | 5.530±0.194 |

## B. Active exploration efficiency

| Method | Final visible surface ↑ | Visible-area AUC ↑ | Recall@5 AUC ↑ | F-score@5 AUC ↑ | Cumulative active F@5 gain ↑ |
|---|---:|---:|---:|---:|---:|
| Pose-Novelty | 0.2984±0.0520 | 0.2195±0.0473 | 0.6015±0.0123 | 0.7193±0.0115 | 0.5141±0.0142 |
| Adapted ER-GPIS | 0.2931±0.0515 | 0.2154±0.0469 | 0.5876±0.0134 | 0.7088±0.0122 | 0.5037±0.0148 |
| Novelty only | 0.2874±0.0512 | 0.2113±0.0464 | 0.5637±0.0154 | 0.6884±0.0139 | 0.4950±0.0165 |
| Uncertainty only | 0.2876±0.0510 | 0.2134±0.0468 | 0.5940±0.0134 | 0.7137±0.0122 | 0.5116±0.0159 |
| Pointwise GP | 0.2886±0.0511 | 0.2140±0.0468 | 0.5969±0.0133 | 0.7158±0.0122 | 0.5137±0.0156 |
| Hit rays only | 0.2926±0.0514 | 0.2114±0.0449 | 0.5859±0.0123 | 0.7082±0.0111 | 0.5184±0.0146 |
| Full Ray-GPIS | 0.2890±0.0511 | 0.2141±0.0468 | 0.5964±0.0135 | 0.7159±0.0122 | 0.5131±0.0157 |

Visible surface is the cumulative area of GT mesh triangles hit by the actually acquired camera rays, divided by total GT mesh area. AUC is the trapezoidal mean over six cumulative reconstruction states: one initial state and five active-action states. The initial state contains 1 shared observation(s).

## C. Active decision policy

| Method | Selected gain/action ↑ | Oracle gain/action ↑ | Score–gain Spearman ↑ | Valid corr. steps ↑ | Oracle regret ↓ | Oracle accuracy ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Pose-Novelty | 0.1028±0.0028 | 0.1118±0.0031 | 0.5017±0.0454 | 100.0±0.0% | 0.0090±0.0018 | 66.2±3.6% |
| Adapted ER-GPIS | 0.1007±0.0030 | 0.1151±0.0036 | 0.3575±0.0520 | 100.0±0.0% | 0.0144±0.0030 | 60.8±3.8% |
| Novelty only | 0.0990±0.0033 | 0.1210±0.0042 | 0.3992±0.0668 | 87.8±3.5% | 0.0220±0.0042 | 54.8±4.4% |
| Uncertainty only | 0.1023±0.0032 | 0.1149±0.0033 | 0.4443±0.0638 | 99.2±0.7% | 0.0126±0.0032 | 62.0±4.3% |
| Pointwise GP | 0.1027±0.0031 | 0.1146±0.0032 | 0.4283±0.0661 | 100.0±0.0% | 0.0118±0.0030 | 62.5±4.5% |
| Hit rays only | 0.1037±0.0029 | 0.1191±0.0035 | 0.4789±0.0514 | 100.0±0.0% | 0.0154±0.0029 | 61.8±4.0% |
| Full Ray-GPIS | 0.1026±0.0031 | 0.1145±0.0032 | 0.4308±0.0653 | 100.0±0.0% | 0.0119±0.0031 | 63.5±4.5% |

At each state, all three executable actions {-x, -y, +z} are rolled out counterfactually. Spearman correlation compares their action scores with their realized F@5 gains. Correlation is averaged over steps with non-constant finite action scores and realized gains.

## D. Runtime per planning step

| Method | Representation update (s) ↓ | Candidate scoring (s) ↓ | Total planning (s) ↓ |
|---|---:|---:|---:|
| Pose-Novelty | 0.000±0.000 | 0.000251±0.000009 | 0.000±0.000 |
| Adapted ER-GPIS | 0.095±0.002 | 0.003853±0.000082 | 0.098±0.002 |
| Novelty only | 0.139±0.013 | 0.000076±0.000002 | 0.139±0.013 |
| Uncertainty only | 0.137±0.013 | 0.000076±0.000001 | 0.137±0.013 |
| Pointwise GP | 0.135±0.013 | 0.000072±0.000001 | 0.135±0.013 |
| Hit rays only | 0.175±0.013 | 0.000084±0.000001 | 0.175±0.013 |
| Full Ray-GPIS | 0.136±0.013 | 0.000074±0.000002 | 0.136±0.013 |

Total planning time is representation update plus candidate scoring, with CUDA synchronization at both phase boundaries.

## Consistency checks

- Episodes checked: 840
- Maximum |saved selected gain - actual next-state F@5 difference|: 0.000e+00
- Maximum |legacy evaluator surface_coverage - Recall@5| (backward-compatibility check): 0.000e+00
- Minimum visibility-coverage step increment: 0.000e+00
