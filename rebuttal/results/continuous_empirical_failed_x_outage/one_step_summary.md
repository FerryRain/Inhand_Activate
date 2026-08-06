# Six-second continuous one-decision summary

Primary analysis: selected F@5 gain over the complete six-second empirical trajectory. Positive paired improvement means Ray-GPIS is better.

| Planner | Selected F gain | Recall gain | Missing-surface recovery | Chamfer reduction (mm) | Corr. | Regret | Accuracy | Accepted frames | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pose_novelty | 0.2991+-0.0153 | 0.4182+-0.0208 | 0.8006+-0.0320 | 7.739+-0.564 | 0.3667+-0.1193 | 0.0223+-0.0090 | 0.6833+-0.0836 | 17.14+-1.02 | 0.000+-0.000 |
| ray_gpis | 0.2965+-0.0131 | 0.4117+-0.0174 | 0.7895+-0.0261 | 7.803+-0.545 | 0.2894+-0.1176 | 0.0248+-0.0072 | 0.5417+-0.0895 | 16.30+-1.01 | 0.546+-0.079 |

The hard-motion stratum is the predeclared top quartile of scene-level mean first-second transition error (threshold 51.58 deg); membership uses trajectory metadata only, not reconstruction outcomes.

| Subset | Metric | Pairs | Ray improvement [95% CI] | W/T/L | p |
|---|---|---:|---:|---:|---:|
| all | selected_gain_f@5 | 120 | -0.00253 [-0.01412, 0.00975] | 33/51/36 | 0.275 |
| all | selected_recovery_recall@5 | 120 | -0.01111 [-0.04660, 0.02558] | 33/51/36 | 0.3 |
| all | score_gain_correlation | 120 | -0.07723 [-0.23862, 0.08527] | 36/44/40 | 0.349 |
| all | oracle_regret_f@5 | 120 | -0.00253 [-0.01412, 0.00975] | 33/51/36 | 0.275 |
| hard_first_second_quartile | selected_gain_f@5 | 30 | 0.01804 [-0.00629, 0.04610] | 11/12/7 | 0.393 |
| hard_first_second_quartile | selected_recovery_recall@5 | 30 | 0.04838 [-0.02314, 0.12722] | 11/12/7 | 0.347 |
| hard_first_second_quartile | score_gain_correlation | 30 | 0.22887 [-0.11667, 0.56220] | 15/7/8 | 0.223 |
| hard_first_second_quartile | oracle_regret_f@5 | 30 | 0.01804 [-0.00629, 0.04610] | 11/12/7 | 0.393 |
| planner_action_disagreement | selected_gain_f@5 | 69 | -0.00440 [-0.02462, 0.01792] | 33/0/36 | 0.275 |
| planner_action_disagreement | selected_recovery_recall@5 | 69 | -0.01932 [-0.08092, 0.04721] | 33/0/36 | 0.3 |
| planner_action_disagreement | score_gain_correlation | 69 | -0.13962 [-0.40580, 0.12849] | 22/21/26 | 0.251 |
| planner_action_disagreement | oracle_regret_f@5 | 69 | -0.00440 [-0.02462, 0.01792] | 33/0/36 | 0.275 |
