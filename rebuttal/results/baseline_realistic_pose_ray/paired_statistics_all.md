| Metric | Baseline | Pairs | Ray improvement (95% CI) | W/T/L | p (two-sided) | p (Holm) |
|---|---|---:|---:|---:|---:|---:|
| final_recall@5 | pose_novelty | 120 | -0.004175 [-0.016582, 0.007622] | 53/12/55 | 0.679 | 0.679 |
| final_f@5 | pose_novelty | 120 | -0.000986 [-0.009975, 0.007606] | 56/12/52 | 0.902 | 0.902 |
| final_chamfer_mm | pose_novelty | 120 | 0.042552 [-0.082762, 0.160874] | 57/12/51 | 0.571 | 0.571 |
| final_visibility_coverage | pose_novelty | 120 | -0.009406 [-0.016483, -0.003507] | 49/16/55 | 0.432 | 0.432 |
| visibility_coverage_auc | pose_novelty | 120 | -0.005410 [-0.009509, -0.002010] | 51/14/55 | 0.54 | 0.54 |
| recall@5_auc | pose_novelty | 120 | -0.005138 [-0.011806, 0.000772] | 49/12/59 | 0.242 | 0.242 |
| f@5_auc | pose_novelty | 120 | -0.003453 [-0.008629, 0.001033] | 48/12/60 | 0.29 | 0.29 |
| selected_gain_f@5_per_action | pose_novelty | 120 | -0.000197 [-0.001958, 0.001503] | 56/12/52 | 0.902 | 0.902 |
| score_gain_correlation | pose_novelty | 120 | -0.070833 [-0.136667, -0.006646] | 45/18/57 | 0.142 | 0.142 |
| oracle_regret_f@5 | pose_novelty | 120 | -0.002906 [-0.006090, -0.000159] | 50/12/58 | 0.392 | 0.392 |
| oracle_action_accuracy | pose_novelty | 120 | -0.026667 [-0.071667, 0.018333] | 38/43/39 | 0.389 | 0.389 |
| consecutive_score_map_correlation | pose_novelty | 120 | 0.534958 [0.498651, 0.570727] | 120/0/0 | 1.97e-21 | 1.97e-21 |
| planning_time_s | pose_novelty | 120 | -0.129485 [-0.142944, -0.117850] | 0/0/120 | 1.97e-21 | 1.97e-21 |
