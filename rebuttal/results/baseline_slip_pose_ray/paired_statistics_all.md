| Metric | Baseline | Pairs | Ray improvement (95% CI) | W/T/L | p (two-sided) | p (Holm) |
|---|---|---:|---:|---:|---:|---:|
| final_recall@5 | pose_novelty | 120 | -0.011473 [-0.022640, -0.001065] | 33/52/35 | 0.204 | 0.204 |
| final_f@5 | pose_novelty | 120 | -0.009103 [-0.016647, -0.002077] | 31/52/37 | 0.127 | 0.127 |
| final_chamfer_mm | pose_novelty | 120 | -0.136688 [-0.273733, -0.003435] | 36/52/32 | 0.243 | 0.243 |
| final_visibility_coverage | pose_novelty | 120 | -0.006511 [-0.012699, -0.001014] | 30/55/35 | 0.133 | 0.133 |
| visibility_coverage_auc | pose_novelty | 120 | -0.003465 [-0.006187, -0.001187] | 28/55/37 | 0.0555 | 0.0555 |
| recall@5_auc | pose_novelty | 120 | -0.008822 [-0.014338, -0.004286] | 29/52/39 | 0.00815 | 0.00815 |
| f@5_auc | pose_novelty | 120 | -0.007652 [-0.012551, -0.003644] | 28/52/40 | 0.00485 | 0.00485 |
| selected_gain_f@5_per_action | pose_novelty | 120 | -0.001821 [-0.003353, -0.000414] | 31/52/37 | 0.127 | 0.127 |
| score_gain_correlation | pose_novelty | 120 | -0.014777 [-0.066226, 0.036058] | 51/22/47 | 0.965 | 0.965 |
| oracle_regret_f@5 | pose_novelty | 120 | -0.002580 [-0.004390, -0.001015] | 30/52/38 | 0.0234 | 0.0234 |
| oracle_action_accuracy | pose_novelty | 120 | -0.035000 [-0.073333, 0.000000] | 20/72/28 | 0.211 | 0.211 |
| consecutive_score_map_correlation | pose_novelty | 120 | 0.347038 [0.325106, 0.369205] | 120/0/0 | 1.97e-21 | 1.97e-21 |
| planning_time_s | pose_novelty | 120 | -0.140761 [-0.157497, -0.125712] | 0/0/120 | 1.97e-21 | 1.97e-21 |
