| Metric | Baseline | Pairs | Ray improvement (95% CI) | W/T/L | p (two-sided) | p (Holm) |
|---|---|---:|---:|---:|---:|---:|
| final_recall@5 | ray_gpis_uncertainty_only | 64 | -0.006683 [-0.014513, 0.000616] | 18/18/28 | 0.111 | 0.111 |
| final_f@5 | ray_gpis_uncertainty_only | 64 | -0.003659 [-0.007659, 0.000038] | 18/18/28 | 0.0766 | 0.0766 |
| final_chamfer_mm | ray_gpis_uncertainty_only | 64 | -0.041603 [-0.091754, 0.008364] | 17/18/29 | 0.114 | 0.114 |
| final_visibility_coverage | ray_gpis_uncertainty_only | 64 | 0.002541 [-0.000691, 0.007572] | 22/19/23 | 0.623 | 0.623 |
| visibility_coverage_auc | ray_gpis_uncertainty_only | 64 | 0.000695 [-0.000162, 0.001649] | 25/19/20 | 0.238 | 0.238 |
| recall@5_auc | ray_gpis_uncertainty_only | 64 | -0.003358 [-0.008590, 0.001680] | 17/18/29 | 0.0553 | 0.0553 |
| f@5_auc | ray_gpis_uncertainty_only | 64 | -0.002221 [-0.005263, 0.000627] | 18/18/28 | 0.0629 | 0.0629 |
| selected_gain_f@5_per_action | ray_gpis_uncertainty_only | 64 | -0.000732 [-0.001495, 0.000001] | 18/18/28 | 0.0766 | 0.0766 |
| score_gain_correlation | ray_gpis_uncertainty_only | 64 | -0.021551 [-0.080593, 0.036441] | 28/5/31 | 0.548 | 0.548 |
| oracle_regret_f@5 | ray_gpis_uncertainty_only | 64 | -0.001414 [-0.003123, 0.000225] | 19/18/27 | 0.0971 | 0.0971 |
| oracle_action_accuracy | ray_gpis_uncertainty_only | 64 | -0.046875 [-0.096875, 0.000000] | 12/31/21 | 0.0831 | 0.0831 |
| consecutive_score_map_correlation | ray_gpis_uncertainty_only | 64 | -0.267130 [-0.300375, -0.234797] | 1/0/63 | 3.7e-12 | 3.7e-12 |
| planning_time_s | ray_gpis_uncertainty_only | 64 | -0.000731 [-0.006926, 0.005550] | 25/0/39 | 0.0388 | 0.0388 |
