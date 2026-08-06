| Metric | Baseline | Pairs | Ray improvement (95% CI) | W/T/L | p (two-sided) | p (Holm) |
|---|---|---:|---:|---:|---:|---:|
| final_recall@5 | ray_gpis_novelty_only | 64 | 0.000887 [0.000076, 0.001904] | 14/43/7 | 0.0646 | 0.0646 |
| final_f@5 | ray_gpis_novelty_only | 64 | 0.000451 [0.000042, 0.000971] | 14/43/7 | 0.0595 | 0.0595 |
| final_chamfer_mm | ray_gpis_novelty_only | 64 | 0.012336 [-0.000229, 0.025793] | 14/41/9 | 0.105 | 0.105 |
| final_visibility_coverage | ray_gpis_novelty_only | 64 | 0.000198 [-0.000235, 0.000668] | 11/41/12 | 0.754 | 0.754 |
| visibility_coverage_auc | ray_gpis_novelty_only | 64 | 0.000210 [-0.000070, 0.000580] | 13/41/10 | 0.393 | 0.393 |
| recall@5_auc | ray_gpis_novelty_only | 64 | 0.002236 [0.000152, 0.005325] | 14/43/7 | 0.119 | 0.119 |
| f@5_auc | ray_gpis_novelty_only | 64 | 0.001377 [0.000075, 0.003311] | 14/43/7 | 0.119 | 0.119 |
| selected_gain_f@5_per_action | ray_gpis_novelty_only | 64 | 0.000090 [0.000008, 0.000193] | 14/43/7 | 0.0595 | 0.0595 |
| score_gain_correlation | ray_gpis_novelty_only | 64 | 0.057133 [0.015625, 0.102447] | 27/24/13 | 0.0259 | 0.0259 |
| oracle_regret_f@5 | ray_gpis_novelty_only | 64 | 0.000728 [0.000003, 0.001800] | 14/43/7 | 0.103 | 0.103 |
| oracle_action_accuracy | ray_gpis_novelty_only | 64 | 0.028125 [0.003125, 0.053125] | 10/51/3 | 0.0574 | 0.0574 |
| consecutive_score_map_correlation | ray_gpis_novelty_only | 64 | 0.089773 [0.071162, 0.107399] | 58/0/6 | 7.63e-10 | 7.63e-10 |
| planning_time_s | ray_gpis_novelty_only | 64 | -0.005316 [-0.012052, 0.000503] | 21/0/43 | 0.00274 | 0.00274 |
