# Paired 6D pose-noise planner ablation

All variants use the same deterministic 6.0 deg / 3.0 mm tracking perturbation at each post-action state. Action execution and RGB-D are unchanged.

| Variant | Clean F-AUC | Pose-noise F-AUC | F-AUC degradation | Noise Corr. | Noise Regret |
|---|---:|---:|---:|---:|---:|
| Novelty only | 0.8949 | 0.8895 | +0.0054 | 0.3806 | 0.0099 |
| Uncertainty only | 0.8953 | 0.8892 | +0.0062 | 0.4302 | 0.0096 |
| w/o receptive-field integration | 0.8991 | 0.8936 | +0.0055 | 0.4596 | 0.0074 |
| w/o miss-ray interpolation | 0.8791 | 0.8744 | +0.0046 | 0.4304 | 0.0169 |
| Full Ray-GPIS | 0.8987 | 0.8932 | +0.0055 | 0.4066 | 0.0076 |

## Full Ray-GPIS paired improvements under pose noise

| Metric | Ablation | Full improvement (95% CI) | W/T/L | p (Holm) |
|---|---|---:|---:|---:|
| f@5_auc | Novelty only | +0.003787 [+0.000811, +0.007932] | 41/55/24 | 0.277 |
| f@5_auc | Uncertainty only | +0.004094 [+0.001194, +0.007437] | 50/19/51 | 0.495 |
| f@5_auc | w/o receptive-field integration | -0.000348 [-0.001817, +0.001097] | 32/49/39 | 0.495 |
| f@5_auc | w/o miss-ray interpolation | +0.018835 [+0.013167, +0.024828] | 46/63/11 | 1.14e-07 |
| final_f@5 | Novelty only | -0.000215 [-0.001850, +0.001670] | 29/56/35 | 0.997 |
| final_f@5 | Uncertainty only | +0.000349 [-0.002485, +0.003385] | 41/19/60 | 1 |
| final_f@5 | w/o receptive-field integration | -0.002026 [-0.003959, -0.000442] | 28/49/43 | 0.106 |
| final_f@5 | w/o miss-ray interpolation | -0.002772 [-0.008519, +0.002182] | 29/63/28 | 1 |
| score_gain_correlation | Novelty only | +0.026042 [-0.016443, +0.068914] | 47/40/33 | 0.596 |
| score_gain_correlation | Uncertainty only | -0.023595 [-0.073408, +0.026733] | 51/13/56 | 0.596 |
| score_gain_correlation | w/o receptive-field integration | -0.052932 [-0.098562, -0.008940] | 38/31/51 | 0.142 |
| score_gain_correlation | w/o miss-ray interpolation | -0.023780 [-0.065897, +0.017669] | 31/57/32 | 0.596 |
| oracle_regret_f@5 | Novelty only | +0.002354 [+0.000718, +0.004478] | 43/55/22 | 0.0707 |
| oracle_regret_f@5 | Uncertainty only | +0.002014 [+0.000459, +0.003849] | 51/19/50 | 0.804 |
| oracle_regret_f@5 | w/o receptive-field integration | -0.000202 [-0.001012, +0.000544] | 36/49/35 | 0.804 |
| oracle_regret_f@5 | w/o miss-ray interpolation | +0.009377 [+0.006280, +0.012533] | 48/63/9 | 6.71e-07 |
| consecutive_score_map_correlation | Novelty only | +0.080925 [+0.067242, +0.093869] | 106/0/14 | 2.68e-17 |
| consecutive_score_map_correlation | Uncertainty only | -0.246118 [-0.268331, -0.223686] | 0/0/120 | 7.89e-21 |
| consecutive_score_map_correlation | w/o receptive-field integration | +0.116825 [+0.103228, +0.131243] | 116/0/4 | 1.78e-20 |
| consecutive_score_map_correlation | w/o miss-ray interpolation | -0.005129 [-0.034217, +0.022704] | 79/0/41 | 0.231 |

## Validation

- Valid: `True`
- Episodes: `600`
- Checked states: `3600`
- Devices: `cuda`
