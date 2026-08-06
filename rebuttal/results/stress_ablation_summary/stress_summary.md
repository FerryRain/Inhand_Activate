# Pre-registered special-case stress ablation

Positive improvements mean Full is better. Holm correction is applied jointly to the four pre-registered primary hypotheses.

| Stress | Primary metric | Full | Ablated | Improvement (95% CI) | p (Holm) | Supported |
|---|---|---:|---:|---:|---:|---:|
| sparse | target_recall@5_auc | 0.589896 | 0.592041 | -0.002145 [-0.019443, 0.011068] | 0.821 | False |
| ghost | post_outlier_oracle_regret | 0.013044 | 0.011282 | -0.001762 [-0.003459, -0.000176] | 0.0762 | False |
| hole | target_recall@5_auc | 0.571520 | 0.549799 | 0.021721 [0.005943, 0.037261] | 0.00487 | True |
| pose_stability | severe_score_map_correlation | 0.502829 | 0.272422 | 0.230407 [0.205861, 0.253509] | 1.55e-11 | True |
