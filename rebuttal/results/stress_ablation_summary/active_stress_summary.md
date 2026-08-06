# Targeted active stress ablations

Positive paired improvements mean Full is better. Values are pre-Holm; pose stability is added before final correction.

| Stress | Primary metric | Full | Ablated | Improvement (95% CI) | p | W/T/L |
|---|---|---:|---:|---:|---:|---:|
| sparse | target_recall@5_auc | 0.589896 | 0.592041 | -0.002145 [-0.019443, 0.011068] | 0.821 | 8/48/8 |
| ghost | post_outlier_oracle_regret | 0.013044 | 0.011282 | -0.001762 [-0.003459, -0.000176] | 0.0381 | 18/18/28 |
| hole | target_recall@5_auc | 0.571520 | 0.549799 | 0.021721 [0.005943, 0.037261] | 0.00162 | 29/21/14 |
