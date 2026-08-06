# ER-GPIS versus Ray-GPIS: visited-view registration gap

The primary metric is one-action Recall@5 on the missing prefix patch. Positive paired improvement means Ray-GPIS is better.

| Planner | Missing-patch Recall@5 | F@5 gain | Corr. | Regret | Time (s) |
|---|---:|---:|---:|---:|---:|
| er_gpis | 0.5321+-0.0960 | 0.1863+-0.0138 | 0.4922+-0.1406 | 0.0196+-0.0088 | -- |
| ray_gpis | 0.5066+-0.0971 | 0.1885+-0.0144 | 0.5625+-0.1423 | 0.0174+-0.0088 | -- |

Primary Ray improvement: -0.02545 [-0.09830, 0.04551], W/T/L 4/55/5, p=0.426, supported=False.

Conclusion: this condition does not support a Ray-over-ER claim; the ER comparison is therefore omitted from the rebuttal.
