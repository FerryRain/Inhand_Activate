# Visited-view local registration-gap robustness

A tracked prefix view loses one contiguous 70% depth region before fusion. All subsequent branches are fault-free and shared. The primary metric is one-action Recall@5 on points that remained missing after the faulty prefix.

| Planner | Missing-patch Recall@5 | Point-weighted Recall@5 | F@5 gain | Regret |
|---|---:|---:|---:|---:|
| pose_novelty | 0.2827+-0.0759 | 0.2763 | 0.1816+-0.0145 | 0.0243+-0.0096 |
| ray_gpis_novelty_only | 0.5066+-0.0971 | 0.5775 | 0.1885+-0.0144 | 0.0174+-0.0088 |
| ray_gpis_uncertainty_only | 0.5030+-0.0963 | 0.5768 | 0.1874+-0.0141 | 0.0185+-0.0092 |
| ray_gpis_pointwise | 0.5404+-0.0980 | 0.6161 | 0.1908+-0.0130 | 0.0151+-0.0077 |
| ray_gpis_hit_only | 0.4833+-0.0956 | 0.4962 | 0.1827+-0.0151 | 0.0232+-0.0106 |
| ray_gpis | 0.5066+-0.0971 | 0.5775 | 0.1885+-0.0144 | 0.0174+-0.0088 |

Positive improvement means Full Ray-GPIS is better. Holm correction is applied jointly to the five missing-patch recovery comparisons.

| Comparator | Full improvement [95% bootstrap CI] | W/T/L | p (Holm) | Supported |
|---|---:|---:|---:|---:|
| Pose-Novelty | 0.2239 [0.1398, 0.3136] | 21/42/1 | 7.15e-06 | True |
| Novelty only | 0.0000 [0.0000, 0.0000] | 0/64/0 | 1 | False |
| Uncertainty only | 0.0036 [-0.0420, 0.0525] | 3/57/4 | 1 | False |
| w/o receptive-field integration | -0.0338 [-0.0755, -0.0018] | 0/60/4 | 0.5 | False |
| w/o miss-ray interpolation | 0.0233 [-0.0729, 0.1195] | 11/44/9 | 1 | False |
