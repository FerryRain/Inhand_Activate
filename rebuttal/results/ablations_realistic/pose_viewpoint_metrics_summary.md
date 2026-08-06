# Pose-space viewpoint coverage

Coverage is the fraction of the shared candidate sphere within 30.0 degrees of at least one acquired tracked-pose viewing direction.

| Planner | Coverage AUC | Final coverage | Incremental novelty (deg) | Mean pairwise distance (deg) |
|---|---:|---:|---:|---:|
| er_gpis | 0.2181 | 0.3462 | 64.90 | 94.18 |
| pose_novelty | 0.2294 | 0.3776 | 71.88 | 96.17 |
| ray_gpis | 0.2186 | 0.3458 | 64.94 | 93.77 |
| ray_gpis_hit_only | 0.2181 | 0.3497 | 64.21 | 94.64 |
| ray_gpis_novelty_only | 0.2106 | 0.3364 | 61.49 | 91.77 |
| ray_gpis_pointwise | 0.2185 | 0.3451 | 64.55 | 93.59 |
| ray_gpis_uncertainty_only | 0.2170 | 0.3424 | 64.36 | 93.09 |
