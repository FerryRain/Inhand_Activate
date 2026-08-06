# Pose-space viewpoint coverage

Coverage is the fraction of the shared candidate sphere within 30.0 degrees of at least one acquired tracked-pose viewing direction.

| Planner | Coverage AUC | Final coverage | Incremental novelty (deg) | Mean pairwise distance (deg) |
|---|---:|---:|---:|---:|
| actnerf | 0.1947 | 0.2954 | 53.26 | 86.15 |
| er_gpis | 0.2110 | 0.3248 | 62.48 | 93.21 |
| fixed | 0.1528 | 0.2628 | 42.70 | 80.02 |
| pb_nbv | 0.1324 | 0.1985 | 29.24 | 53.36 |
| pose_novelty | 0.2348 | 0.4010 | 77.75 | 98.14 |
| ray_gpis | 0.2115 | 0.3352 | 63.58 | 91.51 |
