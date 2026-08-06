# Pose-space viewpoint coverage

Coverage is the fraction of the shared candidate sphere within 30.0 degrees of at least one acquired tracked-pose viewing direction.

| Planner | Coverage AUC | Final coverage | Incremental novelty (deg) | Mean pairwise distance (deg) |
|---|---:|---:|---:|---:|
| er_gpis | 0.2140 | 0.3403 | 65.17 | 94.12 |
| pose_novelty | 0.2282 | 0.3826 | 73.97 | 97.72 |
| ray_gpis | 0.2149 | 0.3417 | 66.06 | 94.05 |
| ray_gpis_hit_only | 0.2178 | 0.3498 | 64.91 | 94.76 |
