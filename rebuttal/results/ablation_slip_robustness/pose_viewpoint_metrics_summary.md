# Pose-space viewpoint coverage

Coverage is the fraction of the shared candidate sphere within 30.0 degrees of at least one acquired tracked-pose viewing direction.

| Planner | Coverage AUC | Final coverage | Incremental novelty (deg) | Mean pairwise distance (deg) |
|---|---:|---:|---:|---:|
| ray_gpis | 0.1951 | 0.3152 | 49.82 | 88.29 |
| ray_gpis_hit_only | 0.1901 | 0.3066 | 46.94 | 84.11 |
| ray_gpis_novelty_only | 0.1835 | 0.2995 | 46.49 | 82.83 |
| ray_gpis_pointwise | 0.1957 | 0.3160 | 50.10 | 88.85 |
| ray_gpis_uncertainty_only | 0.1940 | 0.3141 | 49.39 | 87.73 |
