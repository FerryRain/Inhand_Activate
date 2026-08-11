# Joint pose/depth-gap active-planner robustness

A shared tracked prefix view loses one contiguous 70% depth region. The prefix and each recovery branch have exactly 6 deg / 3 mm pose error. The depth gap is transient and no method is prevented from revisiting the missing region.

| Method | Missing-patch R@5 | Gap corr. | Gap regret | Gap accuracy |
|---|---:|---:|---:|---:|
| Pose-Novelty | 0.312$\pm$0.064 | -0.574$\pm$0.126 | 0.608$\pm$0.073 | 0.078$\pm$0.066 |
| Adapted ActNeRF | 0.530$\pm$0.089 | -0.145$\pm$0.164 | 0.390$\pm$0.091 | 0.328$\pm$0.116 |
| Adapted PB-NBV | 0.716$\pm$0.068 | 0.148$\pm$0.189 | 0.204$\pm$0.064 | 0.297$\pm$0.113 |
| Ray-GPIS | 0.478$\pm$0.086 | -0.272$\pm$0.156 | 0.442$\pm$0.088 | 0.250$\pm$0.107 |

Gap correlation and regret compare action scores with recovery of the specific missing prefix region. The next table instead evaluates global F@5 gain over the complete reconstruction.

| Method | Global F gain | Global corr. | Global regret | Time (s) |
|---|---:|---:|---:|---:|
| Pose-Novelty | 0.214$\pm$0.009 | 0.836$\pm$0.058 | 0.004$\pm$0.003 | 0.000$\pm$0.000 |
| Adapted ActNeRF | 0.201$\pm$0.012 | 0.656$\pm$0.121 | 0.018$\pm$0.008 | 3.512$\pm$0.172 |
| Adapted PB-NBV | 0.145$\pm$0.018 | -0.391$\pm$0.168 | 0.073$\pm$0.017 | 0.612$\pm$0.154 |
| Ray-GPIS | 0.201$\pm$0.013 | 0.562$\pm$0.142 | 0.017$\pm$0.009 | 0.200$\pm$0.023 |

Paired differences below are Ray-GPIS minus each comparator on missing-patch R@5; Holm correction covers the three comparisons.

| Comparator | Difference [95% bootstrap CI] | W/T/L | p (Holm) |
|---|---:|---:|---:|
| Pose-Novelty | 0.1657 [0.0913, 0.2486] | 17/46/1 | 4.58e-05 |
| Adapted ActNeRF | -0.0525 [-0.1551, 0.0479] | 11/38/15 | 0.353 |
| Adapted PB-NBV | -0.2382 [-0.3535, -0.1169] | 16/16/32 | 0.000447 |
