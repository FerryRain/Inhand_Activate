# Final strict-six-image baseline result

The final Level-A benchmark contains 8 objects and 15 shared initial poses, for
120 paired scenes. Every episode contains exactly one initial image and five
planner-selected images. Fixed, PB-NBV, and Ray-GPIS run once per scene;
ActNeRF runs three initialization seeds which are averaged inside each scene.
This produces 720 stored runs and 120 method-level paired units.

| Method | F@5 AUC | Recall AUC | Final F@5 | Corr. | Regret | Accuracy | Time/step (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed schedule | 0.7948±0.0061 | 0.6780±0.0078 | 0.9401±0.0046 | N/A | 0.0806±0.0025 | 29.0±1.8% | 0.000 |
| Adapted PB-NBV (d/70) | 0.7475±0.0193 | 0.6188±0.0258 | 0.8496±0.0234 | -0.3563±0.0731 | 0.1076±0.0124 | 19.2±3.8% | 0.391±0.024 |
| Adapted ActNeRF | 0.8665±0.0062 | 0.7847±0.0095 | 0.9651±0.0050 | 0.1619±0.0337 | 0.0317±0.0029 | 42.3±2.4% | 2.238±0.018 |
| **Ray-GPIS** | **0.8987±0.0091** | **0.8397±0.0136** | **0.9814±0.0100** | **0.4722±0.0667** | **0.0088±0.0051** | **68.0±4.1%** | **0.261±0.024** |

Against adapted ActNeRF, Ray-GPIS improves paired F@5 AUC by 0.03227
(95% bootstrap CI 0.02258 to 0.04038), wins/ties/loses 106/1/13 scenes, and
has a two-sided Wilcoxon p-value of 8.73e-15 after Holm correction. It improves
score-gain Spearman correlation by 0.31031 and reduces oracle regret by
0.02294. Final F@5 improves by 0.01634 and Chamfer decreases by 0.33064 mm.

Timing uses PyTorch 2.4.1+cu121 and GPyTorch 1.11 on an RTX 4090 D. CUDA timers
are synchronized at phase boundaries. All 600 Ray-GPIS planning steps and all
1,800 ActNeRF planning steps record `device=cuda`; PB-NBV and Fixed are native
CPU planners measured on the same machine. Ray-GPIS uses GPyTorch ExactGP and
averages 0.261 s per planning step (0.261 s representation update and 0.000072
s candidate lookup), versus 2.238 s for adapted ActNeRF.

PB-NBV sensitivity confirms that the main result uses its strongest tested
configuration: F@5 AUC is 0.6939 for d/30, 0.7212 for d/50, 0.7475 for d/70,
and 0.6863 for d/50 without partition.

Validation status:

- 720/720 formal runs and 480/480 PB sanity runs are complete;
- all 120 paired initial states are identical across methods;
- every active state contains 256 candidate scores and three counterfactual actions;
- saved selected gains equal next-state F@5 changes exactly;
- independent visible-triangle coverage is monotonic;
- all four core unit tests pass.

The complete categorized report is under
`results/formal_120_sixview_gpu/all_metrics_summary.md`; paired confidence
intervals and corrected tests are in `paired_statistics_all.md`. The ActNeRF
baseline preserves ensemble RGB-variance planning but uses compact PyTorch
radiance fields, not the original Instant-NGP implementation.
