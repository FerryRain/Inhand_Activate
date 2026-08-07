# Sequential place-and-regrasp results

Ten YCB objects, three paired reconstructions/object, ten execution perturbations/reconstruction (30 trials/object/method).

| Reconstruction source | Placement (%) | Stage-2 regrasp (%) | Sequential (%) |
|---|---:|---:|---:|
| Single RGB-D | 39.7 | 20.0 | 6.7 |
| SPAR3D (oracle-align) | 30.0 | 15.0 | 5.0 |
| TRELLIS.2 (oracle-align) | 19.7 | 7.7 | 0.0 |
| Fixed schedule | 50.0 | 44.0 | 24.0 |
| Adapted PB-NBV | 43.3 | 40.0 | 23.0 |
| Adapted ActNeRF | 56.0 | 72.0 | 41.7 |
| Pose-Novelty | 56.7 | 79.3 | 44.7 |
| Full Ray-GPIS | 57.7 | 77.7 | 45.0 |

## Realistic continuous-acquisition reconstruction

Thirty paired episodes/method (10 YCB objects x 3 initial poses); five 6 s primitives at 15 FPS with dynamic occlusion and manipulation errors.

| Planner | F@5 AUC | Final F@5 | Accepted frames | Planning (s/step) |
|---|---:|---:|---:|---:|
| Fixed schedule | 0.5684 | 0.7404 | 21.6 | 0.000 |
| Adapted PB-NBV | 0.5213 | 0.6583 | 14.0 | 0.753 |
| Adapted ActNeRF | 0.6787 | 0.8465 | 26.8 | 7.422 |
| Pose-Novelty | 0.7231 | 0.9152 | 31.9 | 0.000 |
| Full Ray-GPIS | 0.7052 | 0.9052 | 30.6 | 0.968 |

## Paired object-level comparisons

- Ray-GPIS vs Fixed schedule on `placement_success`: +7.7 pp (object-level Wilcoxon p=0.4728).
- Ray-GPIS vs Fixed schedule on `conditional_regrasp_success`: +33.7 pp (object-level Wilcoxon p=0.0117).
- Ray-GPIS vs Fixed schedule on `sequential_success`: +21.0 pp (object-level Wilcoxon p=0.1035).
- Ray-GPIS vs Single RGB-D on `placement_success`: +18.0 pp (object-level Wilcoxon p=0.0394).
- Ray-GPIS vs Single RGB-D on `conditional_regrasp_success`: +57.7 pp (object-level Wilcoxon p=0.0076).
- Ray-GPIS vs Single RGB-D on `sequential_success`: +38.3 pp (object-level Wilcoxon p=0.0115).

## Existing real-mesh transfer subset

Six meshes x 30 perturbations: placement 66.7%, stage-2 regrasp 72.2%, sequential 41.7%.
