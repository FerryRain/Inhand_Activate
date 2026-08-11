# FaCE + NKSR evaluation of seed reconstructions

## Protocol

- Inputs are the four point clouds supplied on 2026-08-11.
- Normals: FaCE in headless mode, without interior-point filtering.
- Reconstruction: NKSR on `cuda:0`, `detail_level=0.4`, `mise_iter=1`.
- Physical GT scale is fixed from the supplied URDF scale (`0.03`), with no
  scale fitting during alignment:
  - mustard: `60.000 x 41.125 x 118.089 mm`;
  - pitcher: `61.527 x 59.751 x 99.999 mm`.
- Alignment follows `5_computer_metric_nskr.py`: 24 PCA hypotheses, 1 mm
  fallback voxel, three ICP stages of 300 iterations, and hypothesis selection
  by Precision@5 mm. Each reconstruction is aligned independently using a
  rigid transform only.
- Mesh metrics use 20,000 uniform surface samples in each direction and report
  mean +/- sample standard deviation over five repeats.

FaCE's constrained Delaunay refinement was numerically unstable for
`mustard_seed5`, `mustard_seed8`, and `pitcher_norf_seed1_0740` in their
original coordinate-axis orientation. For those three clouds, a fixed rigid
rotation was used only as numerical preconditioning before FaCE, and both
points and normals were inverse-rotated before NKSR. No points were removed and
no geometric scaling was applied. The maximum point-set round-trip error was
`0.000085 mm`, far below all metric thresholds. The exact transform and errors
are recorded in `face_numeric_preconditioning.json`.

## Results

| Input | Precision@5 mm | Recall@5 mm | F-score@5 mm | Chamfer-L1 (mm) | F@2 mm | F@10 mm |
|---|---:|---:|---:|---:|---:|---:|
| `Pitcher_missray_seed4_0821.ply` | 0.7567 +/- 0.0013 | 0.6041 +/- 0.0004 | **0.6718 +/- 0.0005** | 8.4663 +/- 0.0075 | 0.4505 +/- 0.0007 | 0.8731 +/- 0.0003 |
| `Pitcher_norf_seed1_0740.ply` | 0.8296 +/- 0.0007 | 0.5196 +/- 0.0005 | **0.6390 +/- 0.0002** | 13.1212 +/- 0.0048 | 0.4264 +/- 0.0007 | 0.7300 +/- 0.0002 |
| `mustard_seed5.ply` | 0.9844 +/- 0.0004 | 0.9483 +/- 0.0003 | **0.9660 +/- 0.0003** | 2.6941 +/- 0.0040 | 0.8051 +/- 0.0004 | 1.0000 +/- 0.0000 |
| `mustard_seed8.ply` | 0.8784 +/- 0.0008 | 0.7510 +/- 0.0001 | **0.8097 +/- 0.0003** | 6.1735 +/- 0.0023 | 0.4287 +/- 0.0009 | 0.9725 +/- 0.0003 |

A separate 100,000-sample check gives F@5 values of `0.6718`, `0.6395`,
`0.9658`, and `0.8095`, respectively, confirming that the reported ordering
and differences are not caused by 20,000-point surface-sampling noise.

## Artifacts

- Complete five-repeat metrics: `strictref_known_urdf_scale/metrics.csv`
- Protocol metadata and metrics: `strictref_known_urdf_scale/evaluation.json`
- 100k-sample check: `strictref_known_urdf_scale/check_100k_samples.json`
- Rigidly aligned meshes: `strictref_known_urdf_scale/meshes_aligned_per_variant/`
- Alignment transforms: `strictref_known_urdf_scale/transforms/`
- Raw NKSR meshes: `mesh_nksr/`
- FaCE point clouds with normals: `face_normals/`
