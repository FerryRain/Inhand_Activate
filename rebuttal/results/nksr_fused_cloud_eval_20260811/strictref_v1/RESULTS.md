# FaCE + NKSR evaluation with `5_computer_metric_nskr.py` strictref

> **Rigid-only diagnostic:** this pass did not yet apply the known object scale.
> Use `../strictref_known_urdf_scale/RESULTS.md` for the final metrics.

## Protocol

- The alignment and metric functions are imported directly from
  `reconstruction/offline/Compute_Metric/5_computer_metric_nskr.py`.
- Each fused reconstruction is treated as an independent one-state episode and
  receives one rigid reconstruction-to-GT transform.
- Alignment uses the original defaults: unit heuristic, 24-way PCA
  initialization, 1 mm fallback voxel, three ICP levels with 300 iterations
  each, and transform selection by one-way Precision@5 mm after 5 mm evaluation
  downsampling.
- The YCB `nontextured.ply` is read both as the GT alignment point cloud
  (8192/8194 mesh vertices with normals) and as the GT triangle mesh because
  these YCB assets do not include a separate `GT_normal.ply`.
- Mesh metrics use the original 2/5/10 mm thresholds and 20,000 uniform surface
  samples in each direction. Five sampling repeats are reported as mean +/-
  sample standard deviation. The unit heuristic returned scale 1.0 for every
  reconstruction and GT; no scale fitting was performed.

## Results

| Object | Variant | P@5 | R@5 | F@2 | F@5 | F@10 | Chamfer-L1 (mm) |
|---|---|---:|---:|---:|---:|---:|---:|
| Pitcher | default | 0.3436 +/- 0.0017 | 0.0551 +/- 0.0005 | 0.0435 +/- 0.0006 | 0.0949 +/- 0.0008 | 0.1744 +/- 0.0003 | 78.21 +/- 0.02 |
| Pitcher | light | 0.2890 +/- 0.0006 | 0.0597 +/- 0.0006 | 0.0469 +/- 0.0006 | 0.0989 +/- 0.0008 | 0.1739 +/- 0.0004 | 78.01 +/- 0.01 |
| Mustard | default | 0.4524 +/- 0.0012 | 0.1141 +/- 0.0006 | 0.0842 +/- 0.0010 | 0.1822 +/- 0.0007 | 0.2816 +/- 0.0006 | 44.56 +/- 0.00 |
| Mustard | light | 0.4586 +/- 0.0007 | 0.1707 +/- 0.0008 | 0.1165 +/- 0.0005 | 0.2488 +/- 0.0009 | 0.3705 +/- 0.0002 | 41.11 +/- 0.01 |

The independent strictref protocol still favors `light`: F@5 rises by 0.0040
absolute for Pitcher and by 0.0665 absolute for Mustard. Pitcher remains strongly
recall-limited because both reconstructions cover only a small fraction of the
complete YCB surface.

Machine-readable values are in `metrics.csv` and `evaluation.json`. Aligned
meshes and transforms are stored in `meshes_aligned_per_variant/` and
`transforms/`, respectively.
