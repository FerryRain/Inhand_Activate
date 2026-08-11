# FaCE normals + NKSR reconstruction evaluation

> **Superseded scale setting:** this first pass used the unscaled YCB meshes.
> The final evaluation using the known physical URDF scale is in
> `strictref_known_urdf_scale/RESULTS.md`.

## Protocol

- Inputs: the four fused point clouds supplied in `/home/ferry/下载`.
- Normal estimation: FaCE in headless mode, using its default estimator and no
  interior-point filtering. The binary PLY inputs were losslessly rewritten as
  ASCII PLY because this FaCE build does not parse Open3D's double-precision
  binary vertex properties; point count and coordinates were otherwise kept
  unchanged (maximum round-trip coordinate error below `5e-8 m`).
- Reconstruction: NKSR on `cuda:0`, `detail_level=0.4`, `mise_iter=1`.
- Ground truth: the YCB `019_pitcher_base` and `006_mustard_bottle` meshes.
- Alignment: one rigid `GT <- reconstruction` transform was estimated from the
  more complete `light` FaCE cloud for each object, then frozen and applied to
  both that object's `default` and `light` meshes. No scale was fitted.
- Evaluation: exact point-to-triangle-mesh distances, threshold `5 mm`; five
  deterministic repeats with 100,000 uniformly sampled surface points in each
  direction per repeat. Values below are mean +/- sample standard deviation.

## Results

| Object | Variant | Precision@5 | Recall@5 | F-score@5 |
|---|---:|---:|---:|---:|
| Pitcher | default | 0.4001 +/- 0.0009 | 0.0614 +/- 0.0010 | 0.1065 +/- 0.0016 |
| Pitcher | light | 0.4231 +/- 0.0016 | 0.0737 +/- 0.0009 | 0.1255 +/- 0.0012 |
| Mustard | default | 0.5340 +/- 0.0016 | 0.1489 +/- 0.0005 | 0.2329 +/- 0.0008 |
| Mustard | light | 0.5463 +/- 0.0014 | 0.2091 +/- 0.0011 | 0.3024 +/- 0.0012 |

Relative to `default`, `light` improves F-score@5 by 0.0190 absolute (17.8%)
for Pitcher and by 0.0695 absolute (29.9%) for Mustard. The low absolute scores
are primarily recall-limited, especially for the Pitcher, rather than caused by
the reconstructed surfaces having uniformly low precision.

Machine-readable results are in `metrics.csv` and `evaluation.json`. Raw NKSR
outputs are under `mesh_nksr/`; final meshes in the shared GT frame are under
`meshes_aligned_shared_rigid/`.
