# Final FaCE + NKSR evaluation with known URDF scale

## Scale conversion

The reconstruction point clouds are already in physical metres. The original
URDF meshes were scaled by `0.03`, giving the following physical dimensions:

| Object | Original maximum | URDF scale | Physical maximum |
|---|---:|---:|---:|
| Pitcher | 3.3333 m | 0.03 | 99.999 mm |
| Mustard | 3.9363 m | 0.03 | 118.089 mm |

The available YCB `nontextured.ply` files use a different metre normalization,
so multiplying them directly by `0.03` would be incorrect. They are uniformly
rescaled to the known physical maximum instead:

- Pitcher GT scale: `99.999 / 242.387 = 0.4125592542`, producing dimensions
  `61.527 x 59.751 x 99.999 mm`.
- Mustard GT scale: `118.089 / 191.301 = 0.6172941795`, producing dimensions
  `60.000 x 41.125 x 118.089 mm`.

These are fixed, known scales; no scale fitting or GT-metric scale selection is
performed. After scaling the GT, each reconstruction is rigidly aligned with
the exact strictref pipeline from `5_computer_metric_nskr.py`.

## Results

Metrics use 20,000 uniformly sampled surface points in each direction and five
sampling repeats. Values are mean +/- sample standard deviation.

| Object | Variant | P@5 | R@5 | F@2 | F@5 | F@10 | Chamfer-L1 (mm) |
|---|---|---:|---:|---:|---:|---:|---:|
| Pitcher | default | 0.9270 +/- 0.0008 | 0.7599 +/- 0.0003 | 0.5901 +/- 0.0009 | 0.8352 +/- 0.0003 | 0.9195 +/- 0.0004 | 5.628 +/- 0.006 |
| Pitcher | light | 0.9773 +/- 0.0003 | 0.9882 +/- 0.0002 | 0.7970 +/- 0.0011 | 0.9827 +/- 0.0002 | 0.9950 +/- 0.0000 | 2.791 +/- 0.002 |
| Mustard | default | 0.8018 +/- 0.0012 | 0.6150 +/- 0.0008 | 0.4132 +/- 0.0003 | 0.6961 +/- 0.0008 | 0.8790 +/- 0.0004 | 8.530 +/- 0.006 |
| Mustard | light | 0.8791 +/- 0.0007 | 0.8771 +/- 0.0005 | 0.7057 +/- 0.0006 | 0.8781 +/- 0.0004 | 0.9570 +/- 0.0002 | 4.435 +/- 0.004 |

An independent deterministic check using 100,000 surface samples per direction
gave F@5 values `0.8359`, `0.9822`, `0.6967`, and `0.8778`, respectively.

Machine-readable values are in `metrics.csv` and `evaluation.json`; aligned
meshes and rigid transforms are in `meshes_aligned_per_variant/` and
`transforms/`.
