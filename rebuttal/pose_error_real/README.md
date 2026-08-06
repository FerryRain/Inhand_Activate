# Real-capture 6D pose-error experiment

This directory answers the reviewer comment:

> More experiments are how AURORA deals with errors in object 6D poses tracking will be very meaningful.

The experiment reuses the three real in-hand RGB-D sequences currently stored
under `Tracking/BundleTrack/results/{001,002,xyz}/keyframes` (401 matched
BundleTrack keyframes in total). It does not render new observations or change
the captured action trajectories.

## What is tested

For every sequence, the unmodified BundleTrack `T_CO` trajectory produces the
clean-pose reference reconstruction. The experiment then adds controlled
rotation and translation errors to each tracked object pose before the same
masked RGB-D points are fused in the object frame.

- `T_CO` maps object-frame points to the camera frame, matching the repository's
  BundleTrack convention.
- The first pose is left unchanged to anchor the object coordinate system. This
  avoids scoring an arbitrary global-frame offset as a geometry failure.
- Later errors follow a deterministic Gauss-Markov process with temporal
  correlation 0.85, representing tracking jitter plus drift.
- Each trace is normalized to the configured RMS geodesic rotation and object-
  origin translation error. The default levels are 0/0, 1 deg/1 mm,
  2 deg/2 mm, 5 deg/5 mm, and 10 deg/10 mm.
- All levels reuse exactly the same RGB-D pixels, masks, frame sampling, and
  trajectory; only the pose supplied to fusion changes.

The primary outcome is point-cloud F-score at 5 mm against the clean-pose
reconstruction. F@2, F@10, and symmetric mean nearest-surface distance are
also reported. This is a controlled sensitivity study, not a claim that the
unperturbed BundleTrack reconstruction is CAD ground truth.

## Reproduce

From the repository root:

```bash
conda run --no-capture-output -n robosyn python \
  rebuttal/pose_error_real/pose_error_experiment.py
```

A quick two-trial smoke run is:

```bash
conda run --no-capture-output -n robosyn python \
  rebuttal/pose_error_real/pose_error_experiment.py --trials 2
```

Run the unit tests with:

```bash
conda run -n robosyn python -m unittest discover \
  -s rebuttal/pose_error_real -p 'test_*.py' -v
```

The script fails early if any local archived capture is unavailable. Paths,
error levels, sampling, and fusion parameters are explicit in `config.json`.
It writes raw samples, aggregate/per-sequence tables, a curve, run metadata,
and ready-to-use reviewer wording under `results/`.

## Reported files

- `results/SUMMARY.md`: concise result table and rebuttal paragraph.
- `results/pose_error_curve.png`: F@5 and symmetric-distance sensitivity curve.
- `results/pose_error_samples.csv`: every sequence/condition/trial result and
  its realized RMS/p95 pose error.
- `results/pose_error_aggregate.csv`: aggregate curve values and confidence
  intervals.
- `results/pose_error_per_sequence.csv`: sequence-level means.
- `results/run_metadata.json`: data counts, conventions, configuration, and
  software versions.
- `results/validation_report.json`: completeness, target-error, identity, and
  aggregate-trend checks.
