# Real-capture pose-error result

## Reviewer response

We thank the reviewer for this suggestion. We added a controlled 6D pose-tracking error study on 3 previously captured real in-hand RGB-D sequences (401 BundleTrack keyframes). We keep the observations, masks, and executed trajectories fixed and perturb only the tracked object-to-camera transform before AURORA's object-centric fusion. The first frame anchors the object coordinate system, and subsequent errors follow a temporally correlated SE(3) process; each nonzero level uses 20 deterministic trials per sequence. At 5 deg / 5 mm RMS error, AURORA retains 96.3% F-score at the 5 mm threshold, with a 1.57 mm symmetric mean distance to the clean-pose reconstruction. Even at 10 deg / 10 mm RMS, it retains 90.3% F-score (2.31 mm symmetric distance). This real-data sensitivity curve shows graceful degradation under moderate tracking noise while making clear that large pose drift remains a failure mode.

## Aggregate table

| Pose error (deg/mm RMS) | F@2 (%) | F@5 (%) | F@10 (%) | Sym. distance (mm) |
|---:|---:|---:|---:|---:|
| 0 / 0 | 100.0 | 100.0 | 100.0 | 0.00 |
| 1 / 1 | 97.2 | 99.7 | 100.0 | 0.84 |
| 2 / 2 | 90.2 | 98.8 | 99.7 | 1.11 |
| 5 / 5 | 79.8 | 96.3 | 98.8 | 1.57 |
| 10 / 10 | 68.6 | 90.3 | 97.4 | 2.31 |

F-scores and symmetric distances are measured against the reconstruction obtained from the same real RGB-D keyframes using the unperturbed BundleTrack poses. This isolates pose sensitivity; it is not an absolute CAD-ground-truth accuracy claim. Error bars in the plot are 95% normal confidence intervals over sequence/trial perturbation realizations.
