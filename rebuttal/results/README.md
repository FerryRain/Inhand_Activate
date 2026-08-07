# Result directory policy

Only reportable completed experiments are retained here:

- `pipeline_runtime/`: module-level runtime audit using 120 synchronized
  SAM2-tiny CUDA measurements, 2,041 online tracking calls, 36 fusion updates,
  and the formal Ray-GPIS GPU timings;
- `formal_120_sixview_gpu/`: final 120-pair baseline suite and all derived metrics;
- `pb_sanity_120_gpu/`: final 120-pair PB-NBV scale/partition sweep;
- `ablations/`: final 120-pair clean component ablations and derived metrics;
- `ablations_realistic/`: strict-six-image component ablations under the frozen moderate noise profile.
- `stress_sparse/`: fresh-seed Full/Novelty-only observed-sparse test;
- `stress_ghost/`: fresh-seed Full/Uncertainty-only transient-outlier test;
- `stress_hole/`: fresh-seed Full/Hit-only contiguous-occlusion test;
- `stress_pose_reference/`: shared trajectories for stability evaluation;
- `stress_pose_stability/`: paired Full/Pointwise perturbation samples;
- `stress_ablation_summary/`: joint corrected table and GPU speed audit;
- `visited_registration_gap/`: 64-scene Pose/Hit/Full visited-view gap test,
  paired statistics, and branch-consistency audit;
- `visited_registration_gap_ablation/`: exhaustive five-variant rerun under
  the identical visited-view gap scenes;
- `visited_registration_gap_er_ray/`: same-scene ER-GPIS/Ray-GPIS check,
  retained as a completed negative comparison.
- `downstream_ycb_reconstruction/`: 150 complete realistic continuous-acquisition
  episodes for Fixed, Pose-Novelty, PB-NBV, ActNeRF, and Ray-GPIS;
- `downstream_ycb_meshes/`: 30 paired meshes for each of eight reconstruction
  sources, plus the GT-oracle inputs;
- `downstream_ycb_task/`: 2,400 final place-and-regrasp trials, paired tests,
  configuration, tables, and mesh montage;
- `downstream_ycb_task_oracle/`: 100-trial GT-mesh reachability sanity check;
- `downstream_real_meshes/` and `downstream_real_task/`: six prior real AURORA
  meshes and 180 separate scanner-GT transfer trials;
- `downstream_single_view_raw/` and `downstream_single_view_raw_24k/`: formal
  SPAR3D and uniform-configuration TRELLIS.2 raw single-image outputs.

The `baseline_slip_pose_ray/`, `ablation_slip_robustness/`, and
`continuous_empirical_failed_x_outage/` trees are retained as completed
negative diagnostics for auditability, but their results are explicitly not
used as Ray-over-Pose evidence in the rebuttal.

Development, CPU-timing, incomplete, and superseded output directories must
not be mixed with these reportable results. Smoke episodes inside a formal root
are retained only when their resolved config exactly matches the frozen formal
config and are subsequently covered by the same completeness audit.
