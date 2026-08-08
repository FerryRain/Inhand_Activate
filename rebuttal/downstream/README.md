# Sequential quasi-static place-and-regrasp evaluation

This experiment tests whether reconstruction quality is useful beyond surface
metrics.  It uses ten metric YCB models and three paired initial orientations.
Every active method receives one initial RGB-D observation followed by five
planner-selected 6 s in-hand rotations.  During every primitive, the renderer
produces 15 RGB-D frames/s (90 frames/action).  Hand occlusion, depth noise,
under-rotation, stalls, and persistent slip are simulated; tracking inference
is omitted and fusion receives the exact executed object pose.

## Reconstruction sources

- `single_view_depth`: the initial RGB-D observation only;
- `fixed`: six observations obtained by a fixed, non-active schedule;
- `pose_novelty`, `pb_nbv`, `actnerf`, and `ray_gpis`: the shared active loop
  with only the high-level planner replaced;
- `spar3d` and `trellis2`: one clean high-resolution image.  Following the
  paper's original single-image geometry comparison, these meshes receive an
  oracle *isotropic* similarity alignment.  This is favourable to the
  single-image methods and does not non-uniformly warp their predicted shape.
  TRELLIS.2 uses the released sparse/shape latent path and shape decoder only;
  texture-latent sampling is skipped because no downstream metric consumes
  appearance.  A same-seed check against the full pipeline gave a bidirectional
  surface distance equal to independent mesh-sampling noise.  Decoder-AABB
  boundary removal remains as a guard before mesh decimation.  All 30 inputs
  use the same 1024-cascade configuration with the official cascade token cap
  set to 24,576 to fit the available GPU; no per-object fallback is used.

All RGB-D methods use the same GPU NKSR backend to convert the final fused
cloud to a mesh.  Each reconstructed mesh is evaluated for ten deterministic
execution perturbations, giving 3 reconstructions x 10 trials = 30 trials per
YCB object and method.

## Task definition

The planner first selects the support pose with the largest predicted
anti-tipping angle among poses admitting a reachable grasp, then plans a
top-down antipodal parallel-jaw grasp in that placed configuration.  This
lexicographic priority reflects the sequential task: the object must first
remain placed before it can be regrasped.  Neither stage accesses ground-truth geometry.  Execution is
checked against the metric ground-truth mesh:

- placement succeeds when the true centre of mass remains inside the true
  support polygon after a clipped 2.5-degree roll/pitch perturbation;
- regrasp succeeds when the planned fingers make reachable antipodal contacts
  on the true object within the gripper and pad-compliance limits, after
  1.5 mm translational and 1.5 degree yaw execution noise;
- sequential success requires both stages.

The support computation uses the convex hull, which is physically equivalent
for static support and avoids dependence on triangle density.  A GT-mesh oracle
passes all ten YCB objects nominally and 99/100 trials with execution noise,
confirming that failures are caused by reconstructed geometry rather than an
unreachable task definition.

The six real meshes are read only from
`reconstruction/offline/result/offline_tracking` and reported as a separate
transfer subset; they are never mixed into the paired YCB aggregate.

The benchmark, task evaluator, statistics, and visualization run in the
CUDA-enabled `robosyn_gpu` environment.  Only the released method-specific
dependency stacks use separate existing environments: `trellis2` for the 4B
TRELLIS.2 model and `nksr` for the common GPU meshing backend.

## Reproduction commands

```bash
conda run --no-capture-output -n robosyn_gpu python -m rebuttal.scripts.runners.run_baselines \
  --config rebuttal/configs/downstream_ycb_reconstruction.yaml \
  --planners fixed,pose_novelty,pb_nbv,ray_gpis,actnerf

conda run --no-capture-output -n nksr python -m rebuttal.scripts.preparation.build_downstream_meshes \
  --backend nksr --overwrite

conda run -n robosyn_gpu python -m rebuttal.scripts.preparation.render_ycb_single_view_inputs
conda run --no-capture-output -n spar3d python -m rebuttal.scripts.runners.run_spar3d_batch
conda run --no-capture-output -n trellis2 python -m rebuttal.scripts.runners.run_trellis2_batch \
  --output rebuttal/results/downstream_single_view_raw_24k/trellis2 \
  --max-num-tokens 24576 --reload-every 0

conda run -n robosyn_gpu python -m rebuttal.scripts.preparation.align_downstream_meshes \
  --methods spar3d --overwrite
conda run -n robosyn_gpu python -m rebuttal.scripts.preparation.align_downstream_meshes \
  --raw-root rebuttal/results/downstream_single_view_raw_24k \
  --methods trellis2 --overwrite
conda run -n robosyn_gpu python -m rebuttal.scripts.evaluation.evaluate_downstream \
  --methods single_view_depth,spar3d,trellis2,fixed,pose_novelty,pb_nbv,actnerf,ray_gpis
conda run -n robosyn_gpu python -m rebuttal.scripts.summarization.summarize_downstream
```
