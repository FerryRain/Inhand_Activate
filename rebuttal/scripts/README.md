# Experiment scripts

Run every entry point from the repository root with `python -m`; this keeps
imports and repository-relative config paths independent of the script's
physical directory.

| Directory | Purpose |
|---|---|
| `runners/` | Execute baseline, ablation, robustness, and external-model runs |
| `preparation/` | Calibrate actions and prepare assets, inputs, and meshes |
| `evaluation/` | Compute reconstruction, coverage, downstream, and runtime metrics |
| `summarization/` | Aggregate episodes and produce statistical result tables |
| `validation/` | Audit environments, CUDA timing, and result consistency |
| `visualization/` | Generate quantitative plots and mesh montages |

For example:

```bash
python -m rebuttal.scripts.runners.run_baselines \
  --config rebuttal/configs/formal_120_sixview_gpu.yaml
```

The reusable implementation remains in `rebuttal/benchmark/` and
`rebuttal/downstream/`; the files here are command-line orchestration only.
