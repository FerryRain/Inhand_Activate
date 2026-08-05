# Result directory policy

Only reportable completed experiments are retained here:

- `formal_120_sixview_gpu/`: final 120-pair baseline suite and all derived metrics;
- `pb_sanity_120_gpu/`: final 120-pair PB-NBV scale/partition sweep.

Future formal runs use `ablations/` and `robustness/`. Development, smoke-test,
CPU-timing, incomplete, and superseded output directories must not be mixed
with these reportable results.
