# Persistent action-mismatch summary

Declared simulation stress: primitive under-rotation/stall plus grip-slip-induced persistent axis drift; it is not claimed as a fit to measured hardware errors.

| Planner | SO(3) action error (deg) | Target-view error (deg) | Stall rate | Slip rate | Persistent-axis rate | Post-fault next gain | Post-fault next regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| pose_novelty | 56.74+-2.54 | 52.38+-2.51 | 0.267+-0.035 | 0.188+-0.029 | 0.335+-0.057 | 0.0965+-0.0104 | 0.0056+-0.0018 |
| ray_gpis | 56.85+-2.22 | 50.55+-2.52 | 0.267+-0.035 | 0.188+-0.029 | 0.335+-0.057 | 0.0951+-0.0103 | 0.0066+-0.0022 |
