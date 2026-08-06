# Persistent action-mismatch summary

Declared simulation stress: primitive under-rotation/stall plus grip-slip-induced persistent axis drift; it is not claimed as a fit to measured hardware errors.

| Planner | SO(3) action error (deg) | Target-view error (deg) | Stall rate | Slip rate | Persistent-axis rate | Post-fault next gain | Post-fault next regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| ray_gpis_novelty_only | 54.15+-2.29 | 45.26+-2.82 | 0.267+-0.035 | 0.188+-0.029 | 0.335+-0.057 | 0.0903+-0.0097 | 0.0135+-0.0053 |
| ray_gpis_uncertainty_only | 56.25+-2.19 | 50.16+-2.39 | 0.267+-0.035 | 0.188+-0.029 | 0.335+-0.057 | 0.0939+-0.0102 | 0.0078+-0.0025 |
| ray_gpis_pointwise | 56.53+-2.23 | 50.16+-2.55 | 0.267+-0.035 | 0.188+-0.029 | 0.335+-0.057 | 0.0944+-0.0104 | 0.0074+-0.0024 |
| ray_gpis_hit_only | 54.32+-2.19 | 48.10+-2.28 | 0.267+-0.035 | 0.188+-0.029 | 0.335+-0.057 | 0.0920+-0.0103 | 0.0137+-0.0043 |
| ray_gpis | 56.85+-2.22 | 50.55+-2.52 | 0.267+-0.035 | 0.188+-0.029 | 0.335+-0.057 | 0.0951+-0.0103 | 0.0066+-0.0022 |
