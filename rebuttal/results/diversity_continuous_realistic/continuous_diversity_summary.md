# Continuous AURORA-style challenging-object evaluation

Each episode uses one initial observation and five complete 6 s active primitives. The fixed RGB-D camera renders at 15 FPS (90 frames/action); full AURORA keyframe filtering and fusion run throughout, and Ray-GPIS replans only at primitive boundaries. Tracking inference is omitted and exact executed pose is supplied.

Stressors: dynamic palm/two-finger occlusion, depth/mask corruption, 55--85% non-stalled progress, 25% configured stall probability, and 22% configured grip-slip probability with persistent axis drift.

| Object | Episodes | F-AUC | Final F@5 | Final Recall@5 | Chamfer (mm) | Accepted keyframes | Visibility | Stall | Slip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Bowl | 15 | 0.6402+-0.0534 | 0.8107+-0.0695 | 0.6992+-0.0831 | 4.758+-0.967 | 20.6+-4.3 | 0.511+-0.009 | 0.280+-0.126 | 0.200+-0.108 |
| Thin_Irregular | 15 | 0.7046+-0.0306 | 0.9274+-0.0219 | 0.8673+-0.0360 | 3.579+-0.602 | 31.1+-3.4 | 0.575+-0.004 | 0.293+-0.100 | 0.173+-0.084 |
| All added objects | 30 | 0.6724+-0.0324 | 0.8690+-0.0416 | 0.7832+-0.0540 | 4.168+-0.599 | 25.8+-3.3 | 0.543+-0.013 | 0.287+-0.079 | 0.187+-0.068 |

Audit: 30 episodes; every episode contains 450 continuously rendered action frames and zero simulated tracking error.
