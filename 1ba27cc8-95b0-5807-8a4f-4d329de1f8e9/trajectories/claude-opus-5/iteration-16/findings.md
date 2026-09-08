Submission is in place and validated.

**`/workspace/refine.py`** (self-contained, stdlib only, writes `proposal.json` beside itself, echoes `summary_digest`, `inherits_from: []`). Verified with `bia_recipe.normalize`: no frozen-axis writes, no unknown keys.

Recipe (starts from iteration 13, the 0.9631 / step-1190 best):

| key | value | rationale |
|---|---|---|
| optimizer | `ortho_momentum`, beta1 0.92, beta2 0.95, ns_steps 6 | proven core, unchanged |
| lr | 0.017 | history shows this axis is flat at embed_mult 32; held fixed |
| schedule | wsd, warmup 0.02, **decay_frac 0.72** | quadratic fit through the measured points (0.6→1260, 0.7→1190, 0.9→1290) puts the vertex at ≈0.74; 0.72 stays on the proven side of it |
| embed/head lr mult | 32 / 5 | saturated per probes |
| **max_steps** | **min(ceiling, budget-affordable, 1750)** | the changed axis |

Two deliberate decisions:

1. **Horizon compression to 1750 instead of 2000.** decay-shape tuning is exhausted (≤6 steps of headroom left by the fit), so the remaining lever is the anneal horizon itself: at 1750 steps the LR at step ~1150 is annealed to ~0.35 of peak versus ~0.58 before, which pushes the crossing earlier. The margin is large — the expected crossing (~1100–1200) sits ~550 steps inside the run, unlike iteration 4, which cut the horizon *below* its own crossing and forfeited the band.

2. **Budget is read from `state.json`, not assumed.** Iteration 15 was zeroed by `attempt-budget-exceeded` because it hard-coded the stale 2000-step figure. `refine.py` now computes `affordable = budget_seconds * 0.90 / 0.170 s`: with 0.12 h it yields 1750 (≈290 s), with 0.0833 h it clamps to 1588 (≈264 s). Either way the run fits with margin and the crossing stays comfortably inside it.