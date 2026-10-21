Submission is written and validated.

**What I shipped (`/workspace/refine.py`):** iteration 13's best recipe (score 0.9631, crossing 1190) with a single-axis move along the one gradient the history shows as monotone and real:

| axis | iter 12 | iter 13 (best) | this attempt |
|---|---|---|---|
| `decay_frac` | 0.6 → crossing 1260 | 0.7 → crossing **1190** | **0.9** |

Everything else held fixed: `ortho_momentum`, `lr` 0.017, `beta1` 0.92, `beta2` 0.95, `wsd` with `warmup_frac` 0.02, `ns_steps` 6, `embed_lr_mult` 32, `head_lr_mult` 5, `scaled_normal` init, `max_steps` = 2000 (frozen ceiling, read from `state.json`, never shortened — iteration 4 proved shortening forfeits the crossing entirely).

**Reasoning:** the campaign shows the peak-LR and beta1 micro-axes are flat-to-noisy (0.018 vs 0.020 → 1360 vs 1370; beta1 0.85 regressed), while the stable/decay split moved the crossing 70 steps per 0.1 of `decay_frac` in the improving direction. `decay_frac` 0.9 starts the anneal at ~step 240 instead of ~600, so LR at the crossing region (~1190) drops from 0.0100 to 0.0076 — earlier annealing is exactly what pulls the loss below target sooner — while still leaving a long non-zero tail (LR 0.0009 at step 1900) so the three-eval sustain window completes well inside the run. I isolated this one axis rather than confounding it with an LR bump, so the next run in the campaign gets a clean read on whether the decay-shape gradient continues past 0.7 or has turned.

I verified schedule bounds against `recipe_schema.json` (all values in range, no frozen-axis keys), confirmed the WSD semantics in `bia_optim.py:38-40`, and test-ran `refine.py` in a clean temp dir with a synthetic `state.json` — it emits a valid `proposal.json` beside itself, echoes `summary_digest` verbatim, `inherits_from: []`, and reports a null `crossing_step` so nothing can diverge from the verifier's measurement. Budget: 2000 steps × ~0.166 s ≈ 332 s, inside the 432 s bound.

I deliberately skipped a local probe: the only informative one (full-horizon dev run comparing decay shapes) costs more than the wall clock I had left, and the history is explicit that short constant-LR probes have repeatedly mispredicted the crossing. A validated submission on a proven gradient beats an unfinished sweep.