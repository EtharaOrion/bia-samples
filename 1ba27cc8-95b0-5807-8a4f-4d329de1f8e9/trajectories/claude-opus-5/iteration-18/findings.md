Probe result: `head_lr_mult=14` is clearly worse (dev loss 5.985 vs 5.723 at step 300), so I kept the shipped recipe, which the probe confirms trains stably at the higher peak LR.

**Submitted `/workspace/refine.py`** — starting from iteration 17's best (graded step 1180, score 0.9723) with three small, trend-following changes:

| axis | it-17 | this attempt | rationale |
|---|---|---|---|
| `lr` | 0.019 | 0.020 | early phase is token/LR-limited; probe shows no instability |
| `warmup_frac` | 0.02 | 0.015 | reach peak ~7 steps sooner, front-load progress |
| `decay_frac` | 0.72 | 0.75 | anneal onset earlier; 0.7→1190, 0.72→1180, 0.9→1290 puts the optimum just above 0.72 |
| `max_steps` | 1500 | 1420 | shortening the horizon monotonically moved the crossing earlier (1750→1190, 1500→1180); 1420 keeps ~240 steps of headroom past the 1180 crossing region, far more than the 30-step sustain window |

Core untouched: `ortho_momentum` (Newton–Schulz, `ns_steps` 6), `beta1` 0.92 / `beta2` 0.95, `embed_lr_mult` 32 / `head_lr_mult` 5, `scaled_normal` init.

Budget: 1420 steps × ~0.17 s ≈ 241 s, inside the 300 s bound, and `max_steps` is additionally clamped by `min(ceiling, budget×0.90/0.17, 1420)` read from `state.json`. `summary_digest` is echoed verbatim, `inherits_from` is `[]`, and `report.crossing_step` is null so nothing can diverge from the verifier's measurement. Verified end-to-end against a mock `state.json` in an isolated directory.