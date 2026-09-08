`/workspace/refine.py` is written, smoke-tested end-to-end, and produces a valid `proposal.json`.

**Submission:** iteration 8's ortho_momentum/WSD core (graded 1350, score 0.8154) with `beta1` lowered 0.92 → 0.85, `max_steps` held at the 2000 ceiling.

**Reasoning:**
- Code read of `bia_optim.py` confirmed the orthogonalized update is *scale-invariant* (Newton–Schulz normalizes by `x.norm()`), so for all 2D params `beta1` changes only the momentum averaging horizon and is fully decoupled from `lr`. That makes it the one axis safe to move alone.
- The same read showed `grad_clip` is a global uniform rescale, hence a no-op for every 2D param and only touching the handful of 1D LayerNorm gains — a dead knob, so I did not spend the run on it. `weight_decay` is likewise dead for `ortho_momentum`.
- History gives a monotone trend on beta1: 0.96 → 1410, 0.95 → 1360/1370, 0.92 → 1350.
- Two concurrent 200-step dev probes at constant LR: beta1 0.88 → 6.2058, beta1 0.78 → 6.1988. Flat but faintly favoring lower, consistent with the history trend and with no sign of a noise blow-up down there. 0.85 sits inside the probed interval.

**Honest caveat:** the probe separation (0.007 nats) is within noise, so this is a small-expected-gain move, not a structural one. The history's own evidence argues the crossing region is token-limited rather than LR-limited — `decay_frac` 0.5 and 0.8 give materially different LRs at step 1350 yet both crossed at 1370 — so schedule reshaping looks exhausted, and a genuinely large gain likely needs a different update rule than the closed schema offers. I kept `max_steps` at 2000 to avoid iteration 4's failure mode, where shortening the horizon moved the crossing past the end of the run and forfeited the entire upper band.