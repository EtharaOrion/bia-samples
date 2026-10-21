`/workspace/refine.py` is written, smoke-tested in an isolated directory, and valid.

**What I shipped (built on iteration 8, best at 0.8154 / step 1350):**
- Kept the ortho_momentum (Muon-style Newton–Schulz) core, WSD schedule (warmup 0.02, decay_frac 0.6), beta1 0.92, `max_steps` = 2000 ceiling (never shortened — iteration 4 proved shortening kills the crossing).
- **Structural change (the axis history said was untried):** decoupled the per-group LR multipliers instead of the tied 10/10 — `embed_lr_mult` 16, `head_lr_mult` 6. Rationale from reading `bia_optim.py`: the orthogonalized update is norm-normalized, so a 50304×384 embedding receives per-element updates ~1/√50304 and needs a much larger multiplier than the head, which the tied setting cannot express (the earlier tied-24 probe regressed precisely because it over-drove the head).
- Secondary: `ns_steps` 5→6 (cleaner orthogonalization) and peak lr 0.018→0.017, following the only monotone trend in the history (0.020→1370, 0.018→1360).

**Local evidence:** two concurrent 200-step dev probes finished inside the command bound — new recipe 6.1923 vs the iteration-8 recipe 6.2165 at the same step, so the change is healthy and mildly ahead, not divergent.

Report fields are left null with a raw readout, so the verifier's own unsmoothed measurement is the only claimed crossing.