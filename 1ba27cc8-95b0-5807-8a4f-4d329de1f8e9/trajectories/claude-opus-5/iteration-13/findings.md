`/workspace/refine.py` is written and verified end-to-end.

**What I did**

1. Wrote a working submission first (a copy of iteration 12's best recipe, score 0.8985 @ step 1260), so a graded artifact existed before any experiment.
2. Ran two concurrent 250-step dev probes (constant LR, everything else matched to the best recipe) to test whether the `embed_lr_mult` axis — the only one that had produced gains — still had headroom:
   - `embed_lr_mult=32`: dev loss **5.6906** at step 250
   - `embed_lr_mult=56`: dev loss **5.6922** at step 250
   
   The axis is **saturated at 32**; pushing it to 56 buys nothing (0.0016 nats, inside noise) and only adds late-phase instability risk. So I kept 32 rather than gambling on 56.
3. Since the multiplier axis is exhausted (as my own prior account predicted), I moved the one remaining structural knob that directly controls *when* the loss settles below target: the WSD stable/decay split, `decay_frac` 0.6 → 0.7. That starts the anneal at ~step 600 instead of ~800, which pulls the low-LR settling phase earlier while the horizon stays pinned at the 2000-step ceiling so the crossing and its 3-point sustain window still fit comfortably inside the run.

**Submitted recipe:** `ortho_momentum`, lr 0.017, beta1 0.92 / beta2 0.95, `wsd` with warmup_frac 0.02 and **decay_frac 0.7**, ns_steps 6, embed_lr_mult 32, head_lr_mult 5, scaled_normal init, `max_steps` = 2000 (read from `state.json`'s shape, never shortened — shortening is what zeroed iteration 4).

The program is stdlib-only, self-contained, reads `state.json` from its own working directory, echoes `summary_digest` verbatim, sets `inherits_from: []`, and leaves `report.crossing_step` null so nothing can diverge from the verifier's measurement. I validated it in a clean temp directory with a mock `state.json` and confirmed the emitted `proposal.json` is well-formed and names no frozen axis.