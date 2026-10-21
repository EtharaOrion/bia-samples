`/workspace/refine.py` is written, verified to run standalone, and emits a valid `proposal.json`.

**What I did this attempt**

- Started from the best known recipe (iterations 3/5: `ortho_momentum`, lr 0.021, wsd, embed/head mult 10, max_steps 2000 → crossing 1370).
- Ran two concurrent 250-step dev probes to test whether the crossing was LR-limited: baseline (lr 0.021, mult 10) vs aggressive (lr 0.030, mult 16, warmup 1%). Result at step 250: **6.17 (baseline) vs 6.54 (aggressive)** — the aggressive direction is clearly worse, so the earlier plan of "raise peak LR / raise embed mult" is refuted by measurement, not just by prior.
- Accordingly I reverted the aggressive LR/mult bump I had staged, and shipped the known-good core with two small, low-risk changes in the remaining directions: `beta1` 0.95 → 0.96 (slightly longer momentum horizon, i.e. more effective averaging per step at the same LR) and `warmup_frac` 0.02 → 0.01 with `decay_frac` 0.7 (anneal begins ~step 600, so the LR through the 1100–1400 crossing region is lower than iteration 3's while the full 2000-step horizon is kept intact so the sustain window always exists).
- `max_steps` stays at the frozen ceiling of 2000 (~330 s, inside the 432 s bound). Iteration 4's lesson — shortening the horizon destroyed the crossing entirely and cost the whole upper band — is respected.

**Key finding for the next run:** the aggressive-LR hypothesis that iterations 1, 3 and 5 all recommended is now measured and wrong at this operating point; explore *downward* LR with higher `beta1`, or a different optimizer family, rather than pushing lr/mult up again.