`/workspace/refine.py` is written and verified (runs standalone in a temp dir, emits a valid `proposal.json`).

**What I changed vs. iteration 1 (score 0.76, crossing 1410):**
- Kept the proven core: adamw, `lr=0.004`, β=(0.9, 0.95), `eps=1e-10`, `wd=0.05`, `grad_clip=1.0`, scaled_normal init, `max_steps=2000` (never shortened).
- Switched the schedule from `wsd` (decay_frac 0.4, i.e. LR held flat until step 1200) to **cosine** with 2% warmup. Cosine anneals continuously, so at any intermediate step the LR is far lower (≈0.0008 vs ≈0.003 at step 1400) and the *anytime* loss is lower — which is exactly what an earliest-sustained-crossing metric favors.

**Probe evidence (250-step dev runs, 3 concurrent):** the aggressive variant (lr 0.006 + `embed_lr_mult=2.0`) was clearly worse at every eval point (6.807 vs 6.470 at step 210) than the lr 0.004 / embed_mult 1.0 configuration, so I rejected the "higher peak LR + boosted embedding LR" direction that the iteration-1 notes suggested and kept the LR at its proven value, changing only the schedule shape. A third probe (lr 0.009, embed_mult 4) OOMed from GPU contention and returned nothing.