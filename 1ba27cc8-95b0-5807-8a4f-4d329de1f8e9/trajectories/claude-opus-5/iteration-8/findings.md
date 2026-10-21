`/workspace/refine.py` is written and verified end-to-end.

**Submitted recipe:** iteration-7's best core (ortho_momentum, wsd, lr 0.018, warmup 0.02, decay_frac 0.6, embed/head mult 10, scaled_normal, max_steps 2000 — full ceiling so the crossing always exists) with **beta1 lowered 0.95 → 0.92**.

**What I measured this run (200-step dev probes, concurrent):**
- head_lr_mult 10 → loss 6.341; head_lr_mult 4 → 6.499. Lowering the head LR hurts, so the mults stay at 10.
- Reading `bia_optim.py`: `weight_decay` is applied **only** for `adamw` and `lion` (and `adam` via the update term). For `ortho_momentum` it is a **dead knob** — no future iteration should spend a change on it.

**Rationale for beta1:** the only prior evidence on this axis is iteration 6 (beta1 0.96 → crossing 1410, worse than 0.95's 1360–1370), so the measured gradient points downward; with orthogonalization normalizing update scale, beta1 changes direction/horizon rather than effective step size, making it a safe structural axis unlike peak LR, which the history shows is flat (~10 steps per probe).