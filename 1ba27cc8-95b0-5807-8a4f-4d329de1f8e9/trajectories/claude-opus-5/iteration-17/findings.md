Submission is in place and verified.

**What I submitted (`/workspace/refine.py`)**

Starting from the best-known core (iteration 13/16: `ortho_momentum` + WSD, `decay_frac 0.72`, `embed_lr_mult 32`, `head_lr_mult 5`, `ns_steps 6`, `beta1 0.92`), I changed two coupled axes:

- `max_steps` 1588 → **1500** (still ~255 s at 0.170 s/step, comfortably inside the 300 s budget, and far above the known 1190 crossing plus its 30-step sustain window, so the crossing cannot be cut off)
- peak `lr` 0.017 → **0.019**

**Reasoning from the history:** the derived anchors are `baseline ≈ 1800`, `target ≈ 1150`, so I only need ~40 steps of improvement over 1190 to reach score 1.0. Compressing the horizon 2000 → 1588 alone produced *no* change (1190 both times), which says the crossing region is token-limited, not tail-LR-limited; and pushing the LR down at the crossing (decay_frac 0.9) regressed to 1290. So I raised the peak LR to learn faster early while shortening the horizon just enough that the annealed LR near step 1190 (0.287 × 0.019 ≈ 0.0055) stays close to the 1190-crossing recipe's value (0.348 × 0.017 ≈ 0.0059) — more early progress with an unchanged effective tail.

**Validation:** a 250-step dev probe at `lr 0.019` with `embed_lr_mult 32` ran cleanly to loss 5.928 with no instability, confirming the higher peak LR does not diverge under the scale-invariant orthogonalized update.

I kept `report.crossing_step` null (verifier owns the measurement), echoed `summary_digest` verbatim, and left `inherits_from` empty — the protocol fields that caused the earlier gate failure are correct, and the step count is sized against the *current* 300 s budget rather than the stale 0.12 h figure that busted iteration 15.