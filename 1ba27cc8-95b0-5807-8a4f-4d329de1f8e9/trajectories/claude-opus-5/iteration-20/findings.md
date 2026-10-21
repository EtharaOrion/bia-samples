Submission complete and verified end-to-end.

**What I shipped** (`/workspace/refine.py`): the iteration-18/19 lineage — `ortho_momentum` (Newton–Schulz orthogonalized momentum, `ns_steps` 6, β₁ 0.92 / β₂ 0.95), WSD schedule, `scaled_normal` init, per-group LR multipliers `embed 32 / head 5` — advanced one notch along the trend that has been monotonically improving the crossing step:

| iter | lr | decay_frac | max_steps | crossing |
|---|---|---|---|---|
| 18 | 0.020 | 0.75 | 1420 | 1150 |
| 19 | 0.021 | 0.78 | 1380 | 1140 |
| **20** | **0.022** | **0.80** | **1340** | — |

**Reasoning about the risk.** Score is already saturated at 1.0, so the dominant consideration was not overshooting into a run with no crossing — the single most expensive failure available here (iteration 4) and the one gate failure in the history (iteration 15, `attempt-budget-exceeded`). Both are guarded:

- Iteration 19 crossed at 1140 with a 1380-step horizon, leaving ~240 steps of slack. Trimming to 1340 shifts the anneal marginally earlier while still leaving roughly 200 steps past the expected crossing — far more than the 30 steps the sustain window needs.
- `decay_frac` 0.80 sits between the proven 0.78 and the known-bad 0.90 (which regressed to 1290), so it stays on the good side of the turning point rather than extrapolating past it.
- `max_steps` is computed as `min(ceiling, budget×0.90/0.170, 1340)` read from `state.json`, so it self-clamps if the budget shrinks again. At 1340 × 0.166 s = **222 s against the 300 s bound**.

**Probes run** (two short ones, both returned, per the guidance against single long sweeps): on fully-annealed 260-step dev horizons, the new settings gave 5.9006 vs 5.9020 for iteration 19's exact settings — a tie within noise, but importantly confirming **no instability at lr 0.022**, which was the real question. I deliberately did not read a crossing-step prediction into that gap; the history shows repeatedly that short-probe loss deltas do not translate proportionally to crossing-step movement, so the probe was used only as a divergence check.

**Protocol correctness** (the gate-failure axis): `summary_digest` is echoed verbatim from the handed `state.json`, `inherits_from` is `[]`, `report.crossing_step` is `null` so nothing can diverge from the verifier's measurement, no frozen-axis key is named, and the file is stdlib-only and self-contained, reading `state.json` from its own working directory. I validated it by copying it alone into a fresh temp directory and running `python refine.py`, matching how the harness invokes it.

One caveat worth flagging: since the score caps at 1.0 and iterations 18–19 already reached it, this attempt cannot beat 1.0000 numerically — the only real outcomes are "hold 1.0" or "regress." I therefore optimized for a modest, defensible step forward in the crossing step rather than an aggressive horizon cut that could have forfeited the upper band entirely.