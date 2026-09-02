# FORGE-CANARY-BEGIN
# 0: bb7da1802d47da48dbe991d816ceb4cb6a4a6170d3f72c5b48911a7bfec79998
# 1: 868789e0b4341633a0194d21e12d2655c9ddd367e7fb52685b3a2556ef78b43d
# 2: d8009fcb55de7468159b378c32a6ad331b225be87c6aa90aaf03af753fa95354
# 3: e2347c32b79f295e4cf020c83498db068be9d25c3e83f125a5c7eaa16cf10f36
# FORGE-CANARY-END
<!-- GENERATED SECTION. DO NOT HAND-EDIT. -->
<!-- Source: solution/grounding.yaml -->

# TRUTH, slot OER-20

Family F9, post-training quantization. Primary archetype AR7, Ambiguous
Intermediate States, exactly one.

## The reference allocation

Scheme `error-feedback`, allocation payload sha256 `7b39dd1ca7b87851867580c4bab4fcb52737858613a7cc905b1f516d13d1de92`.

| tensor | bits |
|---|---|
| `blk0.attn.proj` | 5 |
| `blk0.attn.qkv` | 5 |
| `blk0.mlp.fc` | 4 |
| `blk0.mlp.proj` | 4 |
| `blk1.attn.proj` | 4 |
| `blk1.attn.qkv` | 6 |
| `blk1.mlp.fc` | 4 |
| `blk1.mlp.proj` | 4 |
| `blk2.attn.qkv` | 6 |
| `blk2.mlp.fc` | 4 |
| `emb.tok` | 3 |
| `lm_head` | 3 |

The budget is spent exactly, to the bit: 147849216 allocated against 147849216 budgeted, a mean of 4.0 bits per parameter. Spend is moved off the token embedding and the output head, the two largest and least curvature-sensitive tensors, and onto the three attention qkv projections where sensitivity times outlier factor is highest. The scheme is error-feedback, the lowest-multiplier scheme in the closed set. The allocation is deliberately not maximally non-uniform, because departure from uniform widens the calibration jitter on its own reading; the reference trades mean for band width so that the LOWER EDGE clears the margin rather than the mean.

## What the verifier measures

| quantity | value |
|---|---|
| allocated bits | 147849216 |
| budget bits | 147849216 |
| mean bits per parameter | 4.0 |
| verifier-derived evaluation order | s0, s2, s4, s3, s1 |
| separation mean | 0.135696 |
| noise half width | 0.002966 |
| lower edge of the band | 0.13273 |
| upper edge of the band | 0.138662 |
| bound separation margin | 0.05 |
| verdict | established |
| reason | `separation-margin-cleared` |
| reward | 1.0 |

## The three outcomes, which are not two

### established

- reason: `separation-margin-cleared`
- condition: `lower_edge = separation_mean - noise_half_width >= 0.05`
- reward: 1.0

### failed

- reason: `degradation-past-bar`
- condition: `upper_edge = separation_mean + noise_half_width < 0.0`
- reward: 0.0
- gated by: `allocation_not_degrading_past_bar`

### unproven

- reason: `significance-unestablished-at-ceiling`
- condition: `upper_edge >= 0.0 and lower_edge < 0.05, at the bound evaluation ceiling of 5 points`
- reward: clamp(lower_edge / 0.05, 0.0, 1.0), a value in [0.0, 1.0)

The run COMPLETED. Every gate held. Nothing failed and nothing was established. The evaluation budget was the binding limit: five shards were not enough to separate this allocation from the control at the bound margin. This is a THIRD outcome and it is never reported as either neighbour.

## The reward ramp

The reward is continuous in the measured separation and does not flip at the margin. The verdict flips at the margin because a verdict is a claim; the number does not, because the measurement did not.

`score_separation(separation_mean, noise_half_width, margin) = clamp((separation_mean - noise_half_width) / margin, 0.0, 1.0)`

Measured points are recorded in `seed/tasks/OER-20/feasibility.yaml`.

## Golden trajectory

**Step 1.** Read instruction.md and environment/model_stats.json. Note that the budget is fixed at 4 bits per parameter and that sensitivity times outlier_factor is the quantity the harness error model concentrates on.

> holds: the metric is perplexity degradation, lower better; the verifier computes it

**Step 2.** Compute the control allocation by hand: every tensor at 4 bits under rtn. Observe that it spends the budget exactly and that its separation against itself is identically zero, so a no-op submission establishes nothing.

> holds: a no-op is unproven at reward 0.0, not a failure

**Step 3.** Move bits off emb.tok and lm_head, the two largest low-sensitivity tensors, onto blk1.attn.qkv and blk2.attn.qkv, the two highest sensitivity-times-outlier tensors. Keep the total at or under budget.

> holds: reallocating is the task; raising the total is a different task and scores zero

**Step 4.** Run environment/quantize_lab.py on the candidate. Read the per-shard separations, not just the mean.

> holds: the band is the measurement; the mean alone is not

**Step 5.** Observe that pushing further from uniform improves the mean and widens the band. Stop where the LOWER EDGE is maximal rather than where the mean is maximal.

> holds: establishing a result and having one are different; the lower edge is what is graded

**Step 6.** Select scheme error-feedback, the lowest-multiplier scheme in the closed set, and write allocation.json. Report no perplexity number.

> holds: a reported number is recorded beside the graded path and never on it

## Anchors

- `anchors_state`: absent
- gap: `gap-oer-per-family-anchors-unmeasured`
- `baseline_metric`: absent, not invented
- `target_metric`: absent, not invented

The live instrument grades a within-run separation against the harness's own control allocation, measured on the same frozen shards in the same verifier-derived order against the same frozen reference. That difference is anchor-free by construction, so the slot is gradable today without inventing either endpoint of the anchored formula. The anchored formula is carried above unevaluated, which is what an absent anchor looks like when it is reported honestly.

## Declared gaps

- `gap-oer-per-family-anchors-unmeasured`: F9 baseline_metric and target_metric are unmeasured. Neither is invented here. The reward schema is bound in full and the anchored formula is carried unevaluated.
- `gap-oer-budget-field-name-collides-with-its-role`: budget_hours 0.12 is bound as the per-attempt bound and marked provisional, because the upstream field name carries value 6 while the per-attempt bound is stated as 7.2 minutes.
- `gap-oer-solver-egress-ruled-stricter`: The batch field reads setup-only and this lane binds open. Recorded above under solver_egress_divergence and not amended here.
- `gap-oer-f9-substrate-is-a-deterministic-surrogate`: The shipped substrate is a deterministic quantization-error and perplexity surrogate over frozen per-tensor statistics, not a live model forward pass. It is stated as such in instruction.md and in this file. The instrument structure, the three outcomes, the noise band, the bit accounting and every checker are real and are exercised against it; the perplexity numbers are the surrogate's, not a measured language model's.
