# OER-11 — grounding

## What is graded

The verifier's own validation cross-entropy, in nats, on a held-out FineWeb slice that
exists only inside the verifier image. The reward is

```
reward = clamp( (default_loss - agent_loss) / (default_loss - reference_loss), 0, 1 )
```

and **both** endpoints are measured by real from-scratch training runs on the grading
run itself. There is no literal for either anchor anywhere in this bundle. This is the
substantive change from the bundle that stood in this slot before, which declared its
anchors unmeasured (`gap-oer-per-family-anchors-unmeasured`) and therefore refused
every possible submission at 0.0 — a constant reward, and a dead task.

## The authoring search

Every loss below was produced by a run of this bundle's own harness under the frozen
budget, evaluated on `environment/data/devset_slice.bin`. The graded split is a
different slice of FineWeb, so the graded numbers sit at a small offset from these;
what carried the decisions is the ordering, which the offset does not disturb.

| loss | candidate | what the run established |
|---|---|---|
| 4.93091 | FAMILY A: the two residual projections to 0.008, a divisor of 2.5 | the best point found: a MILDER taper than the textbook one, and the optimum is sharp |
| 4.94711 | FAMILY A: embedding to 0.014 | inside the measurement floor of flat 0.02; not a real gain |
| 4.94938 | FAMILY A, flat 0.02 everywhere -- what GPT-2 uses | 88 percent of the framework gap, from one number applied everywhere |
| 4.95639 | FAMILY A: embedding to 0.01 | shrinking the embedding does not pay either |
| 4.95960 | FAMILY B, on the framework default: embedding to 0.014 and projections to 0.008 | fixing the embedding is where essentially all of the framework gap lives |
| 4.96277 | FAMILY A: q, k, v and the MLP input to 0.015 | the block matrices want 0.02 as well |
| 4.96427 | FAMILY A: norm_gain 0.8 | the RMSNorm gains want to start at one |
| 4.97758 | FAMILY A: residual_depth_power 0.5, the textbook 1/sqrt(2L) taper | the standard depth taper OVERSHOOTS at six layers and loses 0.028 |
| 4.99565 | FAMILY A: flat 0.014 on every weight role | and uniformly smaller is worse too, so 0.02 is not an accident |
| 5.02346 | FAMILY A: flat 0.06 on every weight role | uniformly larger is worse |
| 5.07217 | FAMILY B, on the framework default: taper the projections to 0.008 | 0.010 nats for the best taper, against 0.133 for fixing the embedding: the two are not comparable |
| 5.08238 | FAMILY B, the framework's own initialisation -- the shipped default | the floor; nn.Embedding's unit normal is fifty times the rest of the network |
| 5.08986 | FAMILY B, on the framework default: taper the projections to 0.011 | same conclusion from the other side |
| 5.10488 | FAMILY A: embedding to 0.05 | already past the optimum at two and a half times |
| 5.10601 | FAMILY B, on the framework default: taper the projections to 0.006 | on the framework base the projection taper is nearly inert -- the embedding is the problem |
| 5.13267 | FAMILY A: embedding to 0.10 | the embedding is monotone the other way too |
| 5.16010 | FAMILY A: output projection to 0.01 | still losing; nothing below 0.02 pays on the head |
| 5.22347 | FAMILY A: output projection to 0.005 | the head is the most sensitive role and it wants to be LARGE |
| 5.31025 | FAMILY A, from a flat 0.02: output projection to 0.0 | the standard zero-init-the-head trick is the single worst move in this space |

### The measurement floor

This slot's spread is an order of magnitude wider than its sibling slots' and the span was sized against it rather than against a hoped-for figure. The INITIALISATION is exactly reproducible -- the reference document was applied three times in one process and produced a bit-identical parameter digest and a bit-identical step-zero loss of 10.892676 every time -- but the TRAJECTORY is not, because the frozen optimizer is hot and bfloat16 reductions do not associate. Those same three runs finished at 4.928943, 4.939727 and 4.935169, a spread of 0.0108 nats. The span between the two anchors is 0.1515 nats, fourteen times that spread, so a candidate has to be better by roughly 0.02 nats before the difference is worth believing, and the search below only ever acted on differences several times larger than that.

That floor has a consequence the gate made visible and it is recorded here rather than smoothed over. The oracle and the private reference are separated by 0.0185 nats on the devset, but on the graded split three gate runs measured that separation at 0.0065, 0.0136 and below zero, and on the third run the oracle therefore scored exactly 1.0 with reason `reference-reached`. The oracle's reward on this slot is a range, 0.87 to 1.0, and not a point. That is a property of a bar-shaped reward with a 0.011-nat measurement floor and a 0.15-nat span: the DEFAULT-to-reference distance is fourteen times the floor and is never in doubt, but the last two percent of the scale is inside it. A submission that lands anywhere in the top two percent of the range is scored as having reached the bar, which is what `target_is_a_bar` in task.toml means and is not a defect of the measurement.

## The two anchors and the oracle

| artifact | where it lives | measured devset loss |
|---|---|---|
| the shipped default, LOW anchor | `environment/default_init.json` | 5.08238 |
| the private reference, the BAR | `tests/private/reference_init.json` | 4.93091 |
| the oracle's submission | written by `solution/reference.py` | 4.94938 |

The oracle is **not** the reference. They were arrived at on separate legs of the
search and differ in the two residual-writing projections -- the oracle leaves them at the same 0.02 as everything else, the reference tapers them to 0.008. On the devset numbers above the oracle closes

```
(5.08238 - 4.94938) / (5.08238 - 4.93091) = 0.878
```

of the gap, so the reference arm of the gate scores on a measurement rather than by
construction, and a neutralised submission is refused by the schema at 0.0.

## Why this is hard

The framework's own initialisation is genuinely bad and genuinely the thing you get for free: nanogpt.py calls no init function, so the token embedding starts at unit standard deviation, fifty times what the rest of the network uses. Correcting that is most of the available gap, and a submission that simply writes 0.02 across every role collects 88 percent of it. The last 12 percent is where the slot bites, because every refinement that a practitioner would reach for next is measurably WRONG here. Zeroing the output projection -- the standard trick -- is the worst single move available, costing 0.36 nats. Shrinking it at all costs: 0.005 loses 0.27 and 0.01 loses 0.21. Enlarging the embedding relative to the rest costs monotonically (0.05 loses 0.16, 0.10 loses 0.18). The textbook 1/sqrt(2L) residual taper, at residual_depth_power 0.5, LOSES 0.028. What actually pays is a milder taper of the same two tensors, 0.02 down to 0.008, a divisor of 2.5 where the textbook says 3.46 -- and the optimum is sharp enough that 0.006 and 0.011 are both far worse than 0.008. Nothing about that is deducible; it has to be measured, and measured against a noise floor wide enough that a careless search will chase its own variance.

## Cost

The graded path is three from-scratch training runs of 3072 micro-batches of 16x512
plus three evaluations over 2,097,152 tokens, sharing one compiled graph. Measured wall
clock is recorded in `task.toml` as `graded_path_seconds_measured`, and it is the
gate's own number rather than a declaration.
