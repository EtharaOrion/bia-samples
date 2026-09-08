# OER-05 — grounding

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
| 5.01871 | per-role at 2.4e-2, second moment 0.995 instead of 0.999 | the best point found: SHORTENING the second moment is worth 0.027 on top of everything else |
| 5.03646 | per-role at 2.4e-2, plus decoupled weight decay 0.05 | a little decay pays at this budget |
| 5.04580 | per-role, embed and scalar 2.4e-2, hidden and head 4e-3 | the interior optimum of the embedding boost, at six times the block step |
| 5.05079 | per-role at 2.4e-2, plus sign_mix 0.35 on embed and scalar | sign blending does not help the embedding either |
| 5.06069 | per-role, embed and scalar 2.4e-2, hidden 6e-3, head 2e-3 | pricing the head below hidden does not pay by itself |
| 5.06555 | per-role, embed and scalar 1.2e-2, hidden and head 4e-3 | splitting the roles apart is worth real nats |
| 5.06783 | per-role, embed and scalar 3.2e-2, hidden 5e-3, head 3e-3 | also past it |
| 5.07730 | per-role, embed and scalar 4e-2, hidden and head 4e-3 | past the peak: the embedding boost has an interior optimum |
| 5.12992 | per-role, sign_mix 0.5 on the block matrices | blending toward sign descent on hidden is worth almost nothing |
| 5.13234 | Adam point, uniform 3e-3 -- the shipped default | the floor the submission has to beat |
| 5.15298 | Adam point, uniform 6e-3 | the uniform step size is already near its own optimum; there is no free win here |
| 5.17570 | per-role, first moment 0.95 instead of 0.9 | lengthening the FIRST moment costs |
| 5.30835 | sign momentum, uniform 1e-3 | pure sign descent is a real optimizer here and still well behind |
| 5.43934 | per-role, precond_power 0.65 instead of 0.5 | and it is peaked from the other side too |
| 5.53349 | per-role, precond_power 0.35 instead of 0.5 | the preconditioner exponent is sharply peaked at Adam's value |
| 6.77797 | heavy-ball momentum SGD, uniform 0.15 | the best plain-momentum point found is still 1.6 nats behind Adam |
| 7.24541 | heavy-ball momentum SGD, uniform 0.05 | same, at a larger step |
| 7.61175 | heavy-ball momentum SGD, uniform 0.02 | no preconditioner at all is nowhere near competitive |

### The measurement floor

This harness is deterministic in everything but the accelerator's own reduction order. The shipped default was trained three separate times during this search and produced 5.13234, 5.13296 and 5.13353, a spread of 0.0012 nats. The span between the two anchors here is 0.114 nats, which is ninety-five times that spread, so a difference of more than about 0.004 nats is a real difference and not measurement noise.

## The two anchors and the oracle

| artifact | where it lives | measured devset loss |
|---|---|---|
| the shipped default, LOW anchor | `environment/default_rule.json` | 5.13234 |
| the private reference, the BAR | `tests/private/reference_rule.json` | 5.01871 |
| the oracle's submission | written by `solution/reference.py` | 5.04580 |

The oracle is **not** the reference. They were arrived at on separate legs of the
search and differ in its second-moment decay -- the oracle runs the conventional 0.999, the reference runs 0.995, which was the last thing the search found and the single largest remaining gain. On the devset numbers above the oracle closes

```
(5.13234 - 5.04580) / (5.13234 - 5.01871) = 0.762
```

of the gap, so the reference arm of the gate scores on a measurement rather than by
construction, and a neutralised submission is refused by the schema at 0.0.

## Why this is hard

The default is AdamW at a learning rate that works, so there is no free win from discovering Adam. Everything left has to come from the law itself, and the law does not reward being moved in the obvious directions. Raising or lowering the single uniform step size is worth almost nothing (5.13234 at 3e-3 against 5.15298 at 6e-3). Turning the preconditioner exponent off the Adam value in either direction is a disaster in both (5.53349 at 0.35, 5.43934 at 0.65) even though the interior of that interval sounds like it ought to interpolate. Blending toward sign descent hurts (5.12992), lengthening the first moment hurts (5.17570), and abandoning preconditioning entirely for heavy-ball momentum costs more than a nat and a half at every step size tried. What does pay is splitting the roles apart -- the embedding and the scalars want six times the step of the block matrices -- and then, only after that, shortening the SECOND moment from 0.999 to 0.995, which is the opposite of the direction a short run is usually argued into. Each of those two findings is worth roughly half the available span and neither is visible without running the harness.

## Cost

The graded path is three from-scratch training runs of 3072 micro-batches of 16x512
plus three evaluations over 2,097,152 tokens, sharing one compiled graph. Measured wall
clock is recorded in `task.toml` as `graded_path_seconds_measured`, and it is the
gate's own number rather than a declaration.
