# OER-01 grounding

Every number in this file was produced by running this slot's own harness on this
slot's own substrate during authoring. Nothing here is a target the verifier reads:
the verifier measures the target and both endpoints of its reward scale from scratch
on every grading run, and no literal for any of them appears anywhere in the bundle.

## Three designs were measured and rejected before this one

This slot is a transplant, and the surface it landed on was chosen by measurement
rather than by taste. Three earlier designs were built and measured on this exact
substrate and rejected:

**Optimizer-state precision under a memory ceiling.** Where it is legal the space is
flat: float32 moments everywhere measured 5.045449, naive bfloat16 measured 5.063693,
and bfloat16 with stochastic rounding measured 5.040928 — a total range of 0.023 nats
across the whole non-degenerate region, which is too thin to grade. Where it is not
flat it is a cliff rather than a gradient: float16 second moments measured 15.873891
unscaled, 5.730789 at scale 1e6 and 12.037151 at scale 65536, and linear int8 second
moments diverged outright at 15.677600. A reward scale whose span is one binary
choice between a flat region and a cliff is not a task.

**A tighter ceiling to force int8 quantisation.** Same outcome from the other side:
int8 for the second moment diverges at every block size measured, so the ceiling only
moved the cliff.

**The gradient-accumulation schedule.** Monotone at this budget, with the peak
learning rates frozen at twice the published values: accumulating 1 measured
5.128653, 2 measured 5.321595, 4 measured 5.378550, 8 measured 5.655722 and 16
measured 6.236641, and a 1→2→4 ramp measured 5.190472, losing to constant 1. There is
no interior optimum to find, so `grad_accum` is frozen at 1 in the shipped schema and
the reason is recorded there.

## The design that was kept

Keep the substrate and change the GRADED QUANTITY. The target is the shipped default
recipe's OWN evaluation loss at the end of the ceiling. What is graded is how few
micro-batches a recipe needs to reach it.

## Measured

Every row trained by this harness from the frozen initialisation on the identical
token stream, evaluated every 128 micro-batches on the same 524288-token window:

| recipe | final eval loss | micro-batches to target |
|---|---|---|
| shipped default — constant 3e-4, no warmup | 5.527604 | **2944** |
| hold-then-decay, 10% warmup, 60% hold *(published)* | **5.053879** | 1792 |
| mild hold, 10% warmup, 50% hold *(published)* | 5.054898 | 1792 |
| hot short hold, 5% warmup, 25% hold *(published)* | 5.242303 | 1920 |
| linear to zero, 5% warmup *(the oracle)* | 5.193933 | **1664** |
| cosine to zero, 5% warmup *(the private reference)* | 5.256578 | **1664** |

**The two objectives disagree, and that is the whole slot.** The recipe with the best
final loss — 5.053879, the published hold-then-decay record — crosses the target at
1792. The recipe with nearly the WORST final loss of the tuned set — 5.256578, a
cosine — crosses at 1664. Holding a peak buys quality that arrives after the bar has
already been passed, and this reward does not pay for it.

The target itself is the shipped default's 5.527604, and the default first reaches it
at 2944 rather than at 3072: its constant learning rate makes the curve wander, and it
dips below its own endpoint one grid point early. That is measured, not assumed, and
the grader takes the control endpoint from the same first-crossing rule it applies to
everything else rather than assuming the ceiling.

## The anchors and the span

* **control** — the shipped default's first crossing, measured **2944** micro-batches
* **reference** — the private cosine's first crossing, measured **1664**
* **span** — **1280** micro-batches, ten grid points

## The published-record corpus

Three recipes, all measured above, all real. The corpus contains the strongest final
loss known for this ceiling and the most robust of the published recipes; both of them
cross at 1792, which would be 90% of the reward. They are excluded, so a submission
cannot simply restate the best published recipe and collect. Identity is the graded
fields at 6 significant digits and nothing else, so the corpus excludes exactly what
it lists.

The `measured` field on each record is an authoring measurement kept for a human
reader. Nothing in the grading chain reads it.

## The oracle

`solution/reference.py` writes a linear decay to zero from a 5% warmup, measured to
cross at 1664 — the same grid point as the private cosine, by a different decay
family and at different peak learning rates. The two are different documents and were
arrived at separately. Neither is in the published corpus, and the verifier asserts at
image build that its own reference is not, because a bar that is excluded from the
submission space is a bar nothing could legally reach.

## Cost

Three from-scratch runs of the full ceiling, each with 24 evaluations of a
524288-token window folded in. Each arm measured 101-127s during authoring; a grading
pass is about **360s** on one H100 including the compile, against a declared budget
of 900s.
