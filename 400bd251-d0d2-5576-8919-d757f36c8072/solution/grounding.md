# OER-28 grounding

Every number in this file was produced by running this slot's own harness on this
slot's own substrate during authoring. Nothing here is an anchor the verifier reads:
the verifier times both endpoints of its reward scale itself on every grading run,
and no literal for either appears anywhere in the bundle.

## What the sweep measured

Nine plans, all on the same frozen parameters, timed round-robin over 7 rounds of 24
steps. Latencies below are the per-round medians, which is what the shipped harness
grades on.

| plan | median ms/step | speedup | loss delta | grad relative delta |
|---|---|---|---|---|
| `math`, 1 chunk, fp32 *(the control, and the equivalence reference)* | 107.19 | 1.000x | 0 | 0 |
| `math`, 8 chunks, fp32 | 114.55 | 0.936x | 0.000001 | 0.000122 |
| `flash`, 32 chunks, fp32 | 110.91 | 0.966x | 0.000056 | 0.000260 |
| `flash`, 16 chunks, fp32 | 104.55 | 1.025x | 0.000057 | 0.000152 |
| `flash`, 8 chunks, bf16 rotary + qk | 100.04 | 1.071x | 0.000065 | 0.000131 |
| `flash`, 8 chunks, fp32 | 99.23 | 1.080x | 0.000055 | 0.000106 |
| `flash`, 8 chunks, **bf16 logits** | 97.37 | 1.101x | 0.000112 | **0.001189** |
| `efficient`, 1 chunk, fp32 *(the oracle)* | 92.80 | 1.155x | 0.000074 | 0.000016 |
| `flash`, 1 chunk, fp32 *(the private reference)* | **91.66** | **1.169x** | 0.000055 | 0.000020 |

## Three findings, all of them measurements

**The attention backend is the entire span.** The textbook `math` backend materialises
the full score matrix; either fused backend removes that and is worth about 1.16x on
the whole step. Nothing else in the schema comes close.

**Chunking the cross-entropy is a distractor, and a convincing one.** The unchunked
loss head materialises a 4096-by-50304 float32 logit tensor and then squares it for the
softcap, which looks exactly like the thing to break into pieces. Measured, every chunk
count is slower than none: 8 chunks costs 99.23ms against 91.66ms, 16 costs 104.55ms,
32 costs 110.91ms, monotonically. Chunking trades a few large matmuls for many small
ones and pays in launch overhead. It saves memory. It does not save time. This is the
axis the slot is really testing, because it is the one an engineer will assume.

**Reduced precision is not a shortcut here either.** bfloat16 in the rotary and the qk
normalisation measured slower than float32. bfloat16 in the logit softcap and the
cross-entropy measured slower *and* is the only plan in the table outside the
equivalence tolerance.

## Why the estimator is a median and not a minimum

The harness originally took the minimum over rounds, on the reasoning that contention
can only slow a round down. Measurement contradicted that. Round 1 came in at 80.15ms
for the control against 106.8-107.3ms for rounds 2 through 7, and inside that first
round the plans were measured at different points of the ramp — the control at 80.15ms,
the memory-efficient backend at 63.44ms and the bfloat16-logit plan at 97.39ms. A
minimum therefore picks one anomalous round, and picks it unevenly across plans; it
scored the oracle at a raw 1.46 on round 1 alone.

Rounds after the first are extremely stable. Reward computed from each of rounds 2
through 7 independently:

| round | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|
| reward | 0.9274 | 0.9281 | 0.9286 | 0.9244 | 0.9285 | 0.9243 |

A spread of 0.004 on a reward of about 0.926, measured while another tenant was
training on the same accelerator. That is why the shipped harness warms every plan
before timing any of them, runs 9 rounds, and grades on the median. `min_ms` is still
recorded in the score document for a reader, and nothing grades on it.

## The tolerances

Set from the table above rather than chosen. Across every plan that computes the same
function the worst disagreement measured was 5.7e-5 nats of loss and 1.5e-4 relative
gradient norm, once `loss_chunks: 32` was removed from the option set for being both
the slowest and the loosest. Carrying the logit softcap and the cross-entropy in
bfloat16 measured 1.1e-4 and 1.2e-3.

Both tolerances are set at **5e-4**, which leaves more than a threefold margin above
every equivalent plan and more than a twofold margin below the one that is not. The
bfloat16-logit plan is refused on the gradient, which is the term that actually
notices it.

## The anchors and the span

* **control** — `math`, 1 chunk, fp32, measured **107.19ms**
* **reference** — `flash`, 1 chunk, fp32, measured **91.66ms**
* **span** — **15.53ms**, about 14.5% of the control

The control is not a straw man. It is the canonical, most conservative execution of the
frozen computation and it is also the plan every submission is checked against for
correctness, which is why it is public. It runs, it is exact, and it is 17% slower.

## The oracle

`solution/reference.py` writes `efficient`, 1 chunk, fp32, measured 92.80ms, which
closes **(107.19 - 92.80) / 15.53 = 92.7%** of the gap. It is a different document from
the private reference and picks a different fused backend. The two land 1.14ms apart,
which is the honest size of the remaining decision once the real one has been made.

## Cost

Three plans, each checked for equivalence over 4 fixed batches and then timed for 24
warmup steps plus 9 rounds of 24 steps. Measured **about 150s** end to end on one
H100 while sharing it with another tenant, against a declared budget of 900s.

## A note on memory

The step is 8 sequences rather than a training-sized 16 because the unchunked control
plan materialises a rows-by-50304 float32 logit tensor, and at 16 sequences that peak
made the timed run fail with CUDA out-of-memory on a grading host already carrying
another tenant. Halving the step keeps every plan's relative cost intact — the loss
head still dominates — while keeping the whole measurement inside a few gigabytes.
There is a certain irony in the fact that chunking the loss would have fixed the memory
problem and would have cost time, which is precisely the trade this slot is about.
