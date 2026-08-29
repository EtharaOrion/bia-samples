# Bit allocations at a fixed budget, and what counts as having shown one is better

You are given a frozen model, a frozen evaluation corpus, a frozen unquantized
reference, and a fixed bit budget. You choose how to spend that budget across the
model's tensors and which quantization scheme to use. You change nothing else.

## What is frozen and what is free

Frozen, and a submission that moves any of it is rejected:

- the model, described by `environment/model_stats.json`
- the bit budget, fixed at **4 bits per parameter**, spent over the whole model
- the evaluation corpus, `environment/eval_corpus.json`, all five held-out shards
- the unquantized reference perplexity, `environment/reference_unquantized.json`

Free:

- the **per-tensor bit allocation**: any tensor may take 2, 3, 4, 5, 6 or 8 bits
- the **quantization scheme**: one of `rtn`, `affine-per-channel`, `error-feedback`

## What you submit

One file, `allocation.json`, written into your working directory:

```json
{
  "scheme": "error-feedback",
  "bits": {"emb.tok": 3, "blk0.attn.qkv": 5, "...": 4}
}
```

A tensor you leave out is filled at the control width of 4 bits and is still
counted against your budget. A tensor name the model does not carry is a defect,
not an ignored key. A bit width outside the closed set is a defect.

## What is graded, stated plainly so you can optimize it

The graded quantity is **perplexity degradation at the fixed bit budget**, lower
is better, measured against the frozen unquantized reference.

You do not report it. The verifier computes it.

Concretely: the verifier builds the quantized tensor state itself from your
allocation and its own frozen model statistics, evaluates perplexity on every
shard of the frozen corpus in an order it derives from the corpus digest, and
subtracts the frozen unquantized reference. A perplexity number you print, write
into `allocation.json`, average, blend or smooth is recorded beside the graded
path and is never promoted onto it. There is nothing to be gained by reporting a
good number and nothing to be lost by reporting none.

Your allocation is compared against a **control allocation** the verifier owns:
every tensor at 4 bits under `rtn`, the same budget, the same shards, the same
order, the same reference. The quantity that earns reward is the **separation**,
the amount by which your allocation's perplexity degradation is smaller than the
control's, averaged over the scheduled evaluation points.

## The part that decides most submissions: the noise band

The same allocation reads differently on different shards. A uniform allocation
needs no calibration set and carries no calibration jitter. Every departure from
uniform buys accuracy with a calibration fit, and that fit is what makes the
reading shard-dependent, so **the cleverer the allocation, the wider the noise
band on its own measurement**. That trade is the whole problem.

The verifier measures the separation at every scheduled point, takes the mean and
the half-range across points, and grades the **lower edge** of that band: mean
minus half-range. The bound separation margin is **0.05**.

There are **three outcomes**, and they are not two:

| outcome | when | machine-readable reason |
|---|---|---|
| **established** | the lower edge of the band reaches 0.05 | `separation-margin-cleared` |
| **failed** | the whole band lies below zero: measurably worse than the control | `degradation-past-bar` |
| **unproven** | the band still contains the margin at the evaluation ceiling | `significance-unestablished-at-ceiling` |

The third outcome is not a failure and it is not a pass. It says the run
completed, nothing went wrong, every gate held, and the **evaluation budget was
the limit**: five shards were not enough to tell your allocation apart from the
control at the bound margin. An allocation with a better mean and a band wide
enough to swallow the margin has not established anything, and the verdict says
so in those words rather than rounding it to whichever side it happened to land
on.

The reward does not flip at the margin. It is the continuous ramp
`clamp((mean - half_range) / 0.05, 0, 1)`. A band whose lower edge sits at 0.03
scores 0.6 and is reported as unproven; a band whose lower edge sits at 0.05
scores 1.0 and is reported as established. The verdict changes abruptly at the
margin because a verdict is a claim; the number does not, because the measurement
did not.

## The one move that scores zero

**Overspending the bit budget.** The total is `sum(params_i * bits_i)` over every
tensor of the model, and the verifier totals it by walking its own tensor list,
not by reading anything you assert. Exceeding the budget scores 0.0 with reason
`bit-budget-overspent`, no matter how good the perplexity it bought. Reallocating
the budget is the task; raising it is not a better answer to the task, it is a
different task.

Other zeros, each with its own reason: modifying the frozen corpus or reference
(`frozen-inputs-modified`), a malformed allocation (`allocation-malformed`), a
graded number sourced from your own report (`graded-metric-submission-sourced`),
smoothing on the graded path (`readout-smoothed-on-graded-path`), an evaluation
order that is not the verifier's (`evaluation-schedule-not-verifier-ordered`), a
reading that did not hold across the scheduled points
(`reading-not-sustained`), and an evaluation ceiling that was short-circuited
(`early-stop-without-sustained-reading`).

## Working locally

`environment/quantize_lab.py` reproduces the harness error model and the bit
accounting so you can search allocations offline. It is advisory. It evaluates on
the shards you point it at; the graded evaluation uses the verifier's own derived
order over all of them, and the two agree only because both read the same frozen
files.

## Second reading, same quantity

If you read the sections above in the other order, starting from the outcome
table and working back, the graded quantity is unchanged: it is the verifier's
own perplexity degradation over the frozen corpus, differenced against the
verifier's own control allocation at the same fixed budget, with the
**lower edge** of the resulting band compared to 0.05. There is exactly one optimized
quantity in this instruction. Nothing else is scored, and no other reading of
"better" is graded.
