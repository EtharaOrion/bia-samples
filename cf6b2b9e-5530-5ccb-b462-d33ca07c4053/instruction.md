# Post-training quantization at a fixed bit budget

You are given a frozen model, a frozen bit budget, a frozen evaluation corpus and a
frozen unquantized reference. You choose the **per-tensor bit allocation** and the
**quantization scheme**. Nothing else is yours to move.

## What is graded, stated plainly

The graded quantity is **perplexity degradation at the fixed bit budget**, which is

```
degradation = ppl(quantized) - ppl(unquantized reference)
```

measured over the frozen evaluation corpus, averaged over every evaluation point the
verifier schedules. **Lower is better.**

Three things about that number, so you optimize it rather than discover it:

1. **The verifier computes it, not you.** It is recomputed inside the grading process
   from the grader's own pristine copy of the model under your allocation, evaluated
   against the held-out evaluation payload. No number you report, print or write into
   your submission is ever read as the graded degradation. Reporting one is optional
   and it changes nothing.
2. **It is not smoothed.** The graded reading is the raw evaluation. You may filter
   your own readings for your own use, but the readout filter you declare must be
   `none`; a declared EMA, moving average or any other blend scores zero.
3. **It must hold everywhere.** The bar must be met at **every** evaluation point the
   verifier schedules, not at the best one. A run that halts at a favourable point
   has not established the metric and is graded as such, with a reason, rather than as
   an absent result.

Two readings of this section reduce to the same graded quantity: "the mean of the
per-point degradations" and "the mean quantized perplexity minus the reference
perplexity" are equal, because the reference perplexity is a constant. The grader
computes both and records them side by side.

## The calibration set is versioned, and it moves

You fit your allocation against a **calibration set**, which is not the evaluation
corpus. The calibration set is **harness-owned and versioned**. It can move between
the moment you fit an allocation against it and the moment your allocation is graded,
and **nothing announces the move**.

An answer the calibration set gave you at iteration three is an answer about the
version that was in force *then*. It is not a timeless fact.

Ask it what version it is at:

```
python3 environment/calibration_probe.py
```

It prints the version in force, the harness tick that version was issued at, and the
sha256 over the canonical bytes of that version's per-tensor sensitivity vector.
Ordering is carried by the tick, which is an integer handle the harness issues. No
wall clock is involved anywhere, in the probe or in the grading.

Your submission must record which version you actually fitted against, and prove it
on the bytes.

## The substrate

- `environment/substrate.json` - the tensors, their sizes, the admissible bit widths,
  the group size, the scale width, the bit budget, and how the budget is accounted.
- `environment/reference_model.json` - the unquantized reference perplexity. Frozen.
- `environment/eval_corpus_manifest.json` - the evaluation points and their sizes.
  The payload itself is held out and you never see it. Frozen.
- `environment/calibration_state.json` - the harness calibration state: the version in
  force, its tick, and the ledger.
- `environment/calibration_stats.json` - the per-version estimated per-tensor
  sensitivity vectors. A version that is **not** in force is retained for the record
  and is not the version you are graded against.
- `environment/calibration_probe.py` - the handle described above.

Degradation over this substrate is a closed-form deterministic function of your
allocation and a frozen per-tensor sensitivity vector. It is a deterministic
surrogate, not a live forward pass, and `environment/substrate.json` says so.

The bit budget is accounted over the quantized tensors **including the per-group
scale tensors**. `environment/substrate.json` records the accounting in force. Read
it rather than assuming; it is not the only accounting that has ever been in force
here.

## What to submit

Write `submission.json` into your workspace:

```json
{
  "schema": "oer22.submission/v1",
  "allocation": {"<tensor id>": <bits>, "...": 0},
  "quantization_scheme": {"group_size": <int>, "scale_bits": <int>},
  "derived_against_calibration_version": <int>,
  "calibration_fit_witness": "<sha256 over the canonical bytes of the sensitivity vector you fitted against>",
  "observed_calibration_ledger": [<the versions you observed through the probe, in the order you observed them>],
  "protocol": {"points_completed": <int>, "halted_early": false},
  "readout": {"filter": "none", "reported_degradation": null}
}
```

Every tensor named in the substrate must carry a bit width from the admissible set.
There is no default and a partial allocation is refused.

`calibration_fit_witness` is `sha256` over
`json.dumps(vector, sort_keys=True, separators=(",", ":"), ensure_ascii=True)` with
each element rounded to 9 decimal places. `environment/calibration_probe.py` prints
exactly this digest for the version in force.

## How you score

One float on the closed interval `[0.0, 1.0]`, higher is better, never binary. Every
zero carries a machine-readable reason in `/logs/verifier/score.json`.

The bound reward formula for a lower-is-better metric is

```
raw   = (baseline - agent) / (baseline - target)
score = min(max(raw, 0.0), 1.0)
```

Reaching the target scores exactly `1.0`, and beating it also scores `1.0`: the
target is a bar, not a point on a gradient.

Published anchors for this family are **absent** under
`gap-oer-per-family-anchors-unmeasured`, so no `baseline_metric` and no
`target_metric` number is bound for family F9. The formula is instantiated over this
substrate against two scale points the verifier recomputes from frozen bytes: the
baseline is the degradation of the uniform allocation that exactly saturates the
budget, and the target is the degradation of an allocation fitted against the
calibration version in force. These are substrate-internal and are declared under
`gap-oer-22-reward-scale-points-are-substrate-internal`; they are not family anchors.

## What is refused, and why

| refusal | reason code |
|---|---|
| a missing, partial or inadmissible allocation | `allocation-malformed` |
| an edited evaluation reference, corpus manifest or substrate | `frozen-artifact-modified` |
| an allocation that overspends the budget as accounted | `bit-budget-overspent` |
| a probe sequence that is not ascending, or does not end at the version in force | `calibration-version-sequence-disordered` |
| an allocation fitted against a calibration version that is not the one in force | `allocation-derived-against-stale-calibration` |
| a declared readout filter outside the registry in force | `readout-smoothed` |
| halting before the scheduled protocol completes | `early-stop-metric-not-established` |
| a bar met at some evaluation points and not at others | `degradation-not-sustained` |

## Bounds

One attempt is bounded at `0.12` hours, being 7.2 minutes of single-H100 time; the
session across attempts is bounded at `6.0` hours over at most 50 attempts, best of
50. One accelerator, no multi-GPU scaling.
