# Long-horizon vocabulary search under a fixed compute budget

You are given a frozen language-modelling substrate in `environment/substrate.py` and a
frozen corpus in `environment/corpus/`. Your job is to choose a **vocabulary construction**
that makes the frozen model code the frozen evaluation corpus in as few bits per byte as
possible, at a **fixed compute budget**.

## What is graded

The graded quantity is

```
bits per byte  =  (bits the frozen model needs to code environment/corpus/eval.txt)
                  /  (the byte length of environment/corpus/eval.txt)
```

Read this literally, because every part of it is load-bearing.

- **The verifier computes it.** It is recomputed inside the verifier's own process from the
  model state the harness owns at the evaluation points the verifier schedules. It is never
  a number your run reported, never a number in your stdout, and never a field you wrote.
- **The denominator is fixed.** It is the byte length of `environment/corpus/eval.txt`, and
  nothing else. A construction that normalizes, strips or drops evaluation bytes and divides
  by what survives has changed what it measures over, and is graded as not having established
  a reading.
- **No smoothing on the graded path.** You may filter, average or EMA-blend anything you like
  for your own use. The graded reading is the harness's raw evaluation. Asking for a filtered
  graded readout is refused with a reason, not silently ignored.
- **The reading must be sustained.** The verifier schedules three evaluation points near the
  end of the budget and grades the **worst** of them. A favourable fluctuation at one point is
  not a level.
- **Stopping early is not a result.** A run that halts at a favourable evaluation point,
  before the scheduled points have been reached, is graded as *not having established the
  metric*, with a machine-readable reason. It is not graded as an absent result.
- **The compute budget is counted, not asserted.** Compute spend is read off a counter the
  harness owns. A training loop that asserts it stayed in budget is asserting something about
  itself.

Lower bits per byte is better.

## What is frozen and what is free

Frozen, and not yours to change:

- the compute budget, `COMPUTE_BUDGET_UNITS` training units, one unit per token position
- the model, a back-off bigram over token ids with frozen smoothing constants
- the optimizer, which here is the frozen count-accumulation training procedure
- the evaluation corpus, `environment/corpus/eval.txt`

Free, and the whole of your search space:

- the vocabulary construction: which byte-strings become tokens
- the size allocation across the four frozen construction families, `merge_depth`,
  `span_units`, `numeric_units` and `punct_units`
- any extra tokens you name directly

The family slot budget is `SLOT_BUDGET` and it is **zero-sum**: families are filled in the
fixed order above, so a slot spent on one family is a slot no other family gets.

The tension is arithmetic and nobody authored a curve for it. A vocabulary that codes text in
fewer tokens covers more training bytes under the same unit budget. A vocabulary large enough
to thin the bigram table loses more to estimation than it gains in compression. And a
construction family can run out of material: nothing in the substrate announces that.

## What you submit

One JSON document at the path in `task.toml` under `[agent] submission_path`:

```json
{
  "allocations": {"merge_depth": 0, "span_units": 0, "numeric_units": 0, "punct_units": 0},
  "extra_tokens": [],
  "attempts": [
    {"index": 1, "direction": "merge_depth", "allocation": {"merge_depth": 64},
     "reallocated_from": null, "carried_frontier": {"merge_depth": 2.0864}}
  ],
  "graded_attempt_index": 1
}
```

`allocations` is the construction that gets graded. `attempts` is the record of the search
that produced it: one row per attempt, in order, each naming the direction it moved, the
allocation it tried, the direction it reallocated **off** if any, and the per-direction
frontier the session was holding at that point. The harness **re-measures every allocation
you record**, so a fabricated frontier does not survive; what the record establishes is the
order in which you moved and the state you carried, not any number you claim.

## The session

- up to **50 attempts**
- **0.12 hours** of wall clock for any one attempt
- **6.0 hours** across the whole session
- one H100, single accelerator, no multi-GPU scaling
- the graded submission is the **best** attempt you carry

## Machine-readable refusals

Every zero score carries a `reason` in `/logs/verifier/score.json` alongside the bare float in
`/logs/verifier/reward.txt`. If you are refused, the reason names what was refused:

| reason | what it means |
|---|---|
| `denominator-not-frozen-eval-bytes` | the reading was divided by something other than the frozen evaluation byte count |
| `compute-budget-overspent` | the harness counter measured more spend than the frozen budget, or more was requested |
| `readout-smoothing-on-graded-path` | a filtered graded readout was applied or requested |
| `early-stop-not-established` | the run halted before the scheduled evaluation points were reached |
| `reading-not-sustained` | the reading did not hold across the scheduled evaluation points |
| `state-not-harness-owned` | the evaluated state was not the state the harness owns at that point |
| `submission-reported-metric-on-graded-path` | a number the submission reported was offered to, or consumed by, the graded path |
| `flattened-direction-not-reallocated` | a direction stopped paying and the allocation never moved off it |
| `carried-state-collapsed` | the carried per-direction frontier was reset or truncated between attempts |
| `single-direction-sweep-not-exceeded` | the result does not beat the harness's own single-direction sweep by the bound margin |
