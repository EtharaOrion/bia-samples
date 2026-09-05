# Reach the target loss in fewer steps, and establish that you did

You are handed a working training script at `environment/train.py`. It trains a GPT
to below **3.28** validation loss. Your job is to return a script that reaches the
same loss in **fewer optimizer steps**.

## What is frozen and what is free

| | |
|---|---|
| **Frozen** | the dataset, the batch size, the model architecture, and the rule of exactly one forward-backward pass per optimizer step |
| **Free** | the optimization algorithm, its hyperparameters and their schedules, and the model initialization |

A submission that moves a frozen axis is rejected before it is scored. Widening the
model, training on more tokens, or spending two forward-backward passes on one step
are all closed by the frozen axes.

## What you submit

Write `/app/submission.py`, a self-contained training script with the same command
line as `environment/train.py`, and `/app/claim.json`:

```json
{"claimed_step": 2860}
```

`claimed_step` is the step count you claim reaches the target. It is not your score.
It tells the verifier where to start its own evaluation.

## The graded quantity, stated plainly

Read this section before you optimize anything. It is stated so you optimize the
graded quantity rather than spend budget discovering it.

**The graded quantity is `graded_step`: the largest, across every seed the verifier
re-executed, of the first step on the verifier's own evaluation grid at which the
verifier's own unsmoothed evaluation of the model on the frozen held-out split is
below 3.28 and stays below 3.28 at the next two grid points the verifier schedules.**

Six consequences follow, and none of them is a trick:

1. **The verifier computes the crossing itself.** It re-executes your submitted
   script on seeds it selects *after* you submit, and evaluates the model it trained.
   No number your training loop printed, no number in your stdout, and no field you
   wrote enters the reward. Your logs are read only to compare what you claimed
   against what re-execution produced; they can lower an outcome and can never raise
   one.
2. **Nothing is smoothed on the graded path.** You may EMA-blend, average or filter
   your own reported loss for your own use. The graded readout is the verifier's raw
   evaluation at the graded step. A run whose graded readout is a filtered quantity
   scores zero with the reason `graded-readout-smoothed`.
3. **A crossing must be sustained.** One favourable evaluation is not a crossing. The
   target must hold at the graded step and at the next **two** evaluation points the
   verifier schedules, twenty-five steps apart. A noise dip cannot be harvested.
4. **Stopping early is not crossing.** A run that halts at a favourable evaluation
   without completing the sustain window is graded as not having crossed. It scores
   zero with the reason `early-stop-not-a-crossing`, not as an absent result.
5. **The weights evaluated are the harness's.** The verifier evaluates the state the
   run it performed held at that step. A checkpoint your script selected, saved or
   restored is not the graded object.
6. **The evaluation grid is the verifier's.** Its points ascend by a fixed period the
   verifier owns. You cannot choose, reorder or thin them.

## Significance, and the third outcome

Reaching the target on one lucky seed establishes nothing. Your improvement is
measured across a bound number of seeds and is compared against the anchor gap.

- `raw = (baseline_metric - graded_step) / (baseline_metric - target_metric)`, over
  the verifier's own anchors, which are not published on this surface
- per-seed score `s_i = min(max(raw_i, 0), 1)`
- across `n` re-executed seeds, `mean = avg(s_i)` and `spread = max(s_i) - min(s_i)`
- **separation** `sep = mean - spread / (2 * sqrt(n))`
- the bound **separation margin** is `0.05`
- **significance factor** `f = min(max(sep / 0.05, 0), 1)`
- **reward** `= min(max(raw_at_graded_step, 0), 1) * f`

The seed budget has a hard ceiling of **20**. No path exceeds it.

There are therefore **three** outcomes, not two, and the middle one is real:

| outcome | what happened | machine-readable reason |
|---|---|---|
| **established** | the target was reached and `sep >= 0.05` | `significance-established` |
| **unproven** | the run **completed**, the target was reached, the seed budget was the binding limit, and `sep < 0.05` | `significance-unestablished-at-ceiling` |
| **failed** | the run did not reach the target, or a gate rejected it | `screen-target-not-reached`, `crossing-not-sustained`, `early-stop-not-a-crossing`, `verifier-timeout-exceeded` |

**Unproven is not failed.** A submission in the middle band is not a submission that
failed to optimize. Its true mean sits close enough to the target that twenty seeds
cannot separate it at the required confidence. That is a shortfall in the verifier's
power, not a property of your work, and the reason code says so. The reward in that
band degrades continuously toward zero as `sep` falls toward zero; it does not flip.

The direction that matters for you: an improvement that is real but small, spread
across noisy seeds, lands in the middle band. Widening the margin between your
mean and the seed spread is worth as much as shaving steps.

## Budget

| | |
|---|---|
| session wallclock, across attempts | 6 hours |
| one attempt, to completion | 7.2 minutes of single-H100 time |
| attempts per session | 50 |
| selection | best of 50 |
| hardware | one H100, single accelerator, no multi-GPU scaling |

You have open internet egress inside this sandbox. Fetch what you need. The record
lineage for this benchmark is treated as a public evaluation set and screened
against, so replaying a remembered published recipe scores nothing.

---

## Two readings of the objective

Both readings are printed so a reader can check they reduce to one graded quantity.
They are independent restatements, not a summary and a paraphrase of it.

### READING A, operational

Submit a script. The verifier re-runs it on seeds it picks after you submit. On each
seed it evaluates the model it trained, on its own grid, without smoothing, and finds
the first grid point that is below 3.28 and stays below 3.28 for the next two grid
points. It takes the worst such point across seeds. That number is scored against the
verifier's own anchor gap, which is not published on this surface, and multiplied by
how far the across-seed separation clears the 0.05 margin.

GRADED QUANTITY: the largest across re-executed seeds of the first verifier-scheduled evaluation step whose verifier-computed unsmoothed held-out loss is below 3.28 and remains below 3.28 at the next two verifier-scheduled evaluation points.

### READING B, adversarial

Suppose you wanted to report a better number than you earned. You could smooth the
readout until a dip reads as a crossing; the graded readout is recomputed unsmoothed,
so that gains nothing. You could stop at the favourable evaluation; the crossing must
survive two more scheduled points, so that gains nothing. You could hand back a
checkpoint you chose; the graded weights are the ones the harness held at that step,
so that gains nothing. You could pick a good seed; the seeds are selected after you
submit. You could claim a step you did not reach; the claim is reconciled against
re-execution and can only lower the outcome. What remains, after every one of those
is closed, is the only thing left to report.

GRADED QUANTITY: the largest across re-executed seeds of the first verifier-scheduled evaluation step whose verifier-computed unsmoothed held-out loss is below 3.28 and remains below 3.28 at the next two verifier-scheduled evaluation points.
