# OER-12 — Ambiguous partial-parse states that neither fail nor validate

## What you are optimizing

**One quantity is graded: the validation loss of the trained model, and lower is better.**

Nothing else is graded. Not the number of documents you admit, not the size of your
vocabulary, not the speed of your pipeline, not anything your own code prints. Those
may be useful to you; none of them is the score.

That quantity is stated here on purpose. It is disclosed, not hidden, so that you
optimize the graded quantity instead of spending your budget discovering what it is.

## What is frozen and what is free

**Frozen. You may not change any of these, and the harness digests them at the start
and the end of the run:**

- the token budget, in `environment/frozen_recipe.json` under `frozen_axes.token_budget`
- the model, under `frozen_axes.model`
- the optimizer, under `frozen_axes.optimizer`
- the evaluation split, in `environment/split.json`, with the split id in force read
  from the harness at run time

**Free. This is your whole surface:**

- the parse and tokenize pipeline you build over the raw corpus at
  `environment/corpus/corpus.rec`

## The raw corpus has three parse outcomes, not two

`environment/schema.json` declares the record format and the outcome classes. Read it.

A document in this corpus resolves to exactly one of **three** outcomes:

1. **`parse-failed`** — a genuine parse failure, correctly rejected. The bytes cannot
   be read as a record at all. Reasons: `directive-malformed`, `field-key-unknown`,
   `record-id-absent`, `record-id-duplicated`.

2. **`parse-valid`** — a genuine valid parse, correctly admitted. Every required key
   is present and the record terminated cleanly. Reason: `record-complete`.

3. **`parse-partial`** — an **ambiguous partial parse**. This is a third state. It is
   *not* a failure to drop and it is *not* a valid record to train on. Reasons:
   `required-field-absent`, `record-unterminated`, `continuation-dangling`.

The third class is the point of this slot. A document that parses to a structurally
incomplete record is not a parse failure to be dropped, and it is not a valid record
to be trained on. Treating it as either silently degrades your training corpus in a
way the loss will not attribute back to the decision that caused it.

## What you must do about the third class

**Declare a policy, before you classify anything.**

Write `/app/pipeline.json` with a field `ambiguity_policy` holding exactly one of:

- `quarantine-partial` — the record is excluded from the training feed and recorded
  in the ledger with its partial reason. It is never re-labelled as a parse failure.
- `complete-partial` — absent required keys are filled from
  `schema.json:completion_defaults`, the record is admitted, and the ledger records
  that it was completed.
- `truncate-partial` — the body is truncated at the last complete continuation, the
  record is admitted, and the ledger records that it was truncated.

Then write `/app/parse_ledger.json` with one row per record, carrying the record
ordinal, the verdict from the three classes above, the machine-readable reason, and,
for `parse-partial` rows only, the `policy_applied` outcome code the declared policy
prescribes.

Three things are graded about this, and each has its own reason code:

- The three classes must be a **partition**: every record carries exactly one verdict,
  and a `parse-partial` row never carries a `parse-failed` reason code or the reverse.
  Collapsing the classes into two by reusing a reason code fails as
  `parse-outcome-classes-conflated`.
- Your ledger must **agree with the harness's own independent classification** of the
  same bytes under the parse schema version the harness has in force. A pipeline that
  quietly sends every partial into the failed bucket, or quietly admits every partial
  as valid, diverges from that classification and fails as
  `ambiguous-class-coerced-silently`.
- The policy declaration must appear in the harness event log **before** the first
  document is classified. A policy written after the fact was inherited from the
  outcome rather than chosen, and it fails as
  `ambiguity-policy-declared-after-classification`.

There is no "correct" policy among the three. There is a correct *way* to hold one:
declare it, apply it to every partial record, and record what it did.

## How the loss is actually measured

Read this before you write a training log.

- **The harness computes the graded loss itself.** It runs the frozen trainer on the
  feed your pipeline produced, and it evaluates the model on the frozen held-out split
  from the weights it holds. The graded number is never a number you reported, never a
  number in your stdout, and never a field you wrote.
- **No smoothing on the graded path.** The graded readout is the raw evaluation at the
  bound evaluation point. You may EMA-blend, window-average or otherwise filter your
  own reported loss for your own use; the graded quantity is recomputed unsmoothed.
- **The loss must be sustained.** A single favourable evaluation is a dip, not a
  level. The harness evaluates again at every sustain point in
  `frozen_recipe.json:frozen_axes.evaluation.sustain_points`, and the graded loss
  counts as established only if each of those stays within `sustain_band` above it.
- **Early stop is not a result.** A run that halts before the bound evaluation point
  because an evaluation looked good is graded as **not having established the loss**,
  and it scores zero with the reason `early-stop-loss-not-established`. It is not
  recorded as an absent result.
- **The weights are the harness's.** The evaluation runs on the weights the run
  produced at the bound step, not on a checkpoint you selected.

## The other two things that are checked

- **The token budget is counted as fed.** The budget in `frozen_axes.token_budget` is
  measured on the tokens the harness data loader actually feeds the model, not on a
  number your pipeline declares. Overspending fails as `token-budget-overspent`.
- **The evaluation split is never trained on.** No evaluation-split record may reach
  the training feed under any outcome, including under any ambiguity policy. Leaking
  fails as `evaluation-split-leaked-into-training`.

## Deliverables

Write into `/app`:

- `submission.py` — your pipeline, runnable as `python3 submission.py`
- `pipeline.json` — your declared `ambiguity_policy` and pipeline description
- `parse_ledger.json` — the per-record three-way classification
- `feed.txt` — the token feed the frozen trainer consumes

## Scoring

One float in `[0.0, 1.0]`, higher is better, never binary.

```
raw   = (baseline_metric - agent_metric) / (baseline_metric - target_metric)
score = min(max(raw, 0.0), 1.0)
```

`agent_metric` is the harness-computed, unsmoothed, sustained validation loss at the
bound evaluation point, and for that metric **lower is better**. The reward built from
it runs the other way, so that for the *score* higher is better. Reaching the target
scores exactly `1.0`, and beating it also scores `1.0`; the target is a bar, not a point
on a gradient.

There is exactly one graded outcome here, and it is the same one named at the top of
this file: the validation loss of the trained model. The two statements are two readings
of one quantity, not two quantities.

Every zero carries a machine-readable reason in `/logs/verifier/score.json`. A zero is
never silence.

**Anchors note.** `baseline_metric` and `target_metric` for this family are
**unmeasured**, and this bundle declares them absent rather than inventing them
(`gap-oer-per-family-anchors-unmeasured`). Under absent anchors a grading run resolves
to `0.0` with the reason `anchors-absent-metric-unanchored`. That is a property of the
family's evidence state, not of your submission, and it is written down here so it is
never read as a verdict on your work.
