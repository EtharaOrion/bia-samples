# OER-12: Ambiguous partial-parse states that neither fail nor validate

## What you are optimizing

**One quantity is graded: the validation loss of the trained model, and lower is better.**

Nothing else is graded. Not the number of documents you admit, not the size of your vocabulary, not the speed of your pipeline, not anything your own code prints. Those may be useful to you; none of them is the score.

That quantity is stated here on purpose. It is disclosed, not hidden, so that you optimize the graded quantity instead of spending your budget discovering what it is.

## The model your feed trains

The model is the canonical nanoGPT decoder from the modded-nanogpt record set: a twelve-layer, 768-dimension, six-head causal transformer over a 1024-token context and a 50304-entry GPT-2 BPE vocabulary. Its operating point is declared in `environment/nanogpt_substrate.json`, the frozen axes that bind this slot are in `environment/frozen_recipe.json`, and the trainer that instantiates and trains it is `environment/train_frozen.py`. Read all three. The trainer refuses to run when the recipe and the substrate disagree on any architecture, batch or corpus value, so those numbers are not adjustable by editing a file.

The vocabulary and the encoder are frozen with the model. Your pipeline does not author a vocabulary and does not cap one: the embedding matrix and the output projection are shape-bound to the frozen vocabulary size, and the held-out split the verifier grades on is encoded in that same vocabulary. What your pipeline decides is which documents reach the feed, in what order, and in what form.

## What is frozen and what is free

**Frozen. You may not change any of these, and the harness digests them at the start and the end of the run:**

- the model, in `environment/frozen_recipe.json` under `frozen_axes.model`, pinned to `environment/nanogpt_substrate.json`
- the optimizer, under `frozen_axes.optimizer`
- the token budget, under `frozen_axes.token_budget`, counted as fed
- the evaluation split, which is the verifier's own copy of the FineWeb10B validation shards. It is absent from `environment/` and from this container, its ownership is declared in `environment/split.json`, and the split id in force is read from the harness at run time.

**Free. This is your whole surface:**

- the parse and tokenize pipeline you build over the raw corpus, whose record format is specified by `environment/schema.json` and whose specimen is `environment/corpus/corpus.rec`

## The raw corpus has three parse outcomes, not two

`environment/schema.json` declares the record format and the outcome classes. Read it.

A document in this corpus resolves to exactly one of **three** outcomes:

1. **`parse-failed`**, a genuine parse failure, correctly rejected. The bytes cannot be read as a record at all. Reasons: `directive-malformed`, `field-key-unknown`, `record-id-absent`, `record-id-duplicated`.

2. **`parse-valid`**, a genuine valid parse, correctly admitted. Every required key is present and the record terminated cleanly. Reason: `record-complete`.

3. **`parse-partial`**, an **ambiguous partial parse**. This is a third state. It is *not* a failure to drop and it is *not* a valid record to train on. Reasons: `required-field-absent`, `record-unterminated`, `continuation-dangling`.

The third class is the point of this slot. A document that parses to a structurally incomplete record is not a parse failure to be dropped, and it is not a valid record to be trained on. Treating it as either silently degrades your training corpus in a way the loss will not attribute back to the decision that caused it.

## What you must do about the third class

**Declare a policy, before you classify anything.**

Write `/app/pipeline.json` with a field `ambiguity_policy` holding exactly one of:

- `quarantine-partial`, the record is excluded from the training feed and recorded in the ledger with its partial reason. It is never re-labelled as a parse failure.
- `complete-partial`, absent required keys are filled from `schema.json:completion_defaults`, the record is admitted, and the ledger records that it was completed.
- `truncate-partial`, the body is truncated at the last complete continuation, the record is admitted, and the ledger records that it was truncated.

Then write `/app/parse_ledger.json` with one row per record, carrying the record ordinal, the verdict from the three classes above, the machine-readable reason, and, for `parse-partial` rows only, the `policy_applied` outcome code the declared policy prescribes.

Three things are graded about this, and each has its own reason code:

- The three classes must be a **partition**: every record carries exactly one verdict, and a `parse-partial` row never carries a `parse-failed` reason code or the reverse. Collapsing the classes into two by reusing a reason code fails as `parse-outcome-classes-conflated`.
- Your ledger must **agree with the harness's own independent classification** of the same bytes under the parse schema version the harness has in force. A pipeline that quietly sends every partial into the failed bucket, or quietly admits every partial as valid, diverges from that classification and fails as `ambiguous-class-coerced-silently`.
- The policy declaration must appear in the harness event log **before** the first document is classified. A policy written after the fact was inherited from the outcome rather than chosen, and it fails as `ambiguity-policy-declared-after-classification`.

There is no "correct" policy among the three. There is a correct *way* to hold one: declare it, apply it to every partial record, and record what it did.

## How the loss is actually measured

Read this before you write a training log.

- **The harness trains the model and the verifier grades it.** The harness runs the frozen trainer on the feed your pipeline produced. The trainer encodes that feed with the frozen encoder, trains the frozen decoder with one forward and one backward pass per step, and writes a real parameter snapshot at the bound evaluation point and at every sustain point. The verifier then loads those snapshots and evaluates them itself on its own held-out split. The graded number is never a number you reported, never a number in your stdout, and never a field you wrote.
- **The held-out split is not in this container.** You cannot read it, fit it, or resolve it from any call you can reach. The trainer itself refuses to start if a validation shard is reachable from the container it runs in.
- **No smoothing on the graded path.** The graded readout is the raw evaluation at the bound evaluation point. You may EMA-blend, window-average or otherwise filter your own reported loss for your own use; the graded quantity is recomputed unsmoothed.
- **The loss must be sustained.** A single favourable evaluation is a dip, not a level. The harness evaluates again at every sustain point in `frozen_recipe.json:frozen_axes.evaluation.sustain_points`, and the graded loss counts as established only if each of those stays within `sustain_band` above it.
- **Early stop is not a result.** A run that halts before the bound evaluation point because an evaluation looked good is graded as **not having established the loss**, and it scores zero with the reason `early-stop-loss-not-established`. It is not recorded as an absent result. A feed that runs out of tokens before the schedule ends halts the run the same way, so a pipeline that admits too little never establishes a level at all.
- **The weights are the harness's.** The evaluation runs on the snapshot the run produced at the bound step, not on a checkpoint you selected.

## The other two things that are checked

- **The token budget is counted as fed.** The budget in `frozen_axes.token_budget` is measured on the tokens the harness data loader actually feeds the model, not on a number your pipeline declares. Overspending fails as `token-budget-overspent`.
- **The evaluation split is never trained on.** No evaluation-split record may reach the training feed under any outcome, including under any ambiguity policy. The records withheld from the feed are declared in `environment/split.json`, and they are a leakage control rather than the graded split. Leaking fails as `evaluation-split-leaked-into-training`.

## Deliverables

Write into `/app`:

- `submission.py`, your pipeline, runnable as `python3 submission.py`
- `pipeline.json`, your declared `ambiguity_policy` and pipeline description
- `parse_ledger.json`, the per-record three-way classification
- `feed.txt`, the admitted document text the frozen trainer encodes and feeds, in the order you admit it

## Scoring

One float in `[0.0, 1.0]`, higher is better, never binary.

```
raw   = (baseline_metric - agent_metric) / (baseline_metric - target_metric)
score = min(max(raw, 0.0), 1.0)
```

`agent_metric` is the verifier-computed, unsmoothed, sustained validation loss at the bound evaluation point, and for that metric **lower is better**. The reward built from it runs the other way, so that for the *score* higher is better. Reaching the target scores exactly `1.0`, and beating it also scores `1.0`; the target is a bar, not a point on a gradient.

There is exactly one graded outcome here, and it is the same one named at the top of this file: the validation loss of the trained model. The two statements are two readings of one quantity, not two quantities.

Every zero carries a machine-readable reason in `/logs/verifier/score.json`. A zero is never silence.

**Anchors note.** `baseline_metric` and `target_metric` for this family are **unmeasured**, and this bundle declares them absent rather than inventing them (`gap-oer-per-family-anchors-unmeasured`). Under absent anchors a grading run resolves to `0.0` with the reason `anchors-absent-metric-unanchored`. That is a property of the family's evidence state, not of your submission, and it is written down here so it is never read as a verdict on your work. The 3.28 validation loss the substrate names is the upstream benchmark definition at the record step counts, not a bar this slot claims its own schedule reaches.
