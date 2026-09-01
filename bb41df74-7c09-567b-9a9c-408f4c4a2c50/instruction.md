# OER-18: write a synthetic data generator

You write one program: `generator.py`. It emits a training corpus. A frozen stack trains a
small model on that corpus and scores it on a held-out benchmark you never see.
**That benchmark score is your reward.** Nothing else moves it.

## What is graded, in plain words

The graded quantity is the accuracy of the model **the verifier trains itself**, on the
frozen held-out benchmark, at the evaluation point the verifier binds. Higher is better.

Read that literally, because it closes several things at once:

- The verifier trains the model. It does not load a checkpoint you produced.
- The verifier evaluates the model. It does not read a score you printed, and it does not
  read a score you wrote into a file. If you write one, it is reconciled against the
  verifier's own reading; it is never used as the reading.
- The verifier's reading is raw. If you smooth, blend or average your own reported score,
  that is your business, but the graded number is recomputed unsmoothed.
- The verifier schedules more than one evaluation mark across the frozen token budget. The
  reading at the bound mark must still hold, within a bound tolerance, at every later mark.
  A single favourable evaluation is not a score.
- A run that does not reach every scheduled mark is graded as **not having established** the
  score. Sizing your corpus so the feed runs out just after a good reading does not work.

## The substrate

`environment/train_stack.py` is the frozen model, the frozen optimizer and the frozen
feeder. Read it. It tells you exactly how a token is counted, how a sample becomes features,
and how training consumes your corpus in emission order. The verifier runs its own copy of
those same bytes, so editing your copy changes nothing that is graded.

`environment/dev_benchmark.json` is a small **development** split. It shows you the sample
grammar and the five capability strata. It is not the held-out benchmark, and it shares no
item with it.

## What you emit

Running `python3 generator.py --out corpus.jsonl --manifest coverage_manifest.json
--run-record run.json` must produce:

**`corpus.jsonl`** — one JSON object per line, each with a `text` field and a `label` field.
This is your training corpus.

**`coverage_manifest.json`** — your declared coverage manifest: which capability strata your
corpus covers, and at what proportion. Shape:

```json
{"coverage": {"calc": 0.2, "cmp": 0.2, "fact": 0.2, "neg": 0.2, "seq": 0.2, "unclassified": 0.0}}
```

**`run.json`** — optional. If you report anything about your own run, it goes here.

## How coverage is treated

**Coverage is measured, never declared.** The verifier owns a deterministic classifier and
runs it over the samples you actually emitted. That measured vector is the only coverage
figure that exists for grading purposes.

Your declared manifest is compared against the measurement. If the two disagree by more than
the bound tolerance, the run scores `0.0` and the reason carries **both** vectors, so the
size of the discrepancy is on the record.

Three things follow, and none of them is a trick:

1. **A truthful manifest earns you nothing.** It clears a gate. It is not a scoring
   dimension. Spend your attempt on the corpus.
2. **Declaring nothing is not a way out.** A submission with no manifest is still graded on
   the measurement, the absence is recorded, and the run scores `0.0` under its own reason.
   Silence does not beat a false claim; both reach the floor.
3. **Derive the manifest from the samples you wrote**, by counting them after you write
   them. A manifest derived from what your loop intended can disagree with what your loop
   produced, and the measurement is what counts.

## The frozen budget

Your corpus must offer no more than the frozen token budget, counted the way
`train_stack.py` counts tokens, and it must offer enough to reach every scheduled evaluation
mark. Both directions are read from the feeder's own counters, never from a number you
report.

## The held-out benchmark

You never receive it and you must not try to reconstruct it. Every emitted sample is
compared against every held-out item under a declared deterministic normalization at a bound
similarity threshold. One near-duplicate scores `0.0`.

## Where the score comes from

Coverage is not scored, but it is the lever that moves the score: the model can only answer
a stratum's benchmark items if it saw that stratum's vocabulary during training. What you
choose is how to spend a fixed token budget across strata, words and orderings.

## Reward

One float in `[0.0, 1.0]`, higher is better, never binary. Every zero carries a
machine-readable reason in `/logs/verifier/score.json`; the bare float is written to
`/logs/verifier/reward.txt`.
