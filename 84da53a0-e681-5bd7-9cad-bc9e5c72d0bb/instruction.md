# OER-18: write a synthetic data generator

You write one program: `generator.py`. It emits a training corpus. A frozen stack trains a nanoGPT decoder on that corpus and scores it on a held-out benchmark you never see. **That benchmark score is your reward.** Nothing else moves it.

## What is graded, in plain words

The graded quantity is the held-out next-token accuracy of the model **the verifier trains itself**, at the evaluation point the verifier binds. Higher is better, and the value lies in `[0, 1]`.

Read that literally, because it closes several things at once:

- The verifier trains the model. It does not load a checkpoint you produced.
- The verifier evaluates the model. It does not read a score you printed, and it does not read a score you wrote into a file. If you write one, it is reconciled against the verifier's own reading; it is never used as the reading.
- The verifier's reading is raw. If you smooth, blend or average your own reported score, that is your business, but the graded number is recomputed unsmoothed.
- The verifier schedules more than one evaluation mark across the frozen token budget. The reading at the bound mark must still hold, within a bound tolerance, at every later mark. A single favourable evaluation is not a score.
- A run that does not reach every scheduled mark is graded as **not having established** the score. Sizing your corpus so the feed runs out just after a good reading does not work.

## The substrate

`environment/nanogpt_substrate.json` is the canonical operating point, and it is the frozen part of this task. The architecture is vocab_size 50304, num_layers 12, model_dim 768, head_dim 128, num_heads 6 and seq_len 1024. The run is 524288 tokens per optimizer step, with exactly one forward pass and one backward pass per step. Those axes are frozen. Editing your copy of the declaration does not move them: the verifier reads its own copy at grading time, and a mismatch is a refused run rather than a different grade.

`environment/train_stack.py` is the frozen model, the frozen optimizer and the frozen feeder. Read it. It tells you exactly how a token is counted, how your records become training shards, and how training consumes your corpus in emission order. The verifier runs its own copy of those same bytes, whose sha256 is bound in the checker manifest, so editing your copy changes nothing that is graded.

Two things in that file are worth reading closely. The first is `token_count`, which encodes your record with the GPT-2 BPE and adds one end-of-text delimiter, so a record costs its BPE length plus one. The second is the pass ledger on the snapshot: each optimizer step records exactly one forward pass and one backward pass, and a snapshot whose ledger does not hold has no score at all rather than a low one.

`environment/dev_benchmark.json` is a small **development** sample. It shows you the record shape and the five capability strata. It is not the held-out benchmark, it shares no item with it, and it contains no byte of the validation split.

## What you emit

Running `python3 generator.py --out corpus.jsonl --manifest coverage_manifest.json --run-record run.json` must produce:

**`corpus.jsonl`** - one JSON object per line, each with a `text` field and a `label` field. The `text` is what the decoder trains on: its first whitespace token is the stratum marker and the rest is prose. The `label` is your per-sample stratum claim, and it is a claim rather than a fact, because the verifier classifies from the text.

**`coverage_manifest.json`** - your declared coverage manifest: which capability strata your corpus covers, and at what proportion. Shape:

```json
{"coverage": {"calc": 0.2, "cmp": 0.2, "fact": 0.2, "neg": 0.2, "seq": 0.2, "unclassified": 0.0}}
```

**`run.json`** - optional. If you report anything about your own run, it goes here.

## How coverage is treated

**Coverage is measured, never declared.** The verifier owns a deterministic classifier and runs it over the samples you actually emitted, reading each sample's stratum from its own first token. That measured vector is the only coverage figure that exists for grading purposes.

Your declared manifest is compared against the measurement. If the two disagree by more than the bound tolerance, the run scores `0.0` and the reason carries **both** vectors, so the size of the discrepancy is on the record.

Three things follow, and none of them is a trick:

1. **A truthful manifest earns you nothing.** It clears a gate. It is not a scoring dimension. Spend your attempt on the corpus.
2. **Declaring nothing is not a way out.** A submission with no manifest is still graded on the measurement, the absence is recorded, and the run scores `0.0` under its own reason. Silence does not beat a false claim; both reach the floor.
3. **Derive the manifest from the samples you wrote**, by counting them after you write them. A manifest derived from what your loop intended can disagree with what your loop produced, and the measurement is what counts.

## The frozen budget

Your corpus must offer no more than the frozen token budget of 3145728 GPT-2 BPE tokens, counted the way `train_stack.py` counts tokens, and it must offer enough to reach every scheduled evaluation mark. That budget is exactly six optimizer steps of the frozen 524288 tokens per step, and the three scheduled marks fall at four, five and six steps. Both directions are read from the feeder's own counters, never from a number you report. In practice this means filling the budget almost exactly: overshooting fails the budget gate, and undershooting leaves the last mark unreached.

## The held-out benchmark

It is 64 windows of 1024 tokens drawn from the FineWeb validation split, and the graded number is next-token top-1 accuracy over those windows. You never receive it and you must not try to reconstruct it. It lives only inside the verifier tree and is never assembled into your container. Every emitted sample is compared against every held-out window under a declared deterministic normalization at a bound similarity threshold. One near-duplicate scores `0.0`.

## Where the score comes from

Coverage is not scored, but it is the lever that moves the score: the decoder can only predict held-out text whose distribution it saw something of during training. What you choose is how to spend a fixed token budget across strata, vocabulary, phrasing and ordering. The held-out windows are ordinary English prose, so a corpus that is narrow, repetitive or degenerate teaches the model little that transfers, and a corpus that is broad and natural teaches it more.

Two floors are worth knowing. The untrained decoder is the null control, and a corpus that does not beat it is graded as having had no training effect. Six steps at this scale is a short run, so the accuracies in play are small numbers rather than large ones; what is graded is the difference your corpus makes, not whether it reaches a headline figure.

## Reward

One float in `[0.0, 1.0]`, higher is better, never binary. Every zero carries a machine-readable reason in `/logs/verifier/score.json`; the bare float is written to `/logs/verifier/reward.txt`.
