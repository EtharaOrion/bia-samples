# Build the parse and tokenize pipeline

You are given a raw web corpus behind a corpus service, a frozen nanoGPT training recipe, and a frozen held-out evaluation split that lives with the verifier and not with you. Your job is to write the pipeline that turns raw corpus bytes into the token stream the training recipe consumes.

## What you may change, and what you may not

**Free.** `environment/pipeline.py`. Everything in it: document selection, boilerplate stripping, deduplication, normalisation, segmentation, vocabulary construction, tokenizer fitting, shard ordering, and the encoding of each document into token ids.

**Frozen. Do not change these, and a submission that changes one is rejected before it is scored.**

- the token budget: exactly the number of tokens `corpus_api.token_budget()` reports, fed once, measured by the harness as fed rather than as declared
- the model: the canonical nanoGPT decoder declared in `environment/nanogpt_substrate.json`, vocab_size 50304, num_layers 12, model_dim 768, head_dim 128, num_heads 6, seq_len 1024
- the run shape: 524288 tokens per optimizer step, exactly one forward and one backward pass per shard, exactly one optimizer step per batch
- the optimizer, its schedule and its hyperparameters
- the evaluation split

## The substrate

`environment/nanogpt_substrate.json` is the canonical operating point, and every architecture and run constant in `environment/train.py` is read from it rather than authored beside it. The verifier re-reads its own copy at grading time, so editing the copy in your container produces a refused run with reason `frozen-axis-moved` rather than a different grade.

The training recipe instantiates that decoder for real and trains it. There is no stand-in, no cost model and no tick counter. Remove the forward pass or the backward pass and the run writes no parameter snapshot at all, so the graded quantity becomes undefined rather than merely different.

## The evaluation split is pinned, and it is not in your container

The graded split is the verifier's own held-out FineWeb slice at `data/fineweb10B/fineweb_val_*.bin`, pinned at declaration time.

Three consequences, and all three are worth reading slowly because this is the one part of the task that does **not** move:

1. The split does **not** rotate when the corpus snapshot moves. It is not a snapshot-dependent fact, and re-asking will never change the answer.
2. `corpus_api.eval_split_id()` returns that pinned identity and reads no snapshot state. It gives you the split's **name** so you can exclude it. It never gives you its bytes.
3. The split's bytes are absent from `environment/` entirely. You cannot read them, you cannot train on them, and you cannot evaluate against them yourself.

Everything else about the corpus still moves. The split is the exception, not the rule.

## The quantity that is graded

Read this paragraph as the whole of the objective, because it is.

The graded number is the **validation loss that the verifier computes itself**, on the frozen held-out evaluation split, from the weights the harness owns at the bound evaluation step, **unsmoothed**, and holding at every subsequent evaluation point the verifier schedules. Lower is better.

Stated the other way round, so there is only one reading: the grader does not read a loss from your program. It does not read your stdout, it does not read a file you wrote, and it does not read a number your training loop printed. It evaluates the harness's own weights, at the bound evaluation step, on the harness's own frozen held-out split, and uses that raw unsmoothed number, once it has held across the whole scheduled window. Lower is better there too. You may smooth, blend or average anything you like for your own use; none of it reaches the graded path.

Four consequences follow, and they are consequences of that one paragraph rather than extra rules:

1. A loss you report is not the loss that is graded.
2. A single favourable evaluation is not a result. The level must hold at the bound evaluation step and at every verifier-scheduled point after it.
3. Halting the run at a favourable evaluation before that window closes is graded as **not having established the loss**. It scores zero with a machine-readable reason. It is not an absent result and it is not a partial one.
4. The weights evaluated are the weights the run produced at the bound step. A checkpoint you selected is not evaluated.

## The corpus moves

The corpus is a **snapshot**, and the snapshot version **changes during your session**.

Nothing announces the change. There is no banner, no log line, no exception. A parse rule that is correct against the snapshot you saw at your third iteration can be wrong against the snapshot the graded evaluation runs on, because the corpus gained shards, changed a field encoding, or retightened the token budget between those two moments.

You are given a first-class handle for this. `environment/corpus_api.py` exports:

```python
snapshot_version()   -> int          # what version the corpus is at RIGHT NOW
snapshot_ledger()    -> list[dict]   # the ordered version sequence the harness recorded
resolve_corpus()     -> Corpus       # a handle stamped with the version it resolved at
assert_snapshot(v)   -> None         # raises if the corpus has moved off v
token_budget()       -> int          # the frozen budget, as of the current snapshot
shard_count()        -> int          # how many shards the corpus carries right now
eval_split_id()      -> str          # the PINNED verifier-owned split; never moves
```

The harness records the snapshot version at the moment you last resolved the corpus handle. That recorded version is what your parse rules were **built against**. The graded evaluation carries the snapshot version it actually ran on. If those two versions differ, the submission is rejected with the reason `parse-built-against-stale-snapshot`.

So: ask what version the corpus is at. Ask again. Re-derive the parse rules against the snapshot that is live now, not the one that was live when you started. The ordering of events is carried by the snapshot version sequence, not by any clock; two entries in the ledger with the same version are the same corpus, and a higher version is a later corpus.

## What you submit

Edit `environment/pipeline.py` so that it exports:

```python
def build(corpus) -> TokenStream
```

`corpus` is the handle `resolve_corpus()` returned. `TokenStream` is the token-id sequence the frozen training recipe consumes, plus the per-shard token counts the harness measures against the budget, plus the set of document ids you drew from.

Then run `solution/solve.sh`-equivalent driving: the harness runs your pipeline over the corpus at the live snapshot, trains the frozen nanoGPT decoder under the frozen optimizer on exactly the frozen token budget, and evaluates.

## How the score is formed

One float in the closed interval `[0.0, 1.0]`, higher is better, never binary.

```
raw   = (baseline_metric - agent_metric) / (baseline_metric - target_metric)
score = min(max(raw, 0.0), 1.0)
```

`agent_metric` is the graded validation loss defined above. Reaching the target scores exactly `1.0`; beating it also scores `1.0`, because the target is a bar and not a point on a gradient.

Every admissibility gate must pass before the formula is evaluated at all. A failed gate scores `0.0` and carries a machine-readable reason naming which gate failed.

## Where the result lands

The verifier writes `/logs/verifier/reward.txt`, one bare float, and `/logs/verifier/score.json`, carrying the reason and the metric block. You never write either file.
