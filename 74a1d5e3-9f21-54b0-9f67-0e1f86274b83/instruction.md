# Build the parse and tokenize pipeline

You are given a raw web corpus behind a corpus service, a frozen training recipe, and a
frozen evaluation split. Your job is to write the pipeline that turns raw corpus bytes
into the token stream the training recipe consumes.

## What you may change, and what you may not

**Free.** `environment/pipeline.py`. Everything in it: document selection, boilerplate
stripping, deduplication, normalisation, segmentation, vocabulary construction,
tokenizer fitting, shard ordering, and the encoding of each document into token ids.

**Frozen. Do not change these, and a submission that changes one is rejected before it
is scored.**

- the token budget: exactly the number of tokens `corpus_api.token_budget()` reports,
  fed once, measured by the harness as fed rather than as declared
- the model
- the optimizer, its schedule and its hyperparameters
- the evaluation split

## The quantity that is graded

Read this paragraph as the whole of the objective, because it is.

The graded number is the **validation loss that the verifier computes itself**, on the
frozen held-out evaluation split, from the weights the harness owns at the bound
evaluation step, **unsmoothed**, and holding at every subsequent evaluation point the
verifier schedules. Lower is better.

Stated the other way round, so there is only one reading: the grader does not read a
loss from your program. It does not read your stdout, it does not read a file you
wrote, and it does not read a number your training loop printed. It evaluates the
harness's own weights, at the bound evaluation step, on the harness's own frozen
held-out split, and uses that raw unsmoothed number, once it has held across the whole
scheduled window. Lower is better there too. You may smooth, blend or average anything
you like for your own use; none of it reaches the graded path.

Four consequences follow, and they are consequences of that one paragraph rather than
extra rules:

1. A loss you report is not the loss that is graded.
2. A single favourable evaluation is not a result. The level must hold at the bound
   evaluation step and at every verifier-scheduled point after it.
3. Halting the run at a favourable evaluation before that window closes is graded as
   **not having established the loss**. It scores zero with a machine-readable reason.
   It is not an absent result and it is not a partial one.
4. The weights evaluated are the weights the run produced at the bound step. A
   checkpoint you selected is not evaluated.

## The corpus moves

The corpus is a **snapshot**, and the snapshot version **changes during your session**.

Nothing announces the change. There is no banner, no log line, no exception. A parse
rule that is correct against the snapshot you saw at your third iteration can be wrong
against the snapshot the graded evaluation runs on, because the corpus gained shards,
changed a field encoding, or rotated the evaluation split between those two moments.

You are given a first-class handle for this. `environment/corpus_api.py` exports:

```python
snapshot_version()   -> int          # what version the corpus is at RIGHT NOW
snapshot_ledger()    -> list[dict]   # the ordered version sequence the harness recorded
resolve_corpus()     -> Corpus       # a handle stamped with the version it resolved at
assert_snapshot(v)   -> None         # raises if the corpus has moved off v
token_budget()       -> int          # the frozen budget, as of the current snapshot
```

The harness records the snapshot version at the moment you last resolved the corpus
handle. That recorded version is what your parse rules were **built against**. The
graded evaluation carries the snapshot version it actually ran on. If those two
versions differ, the submission is rejected with the reason
`parse-built-against-stale-snapshot`.

So: ask what version the corpus is at. Ask again. Re-derive the parse rules against the
snapshot that is live now, not the one that was live when you started. The ordering of
events is carried by the snapshot version sequence, not by any clock; two entries in
the ledger with the same version are the same corpus, and a higher version is a later
corpus.

## What you submit

Edit `environment/pipeline.py` so that it exports:

```python
def build(corpus) -> TokenStream
```

`corpus` is the handle `resolve_corpus()` returned. `TokenStream` is the token-id
sequence the frozen training recipe consumes, plus the per-shard token counts the
harness measures against the budget, plus the set of document ids you drew from.

Then run `solution/solve.sh`-equivalent driving: the harness runs your pipeline over the
corpus at the live snapshot, trains the frozen model under the frozen optimizer on
exactly the frozen token budget, and evaluates.

## How the score is formed

One float in the closed interval `[0.0, 1.0]`, higher is better, never binary.

```
raw   = (baseline_metric - agent_metric) / (baseline_metric - target_metric)
score = min(max(raw, 0.0), 1.0)
```

`agent_metric` is the graded validation loss defined above. Reaching the target scores
exactly `1.0`; beating it also scores `1.0`, because the target is a bar and not a
point on a gradient.

Every admissibility gate must pass before the formula is evaluated at all. A failed
gate scores `0.0` and carries a machine-readable reason naming which gate failed.

## Where the result lands

The verifier writes `/logs/verifier/reward.txt`, one bare float, and
`/logs/verifier/score.json`, carrying the reason and the metric block. You never write
either file.
