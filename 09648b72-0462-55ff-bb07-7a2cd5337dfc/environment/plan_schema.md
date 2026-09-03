# The plan grammar the graded harness accepts

Your submission produces exactly one artifact: `plan.json` in its working
directory. The graded harness reads it, feeds the tokens it names, trains the
frozen estimator on those tokens, and evaluates. Your process never touches the
weights and never touches the evaluation split.

`plan.json` is one JSON object.

```json
{ "schema": "oer08.plan/v1", "mode": "constant", "weights": { "<source-id>": <float>, ... } }
```

```json
{ "schema": "oer08.plan/v1", "mode": "schedule",
  "draws": [ { "doc": "<source-id>:<index>", "repeat": <int >= 1> }, ... ] }
```

## `constant` mode

`weights` maps source ids to non-negative shares. The harness feeds
`round(weight * total_tokens)` tokens from each named source, drawn uniformly
across the whole of that source. Because every document in a source is the same
length, "uniformly across the whole source" is exactly the source's aggregate
symbol composition, and the harness feeds it as such. This mode cannot express
any allocation finer than one number per source.

Weights that sum to less than one leave the budget unspent. Weights that sum to
more than one overspend it. Both are graded, and neither is rounded away.

## `schedule` mode

`draws` is an ordered list of document draws. `doc` names one document by
`<source-id>:<index>`. `repeat` is how many times that document is fed. Each
fed instance costs `doc_tokens` tokens against the budget, whether or not the
harness has seen that document before. The budget is a budget on *consumed*
tokens, not on distinct documents.

`schedule` mode is strictly more expressive than `constant` mode: every
constant-mode allocation has a schedule-mode expression, and schedule mode can
also express which documents inside a source are consumed and how many times.

## What is frozen, restated in terms of this grammar

- `total_tokens` is frozen at the value in `corpus_spec.json`. A graded run
  feeds exactly that many tokens.
- The estimator, its smoothing constant and the vocabulary are frozen.
- The evaluation split is frozen, is held by the verifier, and is not in this
  pool. No document id in this pool resolves to it.

## What is free

The allocation of the frozen token budget across sources. Nothing in that
sentence restricts the allocation to being uniform inside a source, and nothing
in it restricts a document to being consumed once.

## Reporting

You may also write `report.json` with whatever you like in it, including your
own smoothed estimate of your loss. Nothing in it is graded. The graded loss is
recomputed by the verifier from the harness-owned weights.
