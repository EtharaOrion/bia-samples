# The plan grammar the graded harness accepts

Your submission produces exactly one artifact: `plan.json` in its working directory. The graded harness reads it, feeds the FineWeb10B tokens it names, trains the frozen decoder on those tokens, and evaluates the parameter snapshot its own trainer produced. Your process never touches the weights and never touches the evaluation split.

`plan.json` is one JSON object.

```json
{ "schema": "oer08.plan/v2", "mode": "constant", "weights": { "<band-id>": <float>, ... } }
```

```json
{ "schema": "oer08.plan/v2", "mode": "schedule",
  "draws": [ { "slice": "<shard-stem>:<slice-index>", "repeat": <int >= 1> }, ... ] }
```

## `constant` mode

`weights` maps band ids to non-negative shares. The harness feeds `round(weight * budget_tokens)` tokens from each named band, drawn uniformly across the whole of that band by cycling its slices in sorted order. Because every slice is the same length, "uniformly across the whole band" is exactly the band's own token composition, and the harness feeds it as such. This mode cannot express any allocation finer than one number per band.

Weights that sum to less than one leave the budget unspent. Weights that sum to more than one overspend it. Both are graded, and neither is rounded away.

## `schedule` mode

`draws` is an ordered list of slice draws. `slice` names one pool slice by `<shard-stem>:<slice-index>`, exactly as `corpus_spec.json` defines the id. `repeat` is how many times that slice is fed. Each fed instance costs `slice_tokens` tokens against the budget, whether or not the trainer has seen that slice before. The budget is a budget on *consumed* tokens, not on distinct slices.

`schedule` mode is strictly more expressive than `constant` mode: every constant-mode allocation has a schedule-mode expression, and schedule mode can also express which slices inside a band are consumed and how many times.

## What is frozen, restated in terms of this grammar

- `budget_tokens` is frozen at the value in `corpus_spec.json`. A graded run feeds exactly that many tokens, in steps of `tokens_per_step`, one forward pass and one backward pass per step.
- The decoder architecture is frozen at the canonical substrate in `environment/nanogpt_substrate.json`: vocab_size 50304, num_layers 12, model_dim 768, head_dim 128, num_heads 6, seq_len 1024. The graded parameter snapshot is shape-bound to those numbers, and a snapshot of any other shape is not a snapshot of this model.
- The optimizer, its schedule and the initialization seed are frozen. There are no optimizer hyperparameters to tune in this slot.
- The evaluation split is frozen, is a slice of a FineWeb10B validation shard held by the verifier, and is not in this pool. No slice id in this pool resolves to it.

## What is free

The allocation of the frozen token budget across bands. Nothing in that sentence restricts the allocation to being uniform inside a band, and nothing in it restricts a slice to being consumed once.

## Reporting

You may also write `report.json` with whatever you like in it, including your own smoothed estimate of your loss. Nothing in it is graded. The graded loss is recomputed by the verifier from the parameter snapshot its own trainer produced.
