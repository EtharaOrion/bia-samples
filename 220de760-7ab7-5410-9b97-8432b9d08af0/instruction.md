# Data mixture search under a fixed token budget

You are given a pool of FineWeb10B training shards and a fixed budget of training tokens. A frozen nanoGPT decoder is trained on exactly that many tokens and then evaluated on a held-out validation split you never see. Your job is to decide where those tokens come from.

## What you produce

Exactly one file, `plan.json`, in your working directory. Its grammar is in `environment/plan_schema.md`. Nothing else you write is read by the grader.

## The graded quantity, stated plainly

The graded quantity is the **validation loss of the trained decoder on the frozen held-out FineWeb10B split, recomputed by the verifier from the parameter snapshot its own trainer produced at the bound evaluation point**. Lower is better.

Everything in that sentence is load-bearing, so it is spelled out:

- **Recomputed by the verifier.** The graded loss is not a number your process reported. It is not a number in your stdout. It is not a field you wrote. The verifier executes your plan itself, builds the frozen decoder itself, trains it itself, and evaluates the resulting parameters itself.
- **Of the trained decoder.** The graded artifact is a real parameter snapshot of the frozen architecture, shape-bound to the declared vocab_size, num_layers, model_dim and head_dim in `environment/nanogpt_substrate.json`. There is no closed form and no stand-in estimator. Remove the forward and backward passes and there is no snapshot, so there is no score rather than a different score.
- **Unsmoothed.** The graded readout is the raw evaluation: the summed cross entropy in nats over the held-out split divided by the number of tokens predicted. You may EMA, average or otherwise filter your own reported loss for your own use and write it into `report.json`; nothing there is graded. Declaring smoothing on the graded readout in `environment/graded_readout.json` does not change what the verifier computes, and a declared value other than `none` is graded as readout smoothing and scores zero.
- **At the bound evaluation point.** The bound evaluation point is budget exhaustion: the moment the trainer has fed exactly the frozen token budget, in steps of `tokens_per_step`, one forward pass and one backward pass per step. A run that stops feeding before that point has not established a loss. It is graded as an early stop and scores zero with that reason. It is not graded as an absent result.
- **Sustained across the verifier's own folds.** The verifier splits the frozen held-out split into two folds and evaluates on both. An improvement that holds on the whole split but fails on either fold is a fold artifact, not a loss, and scores zero with that reason.
- **From the parameters the harness owns.** You cannot hand the verifier a checkpoint. The snapshot the verifier evaluates is the one its own trainer produced from the slices it fed.

## What is frozen

- **The token budget.** `corpus_spec.json` names it. A graded run feeds exactly that many tokens, counted by the harness from the slices it fed, never from a number your loop claimed. Feeding more is an overspend and scores zero. Feeding fewer is an early stop and scores zero.
- **The model.** The canonical nanoGPT decoder declared in `environment/nanogpt_substrate.json`: vocab_size 50304, num_layers 12, model_dim 768, head_dim 128, num_heads 6, seq_len 1024. A snapshot of any other shape is refused.
- **The optimizer.** Fixed algorithm, fixed schedule, fixed initialization seed. There are no optimizer hyperparameters to tune in this slot.
- **The batch rule.** `tokens_per_step` tokens per optimizer step, with exactly one forward pass and one backward pass over each step's batch.
- **The evaluation split.** A slice of a FineWeb10B validation shard, held by the verifier. It is not in the pool, no slice id in the pool resolves to it, and any pool slice whose bytes are a held-out slice's bytes is a leak that scores zero. If a source in your plan turns out to carry held-out bytes, that is your plan's problem to have avoided, not the grader's to forgive.

## What is free

The allocation of the fixed token budget across bands.

That is the whole free variable, and it is worth reading it exactly as written. It constrains the *total* token count. It says nothing about the allocation being one number per band, nothing about a band being indivisible, and nothing about a slice being consumed at most once. `environment/mixture.yaml` hands you a constant-weight-per-band template because that is the parameterisation the literature hands you. It is a template, not the boundary of the search space, and the harness accepts a strictly wider grammar that is documented in full in `environment/plan_schema.md`. If the best constant-weight mixture is not good enough, the constraint you are hitting is the template's, not the task's.

A band is not an authored label. `corpus_spec.json` states the rule that assigns each pool slice to a band, and that rule is a statistic computed from the slice's own tokens, so the bands are a measured partition of FineWeb10B rather than a provenance claim this bundle made up.

`environment/harness.py` is an agent-side simulator. It builds the same frozen decoder, trains it on the same shards, and scores it on an agent-visible probe drawn from a FineWeb10B training shard that carries no pool slice. Use it to measure your own candidate mixtures. It is not the grader and it does not hold the graded split.

## How you are scored

One float in `[0.0, 1.0]`, higher is better, never binary.

Every integrity condition above is a gate: failing one scores `0.0` and the score document names the reason. When every gate passes, the score is a continuous function of your recomputed validation loss, normalised between two calibration points the verifier recomputes for itself from the frozen substrate by training the same decoder on plans it constructs itself: the best point over its declared constant-weight calibration set, and the point reached by a reference plan that allocates at slice granularity. Matching the constant-weight point scores `0.0` with a reason. Matching or beating the reference point scores exactly `1.0`.

The family anchors for this task family are unmeasured, so `baseline_metric` and `target_metric` are declared absent rather than invented, and the calibration points above are substrate-local and are not those anchors. No loss figure for either point is written down anywhere in this bundle, because neither has been measured; both are produced by the verifier at grading time. `task.toml` records those facts and the gap ids that carry them.
