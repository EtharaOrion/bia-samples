# Data-order optimization: choose the order, nothing else

## Objective

Lower the final validation loss of a frozen training run by choosing the order in which the frozen training sequences are consumed. The model, its initialization, the optimizer, its schedule, the corpus and the total token budget are all frozen and harness-owned. You cannot change any of them, and the harness digests them and refuses a run whose digest moves. The one thing you own is the permutation of the training sequences into batches.

## What you submit

Write `submission/order.py`. It must define one function.

```python
def build_order(meta: dict) -> list[list[int]]:
    ...
```

It returns `meta["n_steps"]` batches, each a list of exactly `meta["batch_sequences"]` sequence indices. Taken together those batches must contain every index in `range(meta["n_sequences"])` exactly once. That is the frozen multiset: you may decide when a sequence is seen and what it is seen beside, and you may not decide how often it is seen, whether it is seen at all, or how many tokens the run consumes.

`meta` carries `n_sequences`, `batch_sequences`, `n_steps`, `seq_len`, `n_domains`, and `features`. `features` is a list with one record per training sequence, each carrying `index`, `domain_id`, `unigram_entropy`, `distinct_token_count`, `adjacent_repeat_rate` and `mean_token_id`. Those are the only per-sequence facts you get. None of them is a loss, a gradient or a model output, so an ordering is fixed before training starts and there is no online feedback channel.

`build_order` must be pure and deterministic. The grader derives your order twice, in two independent processes under different working directories and different environment identities, and requires the two derivations to be byte-identical. A `build_order` that reads a clock, a random source without a fixed seed, the environment, or the filesystem will not survive that and scores zero.

Produce your run logs by invoking the runner at `$BIA_RUNNER`. That runner is the only writer of the harness telemetry record. Logs you write by hand are not evidence and will not reconcile.

```sh
python3 "$BIA_RUNNER" grid
python3 "$BIA_RUNNER" run --arm submitted --seed 0
python3 "$BIA_RUNNER" run --arm baseline  --seed 0 --draw 0
python3 "$BIA_RUNNER" order-digest --order-module submission/order.py
```

## How you are graded

Score is a float on the closed interval from 0 to 1, higher is better. It is not binary.

Every graded seed is run several times inside the same attempt. The comparator is an ensemble of independent uniform random shuffles, not a single shuffle, so beating one unlucky draw earns nothing. Every run of a given seed uses the same model initialization. Let `delta` range over every pair of a graded seed and a comparator draw, taking the comparator's final validation loss minus yours on that seed.

Score is `clip(mean(delta) / 0.05, 0, 1)` when every gate below passes, and exactly 0.0 when any of them fails.

Gates, all required.

1. **Frozen substrate.** The harness records a digest over the architecture, the initialization, the optimizer, the schedule, the corpus seed, the total token budget and the deterministic kernel binding on every telemetry record. Any deviation scores zero. Kernel determinism is part of that frozen substrate and the harness binds it after your ordering module has been imported, so a run is a pure function of its seed and its order and repeating one reproduces its loss exactly. Use that: a difference you measure between two orderings on the same seed is the ordering and nothing else.

2. **Frozen multiset.** The harness accumulates every sequence index the trainer actually consumed and digests the sorted result. Anything other than the frozen index set exactly once scores zero.

3. **Order fixed before training.** Exactly one ordering event is recorded per run and it must precede every training step of that run. There is no lawful path that reorders mid-run.

4. **No egress during scored work.** The runner installs a socket hook before the scored interval opens and records every outbound connect attempt. A non-empty audit scores zero. The environment is provisioned during setup and everything you need is already installed.

5. **Your order was actually used.** The grader independently derives your order and requires the submitted arm of the telemetry to carry that exact digest, and to differ from the comparator's order. A submission the trainer ignored cannot score.

6. **Noise floor.** `mean(delta)` must be positive, and its t statistic against the spread of those same observations must be at or above 2.0. The noise floor is therefore not a number you can look up: it is estimated from the comparator ensemble during your own graded attempt, so a difference that is small relative to the run-to-run spread scores zero however favourable its sign. The graded seeds and the comparator draws are chosen by the harness and are never passed to `build_order`, so you cannot fit an ordering to what will grade it.

## What you may use

The published literature on curriculum learning, data ordering, batch composition and sample scheduling is public, and what you recall of it you may use. Be aware of what this interface does and does not expose. The comparator is an ensemble of uniform random shuffles rather than the corpus order, so the gain a published result reports against an unshuffled or naively ordered baseline is already in the comparator and is not available to you a second time, and an arbitrary reshuffle has an expected improvement of exactly zero. The multiset is frozen, so every result that works by reweighting, filtering, deduplicating, repeating or dropping data is out of reach here, whatever it reports. The ordering is fixed before the first step, so every result that selects the next batch against a running loss or gradient signal is also out of reach. What remains is the part of the published claim that is genuinely about order at a fixed multiset, and separating that part from the rest is the intended work.

## Budget

One graded attempt is bounded at 0.12 hours, which is 7.2 minutes of single-H100 time. A graded attempt runs nine training runs: two comparator draws and one submitted run on each of three graded seeds. Your exploratory probes come out of the same session budget, which is 6 hours across at most 50 attempts, and the best attempt of the session is the one that counts.

Time your own probes before committing. An unfinished graded run scores zero. Do not launch the graded run, background a wait, and end your turn: the container is torn down and nothing is graded. Wait for it in the foreground.
