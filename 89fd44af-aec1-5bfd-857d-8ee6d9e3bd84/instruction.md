# Track-3 optimizer search: derive a new update rule

## Objective

Lower the number of optimizer steps needed to reach 3.28 validation loss on the frozen track-3 benchmark. The dataset, the batch size, and the architecture are frozen. You may change only the optimizer, its schedule, its own internal state initialization, and its hyperparameters. Model weight initialization is frozen and harness-owned; you cannot change it. Exactly one forward-backward pass per step. The Muon reference reaches the target in 3500 steps.

## What you submit

Write `submission/optimizer.py`. It must define `build_optimizer(params, lr=0.02, **kwargs)` returning a `torch.optim.Optimizer`. `params` is a list of parameter tensors. This exact entry point is how your optimizer is both trained and evaluated, so a submission that does not expose it cannot be graded and scores zero.

What is graded here is the name, the parameter order, and the acceptance of `**kwargs`. The harness always passes `lr` explicitly, so the default shown is a placeholder you may retune to your own schedule; what you must not do is ignore the value passed in. A `build_optimizer` that silently overrides the caller's `lr` is not gradeable.

Produce your run logs by invoking `runner/run_track3.py`. That runner is the only writer of the harness telemetry record. Logs you write by hand are not evidence and will not reconcile.

## How you are graded

Score is a float on the closed interval from 0 to 1, higher is better. It is not binary. Let `graded_step` be the earliest step at which the seed-mean validation loss reaches 3.28 and clears the noise floor, meaning `(3.28 - mean) * sqrt(n) >= 0.004` across at least two seeds.

Score is `clip((3500 - graded_step) / (3500 - 2900), 0, 1)` when both gates below pass, and exactly 0.0 when either of them fails. Full Score, 1.0, requires `graded_step` at or below 2900.

Two gates, both required:

1. **Integrity.** The harness observes the dataset, the global batch size in tokens, the microbatch shape, the architecture signature, and the forward-backward count per step, and records them in an append-only hash chain. Any deviation from the frozen contract scores zero. So does selecting your stopping point or your seed against validation loss.

2. **Noise floor.** At least two seeds must clear the margin above, and every seed must individually reach 3.28. A single lucky seed is not a result, and a seed-mean that hides one failing seed is not a result either. The crossing must also hold: if the seed-mean, or any individual seed, rises back above 3.28 at any later logged step, that crossing does not count. A transient dip followed by divergence scores nothing.

Your reported logs are reconciled against the harness telemetry. Any divergence scores zero.

## What you may use

The environment is pre-provisioned and egress is restricted to the model bridge, so you cannot download packages, datasets, or models. Everything you need is already installed. The published track-3 record lineage is public, and what you recall of it you may use. Be aware that the published records reach their step counts using the full training recipe, including weight initialization and eval-time weight blending, and this interface exposes neither: you supply an optimizer only, and the harness owns initialization and validation. A record ported through `build_optimizer` therefore lands well short of its published figure. Understanding why those records work, and adapting them to what this interface can express, is the intended path.

## Budget

You have **8 hours of wall clock**. Measured on this exact host, and the cost
depends strongly on how expensive your update rule is:

| run | plain Muon-class rule | with preconditioning (SOAP/Shampoo-class) |
|---|---|---|
| 1 seed x 3500 steps | ~1.1 h | ~2.1 h |
| 2 seeds x 3500 steps (minimum for a GRADED result) | ~2.2 h | ~4.3 h |

A graded submission needs **>= 2 seeds** (MIN_SEEDS=2); a 1-seed run cannot score.
Budget against the SLOWER column if your rule does eigendecompositions or matrix
preconditioning, and time your own probes before committing.
Leave enough time for the graded run to FINISH, because an unfinished run scores zero.

The trial ends when you stop producing foreground work. Do not launch the graded run,
background a wait, and end your turn: the container is torn down and nothing is graded.
Wait for it in the foreground.
