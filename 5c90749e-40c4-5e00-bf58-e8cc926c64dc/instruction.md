# Optimizer search on a frozen substrate: reach the target loss in fewer steps

## Objective

Reach the target validation loss in the fewest optimizer steps on a frozen training substrate. The dataset, the batch shape, the architecture and the model weight initialization are all frozen and harness owned. What you may change is the optimizer update rule and its learning rate schedule, and nothing else.

The frozen baseline recipe is in `environment/baseline_optimizer.py`. It is decoupled AdamW with a warmup and a cosine decay, and it is the anchor your score is measured against. Read it, run it if you want the anchor curve, and then beat it.

## What is frozen

| Axis | Owned by |
|---|---|
| The corpus, a sparse order-two categorical source and the token arrays it produces | `environment/bia_core.py`, pinned by seed and by digest |
| The batch, meaning sequences per forward pass and sequence length, and therefore tokens per step | `environment/bia_core.py` |
| The architecture, meaning the module tree and every parameter shape | `environment/bia_core.py` |
| The model weight initialization | `environment/bia_core.py` function `frozen_init`, seeded by the run seed |
| One forward and backward pass per optimizer step | the runner |
| Evaluation, performed on the live parameters with no averaging and no blending | the runner |

There is no interface through which a submission can change any row of that table, and the verifier records a digest of the initialized weights, of the architecture and of the corpus at every logged step and compares each against the frozen value.

## What you submit

Write `submission/optimizer.py`. It must export exactly two callables.

`build_optimizer(params, **kwargs)` receives `params` as a list of `(name, tensor)` pairs sorted by name, and returns a `torch.optim.Optimizer`. Your rule owns its own learning rate, its own internal state and its own hyperparameters.

`build_schedule(total_steps, **kwargs)` returns a callable taking a zero based step index and returning a non negative multiplier. The runner sets each parameter group's learning rate to its initial learning rate times that multiplier before every step, so the schedule is yours to shape across the whole run.

A submission that does not export both callables cannot be graded and scores zero.

## How you run

`environment/runner/run_bia.py` is the only writer of the harness telemetry record. Logs you write by hand are not evidence and will not reconcile.

```
python3 environment/runner/run_bia.py --mode probe --seeds 0
python3 environment/runner/run_bia.py --mode full  --seeds 0,1
```

Use `--mode probe` for every exploratory run. Probe records are ignored by the verifier. Use `--mode full` only for the run you intend to be graded, with the submission you intend to be graded, because the verifier requires every full mode record in the telemetry file to carry the digest of the `submission/optimizer.py` that is on disk at grading time. A full mode run made with an earlier optimizer invalidates the whole record.

Submit the per seed logs the runner writes under `submission/logs/` exactly as it wrote them, beside `submission/optimizer.py`.

## How you are graded

Score is a single float on the closed interval from zero to one, higher is better, and it is not binary.

Let `graded_step` be the earliest logged step at which the seed mean validation loss is at or below the target loss, every individual seed is at or below the target loss, the margin `(target_loss - mean) * sqrt(n)` is at or above the bound significance margin, and the crossing is sustained, meaning no later logged step rises back above the target.

Score is `clip((baseline_steps - graded_step) / (baseline_steps - target_steps), 0, 1)`. The target loss, the baseline anchor and the target anchor for the graded scale are all published in `environment/frozen_manifest.json`, so nothing about the scoring rule is hidden from you.

A single seed is not a result. At least two seeds are required, and a seed mean that hides one failing seed is not a result either. A transient dip followed by a rise back above the target is not a crossing.

Every gate below scores exactly zero when it fails, and the verifier writes a machine readable reason for every zero it produces.

1. The recorded batch contract, architecture signature, corpus digest and initialization digest all equal their frozen values at every logged step.
2. Every full mode telemetry record binds to the submitted optimizer file.
3. Submitted code opened no file for writing while it was building or stepping the optimizer.
4. The optimizer step actually moved live model parameters.
5. Your reported per seed logs and the harness telemetry agree in both directions, on every step and every value.
6. The submitted file is not a near verbatim copy of the frozen baseline recipe or of any member of the pinned exclusion set of published recipes.

## What you may use

The environment is provisioned before the run and there is no network egress during scored work, so you cannot download packages, datasets or models. Everything you need is already installed.

The published optimizer speedrun record lineage is public and what you recall of it you may use. Be aware of what recall buys you here. Those records reach their step counts with a complete training recipe, and a large share of the reported gain is carried by parts of that recipe this interface does not expose, specifically the weight initialization and the eval-time blending of weights. This task freezes the initialization inside the harness and evaluates the live parameters with no blending, so a record ported through `build_optimizer` arrives stripped of the components that carried its published figure. Working out which part of a record survives that reduction, and deriving a rule against the reduced surface, is the intended path.

## Budget

One graded attempt is designed to fit inside 7.2 minutes of single H100 time, which is a six hour session divided by fifty attempts. Probe before you commit, and leave enough of the attempt for the graded run to finish, because an unfinished run produces no crossing and scores zero.

The session ends when you stop producing foreground work. Do not launch the graded run, background a wait, and end your turn, because the container is torn down and nothing is graded. Wait for it in the foreground.
