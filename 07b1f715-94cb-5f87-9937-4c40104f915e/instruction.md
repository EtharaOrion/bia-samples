# S02: derive an update rule for a schedule you did not choose

## Objective

Reach a validation loss of 5.30 in the fewest optimizer steps on the frozen benchmark in this environment. The dataset and its token order, the batch shape, the architecture, the initialization, the validation path and the learning-rate schedule are all frozen. The one thing you may change is the update rule: the map from gradient and optimizer state to a parameter delta.

The learning-rate schedule is published at `environment/frozen_schedule.py`. Read it. It is the whole point of this task and it is not modifiable. The verifier recomputes it from its own private copy and compares that against the learning rates the runner actually applied, so an edited copy in your workspace changes the score and nothing else.

## What you submit

Write `submission/update_rule.py`. It must define

```python
build_update_rule(param_groups) -> torch.optim.Optimizer
```

`param_groups` is the frozen group list. Each entry is a dict with `name` in `{embed, hidden_matrix, head, vector}`, a `params` list, and a placeholder `lr`. Keep the `name` key on every group you hand to the optimizer, because the runner writes the frozen learning rate into the group that carries it. A submission that does not expose this entry point cannot be graded and scores zero.

A working starting point is at `environment/starter/update_rule.py`. It is a plain AdamW-class rule and it is deliberately unadapted to the schedule you are given. Copy it and improve it.

Produce your run logs by invoking `environment/runner/run_bia_s02.py`. That runner is the only writer of the harness telemetry record. Logs you write by hand are not evidence and will not reconcile.

```sh
python3 environment/runner/run_bia_s02.py --update-rule submission/update_rule.py --seed 0 --out-dir submission
python3 environment/runner/run_bia_s02.py --update-rule submission/update_rule.py --seed 1 --out-dir submission
```

## What is frozen, stated exactly

The schedule is a pure function of the step index and the total step count, times a fixed per-group multiplier, times a fixed base learning rate. The runner evaluates it and writes the result into every parameter group before every step. Three consequences you must design around, all of them enforced by the verifier rather than asked of you politely.

1. The parameter delta your rule applies must be exactly proportional to the `lr` the runner wrote into that group. The verifier builds your rule twice from identical state, steps it at two different learning rates, and checks that the two parameter deltas scale linearly.
2. Your rule must never write `lr` into a parameter group. The runner reads the value back after every single step and counts any disagreement.
3. Your rule is never told the total step count and must not try to obtain it. There is no environment read, no file read, and no import of the runner or the schedule module from inside your submission. A cooldown you cannot locate in time is a cooldown you cannot reimplement.

Your rule also must not touch parameters outside `optimizer.step()`. The runner fingerprints live parameter state at the end of each step and again at the start of the next one, so an averaged or blended weight set applied behind the runner's back is counted and reported.

## How you are graded

Score is a float on the closed interval from 0 to 1, higher is better, and it is not binary. Let `graded_step` be the earliest step at which the seed-mean validation loss reaches 5.30, clears the noise floor, and holds. Concretely, all four of these must be true at that step: the seed mean is at or below 5.30, every individual seed is at or below 5.30, `(5.30 - mean) * sqrt(n) >= 0.023` across at least two seeds, and neither the mean nor any individual seed rises back above 5.30 at any later logged step.

Score is `clip((1400 - graded_step) / (1400 - 980), 0, 1)` when every gate below passes, and exactly 0.0 when any of them fails. Full reward, 1.0, requires `graded_step` at or below 980, which is the last step before the published schedule enters its tail branch, so full reward means reaching the target without spending any of the final thirty percent of the run.

The gates, each one a deterministic check against live state rather than against your description of what you did:

1. **Frozen schedule.** The schedule module is unedited and the learning rates the runner applied match an independent recomputation at every recorded step of every group.
2. **Free axis only.** Your update is linear in the supplied learning rate, writes no learning rate, carries no route to the step horizon, and is not a replay of a pinned published record.
3. **Frozen recipe.** The architecture signature, the batch shape, the data shards and the one-forward-backward-pass-per-step rule are unchanged for the whole run.
4. **Integrity.** The telemetry chain is unbroken and in order, and the losses you report reconcile with the losses the harness recorded, in both directions.
5. **Noise floor.** At least two seeds, every seed individually reaching the target, and the crossing sustained. A single lucky seed is not a result, and a seed mean that hides one failing seed is not a result either.

Every zero score carries a machine-readable reason. If you score zero, read it.

## What you may use

The environment is provisioned before your run begins and there is no network during scored work, so you cannot download packages, datasets or models. Everything you need is installed. The published optimizer literature is public and what you recall of it you may use.

Be aware of what that recall is worth here. Published records co-design the update rule and the schedule it runs under, and the reported gain belongs to the pair. This schedule was not the partner of any of them: the warmup is four tenths of one percent of the run, there is a discontinuous drop at forty percent followed by a rise, and the tail floors at a quarter of peak instead of annealing to zero. A rule ported from a record arrives with its own missing half, and the half it is missing is the one you cannot restore, because you may not change the schedule and you are not told the horizon. Working out which part of a record actually carried its gain, and expressing that part inside an update rule under a schedule that was never tuned for it, is the intended path.

## Budget

One graded attempt is bounded at 7.2 minutes of single-H100 time, which is the per-attempt budget this environment is designed against: a six hour session divided by fifty attempts. The runner enforces it with a 420 second wall-clock guard per invocation. A run that trips the guard stops where it is, records `stop_reason` as `wallclock_guard`, and is graded on the telemetry it managed to produce, which for an unfinished run usually means zero.

A graded result needs at least two seeds, so budget for two full runs and time your own probes before committing. Leave enough room for the graded run to finish. Do not launch it, background a wait and end your turn: the container is torn down and nothing is graded. Wait for it in the foreground.
