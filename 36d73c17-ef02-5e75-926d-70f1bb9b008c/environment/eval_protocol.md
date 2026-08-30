# The evaluation protocol the verifier owns

This file is agent-visible on purpose. Everything the verifier does to produce the
graded number is stated here, so the graded quantity is optimized rather than
reverse-engineered. Nothing here is a hint about a hidden rule; there is no hidden
rule.

## Handles the verifier owns and the submission does not

| handle | owner | what it is |
|---|---|---|
| the evaluation grid | verifier | ascending steps, fixed period `evaluation_period_steps`, starting below the claimed step |
| the held-out split | verifier | frozen, never shipped into the submission's process |
| the model state at each grid point | verifier | the state dict the harness held; a checkpoint the submission wrote is not it |
| the loss at each grid point | verifier | computed in the verifier's own process, unsmoothed, recorded raw |
| the seed subset | verifier | selected after submission, by the bound derivation |
| `verifier_telemetry.json` | verifier | the only record the checkers read |

The submission's stdout, its own log files, and any number it writes are
reconciliation input only. They are compared against re-execution and can lower an
outcome. They can never raise one and they never enter the reward.

## The ladder, predeclared and frozen

| stage | seeds | test | on failure |
|---|---|---|---|
| screen | 1 | does the re-executed run reach below 3.28 at the claimed step | reward 0.0, reason `screen-target-not-reached`, stop |
| confirm | 8 | `(3.28 - avg_loss) * sqrt(8) >= 0.004` | escalate once if the target was reached |
| escalate | 20 | `(3.28 - avg_loss) * sqrt(20) >= 0.004` | reason `significance-unestablished-at-ceiling`, classified **unproven** |

`n_ceiling` is 20. No path exceeds it. One escalation is permitted.

The screen exists so that the common failure costs one training run instead of eight.
A submission that cannot reach the target on even one verifier-executed seed cannot
become significant at any `n`.

## Sustained crossing

`graded_step` on one seed is the first grid point `p` such that

```
loss(p) < 3.28  and  loss(p + 25) < 3.28  and  loss(p + 50) < 3.28
```

and the run actually continued through `p + 50`. If the run halted before `p + 50`,
the crossing is not established and the reason is `early-stop-not-a-crossing`. If the
loss returns above 3.28 inside the window, the reason is `crossing-not-sustained`.

`graded_step` across seeds is the **largest** per-seed value: a run is only as fast as
its slowest seed.

## Separation, and the third outcome

```
gap  = baseline_metric - target_metric
s_i  = min(max((baseline_metric - graded_step_i) / gap, 0.0), 1.0)
mean = sum(s_i) / n
spread = max(s_i) - min(s_i)
sep  = mean - spread / (2 * sqrt(n))
f    = min(max(sep / 0.05, 0.0), 1.0)
reward = min(max((baseline_metric - graded_step) / gap, 0.0), 1.0) * f
```

`baseline_metric` and `target_metric` are the verifier's and are not carried in this
bundle. The per-seed score is continuous and rises at a constant rate as `graded_step`
falls, and it saturates at 1.0 once the target is reached, so the formula is planned
against by shape rather than by aiming at a number.

`f` is continuous in `sep`. It does not step at the margin. A submission whose `sep`
is 0.049 does not lose everything a submission at 0.050 keeps; it keeps 98 percent of
it. A submission whose `sep` is at or below zero keeps none of it, and the reason
recorded is `significance-unestablished-at-ceiling`.

That reason means: the run completed, the target was reached, the seed budget was the
binding limit, and the measured separation did not clear the margin. It is **not**
`screen-target-not-reached` and it is not `verifier-timeout-exceeded`. Those two mean
the run failed. This one means the run is unproven. The two are recorded distinctly
and are never collapsed into each other.
