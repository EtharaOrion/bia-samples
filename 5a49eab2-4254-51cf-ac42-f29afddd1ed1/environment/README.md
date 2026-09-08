# OER-05 — the update rule

`rule_schema.py` and `harness.py` are the two files worth reading first. Between them
they say exactly what a submission is and exactly what will be done with it.

## What you submit

One JSON document at `/workspace/submission/update_rule.json`, carrying a `grad_clip`
and four blocks — `embed`, `hidden`, `head`, `scalar` — one per parameter role. Each
block is seven numbers, and those seven numbers select a point in a single update law:

```
m <- decay1 * m + (1 - decay1) * g
v <- decay2 * v + (1 - decay2) * g*g
m^, v^  bias-corrected when their decay is nonzero
d <- m^ / (v^ ** precond_power + eps)
d <- (1 - sign_mix) * d + sign_mix * sign(m^)
w <- w - step_size * envelope(t) * (d + weight_decay * w)
```

Familiar optimizers are interior points of that law, not special cases beside it:

| rule | `precond_power` | `decay1` | `decay2` | `sign_mix` |
|---|---|---|---|---|
| AdamW | 0.5 | 0.9 | 0.999 | 0 |
| heavy-ball SGD | 0 | 0.9 | — | 0 |
| RMSProp | 0.5 | 0 | 0.99 | 0 |
| sign momentum | 0 | 0.95 | — | 1 |

and the space between them is reachable too. Because the four roles are priced
independently, you may run a genuinely different law on the token embedding than on
the block matrices — which is the axis this slot is about and is not something a
learning-rate sweep can reach.

## What is frozen

The architecture, the initialisation, the token stream, 3072 forward passes, 3072
backward passes, 3072 optimizer steps, and the step-size **envelope**: a 5% linear
warmup, a hold, then a linear decay to zero over the last 45%, declared in
`frozen/task_spec.json` and applied identically to every run. You choose the magnitude
of your step sizes. You do not choose their shape over time — that is deliberately
somebody else's slot.

## The shipped default

`default_rule.json` is AdamW at a uniform 3e-3 on all four roles: a competent,
conventional choice rather than a straw man. It is the LOW anchor of the reward scale
and the verifier retrains it from scratch on every grading run. Beating it means
finding a rule that is better than a well-set Adam, not finding Adam.

## Measuring

`python3 train_local.py mine.json` runs the real harness on the real budget and
reports a real loss, on `data/devset_slice.bin`. Pass several files in one invocation
to pay for `torch.compile` once. The graded split is not in this image; expect an
offset from the devset number and expect the ranking to hold.
