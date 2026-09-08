# Design an update rule that beats AdamW on a frozen nanoGPT at a fixed budget

You have one accelerator, a 49.3M-parameter nanoGPT, and a budget of 25,165,824
training tokens that you cannot change. The architecture is frozen, the initialisation
is frozen, the token stream is frozen, and the shape of the step size over time is
frozen. **The per-parameter update rule is yours.**

Write one JSON document to:

```
/workspace/submission/update_rule.json
```

## The law you are choosing a point in

The harness applies one update law, and your seven numbers per parameter role select a
point in it. For each role, with `g` the gradient, `w` the parameter and `t` the
optimizer step:

```
m  <- decay1 * m + (1 - decay1) * g
v  <- decay2 * v + (1 - decay2) * g * g
m^ <- m / (1 - decay1**t)          when decay1 > 0, else m^ = g
v^ <- v / (1 - decay2**t)          when decay2 > 0, else v^ = g*g
d  <- m^ / (v^ ** precond_power + eps)
d  <- (1 - sign_mix) * d + sign_mix * sign(m^)
w  <- w - step_size * envelope(t) * (d + weight_decay * w)
```

`envelope(t)` is frozen and identical for every run: a 5% linear warmup, a hold, then
a linear decay to zero across the final 45% of the run. It is declared in
`environment/frozen/task_spec.json`.

The four parameter roles are priced **independently**:

| role | tensors |
|---|---|
| `embed` | the token embedding |
| `hidden` | every matrix inside the six blocks |
| `head` | the untied output projection |
| `scalar` | every RMSNorm gain and every bias |

Nothing requires the four roles to run the same law. That is the point of the slot.

## The document

```json
{
  "schema": "oer-nanogpt-updaterule/v1",
  "notes": "free text, never parsed, copied into the score document",
  "grad_clip": 1.0,
  "rules": {
    "embed":  {"step_size": 0.003, "decay1": 0.9, "decay2": 0.999,
               "precond_power": 0.5, "sign_mix": 0.0,
               "weight_decay": 0.0, "eps": 1e-10},
    "hidden": { ... }, "head": { ... }, "scalar": { ... }
  }
}
```

All four roles must be present and all seven keys must be present in each. A missing
role is a refusal, not an inheritance. `environment/rule_schema.py` is the schema the
verifier uses, byte for byte; run it against your candidate before you submit.

## How you are scored

The verifier trains **three** models from scratch on this run:

1. the shipped `environment/default_rule.json` — AdamW at a uniform 3e-3, the LOW anchor
2. a stronger reference rule that exists only inside the verifier image — the BAR
3. your submission

and computes, on a held-out FineWeb slice that is not in your image:

```
reward = clamp( (default_loss - agent_loss) / (default_loss - reference_loss), 0, 1 )
```

Neither anchor is a stored number. Both are measured, from scratch, on every grading
run. Reaching the reference scores 1.0; beating it also scores 1.0; tying the default
scores 0.0.

**The default is not a straw man.** It is AdamW at a learning rate that works. You are
not being asked to discover Adam — you are being asked to beat a well-set Adam by
changing the law, and the roles are where the room is.

## Measuring before you submit

```
python3 /app/train_local.py candidate_a.json candidate_b.json
```

runs the same harness, the same budget and the same frozen initialisation the verifier
will use, and reports the loss on `data/devset_slice.bin` — a proxy split, not the
graded one. One run is roughly a minute on an H100, and several documents in one
invocation share the compiled graph.

## What will get you refused

Refusals score 0.0 and carry a machine-readable reason: a missing submission, an
unparseable document, an unknown or missing key, a value outside the declared bounds,
a role left out. A rule that is *inside* the bounds and trains badly is not refused —
it is a wrong answer and is scored on the loss it produced.
