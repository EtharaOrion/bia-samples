# Submission protocol

## What you submit

One JSON file:

    /workspace/submission/plan.json

`/workspace` is the volume shared with the verifier. `/app` is **not** shared — a
plan left in `/app` is a plan that is never graded. Create the directory if it does
not exist. `/workspace/plan.json` is also accepted, as a fallback.

## The schema

`plan_schema.py` is the *only* definition of what is accepted, it ships on this
surface, and the verifier runs a byte-identical copy of it. If

    python3 -c "import harness, plan_schema; plan_schema.load('my.json', harness.load_spec())"

returns without raising, the verifier will assemble and train your plan.

```json
{
  "schema": "oer08-data-plan/v1",
  "notes": "free text, never parsed, copied into the score document",
  "draws": [
    {"source": "spliced-theta", "tokens": 2097152, "offset": 0},
    {"source": "clean-alpha",   "tokens": 3145728, "offset": 0},
    {"source": "clean-beta",    "tokens": 3145728, "offset": 0}
  ]
}
```

| field | meaning | bound |
|---|---|---|
| `draws` | an **ordered** list of draws | 1 to 32 entries |
| `draws[].source` | which pool source to read | one of the eight declared in `frozen/task_spec.json` |
| `draws[].tokens` | how many tokens to take from it | positive multiple of 8192 |
| `draws[].offset` | where in the source to start | optional, non-negative multiple of 8192, default 0 |

and across the whole plan:

- the draws must total **exactly 8388608** tokens — the frozen budget. Under-spending
  would be training on less compute and over-spending on more, and neither is the
  task, so both are refused rather than clipped;
- unknown keys are refused rather than ignored.

## What the harness does with it

`harness.assemble` concatenates the draws **in the order you wrote them**. A draw
that runs past the end of its source cycles back to the start of that source, so
repetition is an allocation you may choose rather than an error — what it costs is
that the repeated tokens come out of the same fixed budget.

The stream is then fed in order, one 16 x 512 micro-batch per optimizer step, for
1024 steps. **Order is not normalised away**: two plans with identical per-source
totals in different orders are two different training streams and are trained as
such.

## The reward

    reward = clamp( (default_loss - agent_loss) / (default_loss - reference_loss), 0, 1 )

Each `*_loss` is the validation cross-entropy of a model trained from the frozen
initialisation on that plan's assembled stream, measured by the verifier on a
held-out FineWeb slice that is not on this surface. The default anchor is
`default_plan.json`. The reference anchor is a stronger plan held only by the
verifier. Both are retrained from scratch on your grading run. Neither is a stored
number.

Nothing your plan says about itself is graded. `notes` is free text that is copied
into the score document and never parsed.
