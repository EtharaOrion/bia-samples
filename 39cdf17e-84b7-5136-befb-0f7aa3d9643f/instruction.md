# Shape the learning-rate schedule of a frozen nanoGPT at a fixed budget

You have one accelerator, a 49.3M-parameter nanoGPT, and a budget of 25,165,824
training tokens across exactly 3072 optimizer steps. The architecture, the
initialisation, the token stream and **the entire optimizer** are frozen. What is
yours is the shape of the learning-rate envelope over those 3072 steps.

Write one JSON document to:

```
/workspace/submission/schedule.json
```

## What "the entire optimizer is frozen" means

`environment/frozen/task_spec.json` fixes AdamW, four per-role **peak** learning rates,
both moments, epsilon, the weight decay and the gradient clip. Your document produces
a multiplier in [0, 1] for each step, and that multiplier scales those peaks. You
cannot raise a peak. You cannot lower one. You decide when in the run the step size is
allowed to be near the peak, and nothing else.

## The document

```json
{
  "schema": "oer-nanogpt-schedule/v1",
  "notes": "free text, never parsed",
  "schedule": {
    "shape": "wsd",
    "warmup_shape": "linear",
    "warmup_frac": 0.10,
    "warmup_power": 1.0,
    "stable_frac": 0.60,
    "final_frac": 0.0,
    "decay_power": 1.0
  }
}
```

Warmup runs first, over `round(warmup_frac * 3072)` steps, from one step's worth of
the peak up to the peak. `warmup_shape` is `linear`, or `poly` to raise the linear
fraction to `warmup_power` — a power below one front-loads the ramp, above one holds
the run low for longer.

After warmup, `shape` carries the multiplier from 1.0 down to `final_frac`:

| `shape` | the path from peak to floor |
|---|---|
| `constant` | no decay at all; stays at the peak |
| `linear` | a straight line |
| `cosine` | a half cosine |
| `wsd` | hold the peak for `stable_frac` of the remainder, then a straight line |
| `poly` | `(1 - progress) ** decay_power` |
| `exp` | `exp(-decay_power * progress)`, rescaled to land on the floor exactly |
| `inv_sqrt` | `1 / sqrt(1 + decay_power * progress)`, rescaled the same way |

Every family lands on `final_frac` at the last step. They differ in the path, not the
endpoint — which is exactly what is being measured.

`environment/schedule_schema.py` is the schema the verifier uses, byte for byte.

## How you are scored

The verifier trains **three** models from scratch on this run:

1. the shipped `environment/default_schedule.json` — a flat envelope, no warmup, no
   decay, the peaks held for all 3072 steps. The LOW anchor.
2. a stronger reference schedule that exists only inside the verifier image. The BAR.
3. your submission.

and computes, on a held-out FineWeb slice that is not in your image:

```
reward = clamp( (default_loss - agent_loss) / (default_loss - reference_loss), 0, 1 )
```

Neither anchor is a stored number. Both are measured, from scratch, on every grading
run.

The peaks in the frozen spec were chosen so that the flat default **trains** rather
than diverging — so the low anchor is a loss and not a refusal, and so the peaks are
high enough that when you spend your steps genuinely matters.

## Measuring before you submit

```
python3 /app/train_local.py candidate_a.json candidate_b.json
```

runs the same harness, the same budget and the same frozen initialisation the verifier
will use, on `data/devset_slice.bin` — a proxy split, not the graded one. Roughly a
minute per run on an H100; several documents in one invocation share the compiled
graph.

## What will get you refused

Refusals score 0.0 with a machine-readable reason: a missing submission, an
unparseable document, an unknown shape, a missing key, a value out of bounds. A legal
schedule that trains badly is a wrong answer, not a refusal, and is scored on the loss
it produced.
