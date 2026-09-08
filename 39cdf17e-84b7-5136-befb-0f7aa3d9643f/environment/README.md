# OER-07 — the schedule shape

## What you submit

One JSON document at `/workspace/submission/schedule.json`, carrying a single
`schedule` block:

| key | what it does |
|---|---|
| `warmup_frac` | fraction of the 3072 steps spent ramping to the peak, in [0, 0.5] |
| `warmup_shape` | `linear`, or `poly` to bend the ramp by `warmup_power` |
| `warmup_power` | the bend, in [0.25, 4] |
| `shape` | the decay family: `constant`, `linear`, `cosine`, `wsd`, `poly`, `exp`, `inv_sqrt` |
| `decay_power` | the family's exponent, in [0.25, 8] |
| `stable_frac` | for `wsd`, the fraction of the post-warmup run held at the peak |
| `final_frac` | the floor the envelope lands on at the last step, in [0, 1] |

The document returns a multiplier in [0, 1] for each of the 3072 optimizer steps, and
that multiplier scales the four **frozen** per-role peak learning rates.

## What is frozen

Everything else, and in particular the whole optimizer. `frozen/task_spec.json` fixes
AdamW, the four per-role peaks, both moments, epsilon, the weight decay and the
gradient clip. You cannot raise a peak and you cannot lower one. You decide **when in
the run the step size is allowed to be near the peak**, and nothing else.

The peaks were chosen during authoring as the largest values at which the least
forgiving schedule in the space — a flat constant envelope with no warmup at all —
still trains to a finite loss. That is what makes the shipped default a real run
rather than a divergence, and it is also why the peaks are high enough that when the
run spends its steps is worth something.

## The shipped default

`default_schedule.json` is that flat envelope: no warmup, no decay, the peaks held for
all 3072 steps. It is the LOW anchor of the reward scale and the verifier retrains it
from scratch on every grading run.

## Measuring

`python3 train_local.py mine.json` runs the real harness on the real budget against
`data/devset_slice.bin`. Every family lands on `final_frac` at the last step, so the
families differ in the *path* they take and not in where they end. At 3072 steps that
path is worth more than you might expect.
