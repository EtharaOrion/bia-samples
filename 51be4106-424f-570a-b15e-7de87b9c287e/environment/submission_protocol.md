# Submission protocol

## What you submit

One JSON file:

    /workspace/submission/recipe.json

`/workspace` is the volume shared with the verifier. `/app` is **not** shared — a
recipe left in `/app` is a recipe that is never graded. Create the directory if it
does not exist.

`/workspace/recipe.json` is also accepted, as a fallback.

## The schema

`recipe_schema.py` is the *only* definition of what is accepted, it ships on this
surface, and the verifier runs a byte-identical copy of it. If
`python3 -c "import recipe_schema; recipe_schema.load('my.json')"` returns without
raising, the verifier will train your recipe.

```json
{
  "schema": "oer-nanogpt-recipe/v1",
  "notes": "free text, never parsed, copied into the score document",
  "grad_accum": 1,
  "grad_clip": 1.0,
  "optimizer": {
    "lr_embed":      0.001,
    "lr_hidden":     0.001,
    "lr_head":       0.001,
    "lr_scalar":     0.001,
    "beta1":         0.9,
    "beta2":         0.999,
    "eps":           1e-8,
    "weight_decay":  0.1
  },
  "schedule": {
    "shape":        "constant",
    "warmup_frac":  0.0,
    "final_frac":   1.0,
    "stable_frac":  1.0
  }
}
```

Unknown keys are refused rather than ignored, so a typo is reported instead of
silently dropping a setting you thought you had set.

### Fields

| field | meaning | bound |
|---|---|---|
| `grad_accum` | micro-batches accumulated into one optimizer step. `3072 / grad_accum` optimizer steps are taken. | one of 1, 2, 3, 4, 6, 8, 12, 16, 24, 32 |
| `grad_clip` | global grad-norm clip; `0` disables clipping | `[0, 10]` |
| `lr_embed` | peak LR for the token embedding | `(0, 0.1]` |
| `lr_hidden` | peak LR for every transformer block matrix | `(0, 0.1]` |
| `lr_head` | peak LR for the untied output projection | `(0, 0.1]` |
| `lr_scalar` | peak LR for every RMSNorm gain and every bias | `(0, 0.1]` |
| `beta1`, `beta2` | AdamW moment decays | `[0, 0.999]`, `[0.5, 0.99999]` |
| `eps` | AdamW epsilon | `[1e-16, 1e-4]` |
| `weight_decay` | decoupled weight decay, applied to all four roles | `[0, 1]` |
| `shape` | post-warmup LR shape | `constant`, `linear`, `cosine`, `wsd` |
| `warmup_frac` | fraction of optimizer steps spent warming linearly to peak | `[0, 0.5]` |
| `final_frac` | end-of-run LR as a fraction of peak | `[0, 1]` |
| `stable_frac` | `wsd` only: fraction of the post-warmup run held at peak before the decay | `[0, 1]` |

`stable_frac` is read only when `shape` is `wsd`, but it must be present and in
bounds regardless, so that one schema describes every recipe.

## What is frozen

Everything else. `frozen/task_spec.json` states it: the architecture (6 layers,
model_dim 384, head_dim 64, seq_len 512, vocab 50304), the initialisation seed, the
evaluation window, and the compute budget of 3072 micro-batches of 16x512 tokens.

The budget is fixed in **forward and backward passes**, not merely in tokens. A
recipe chooses how many micro-batches accumulate into an optimizer step; it cannot
change how many micro-batches are run. Every recipe therefore costs the same FLOPs.
There is no recipe that buys a lower loss by training longer.

## How you are graded

The verifier trains three models on the run, all from the same frozen
initialisation, on the same tokens, with the same compiled kernels:

1. the shipped `default_recipe.json`,
2. a stronger reference recipe held privately inside the verifier image,
3. your recipe.

It then evaluates all three on a held-out FineWeb slice that is **not** on this
surface, and scores:

    reward = clamp( (default_loss - agent_loss) / (default_loss - reference_loss), 0, 1 )

Both endpoints are measured on your run. Nothing is read from a stored constant, and
nothing your recipe *says* about itself is graded — `notes` is copied into the score
document and never parsed. Matching the default scores 0.0; reaching the reference
scores 1.0; beating it also scores 1.0.

A malformed recipe is refused at 0.0 with a machine-readable reason. Validation
never scales the reward — it is a gate, not a grade.

## Measuring locally

    python3 train_local.py                      # the shipped default
    python3 train_local.py cand_a.json cand_b.json cand_c.json

Several recipes in one invocation is much cheaper than several invocations: the
model is built once, `torch.compile` is paid once, and the corpus is read once.

`train_local.py` evaluates on `data/devset_slice.bin`, a proxy split cut from a
*different* FineWeb training shard. It is not the graded split. Expect a small
constant offset between the two; expect them to rank recipes the same way.
