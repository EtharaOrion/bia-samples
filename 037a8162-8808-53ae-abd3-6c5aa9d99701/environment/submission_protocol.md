# Submission protocol

Write one JSON document to:

```
/workspace/submission/recipe.json
```

`/workspace` is the only filesystem path shared between this container and the
verifier's. `/app` is this image's own, and a file left there is never graded.

## Shape

```json
{
  "schema": "oer-displacement-recipe/v1",
  "notes": "optional free text; copied into the score document, never parsed",
  "grad_accum": 1,
  "grad_clip": 1.0,
  "optimizer": {
    "lr_embed": 0.03, "lr_hidden": 0.006, "lr_head": 0.004, "lr_scalar": 0.03,
    "beta1": 0.9, "beta2": 0.999, "eps": 1e-10, "weight_decay": 0.0
  },
  "schedule": {
    "shape": "constant|linear|cosine|wsd",
    "warmup_frac": 0.05, "final_frac": 0.0, "stable_frac": 0.6
  }
}
```

Bounds, all inclusive unless noted: the four learning rates in (0, 0.1];
`beta1` in [0, 0.999]; `beta2` in [0.5, 0.99999]; `eps` in [1e-16, 1e-4];
`weight_decay` in [0, 1]; `grad_clip` in [0, 10]; `warmup_frac` in [0, 0.5];
`final_frac` and `stable_frac` in [0, 1]. `grad_accum` must be exactly 1.

`stable_frac` is read only by the `wsd` shape; the other shapes ignore it but it
must still be present and in bounds.

## What the verifier refuses

Every refusal is reward 0.0 with a machine-readable reason, never a scaled value:

`submission-absent`, `recipe-unreadable`, `recipe-not-an-object`,
`recipe-block-not-an-object`, `recipe-schema-unrecognised`, `recipe-key-unknown`,
`recipe-key-missing`, `recipe-value-not-a-number`, `recipe-value-out-of-bounds`,
`recipe-grad-accum-not-allowed`, `recipe-schedule-shape-unknown`, `record-replayed`,
`accelerator-absent`, `verifier-substrate-unavailable`,
`verifier-anchor-recipe-invalid`, `anchor-run-failed`, `anchor-diverged`,
`calibration-span-nonpositive`, `target-not-reached`, `agent-run-failed`,
`grading-chain-failed`, `grading-timed-out`, `grading-killed`.

`record-replayed` is the record-displacement gate and `target-not-reached` fires when
a recipe never reaches the target inside the ceiling. Both are gates: they refuse, and
they never scale a reward.

## What is never read

Nothing your submission says about itself. `notes` is copied into the score document
for a human reader and is never parsed. No loss and no step count you report is read.
