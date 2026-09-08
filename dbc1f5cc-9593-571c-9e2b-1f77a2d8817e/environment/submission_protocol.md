# Submission protocol — the batch-geometry plan

Write one JSON document to:

    /workspace/submission/packing.json

`/workspace` is the only path shared with the verifier. A plan left anywhere under
`/app` is never graded.

## The document

```json
{
  "schema": "oer-nanogpt-packing/v1",
  "notes": "free text, never parsed, copied into the score document",
  "phases": [
    {"budget_frac": 0.5, "rows": 64, "seq_len": 128, "grad_accum": 1},
    {"budget_frac": 0.5, "rows": 16, "seq_len": 512, "grad_accum": 1}
  ]
}
```

`schema`, `phases` are required. `notes` is optional. **Any other top-level key is
refused**, and so is any other key inside a phase — unknown keys are refused rather
than ignored, so a typo is a named refusal instead of a silently dropped setting.

## Phases

The phases partition the frozen token budget **in order**. Phase *i* consumes
`budget_frac` of `total_train_tokens` and feeds it as micro-batches of
`rows x seq_len` tokens, `grad_accum` of them per optimizer step.

| field | type | bound |
|---|---|---|
| `budget_frac` | number | `[0.02, 1.0]`; the fractions must sum to `1.0` |
| `rows` | integer | one of `8, 16, 32, 64` |
| `seq_len` | integer | one of `128, 256, 512, 1024` |
| `grad_accum` | integer | one of `1, 2, 4` |

Two further bounds:

* `rows * seq_len` — the micro-batch token count — must lie in `[8192, 32768]`. That
  leaves nine geometries: `8x1024`, `16x512`, `16x1024`, `32x256`, `32x512`,
  `32x1024`, `64x128`, `64x256`, `64x512`. The floor keeps every plan's throughput in
  the same band, so no plan can be much slower per token than any other; the ceiling
  keeps every plan inside the accelerator.
* every phase must buy at least **8 whole optimizer steps** out of its share. A phase
  that cannot is refused by name rather than silently run as something else.

At most **6** phases.

## What is frozen

Everything else, and specifically:

* the architecture, and the initialisation seed — every run starts from the same
  parameter tensors;
* the **token** budget: `total_train_tokens` from `frozen/task_spec.json`, consumed
  in order from position 0 of the same stream by every plan;
* the optimizer — AdamW with four fixed per-role learning rates, fixed betas, eps,
  weight decay and gradient clip;
* the learning-rate envelope. It is **clocked in tokens consumed**, not in step
  index, so two plans see the same peak-relative multiplier at the same point of the
  same budget. On top of it the peak is multiplied by
  `sqrt(rows * seq_len * grad_accum / 8192)`, the standard square-root batch rule for
  adaptive optimizers, so a plan that accumulates into larger steps is not simply
  handed a smaller total parameter movement;
* the **evaluation geometry**, fixed at `16 x 512` for every run whatever geometry it
  trained under. A plan cannot move the measurement.

## How the reward is computed

The verifier trains three models on the grading run from the same frozen
initialisation over the same tokens: the shipped `default_packing.json`, a stronger
reference plan it holds privately, and yours. It evaluates all three on a held-out
FineWeb slice that is not in this image and scores

    reward = clamp( (default_loss - agent_loss) / (default_loss - reference_loss), 0, 1 )

Both endpoints are measured on the grading run. Nothing is read from a stored number,
and nothing your plan says about itself is read at all.

## Refusal reasons

A malformed or out-of-bounds plan is refused at reward 0.0 with a machine-readable
reason and is never partially credited:

`submission-absent`, `plan-unreadable`, `plan-not-an-object`, `plan-block-not-an-object`,
`plan-schema-unrecognised`, `plan-key-unknown`, `plan-key-missing`,
`plan-phases-not-a-list`, `plan-phases-empty`, `plan-too-many-phases`,
`plan-value-not-a-number`, `plan-value-not-an-integer`, `plan-rows-not-allowed`,
`plan-seq-len-not-allowed`, `plan-grad-accum-not-allowed`,
`plan-microbatch-tokens-out-of-bounds`, `plan-budget-frac-out-of-bounds`,
`plan-budget-frac-does-not-sum-to-one`, `plan-phase-buys-too-few-steps`.

## Measuring locally

    python3 train_local.py                      # the shipped default
    python3 train_local.py cand_a.json cand_b.json

`train_local.py` runs the SAME harness the verifier runs. It evaluates on
`data/devset_slice.bin`, a proxy cut from a different FineWeb training shard: expect
an offset against the graded number, expect the same ranking.
