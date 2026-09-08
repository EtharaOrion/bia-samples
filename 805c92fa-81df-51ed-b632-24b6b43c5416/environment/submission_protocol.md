# Submission protocol — the decoder shape

Write one JSON document to:

    /workspace/submission/shape.json

`/workspace` is the only path shared with the verifier. A shape left anywhere under
`/app` is never graded.

## The document

```json
{
  "schema": "oer-nanogpt-shape/v1",
  "notes": "free text, never parsed, copied into the score document",
  "shape": {
    "num_layers": 7,
    "model_dim": 384,
    "head_dim": 64,
    "mlp_ratio": 3
  }
}
```

`schema` and `shape` are required, `notes` is optional, and **any other key — at the
top level or inside `shape` — is refused** rather than ignored.

| field | allowed values |
|---|---|
| `num_layers` | `2,3,4,5,6,7,8,9,10,11,12,14,16` |
| `model_dim` | `256, 320, 384, 448, 512` |
| `head_dim` | `32, 64, 128`, and must divide `model_dim` exactly |
| `mlp_ratio` | `2, 3, 4` |

## The budget your shape must fit

The scarce thing is **non-embedding parameters**: everything in the block stack plus
the two model-level RMSNorm gains. The frozen budget is `10,642,944` — what a 6-layer,
384-wide, 4x-expansion stack costs — and your shape's count must land within **±5%**
of it, i.e. in `[10,110,796, 11,175,091]`.

The token embedding and the untied output projection are **not** counted. They are both
`vocab_size x model_dim`, and at a 50,304-row vocabulary they would otherwise dominate
the budget so completely that the width would be fixed by arithmetic rather than
chosen.

The count is computed by `shape_schema.non_embedding_parameters`, and the verifier
image asserts at build time that this arithmetic equals the parameter count of the
**real constructed module** for every shape on the grid. Nothing you write about your
own parameter count is read. A shape outside the band is refused with
`shape-parameter-budget-violated`; the budget is never rescaled to fit a shape.

There are **25** admissible `(num_layers, model_dim, head_dim, mlp_ratio)` points,
spanning 4 layers at width 512 through 16 layers at width 256.

## What is frozen

* the vocabulary, the sequence length, the rotary positions, the pre-norm residual
  form, the squared-ReLU MLP, the attention scale and the logit softcap — a submission
  chooses how many blocks and how wide, not what a block is;
* the token budget: 6,291,456 tokens as 768 micro-batches of 16x512, consumed in
  order from position 0 by every shape;
* the **training recipe**, identical for every run — four fixed per-role learning
  rates, fixed AdamW constants, fixed clip, and a warmup-hold-decay envelope over the
  same 768 optimizer steps. It is on this surface in full, in
  `frozen/task_spec.json`. This is deliberately one recipe rather than a per-shape
  recipe: a shape that only wins once it is given its own learning rate has not
  answered the question this slot asks;
* the initialisation seed and law, re-applied immediately before each build;
* the evaluation window and geometry.

## How the reward is computed

The verifier builds and trains three decoders on the grading run: the shipped
`default_shape.json`, a stronger reference shape it holds privately, and yours. It
evaluates all three on a held-out FineWeb slice that is not in this image and scores

    reward = clamp( (default_loss - agent_loss) / (default_loss - reference_loss), 0, 1 )

Both endpoints are measured on the grading run. Nothing is read from a stored number.

## Refusal reasons

`submission-absent`, `shape-unreadable`, `shape-not-an-object`,
`shape-block-not-an-object`, `shape-schema-unrecognised`, `shape-key-unknown`,
`shape-key-missing`, `shape-value-not-an-integer`, `shape-value-not-allowed`,
`shape-heads-do-not-divide-width`, `shape-parameter-budget-violated`.

## Measuring locally

    python3 train_local.py                    # the shipped default shape
    python3 train_local.py cand_a.json cand_b.json

`train_local.py` runs the SAME harness the verifier runs. It evaluates on
`data/devset_slice.bin`, a proxy cut from a different FineWeb training shard: expect an
offset against the graded number, expect the same ranking.
