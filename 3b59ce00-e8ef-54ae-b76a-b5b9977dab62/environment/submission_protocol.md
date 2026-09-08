# Submission protocol — the regularisation and output-distribution policy

Write one JSON document to:

    /workspace/submission/policy.json

`/workspace` is the only path shared with the verifier. A policy left anywhere under
`/app` is never graded.

## The document

```json
{
  "schema": "oer-nanogpt-policy/v1",
  "notes": "free text, never parsed, copied into the score document",
  "decay":     {"wd_embed": 0.0, "wd_hidden": 0.0, "wd_head": 0.0, "wd_scalar": 0.0},
  "objective": {"label_smoothing": 0.0, "z_loss": 0.0, "logit_softcap": 15.0}
}
```

`schema`, `decay` and `objective` are required, `notes` is optional, and **any other
key — at the top level or inside either block — is refused** rather than ignored. Both
blocks must carry every field; a missing one is a named refusal, not a default.

| field | bound | what it does |
|---|---|---|
| `wd_embed` | `[0.0, 0.5]` | decoupled weight decay on the token embedding |
| `wd_hidden` | `[0.0, 0.5]` | decoupled weight decay on every block matrix |
| `wd_head` | `[0.0, 0.5]` | decoupled weight decay on the untied output projection |
| `wd_scalar` | `[0.0, 0.5]` | decoupled weight decay on the RMSNorm gains and biases |
| `label_smoothing` | `[0.0, 0.3]` | mixes uniform mass into the training target |
| `z_loss` | `[0.0, 0.02]` | coefficient on `mean(logsumexp(logits)^2)` |
| `logit_softcap` | `[2.0, 64.0]` | the cap `c` in `c * x / sqrt(x^2 + c^2)` on the logits |

The decays are **decoupled and per role**: AdamW's own `weight_decay` is held at zero
and each parameter role carries its own, so these four numbers are the policy's and
nothing else's.

## The asymmetry that is the point

The **training objective** is

    (1 - label_smoothing) * mean NLL
      + label_smoothing * mean uniform NLL
      + z_loss * mean( logsumexp(logits)^2 )

The **graded reading** is plain mean cross entropy — no smoothing term, no penalty —
taken through the same `logit_softcap` the run trained under.

So two of the three objective controls appear only in training and never in the
reading, while the third is part of the trained model and appears in both. A policy
that improves the training objective and hurts the reading scores worse. Setting
`label_smoothing` and `z_loss` to zero recovers plain cross entropy exactly.

## What is frozen

The architecture, the initialisation seed, the token stream, the compute budget
(3072 micro-batches of 16x512, `grad_accum` 1), and the **whole optimizer and
schedule** — four per-role peak learning rates, betas, eps, gradient clip and a
warmup-hold-decay envelope. They are on this surface in full, in
`frozen/task_spec.json`, and they are already tuned. This slot does not grade
optimizer search; it grades what is left once the optimizer is fixed.

Every control reaches the decoder as a 0-dim tensor rather than a Python constant, so
one compiled graph serves every policy and no policy can make grading cost more than
any other.

## How the reward is computed

The verifier trains **three** models on the grading run from the same frozen
initialisation over the same tokens: the shipped `default_policy.json`, a stronger
reference policy it holds privately, and yours. It evaluates all three on a held-out
FineWeb slice that is not in this image and scores

    reward = clamp( (default_loss - agent_loss) / (default_loss - reference_loss), 0, 1 )

Both endpoints are measured on the grading run. Nothing is read from a stored number.

## Refusal reasons

`submission-absent`, `policy-unreadable`, `policy-not-an-object`,
`policy-block-not-an-object`, `policy-schema-unrecognised`, `policy-key-unknown`,
`policy-key-missing`, `policy-value-not-a-number`, `policy-value-out-of-bounds`.

## Measuring locally

    python3 train_local.py                    # the shipped default policy
    python3 train_local.py a.json b.json      # several, one warm process

`train_local.py` runs the SAME harness the verifier runs. It evaluates on
`data/devset_slice.bin`, a proxy cut from a different FineWeb training shard: expect an
offset against the graded number, expect the same ranking. Several policies in one
invocation share the model build and the single compilation.
