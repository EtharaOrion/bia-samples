# Submission protocol

Write one JSON document to:

```
/workspace/submission/shape.json
```

`/workspace` is the only filesystem path shared between this container and the
verifier's. `/app` is this image's own, and a file left there is never graded.

## Shape

```json
{
  "schema": "oer-nanogpt-shape/v1",
  "notes": "optional free text; copied into the score document, never parsed",
  "num_layers": 6,
  "model_dim": 384,
  "head_dim": 64,
  "mlp_ratio": 4
}
```

* `schema` must be exactly `oer-nanogpt-shape/v1`.
* All four shape fields are required and must be integers.
* `num_layers` in [2, 16].
* `model_dim` in [256, 512] and a multiple of 64.
* `head_dim` one of 32, 64, 128, and must divide `model_dim`.
* `model_dim // head_dim` must be at least 2.
* `mlp_ratio` one of 2, 3, 4, 6.
* The INSTANTIATED model must hold between 48,000,000 and 50,600,000 parameters.

`num_heads` is derived and must not be declared.

## What the verifier refuses

Every refusal is reward 0.0 with a machine-readable reason, never a scaled value:

`submission-absent`, `shape-unreadable`, `shape-not-an-object`, `shape-key-unknown`,
`shape-key-missing`, `shape-schema-unrecognised`, `shape-value-not-an-integer`,
`shape-value-out-of-bounds`, `shape-model-dim-not-a-multiple`,
`shape-head-dim-not-allowed`, `shape-mlp-ratio-not-allowed`,
`shape-heads-not-integral`, `shape-too-few-heads`, `shape-under-parameter-band`,
`shape-over-parameter-band`, `shape-not-instantiable`, `accelerator-absent`,
`verifier-substrate-unavailable`, `verifier-anchor-invalid`, `anchor-run-failed`,
`anchor-diverged`, `calibration-span-nonpositive`, `agent-run-failed`,
`agent-diverged`, `grading-chain-failed`, `grading-timed-out`, `grading-killed`.

## What is never read

Nothing your submission says about itself. `notes` is copied into the score document
for a human reader and is never parsed. The parameter count is taken from the model
the verifier built, never from a number you supply.
