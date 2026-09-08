# Submission protocol

Write one JSON document to:

```
/workspace/submission/plan.json
```

`/workspace` is the only filesystem path shared between this container and the
verifier's. `/app` is this image's own, and a file left there is never graded.

## Shape

```json
{
  "schema": "oer-kernel-plan/v1",
  "notes": "optional free text; copied into the score document, never parsed",
  "attention": "flash",
  "loss_chunks": 8,
  "logit_dtype": "fp32",
  "rotary_dtype": "fp32",
  "qk_norm_dtype": "fp32"
}
```

* `schema` must be exactly `oer-kernel-plan/v1`.
* All five plan fields are required.
* `attention` one of `math`, `efficient`, `flash`.
* `loss_chunks` one of 1, 2, 4, 8, 16, 32. It must divide the 8192 rows of a step,
  and every allowed value does.
* `logit_dtype`, `rotary_dtype`, `qk_norm_dtype` each one of `fp32`, `bf16`.

## Two gates, both of which refuse rather than scale

`kernel_plan.py` answers whether the document is well formed and drawn from the
declared option sets. Being well formed is **not** the same as being correct: the
verifier separately checks that the plan computes the same function as the reference
plan, on frozen weights, before it times anything.

## What the verifier refuses

Every refusal is reward 0.0 with a machine-readable reason, never a scaled value:

`submission-absent`, `plan-unreadable`, `plan-not-an-object`, `plan-key-unknown`,
`plan-key-missing`, `plan-schema-unrecognised`, `plan-option-unknown`,
`numerical-output-diverged`, `accelerator-absent`, `verifier-substrate-unavailable`,
`verifier-anchor-plan-invalid`, `anchor-plan-failed`, `agent-plan-failed`,
`timing-failed`, `calibration-span-nonpositive`, `grading-chain-failed`,
`grading-timed-out`, `grading-killed`.

## What is never read

Nothing your submission says about itself. `notes` is copied into the score document
for a human reader and is never parsed. No latency and no speedup you report is read;
the verifier times your plan itself, interleaved with both anchors.
