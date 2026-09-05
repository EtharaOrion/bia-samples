# FORGE-CANARY-BEGIN
# 0: 2a6d592c0c838a04f5c0ef24a720312082b84793f42f34021717ff238d66a469
# 1: 0320d11fd31ef8b4dc7d75e23de3dfc0672d7703e2f583a38fd0be9cb88b0043
# 2: f592f7f96b6ea7701038b3480848b79896ab223a82b82ab4bd71d8b73618e553
# 3: 6f11bdcef1f53c896dafef8278b25aa5b95392e47f1551a5c7ff8dd67c4f9c11
# FORGE-CANARY-END
<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml. -->

# OER-23 reference truth

Slot `OER-23`, family `F11` inference serving throughput, archetype `AR1` long-horizon
state collapse. Provisional identifier is the slot id.

## What is graded

Sustained output tokens per second over the frozen request trace, subject to a hard p99
per-output-token latency objective of 1800 centiticks. Higher is better. The objective is a
constraint and not a penalty: exceeding it scores zero at any throughput.

Both readouts are recomputed by the verifier from harness-owned completion telemetry over
every request in the frozen trace. Nothing the submission reports is substituted for them.

## Who owns the clock

The harness. It advances a virtual integer tick clock inside the serving loop and writes
the result into telemetry records. Every checker reads those records and never calls a
clock, which is what makes the grading pure, deterministic and replayable over the
recorded trace.

## The reference answer

```json
{
  "admission_policy": "defer_when_full",
  "chunked_prefill": true,
  "max_batch_size": 8,
  "max_queue_depth": 16,
  "scheduler": "fcfs",
  "stop_after_requests": null
}
```

| quantity | value |
|---|---|
| attempts in the reference session | 16 |
| selected attempt | 10 |
| baseline tokens per second (default configuration, measured) | 119.550 |
| target tokens per second (best feasible lattice point, measured) | 403.138 |
| reference p99 per-output-token latency | 1790 centiticks |

`baseline_metric` and `target_metric` are ABSENT as published anchors under
`gap-oer-per-family-anchors-unmeasured`. The two figures above are calibration endpoints
the verifier measures from frozen bytes at grade time. Neither appears as a literal in any
graded file.

## Why a sweep does not reach it

`max_batch_size` pays until four-way batching and then stops paying, because every wider
batch pushes the per-output-token tail past the objective. Nothing announces that. Chunked
prefill pays nothing on its own at four-way batching and pays a great deal jointly with an
eight-way batch, because it removes the head-of-line prefill stall that was carrying the
tail over the line. A session that rides one axis to the end of its budget never sees the
joint move; a session that holds its earlier attempts and reallocates does.

## Substrate

A deterministic discrete-event simulator, declared under `gap-oer23-substrate-is-a-simulator`.
It is the right substrate here because it makes grading reproducible and lets a checker read
timing without a clock. It is never a live measurement and `reference_hours` covers simulated
serving only.
