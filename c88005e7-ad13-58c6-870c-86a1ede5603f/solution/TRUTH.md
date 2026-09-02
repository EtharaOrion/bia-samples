# FORGE-CANARY-BEGIN
# 0: d1ec87f142276f1d87148afa0308673cb828e00ac9795096a89f119424a7be0a
# 1: b2a2dc77a06af10fc5eb632ab3b92b10d97164e7fa52a50a80bdbd3c04a78afc
# 2: b874129310791cd4a97503ac3be73503dc0d45c9eb186a8303586a9279709daa
# 3: 5c6fd3f1eb11c72db73ae112be045542a5c11659eea1c8160ec75a5b8bb3acd1
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
