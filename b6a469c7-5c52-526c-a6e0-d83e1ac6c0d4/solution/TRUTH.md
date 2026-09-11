# FORGE-CANARY-BEGIN
# 0: 7a457195751eb83b5e0131fa097c97f03c0a89e245048f4ad37a7f25beaa98eb
# 1: 3be3a1ec778a9b1771645196ba871e3941bdb57358a4ea85bac86dc7ea7e88c2
# 2: a843b5b8b5477d376531826fcee8b255229c0e230dca3184dbdf9c75f50b65ef
# 3: 929cbd32148dc1bad185a96619d067d8e019fc568e7857e4cd1e1f7e02e5db47
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
