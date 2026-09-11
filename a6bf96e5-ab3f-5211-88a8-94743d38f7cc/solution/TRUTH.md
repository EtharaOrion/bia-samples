# TRUTH.md

GENERATED SECTION. DO NOT HAND-EDIT.

Source: `solution/grounding.yaml`. Regenerate with `solution/recompute.py`.

## What is graded

Tokens per second over the frozen request trace, subject to a p99 latency service
level objective of 8800 ms. Direction: higher is better. The objective is a
HARD CONSTRAINT: a configuration that exceeds it scores exactly 0.0 with the reason
`p99-slo-exceeded`, however high its throughput.

Both readouts are computed by the verifier. The verifier re-runs the frozen trace
under the submitted configuration on the harness-owned simulator and derives tokens
per second and p99 latency from its own completion telemetry, over EVERY request in
the trace. No number the submission reported reaches the reward.

## The archetype

AR10, memory degradation. The bound refinement loop hands each iteration a summary of
every prior configuration and its reward, and that summary retains only the last 5
iterations. The drop is silent. An agent that reads the summary as complete loses the
configurations earlier iterations proved to violate the objective, and either re-walks
that ground or contradicts a constraint it had already discovered.

`notes/` is a durable, agent-writable directory the loop never compacts. Recovery from
there is the graded skill; the amnesia is the pressure, not the verdict.

## The reference trajectory

| iteration | max_batch_size | max_admit_per_step | scheduler | tokens/s | p99 ms | objective |
|---|---|---|---|---|---|---|
| 1 | 4 | 2 | fcfs | 141.329203 | 13562 | exceeded |
| 2 | 64 | 8 | fcfs | 189.609104 | 11502 | exceeded |
| 3 | 8 | 2 | fcfs | 164.534135 | 10062 | exceeded |
| 4 | 64 | 2 | fcfs | 188.341689 | 10530 | exceeded |
| 5 | 12 | 2 | fcfs | 173.472159 | 9087 | exceeded |
| 6 | 16 | 2 | fcfs | 178.448356 | 8819 | exceeded |
| 7 | 20 | 2 | fcfs | 181.422214 | 8590 | met |
| 8 | 24 | 2 | fcfs | 183.366829 | 8844 | exceeded |
| 9 | 32 | 2 | fcfs | 186.001359 | 9229 | exceeded |
| 10 | 32 | 1 | sjf | 183.437051 | 17163 | exceeded |
| 11 | 24 | 1 | fcfs | 182.737244 | 8683 | met |
| 12 | 32 | 1 | fcfs | 183.507327 | 8758 | met |

Iterations 1 through 7 fall outside the retained window of the final summary. The
reference reconstructs them from `notes/ledger.jsonl` and reports the full set of
objective-violating configuration digests, which is what the DIVERGENCE checker
compares against the harness-owned authoritative record.

## Anchors

`anchors_state: absent`, gap `gap-oer-per-family-anchors-unmeasured`. F11 carries no
measured family baseline or target, so none is invented. The reward schema is bound
in full and the two normalisation numbers below are measured on this slot's own
frozen substrate; they are instance-local and are NOT family anchors, under gap
`gap-oer-24-instance-local-anchors-not-family-anchors`.

| quantity | value |
|---|---|
| instance_baseline_tps | 141.329203 |
| instance_target_tps | 183.507327 |
| p99 objective, ms | 8800 |
| reference tokens/s | 183.507327 |

raw = (agent_metric - instance_baseline_tps) / (instance_target_tps - instance_baseline_tps)

score = min(max(raw, 0.0), 1.0). Reaching the target scores exactly 1.0 and beating
it also scores 1.0.

## The clock

THE HARNESS OWNS THE CLOCK. `environment/serving_sim.py` advances a virtual integer
millisecond counter derived arithmetically from the frozen envelope and emits one
telemetry record per event. Every checker reads those records and never calls a
clock. `recompute.py` reads no clock either: the golden telemetry above is a
recorded fixture, never a live measurement.

## Substrate limit

The substrate is a deterministic discrete-event simulator, declared under gap
`gap-oer-24-substrate-is-a-simulator-not-a-live-server`. That is the right choice
because it makes grading reproducible, and it never passes as a live measurement.
