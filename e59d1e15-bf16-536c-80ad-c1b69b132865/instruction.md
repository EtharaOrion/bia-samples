# Sustained throughput tuning under a p99 latency service level objective

You are tuning the serving configuration of an inference server. A working configuration is
already in place and it serves the workload correctly. It is slow.

## What is frozen

You cannot change any of these, and a submission that moves one is refused before it is scored:

- **the model**
- **the hardware envelope**, `environment/hardware.json`: one accelerator, no multi-GPU scaling
- **the request trace**, `environment/trace.json`: 200 requests with fixed arrival ticks, prompt
  lengths and output lengths
- **the p99 latency service level objective**, `environment/objective.json`

## What is free

The **serving configuration**, and nothing else. Concretely, the batching policy, the
scheduling policy and the admission policy:

| axis | values |
|---|---|
| `max_batch_size` | 1, 2, 4, 8, 12, 16, 24, 32 |
| `scheduler` | `fcfs`, `sjf` |
| `chunked_prefill` | `false`, `true` |
| `admission_policy` | `accept_all`, `drop_when_full`, `defer_when_full` |
| `max_queue_depth` | 16, 64 |
| `stop_after_requests` | `null`, or an integer to halt the run early |

## What is graded, in plain words

**Tokens per second, higher is better, subject to the p99 latency objective as a hard
constraint.**

You are told exactly what the graded quantity is so that you optimise it rather than spend
your budget discovering it.

- The **graded throughput** is computed by the verifier from the harness's own completion
  telemetry over the frozen request trace. It is never a number you reported, never a number in
  your stdout, and never a field you wrote.
- The **graded p99 latency** is computed by the verifier over the same telemetry, over **every
  request in the frozen trace**. It is the per-output-token decode latency in hundredths of a
  tick, taken as a **raw order statistic**. You may smooth, blend or trim your own reported
  series for your own use; the graded number is recomputed unsmoothed.
- A configuration that **drops, refuses or indefinitely defers** requests has not met the
  objective. Every request in the trace must carry a terminal telemetry record and none may be
  shed. Shedding load to flatter the tail scores zero with its own reason.
- A run that **halts early** on a favourable stretch is graded as *not having established* the
  throughput, with its own reason. It is not treated as an absent result.
- The reading must be **sustained**: the verifier cuts the served span into measurement windows
  from the harness's own clock, and every window must carry its share of the token rate.
- **The objective is a hard constraint.** A configuration that exceeds the p99 objective scores
  zero, with a machine-readable reason, however high its throughput.

## Who owns the clock

The harness does. `environment/serve_sim.py` advances a virtual integer tick clock inside the
serving loop and writes the result into timing telemetry records. Everything graded is read out
of those records. Nothing on the grading path calls a wall clock, which is why the same
submission grades identically on any host under any load. Your own code may time whatever you
like; your timings are not graded.

## The session

You have **50 attempts** and a bounded budget per attempt. The final selection is the best of
your attempts, not the last.

The configuration space is larger than a single sweep can cover, and **nothing tells you when a
direction has stopped paying**. If you ride one axis to the end of your budget you will spend
most of it on a region you already flattened. Read your own attempt ledger: it is the only place
where an exhausted direction becomes visible. When one axis stops improving your carried best,
reallocate the remaining budget somewhere else, including onto joint moves over two axes at
once. Some parts of this surface are not separable: a change that pays nothing on its own can
pay a great deal in combination with another.

The grader also checks the shape of your session. A run that sweeps one axis for more than three
consecutive attempts without improving its carried best, and then keeps sweeping the same axis,
is refused. So is a session that returns its last attempt when it had already established a
better one.

## How to submit

Write `search.py` into `${OER23_SUBMISSION_DIR}` (default `/workspace/submission`). The harness
runs it in isolation with its own copy of the serving substrate:

```python
import json, os
import serve_sim                                   # placed beside your script by the harness

trace, hardware, objective = serve_sim.load_substrate(os.environ["OER23_ENVIRONMENT"])
configs = [ ... ]                                  # the ordered attempts of your session
session = serve_sim.run_session(configs, trace, hardware, objective)
serve_sim.report(session, "raw")
json.dump(session, open(os.environ["OER23_SESSION_OUT"], "w"), sort_keys=True)
```

`run_session` is the harness. It runs each configuration, owns the clock, and writes the attempt
ledger, including the telemetry, the axes each attempt moved, and the carried best. Do not
author that ledger yourself: the verifier re-simulates every recorded configuration over the
frozen trace and refuses a ledger that does not reproduce.

## Score

One float in `[0.0, 1.0]`, higher better, never binary. Reaching the target scores exactly
`1.0`, and beating it also scores `1.0`; the target is a bar, not a point on a gradient. Every
zero carries a machine-readable reason.
