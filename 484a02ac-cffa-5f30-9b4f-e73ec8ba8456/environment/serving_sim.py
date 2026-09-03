#!/usr/bin/env python3
"""The harness-owned inference-serving simulator. THE HARNESS OWNS THE CLOCK.

This module is the only thing in the slot that advances time, and the time it
advances is virtual: an integer millisecond counter driven by the arithmetic cost
model in `envelope.json`. It never calls `time`, `datetime`, `perf_counter` or any
other clock, and it consults no random source. Two runs over the same trace, the
same envelope and the same configuration produce byte-identical telemetry.

That is deliberate and it is what makes this slot gradable. The metric is a timing
metric, and a checker is forbidden from reading a clock, so ownership of the clock
has to sit somewhere a checker can read *afterwards* rather than *alongside*. It
sits here. This module instruments the serving loop and emits one telemetry record
per event; every checker downstream reads those records and computes from them,
which keeps each checker pure, deterministic and replayable over a recorded trace.

A simulator is not a live server and this module never pretends otherwise. The gap
is declared by name in `solution/grounding.yaml` and in `seed/tasks/OER-24/feasibility.yaml`.

Telemetry record kinds, all carrying `t_ms`, the harness virtual clock:

    arrival   a request from the frozen trace entered the system
    admit     the scheduler pulled it into the running batch and paid its prefill
    reject    the admission policy refused it; it will never complete
    step      one decode step over the running batch
    complete  the request produced its last output token

Usage:
    python3 serving_sim.py --trace trace.json --envelope envelope.json \
        --config config.json --telemetry out.jsonl --summary out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = "oer24.telemetry/v1"

# The closed configuration surface. These are the free axes of the slot: batching,
# scheduling and admission. Nothing else about the service is configurable, because
# the model, the hardware envelope, the request trace and the p99 SLO are frozen.
CONFIG_KEYS = ("max_batch_size", "max_admit_per_step", "scheduler", "admission_queue_limit")
SCHEDULERS = ("fcfs", "sjf")


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_config(config, envelope):
    """Fill every unset configuration key from the envelope's own default block.

    A key the submission leaves out is not a neutral choice. It resolves to the
    envelope default, and the envelope default is frozen state the agent does not
    own. This is the seam the admission-policy drift moves through, so it is
    written once, here, rather than reimplemented per call site.
    """
    defaults = dict(envelope.get("default_config") or {})
    resolved = {}
    for key in CONFIG_KEYS:
        resolved[key] = config.get(key, defaults.get(key)) if isinstance(config, dict) else defaults.get(key)
    if resolved["scheduler"] not in SCHEDULERS:
        resolved["scheduler"] = "fcfs"
    for key in ("max_batch_size", "max_admit_per_step"):
        try:
            resolved[key] = int(resolved[key])
        except (TypeError, ValueError):
            resolved[key] = 1
        resolved[key] = max(1, resolved[key])
    resolved["max_batch_size"] = min(resolved["max_batch_size"], int(envelope["max_batch_size_ceiling"]))
    resolved["max_admit_per_step"] = min(
        resolved["max_admit_per_step"], int(envelope["max_admit_per_step_ceiling"])
    )
    limit = resolved["admission_queue_limit"]
    if limit is not None:
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = None
        if limit is not None and limit < 0:
            limit = None
    resolved["admission_queue_limit"] = limit
    return resolved


def simulate(trace, envelope, config):
    """Run the frozen trace under one configuration and return the telemetry list.

    Time is an integer millisecond counter. A decode step over a running batch of
    `b` requests that also pays prefill for `p` freshly admitted prompt tokens
    costs `step_overhead_ms + decode_ms_per_request * b + prefill_ms_per_token * p`
    and produces exactly one output token for every request in the batch. Larger
    batches amortise the overhead, so throughput rises with batch size, and every
    request in the batch waits for the whole batch, so per-request latency rises
    with it too. The p99 SLO is what closes that trade.
    """
    resolved = resolve_config(config, envelope)
    overhead = int(envelope["step_overhead_ms"])
    per_request = int(envelope["decode_ms_per_request"])
    per_prompt_token = int(envelope["prefill_ms_per_token"])

    requests = sorted(trace["requests"], key=lambda row: (int(row["arrival_ms"]), str(row["id"])))
    total = len(requests)
    telemetry = []
    t = 0
    next_arrival = 0
    queue = []
    running = []
    settled = 0
    step_index = 0

    while settled < total:
        while next_arrival < total and int(requests[next_arrival]["arrival_ms"]) <= t:
            row = requests[next_arrival]
            queue.append(
                {
                    "id": str(row["id"]),
                    "arrival_ms": int(row["arrival_ms"]),
                    "prompt_tokens": int(row["prompt_tokens"]),
                    "output_tokens": int(row["output_tokens"]),
                    "produced": 0,
                }
            )
            telemetry.append({"kind": "arrival", "request": str(row["id"]), "t_ms": t})
            next_arrival += 1

        if not running and not queue:
            if next_arrival >= total:
                break
            t = int(requests[next_arrival]["arrival_ms"])
            continue

        limit = resolved["admission_queue_limit"]
        if limit is not None:
            while len(queue) > limit:
                dropped = queue.pop(0)
                telemetry.append({"kind": "reject", "request": dropped["id"], "t_ms": t})
                settled += 1

        if resolved["scheduler"] == "sjf":
            queue.sort(key=lambda row: (row["output_tokens"], row["arrival_ms"], row["id"]))
        else:
            queue.sort(key=lambda row: (row["arrival_ms"], row["id"]))

        prompt_tokens = 0
        admitted = 0
        while (
            queue
            and len(running) < resolved["max_batch_size"]
            and admitted < resolved["max_admit_per_step"]
        ):
            row = queue.pop(0)
            running.append(row)
            prompt_tokens += row["prompt_tokens"]
            admitted += 1
            telemetry.append({"kind": "admit", "request": row["id"], "t_ms": t})

        if not running:
            # Every runnable request is queued behind a full batch that is empty,
            # which cannot happen; guard anyway so the loop can never spin.
            t += overhead
            continue

        cost = overhead + per_request * len(running) + per_prompt_token * prompt_tokens
        t += cost
        step_index += 1
        telemetry.append(
            {
                "kind": "step",
                "index": step_index,
                "t_ms": t,
                "batch_size": len(running),
                "prompt_tokens": prompt_tokens,
                "tokens_emitted": len(running),
            }
        )

        still = []
        for row in running:
            row["produced"] += 1
            if row["produced"] >= row["output_tokens"]:
                telemetry.append(
                    {
                        "kind": "complete",
                        "request": row["id"],
                        "t_ms": t,
                        "arrival_ms": row["arrival_ms"],
                        "output_tokens": row["output_tokens"],
                        "latency_ms": t - row["arrival_ms"],
                    }
                )
                settled += 1
            else:
                still.append(row)
        running = still

    return resolved, telemetry


def run(trace_path, envelope_path, config_path, telemetry_path, summary_path):
    trace = load_json(trace_path)
    envelope = load_json(envelope_path)
    config = load_json(config_path)
    resolved, telemetry = simulate(trace, envelope, config)

    lines = [json.dumps(row, sort_keys=True, separators=(",", ":")) for row in telemetry]
    Path(telemetry_path).parent.mkdir(parents=True, exist_ok=True)
    Path(telemetry_path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    completions = [row for row in telemetry if row["kind"] == "complete"]
    rejects = [row for row in telemetry if row["kind"] == "reject"]
    summary = {
        "schema": SCHEMA,
        "clock_owner": "harness",
        "clock_kind": "virtual-integer-milliseconds",
        "resolved_config": resolved,
        "requests_in_trace": len(trace["requests"]),
        "completions": len(completions),
        "rejects": len(rejects),
        "steps": len([row for row in telemetry if row["kind"] == "step"]),
    }
    if summary_path:
        Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_path).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description="harness-owned serving simulator")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--envelope", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--telemetry", required=True)
    parser.add_argument("--summary", default="")
    args = parser.parse_args()
    summary = run(args.trace, args.envelope, args.config, args.telemetry, args.summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
