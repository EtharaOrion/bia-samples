#!/usr/bin/env python3
"""The bound refinement loop. It runs one iteration and rewrites the context summary.

Each call takes one serving configuration, runs the frozen request trace under it on the
harness-owned simulator, appends an authoritative record to `logs/iterations.jsonl`, and
rewrites `context/summary.json`, which is what the next iteration of the loop is handed.

The summary is LOSSY. It retains only the most recent `retain` iterations, and it says
nothing about the ones it dropped: its `note` is a fixed string that is byte-identical before
and after the first drop, it carries no count of what it once held, and no banner changes. An
iteration that reads the summary as a complete account of the configuration space will
re-walk ground it already covered, or re-propose a configuration an earlier iteration already
proved violates the service level objective.

`notes/` is yours. It is a durable, agent-writable directory that this loop never truncates,
never compacts and never rewrites. Reconstructing what the summary dropped is done from
there.

Usage:
    python3 refine.py --workspace WS --config cfg.json --iteration N \
        --trace trace.json --envelope envelope.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import serving_sim

SUMMARY_SCHEMA = "oer24.summary/v1"
ITERATIONS_SCHEMA = "oer24.iterations/v1"

# How many iterations the context summary retains. Beyond this the oldest entries are
# dropped and nothing announces the drop.
RETAIN = 5

# The fixed note the summary always carries. It does not move when compaction happens,
# which is exactly what makes the compaction silent.
SUMMARY_NOTE = "summary of prior configuration attempts"


def config_digest(config):
    canonical = json.dumps(
        {key: config.get(key) for key in serving_sim.CONFIG_KEYS},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def percentile_99(values):
    """Nearest-rank p99: index ceil(0.99 * n) - 1 over the sorted list."""
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil(0.99 * len(ordered)) - 1
    return ordered[max(0, index)]


def measure(trace, envelope, config):
    """Run one configuration and reduce the harness telemetry to a reading."""
    resolved, telemetry = serving_sim.simulate(trace, envelope, config)
    arrivals = [row for row in telemetry if row["kind"] == "arrival"]
    completions = [row for row in telemetry if row["kind"] == "complete"]
    rejects = [row for row in telemetry if row["kind"] == "reject"]
    if not completions or not arrivals:
        return resolved, telemetry, {
            "tokens_per_second": 0.0,
            "p99_latency_ms": None,
            "completions": 0,
            "rejects": len(rejects),
            "slo_met": False,
        }
    first = min(row["t_ms"] for row in arrivals)
    last = max(row["t_ms"] for row in completions)
    tokens = sum(int(row["output_tokens"]) for row in completions)
    span = last - first
    tps = round(tokens * 1000.0 / span, 6) if span > 0 else 0.0
    p99 = percentile_99([int(row["latency_ms"]) for row in completions])
    slo = int(envelope["p99_latency_slo_ms"])
    return resolved, telemetry, {
        "tokens_per_second": tps,
        "p99_latency_ms": p99,
        "completions": len(completions),
        "rejects": len(rejects),
        "slo_met": bool(p99 is not None and p99 <= slo),
    }


def read_jsonl(path):
    if not Path(path).is_file():
        return []
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    body = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    )
    Path(path).write_text(body, encoding="utf-8")


def run_iteration(workspace, iteration, config, trace, envelope):
    """Run one iteration, extend the authoritative record, rewrite the lossy summary."""
    workspace = Path(workspace)
    resolved, telemetry, reading = measure(trace, envelope, config)
    digest = config_digest(resolved)

    record = {
        "iteration": int(iteration),
        "config": resolved,
        "config_sha256": digest,
        "tokens_per_second": reading["tokens_per_second"],
        "p99_latency_ms": reading["p99_latency_ms"],
        "completions": reading["completions"],
        "rejects": reading["rejects"],
        "slo_met": reading["slo_met"],
    }

    ledger_path = workspace / "logs" / "iterations.jsonl"
    rows = [row for row in read_jsonl(ledger_path) if int(row["iteration"]) != int(iteration)]
    rows.append(record)
    rows.sort(key=lambda row: int(row["iteration"]))
    write_jsonl(ledger_path, rows)

    retained = rows[-RETAIN:]
    summary = {
        "schema": SUMMARY_SCHEMA,
        "note": SUMMARY_NOTE,
        "entries": [
            {
                "iteration": int(row["iteration"]),
                "config": row["config"],
                "config_sha256": row["config_sha256"],
                "tokens_per_second": row["tokens_per_second"],
                "p99_latency_ms": row["p99_latency_ms"],
                "slo_met": row["slo_met"],
            }
            for row in retained
        ],
    }
    summary_path = workspace / "context" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    telemetry_path = workspace / "logs" / "telemetry" / ("iter%03d.jsonl" % int(iteration))
    write_jsonl(telemetry_path, telemetry)
    return record


def write_inputs(workspace, trace_path, envelope_path):
    """Record the digests of the frozen inputs this run was driven by."""
    payload = {
        "schema": ITERATIONS_SCHEMA,
        "trace_sha256": hashlib.sha256(Path(trace_path).read_bytes()).hexdigest(),
        "envelope_sha256": hashlib.sha256(Path(envelope_path).read_bytes()).hexdigest(),
    }
    path = Path(workspace) / "logs" / "inputs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description="bound refinement loop, one iteration")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--envelope", required=True)
    args = parser.parse_args()
    trace = serving_sim.load_json(args.trace)
    envelope = serving_sim.load_json(args.envelope)
    config = serving_sim.load_json(args.config)
    write_inputs(args.workspace, args.trace, args.envelope)
    record = run_iteration(args.workspace, args.iteration, config, trace, envelope)
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
