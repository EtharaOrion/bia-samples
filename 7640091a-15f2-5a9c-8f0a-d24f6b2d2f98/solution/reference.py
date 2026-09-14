#!/usr/bin/env python3
"""The reference solution the live checkers accept.

It does the task rather than describing it. It drives the bound refinement loop over the
recorded exploration schedule, keeps its own durable ledger under `notes/`, notices that the
context summary has stopped covering every iteration it ran, reconstructs the dropped
findings from the ledger rather than from the summary, and writes `submission.json`.

The reconstruction step is the graded skill. A reference that assembled its answer from
`context/summary.json` would produce a submission that validates, reads as internally
consistent, and is short of the durable record by exactly the iterations compaction dropped.
That submission is one of the negative controls in seed/tasks/OER-24/adequacy.py.

No clock is read here either. Timing comes from the harness-owned simulator's virtual clock.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def solve(bundle, workspace):
    bundle = Path(bundle)
    workspace = Path(workspace)
    sys.path.insert(0, str(bundle / "environment"))
    import refine  # noqa: E402  imported here so the bundle path is already on sys.path
    import serving_sim  # noqa: E402

    trace_path = bundle / "environment" / "trace.json"
    envelope_path = bundle / "environment" / "envelope.json"
    trace = load_json(trace_path)
    envelope = load_json(envelope_path)
    plan = load_json(HERE / "exploration.json")

    refine.write_inputs(workspace, trace_path, envelope_path)

    ledger = []
    records = []
    for row in plan["iterations"]:
        config = {key: row[key] for key in serving_sim.CONFIG_KEYS}
        attempt = workspace / ("attempts/iter%03d" % int(row["iteration"])) / "config.json"
        attempt.parent.mkdir(parents=True, exist_ok=True)
        attempt.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        record = refine.run_iteration(workspace, int(row["iteration"]), config, trace, envelope)
        records.append(record)

        # The durable ledger is written on EVERY iteration, before anything is known to have
        # been dropped. Recovery that depends on having been prescient is not recovery.
        ledger.append(
            {
                "iteration": record["iteration"],
                "config_sha256": record["config_sha256"],
                "config": record["config"],
                "tokens_per_second": record["tokens_per_second"],
                "p99_latency_ms": record["p99_latency_ms"],
                "slo_met": record["slo_met"],
            }
        )
        write_jsonl(workspace / "notes" / "ledger.jsonl", ledger)

    summary = load_json(workspace / "context" / "summary.json")
    summary_iterations = sorted(int(entry["iteration"]) for entry in summary["entries"])
    ledger_iterations = sorted(int(entry["iteration"]) for entry in ledger)
    dropped = [n for n in ledger_iterations if n not in summary_iterations]

    # Reconstruct from the ledger, never from the summary.
    violating = sorted(entry["config_sha256"] for entry in ledger if not entry["slo_met"])

    final = records[-1]
    _resolved, telemetry = serving_sim.simulate(trace, envelope, final["config"])
    first_arrival = min(row["t_ms"] for row in telemetry if row["kind"] == "arrival")
    last_completion = max(row["t_ms"] for row in telemetry if row["kind"] == "complete")

    submission = {
        "schema": "oer24.submission/v1",
        "final_iteration": final["iteration"],
        "config": final["config"],
        "reported_tokens_per_second": final["tokens_per_second"],
        "readout": {"smoothing": "none", "window": 0},
        "measurement": {
            "start_ms": first_arrival,
            "end_ms": last_completion,
            "stopped_early": False,
            "covers_full_trace": True,
        },
        "recovered_iterations": ledger_iterations,
        "recovered_slo_violating_configs": violating,
        "recovery_note": (
            "context/summary.json retained only the last %d iterations. The full set of "
            "configurations proved to violate the objective was reconstructed from "
            "notes/ledger.jsonl, which the refinement loop never compacts."
            % len(summary_iterations)
        ),
    }
    (workspace / "submission.json").write_text(
        json.dumps(submission, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "final_iteration": final["iteration"],
        "tokens_per_second": final["tokens_per_second"],
        "p99_latency_ms": final["p99_latency_ms"],
        "iterations_dropped_by_compaction": dropped,
        "recovered_slo_violating_configs": len(violating),
    }


def main():
    parser = argparse.ArgumentParser(description="reference solution for OER-24")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    print(json.dumps(solve(args.bundle, args.workspace), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
