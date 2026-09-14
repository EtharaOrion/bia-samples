"""The reference serving-configuration search. This is the solution the live checkers accept.

It is a staged search, not a sweep, and the staging is the whole point of this slot.

A greedy session picks the axis that paid first -- max_batch_size -- and rides it until the
attempt budget is gone. That axis stops paying at four-way batching, because past there the
per-output-token tail crosses the objective and every wider batch is refused. Nothing in the
environment announces that. The only place it is readable is the harness-owned attempt
ledger, which is why this search reads its own ledger: after two consecutive attempts on one
axis that improve nothing, it abandons that axis and spends the budget somewhere else.

The reallocation stage is the one that finds the answer. Chunked prefill pays NOTHING on its
own at four-way batching -- it is slower there -- and pays a great deal jointly with an
eight-way batch, because it removes the head-of-line prefill stall that was pushing the tail
over the objective. A coordinate descent that tests one axis at a time never sees it. The
budget has to be held and reallocated onto the joint move.

Run as: the harness places this file as search.py beside its own serve_sim.py, points
OER23_ENVIRONMENT at the frozen substrate, and reads the session document back from
OER23_SESSION_OUT. Every number in that document is produced by the harness. This file
reports what the harness measured and never a figure of its own.
"""

from __future__ import annotations

import json
import os

import serve_sim

# One stage is one bounded exploration of one direction. `stale_limit` is the reallocation
# rule: after that many consecutive attempts improving nothing, the stage is over.
STAGES = (
    {"name": "batch-axis-sweep", "override": {}, "axis": "max_batch_size", "values": (2, 4, 8, 12, 16, 24, 32), "stale_limit": 2},
    {"name": "scheduler-axis", "override": {}, "axis": "scheduler", "values": ("sjf",), "stale_limit": 2},
    {"name": "prefill-axis", "override": {}, "axis": "chunked_prefill", "values": (True,), "stale_limit": 2},
    {"name": "admission-axis", "override": {}, "axis": "admission_policy", "values": ("defer_when_full", "drop_when_full"), "stale_limit": 2},
    {"name": "queue-depth-axis", "override": {}, "axis": "max_queue_depth", "values": (64,), "stale_limit": 2},
    {
        "name": "reallocation-joint-batch-and-prefill",
        "override": {"chunked_prefill": True},
        "axis": "max_batch_size",
        "values": (8, 12, 16),
        "stale_limit": 2,
    },
    {"name": "confirmation", "override": {}, "axis": "scheduler", "values": ("sjf",), "stale_limit": 2},
    {"name": "confirmation", "override": {}, "axis": "max_queue_depth", "values": (64,), "stale_limit": 2},
    {"name": "confirmation", "override": {}, "axis": "admission_policy", "values": ("accept_all",), "stale_limit": 2},
)


def plan(trace: dict, hardware: dict, objective: dict) -> list:
    """Decide the ordered configuration list by measuring, never by guessing.

    The decision at every step is taken against harness telemetry for the attempts already
    run. That is what makes this a session rather than a script: the carried best is state,
    and losing it would lose the answer.
    """
    slo = int(objective["p99_tpot_centiticks"])
    carried = dict(serve_sim.DEFAULT_CONFIG)
    configs = [dict(carried)]
    best = serve_sim.simulate(trace, hardware, carried)
    if not serve_sim.feasible(best, slo):
        best = None

    for stage in STAGES:
        stale = 0
        for value in stage["values"]:
            candidate = dict(carried)
            candidate.update(stage["override"])
            candidate[stage["axis"]] = value
            if candidate == carried:
                continue
            configs.append(dict(candidate))
            telemetry = serve_sim.simulate(trace, hardware, candidate)
            if serve_sim.feasible(telemetry, slo) and serve_sim.better(telemetry, best):
                best, carried, stale = telemetry, dict(candidate), 0
            else:
                stale += 1
                if stale >= int(stage["stale_limit"]):
                    break
    return configs


def main() -> int:
    environment = os.environ["OER23_ENVIRONMENT"]
    destination = os.environ["OER23_SESSION_OUT"]
    trace, hardware, objective = serve_sim.load_substrate(environment)
    session = serve_sim.run_session(plan(trace, hardware, objective), trace, hardware, objective)
    serve_sim.report(session, "raw")
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(session, handle, sort_keys=True, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
