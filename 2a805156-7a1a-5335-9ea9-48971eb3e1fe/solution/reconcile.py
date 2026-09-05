#!/usr/bin/env python3
"""One iteration of the recovery discipline. Driven by solution/solve.sh.

Reads the summary the loop handed this iteration, reconciles it against the
durable ledger, and appends a reconstruction record if the summary lost
anything. Reconciliation happens BEFORE anything is proposed, because a
reconciliation that runs after the proposal recovers nothing the proposal could
have used.
"""

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import reference

TELEMETRY = pathlib.Path(os.environ.get("OER_TELEMETRY", "/logs/verifier"))
LEDGER = pathlib.Path(os.environ.get("OER_LEDGER_DIR", "/workspace/ledger")) / "ledger.jsonl"


def main():
    iteration = int(os.environ.get("OER_ITERATION", "1"))
    path = TELEMETRY / "loop" / "summaries" / ("iter-" + str(iteration) + ".json")
    summary = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    outcome = reference.recover(iteration, summary, LEDGER)
    sys.stdout.write(
        json.dumps({"iteration": iteration, **outcome}, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
