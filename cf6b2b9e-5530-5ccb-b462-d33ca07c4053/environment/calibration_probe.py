#!/usr/bin/env python3
"""Ask the calibration set what version it is at. This is the handle, use it.

The calibration set this task is allocated against is versioned and harness-owned.
An answer it gave earlier in the session is an answer about the version that was in
force THEN. This probe is the first-class way to ask what is in force NOW, so
reasoning about when a calibration statistic was true is a thing you can actually
do rather than something you have to guess at.

Temporal ordering is carried by `issued_at_tick`, a monotonically issued integer
handle the harness owns. NO CLOCK IS READ, here or anywhere on the graded path.
Wall time is not the ordering.

Usage:
    python3 environment/calibration_probe.py [--workspace DIR]

Prints one JSON object on stdout:

    {
      "calibration_version_in_force": <int>,
      "issued_at_tick": <int>,
      "sensitivity_digest": "<sha256 over the canonical bytes of the in-force vector>",
      "ledger": [{"version": <int>, "tick": <int>, "label": "<...>"}, ...]
    }

The digest is the value to carry as `calibration_fit_witness` in your submission if
and only if you actually fitted against the in-force vector. Declaring a version you
did not fit against is graded as the disagreement it is.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def resolve(workspace: str | None) -> Path:
    base = Path(workspace).resolve() if workspace else Path(__file__).resolve().parents[1]
    candidate = base / "environment" / "calibration_state.json"
    if candidate.is_file():
        return candidate
    return Path(__file__).resolve().parent / "calibration_state.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--workspace", default=None)
    args = parser.parse_args(argv)
    path = resolve(args.workspace)
    if not path.is_file():
        print(json.dumps({"error": "calibration-state-absent", "looked_at": str(path)}))
        return 2
    state = json.loads(path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "calibration_version_in_force": int(state["calibration_version_in_force"]),
                "issued_at_tick": int(state["issued_at_tick"]),
                "sensitivity_digest": str(state["sensitivity_digest"]),
                "ledger": state["ledger"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
