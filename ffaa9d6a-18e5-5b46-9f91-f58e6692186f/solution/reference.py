#!/usr/bin/env python3
"""The reference solution for slot OER-14. PRIVATE.

It derives the reference session from `solution/grounding.yaml` and nothing else, measures
each attempt against the frozen substrate, and emits ONE document: the submission.

The submission carries the vocabulary construction plan and the record of the search that
produced it. It carries no bits-per-byte figure for itself, because the harness measures
that. The per-attempt numbers it does carry are the CARRIED STATE of the session, and the
harness re-measures every one of them, so a fabricated frontier does not survive.

The reference is a long-horizon trajectory on purpose: nine attempts sweep the direction
that pays until it stops paying, two attempts reallocate off it, and one attempt probes a
direction that turns out not to pay. Nothing in the substrate announces the flattening. The
best carried attempt is the submitted allocation, which is what `final_selection: best` means.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "environment"))

import substrate  # noqa: E402


def grounding() -> dict:
    with (BUNDLE / "solution" / "grounding.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def measure_allocation(allocation: dict, train: bytes, evaluation: bytes) -> float:
    """Bits per byte for one allocation, read the way the harness reads it."""
    spec = substrate.VocabSpec.from_payload({"allocations": allocation})
    meter = substrate.Meter(substrate.COMPUTE_BUDGET_UNITS)
    telemetry = substrate.measure(spec, train, evaluation, meter)
    return substrate.graded_bpb(telemetry["points"], len(evaluation))


def submission_document() -> dict:
    source = grounding()
    train = substrate.load_frozen("train.txt")
    evaluation = substrate.load_frozen("eval.txt")

    attempts, frontier, previous = [], {}, None
    for row in source["golden_trajectory"]["attempts"]:
        allocation = {name: 0 for name in substrate.DIRECTIONS}
        allocation.update(row["allocation"])
        reading = measure_allocation(allocation, train, evaluation)
        direction = row["direction"]
        best = frontier.get(direction)
        frontier[direction] = reading if best is None else min(best, reading)
        attempts.append(
            {
                "index": int(row["index"]),
                "direction": direction,
                "allocation": allocation,
                "reallocated_from": row["reallocated_from"],
                "marginal_gain_bits": 0.0 if previous is None else previous - reading,
                "carried_frontier": dict(sorted(frontier.items())),
            }
        )
        previous = reading

    graded_index = int(source["golden_trajectory"]["graded_attempt"])
    graded = next(row for row in attempts if row["index"] == graded_index)
    return {
        "allocations": graded["allocation"],
        "extra_tokens": [],
        "train_units_requested": substrate.COMPUTE_BUDGET_UNITS,
        "graded_readout_filter": "none",
        "denominator_override": 0,
        "stop_after_point": 0,
        "reported_bpb": None,
        "notes": "reallocated off merge_depth once its marginal gain reached zero",
        "graded_attempt_index": graded_index,
        "attempts": attempts,
    }


def main(argv) -> int:
    parser = argparse.ArgumentParser(description="emit the OER-14 reference submission")
    parser.add_argument("--emit", default="-", help="path to write the submission JSON to")
    args = parser.parse_args(argv)
    payload = json.dumps(submission_document(), indent=2, sort_keys=True) + "\n"
    if args.emit == "-":
        sys.stdout.write(payload)
    else:
        target = Path(args.emit)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
