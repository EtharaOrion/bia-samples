#!/usr/bin/env python3
"""The oracle. It solves the task the way the task asks it to be solved.

It opens the harness handle onto the BUILT fence state, walks the producer chain once from the
head segment to the null link, derives the admitted-kind set from the registry declarations by
the stated seal rule, and folds the four refusal tests over the crossings in chain order while
spending quota as it goes. It reads no answer file, imports nothing from `tests/`, and carries
no literal for either discovery value. Point it at a different built fence state and it produces
a different, equally correct submission.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent.parent
for candidate in ("/opt/fence", str(BUNDLE / "environment")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import fencelib  # noqa: E402

REFUSAL_MALFORMED = "malformed-crossing"
REFUSAL_UNATTESTED = "unattested-origin"
REFUSAL_NOT_ADMITTED = "kind-not-admitted"
REFUSAL_QUOTA = "quota-exhausted"
REFUSAL_REASONS = (REFUSAL_MALFORMED, REFUSAL_UNATTESTED, REFUSAL_NOT_ADMITTED, REFUSAL_QUOTA)

REQUIRED_FIELDS = {"seq", "kind", "size", "origin", "epoch", "sig"}


def well_formed(record) -> bool:
    if set(record) != REQUIRED_FIELDS:
        return False
    for key in ("seq", "size", "epoch"):
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            return False
    for key in ("kind", "origin", "sig"):
        if not isinstance(record.get(key), str):
            return False
    return int(record["size"]) > 0


def enforce(fence) -> dict:
    admitted_kinds = fence.admitted_kinds()
    admitted_set = set(admitted_kinds)
    quotas = {entry["kind"]: int(entry["quota"]) for entry in fence.entries()}
    salt = fence.salt()

    spent = {}
    verdicts = []
    refusals = {name: 0 for name in REFUSAL_REASONS}
    admitted = 0

    for record in fence.crossings():
        kind = record.get("kind")
        if not well_formed(record):
            decision, reason = "refuse", REFUSAL_MALFORMED
        elif fence.attestation(record, salt) != record["sig"]:
            decision, reason = "refuse", REFUSAL_UNATTESTED
        elif kind not in admitted_set:
            decision, reason = "refuse", REFUSAL_NOT_ADMITTED
        elif spent.get(kind, 0) >= quotas.get(kind, 0):
            decision, reason = "refuse", REFUSAL_QUOTA
        else:
            spent[kind] = spent.get(kind, 0) + 1
            decision, reason = "admit", ""

        if decision == "admit":
            admitted += 1
        else:
            refusals[reason] += 1
        verdicts.append(
            {
                "seq": record.get("seq"),
                "kind": kind if isinstance(kind, str) else None,
                "decision": decision,
                "reason": reason,
            }
        )

    return {
        "schema": "oer26.submission/v1",
        "crossings_observed": len(verdicts),
        "admitted_kinds": admitted_kinds,
        "admitted": admitted,
        "refused": len(verdicts) - admitted,
        "refusals": refusals,
        "verdicts": verdicts,
    }


def main():
    parser = argparse.ArgumentParser(description="enforce the OER-26 producer fence")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--state", default=None)
    args = parser.parse_args()

    fence = fencelib.open_fence(args.state)
    submission = enforce(fence)

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "submission.json").write_text(
        json.dumps(submission, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "crossings_observed": submission["crossings_observed"],
                "admitted_kind_count": len(submission["admitted_kinds"]),
                "admitted": submission["admitted"],
                "refused": submission["refused"],
                "refusals": submission["refusals"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
