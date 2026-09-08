#!/usr/bin/env python3
"""The selection schema. A GATE, never a grade.

This module answers exactly one question -- is this a well-formed retention selection
inside the declared storage budget -- and it answers it with a `Refusal` carrying a
machine-readable reason. It never scales a reward, never awards partial credit for a
nearly-valid selection, and never repairs one. A selection that does not validate here
is refused at 0.0 with its reason, and a selection that does validate is combined and
evaluated exactly as written.

This file is BYTE-IDENTICAL on the agent surface and inside the verifier, enforced by
`cmp` at verifier build time, so a selection the agent can validate locally is a
selection the verifier will accept.
"""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA_ID = "oer22-selection/v1"

TOP_LEVEL_REQUIRED = ("schema", "keep", "weights")
TOP_LEVEL_OPTIONAL = ("notes",)

# The weights must sum to one to this tolerance. Tight enough that a rescaling cannot
# hide in it, loose enough that a solver writing six decimal places is not refused.
SUM_TOLERANCE = 1e-6


class Refusal(Exception):
    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


def _spec():
    from pathlib import Path as _P
    with (_P(__file__).resolve().parent / "frozen" / "task_spec.json").open(
            "r", encoding="utf-8") as fh:
        return json.load(fh)


def validate(selection, spec: dict | None = None) -> dict:
    spec = spec if spec is not None else _spec()
    snaps = spec["snapshots"]
    ids = set(int(i) for i in snaps["ids"])
    budget = int(snaps["storage_budget"])

    if not isinstance(selection, dict):
        raise Refusal("selection-not-an-object",
                      f"the submission parsed as {type(selection).__name__}, not an object")
    if selection.get("schema") != SCHEMA_ID:
        raise Refusal("selection-schema-unrecognised",
                      f"schema is {selection.get('schema')!r}, this verifier reads {SCHEMA_ID!r}")
    for key in TOP_LEVEL_REQUIRED:
        if key not in selection:
            raise Refusal("selection-key-missing", f"required key {key!r} is absent")
    for key in selection:
        if key not in TOP_LEVEL_REQUIRED + TOP_LEVEL_OPTIONAL:
            raise Refusal("selection-key-unknown", f"key {key!r} is not part of {SCHEMA_ID}")

    keep = selection["keep"]
    weights = selection["weights"]
    if not isinstance(keep, list) or not keep:
        raise Refusal("selection-keep-not-a-list",
                      "keep must be a non-empty list of snapshot ids")
    if not isinstance(weights, list):
        raise Refusal("selection-weights-not-a-list", "weights must be a list of numbers")
    if len(keep) != len(weights):
        raise Refusal("selection-weights-length-mismatch",
                      f"keep names {len(keep)} snapshots and weights carries {len(weights)} numbers")

    # The storage budget. This is the constraint that makes the task a selection.
    if len(keep) > budget:
        raise Refusal("selection-storage-budget-exceeded",
                      f"keep names {len(keep)} snapshots and the retention budget is {budget}")

    clean_ids = []
    for item in keep:
        if isinstance(item, bool) or not isinstance(item, int):
            raise Refusal("selection-id-not-an-integer",
                          f"snapshot id {item!r} is not an integer")
        if item not in ids:
            raise Refusal("selection-id-unknown",
                          f"snapshot id {item} is not one of the {len(ids)} the frozen run produces")
        if item in clean_ids:
            raise Refusal("selection-id-repeated",
                          f"snapshot id {item} is named more than once; a repeat is a reweighting "
                          "written as a duplicate and the storage budget counts slots, not entries")
        clean_ids.append(item)

    clean_weights = []
    for value in weights:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise Refusal("selection-weight-not-a-number", f"weight {value!r} is not a number")
        value = float(value)
        if value != value or value in (float("inf"), float("-inf")):
            raise Refusal("selection-weight-not-finite", f"weight {value!r} is not finite")
        if value < 0.0:
            raise Refusal("selection-weight-negative",
                          f"weight {value!r} is negative; a combination of snapshots is a convex "
                          "one and an extrapolation is outside this schema")
        clean_weights.append(value)

    total = sum(clean_weights)
    if abs(total - 1.0) > SUM_TOLERANCE:
        raise Refusal("selection-weights-not-normalised",
                      f"the weights sum to {total!r}, which is further than {SUM_TOLERANCE} from 1.0")

    return {
        "schema": SCHEMA_ID,
        "keep": clean_ids,
        "weights": clean_weights,
        "notes": str(selection.get("notes", "")),
    }


def load(path, spec: dict | None = None) -> dict:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        raise Refusal("submission-absent", f"no selection at {path}") from None
    except (ValueError, UnicodeDecodeError) as exc:
        raise Refusal("selection-unreadable", f"{path} is not readable JSON: {exc}") from None
    return validate(payload, spec)
