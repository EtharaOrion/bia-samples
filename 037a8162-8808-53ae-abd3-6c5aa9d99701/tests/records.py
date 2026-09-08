#!/usr/bin/env python3
"""The published-record corpus and the replay check. BOTH surfaces run this file.

BYTE-IDENTICAL on the agent surface and inside the verifier; tests/Dockerfile
asserts that with `cmp` at build time.

WHAT THIS IS FOR
    This slot is a record-displacement task. The corpus in published_records.json
    is the set of recipes that have already been published against this substrate.
    Re-submitting one of them is not a displacement of the record, it is a restatement
    of it, and the verifier REFUSES it with `record-replayed` at reward 0.0.

    That refusal is a hygiene gate and nothing else. It never scales the reward and
    it never contributes a partial credit. A recipe either is a published record or
    it is not; if it is not, it is trained and scored purely on the tokens it needed.

WHAT COUNTS AS THE SAME RECORD
    The graded fields only: `grad_clip`, every key of `optimizer`, and every key of
    `schedule`. `notes` is free text and is not part of the identity of a recipe;
    neither is `schema`. Floats are compared at `PRECISION` significant digits so
    that a published record cannot be replayed by perturbing a learning rate in the
    fourteenth decimal place.

    The comparison is deliberately NOT a similarity score. A recipe that differs from
    a published record in any graded field at this precision is a different recipe and
    is accepted. The corpus excludes exactly what it lists.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORDS_PATH = HERE / "published_records.json"

PRECISION = 6

GRADED_TOP = ("grad_clip", "grad_accum")
GRADED_BLOCKS = ("optimizer", "schedule")


def _canonical_scalar(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(f"{float(value):.{PRECISION}g}")
    return value


def canonical(recipe: dict) -> str:
    """A stable string identity for the graded half of a recipe."""
    body = {}
    for key in GRADED_TOP:
        if key in recipe:
            body[key] = _canonical_scalar(recipe[key])
    for block in GRADED_BLOCKS:
        got = recipe.get(block, {})
        if isinstance(got, dict):
            body[block] = {k: _canonical_scalar(v) for k, v in sorted(got.items())}
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def digest(recipe: dict) -> str:
    return hashlib.sha256(canonical(recipe).encode("utf-8")).hexdigest()[:16]


def load_records(path=RECORDS_PATH) -> list:
    with Path(path).open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    return doc["records"]


def find_replay(recipe: dict, records: list):
    """Return the published record this recipe restates, or None."""
    mine = canonical(recipe)
    for record in records:
        if canonical(record["recipe"]) == mine:
            return record
    return None
