#!/usr/bin/env python3
"""The learning-rate schedule schema, and the only place its bounds are written down.

BYTE-IDENTICAL on the agent surface and inside the verifier; tests/Dockerfile asserts
that with `cmp` at build time. A schedule that `validate` accepts here is a schedule
the verifier will train, so a submission never has to guess what will be accepted.

This module is a HYGIENE GATE and nothing else. It answers one question -- is this a
well-formed schedule inside the declared bounds -- and its answer is yes or no. It
never returns a quality, never returns a partial credit, and no number it produces
reaches the reward. A malformed schedule is REFUSED with a reason at reward 0.0; a
well-formed one is trained and scored purely on the validation loss it produces.

WHAT THE BOUNDS DO NOT DO. They do not fence the submission into a region where every
point is good. `warmup_frac` may be 0.0 and `final_frac` may be 1.0, which together
name the flat envelope the shipped default uses; `warmup_frac` may be 0.5, which
spends half the budget getting to the peak. Both are legal and both are bad. A legal
schedule that trains badly is a wrong answer, not a refused one.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

SCHEMA_ID = "oer-nanogpt-schedule/v1"

# The decay families. Every one of them is evaluated over the post-warmup fraction of
# the run and every one of them lands on `final_frac` at the last step, so the family
# names a PATH between the peak and the floor rather than a different endpoint.
ALLOWED_SHAPES = ("constant", "linear", "cosine", "wsd", "poly", "exp", "inv_sqrt")

# The warmup ramps. `linear` is the usual straight line from one step's worth of peak
# up to the peak; `poly` bends that line by `warmup_power`.
ALLOWED_WARMUP_SHAPES = ("linear", "poly")

# name -> (low, high, low_is_inclusive)
SCHEDULE_BOUNDS = {
    "warmup_frac": (0.0, 0.5, True),
    "warmup_power": (0.25, 4.0, True),
    "stable_frac": (0.0, 1.0, True),
    "final_frac": (0.0, 1.0, True),
    "decay_power": (0.25, 8.0, True),
}


class Refusal(Exception):
    """A machine-readable refusal. `reason` is kebab-case and reaches reward.json."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def _number(where: str, key: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Refusal("schedule-value-not-a-number",
                      f"{where}.{key} is {value!r}, which is not a number")
    value = float(value)
    if not math.isfinite(value):
        raise Refusal("schedule-value-not-a-number",
                      f"{where}.{key} is {value!r}, which is not finite")
    return value


def _bounded(where: str, table: dict, block: dict) -> None:
    missing = sorted(set(table) - set(block))
    if missing:
        raise Refusal("schedule-key-missing", f"{where} is missing {', '.join(missing)}")
    for key, (low, high, low_inclusive) in table.items():
        value = _number(where, key, block[key])
        ok_low = value >= low if low_inclusive else value > low
        if not ok_low or value > high:
            edge = "[" if low_inclusive else "("
            raise Refusal("schedule-value-out-of-bounds",
                          f"{where}.{key} is {value!r}, outside {edge}{low}, {high}]")


def _exact_keys(where: str, allowed: set, block: dict) -> None:
    if not isinstance(block, dict):
        raise Refusal("schedule-block-not-an-object",
                      f"{where} is {type(block).__name__}, not an object")
    unknown = sorted(set(block) - allowed)
    if unknown:
        raise Refusal("schedule-key-unknown",
                      f"{where} carries unknown key(s) {', '.join(unknown)}")


def validate(document) -> dict:
    """Return the document if it is well formed and in bounds. Otherwise raise Refusal."""
    if not isinstance(document, dict):
        raise Refusal("schedule-not-an-object",
                      f"the submission parsed to {type(document).__name__}, not an object")

    _exact_keys("submission", {"schema", "schedule", "notes"}, document)
    for key in ("schema", "schedule"):
        if key not in document:
            raise Refusal("schedule-key-missing", f"the submission is missing {key}")

    if document["schema"] != SCHEMA_ID:
        raise Refusal("schedule-schema-unrecognised",
                      f"schema is {document['schema']!r}, expected {SCHEMA_ID!r}")

    block = document["schedule"]
    _exact_keys("schedule", set(SCHEDULE_BOUNDS) | {"shape", "warmup_shape"}, block)

    for key in ("shape", "warmup_shape"):
        if key not in block:
            raise Refusal("schedule-key-missing", f"schedule is missing {key}")
    if block["shape"] not in ALLOWED_SHAPES:
        raise Refusal("schedule-shape-unknown",
                      f"schedule.shape is {block['shape']!r}, not one of {list(ALLOWED_SHAPES)}")
    if block["warmup_shape"] not in ALLOWED_WARMUP_SHAPES:
        raise Refusal("schedule-warmup-shape-unknown",
                      f"schedule.warmup_shape is {block['warmup_shape']!r}, "
                      f"not one of {list(ALLOWED_WARMUP_SHAPES)}")

    _bounded("schedule", SCHEDULE_BOUNDS, block)
    return document


def load(path) -> dict:
    """Read and validate a schedule from disk. Every failure is a named Refusal."""
    path = Path(path)
    if not path.is_file():
        raise Refusal("submission-absent", f"no schedule at {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise Refusal("schedule-unreadable", f"{path} could not be read: {exc}") from exc
    if not raw.strip():
        raise Refusal("schedule-unreadable", f"{path} is empty")
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise Refusal("schedule-unreadable", f"{path} is not valid JSON: {exc}") from exc
    return validate(parsed)
