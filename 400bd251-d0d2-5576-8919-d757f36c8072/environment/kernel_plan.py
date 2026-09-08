#!/usr/bin/env python3
"""The execution-plan schema. BYTE-IDENTICAL on both surfaces.

tests/Dockerfile asserts that with `cmp` at build time.

THIS MODULE IS A HYGIENE GATE AND NOTHING ELSE. It answers one question -- is this
a well-formed plan drawn from the declared option sets -- and its answer is yes or
no. It never returns a quality and no number it produces reaches the reward.

Being well formed is NOT the same as being correct. Every option here is offered
because it is a plausible implementation of the frozen computation; whether a
particular combination actually computes that function to the tolerance this task
holds is decided by the verifier, against the reference plan, on fixed weights and
a fixed batch, BEFORE anything is timed. A plan that passes this schema and fails
that check is refused with `numerical-output-diverged` and scores exactly 0.0,
however fast it was.
"""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA_ID = "oer-kernel-plan/v1"

OPTIONS = {
    "attention": ("math", "efficient", "flash"),
    "loss_chunks": (1, 2, 4, 8, 16),
    "logit_dtype": ("fp32", "bf16"),
    "rotary_dtype": ("fp32", "bf16"),
    "qk_norm_dtype": ("fp32", "bf16"),
}

KEYS = tuple(OPTIONS)


class Refusal(Exception):
    """A machine-readable refusal. `reason` is kebab-case and reaches reward.json."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def validate(doc) -> dict:
    if not isinstance(doc, dict):
        raise Refusal("plan-not-an-object",
                      f"the submission parsed to {type(doc).__name__}, not an object")
    unknown = sorted(set(doc) - ({"schema", "notes"} | set(KEYS)))
    if unknown:
        raise Refusal("plan-key-unknown", f"plan carries unknown key(s) {', '.join(unknown)}")
    if "schema" not in doc:
        raise Refusal("plan-key-missing", "plan is missing schema")
    if doc["schema"] != SCHEMA_ID:
        raise Refusal("plan-schema-unrecognised",
                      f"plan.schema is {doc['schema']!r}, expected {SCHEMA_ID!r}")
    for key in KEYS:
        if key not in doc:
            raise Refusal("plan-key-missing", f"plan is missing {key}")
        allowed = OPTIONS[key]
        value = doc[key]
        if isinstance(value, bool) or value not in allowed:
            raise Refusal("plan-option-unknown",
                          f"plan.{key} is {value!r}, not one of {list(allowed)}")
    return doc


def load(path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise Refusal("submission-absent", f"no plan at {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise Refusal("plan-unreadable", f"{path} could not be read: {exc}") from exc
    if not raw.strip():
        raise Refusal("plan-unreadable", f"{path} is empty")
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise Refusal("plan-unreadable", f"{path} is not valid JSON: {exc}") from exc
    return validate(parsed)


def as_execution(doc: dict) -> dict:
    """The plan reduced to exactly what the model reads. `notes` never gets here."""
    return {k: doc[k] for k in KEYS}
