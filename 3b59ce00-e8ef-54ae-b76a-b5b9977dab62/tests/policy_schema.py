#!/usr/bin/env python3
"""The policy schema, and the only place its bounds are written down.

BYTE-IDENTICAL on the agent surface and inside the verifier; tests/Dockerfile asserts
that with `cmp` at build time. A policy that `validate` accepts here is a policy the
verifier will train with, so a submission never has to guess what will be accepted.

This module is a HYGIENE GATE and nothing else. It answers one question -- is this a
well-formed policy inside the declared bounds -- and its answer is yes or no. It never
returns a quality, never returns a partial credit, and no number it produces reaches
the reward. A malformed policy is REFUSED with a machine-readable reason at reward 0.0;
a well-formed one is trained and scored purely on the validation cross entropy it
produces.

WHAT A POLICY IS
    Seven numbers. Four decoupled weight decays, one per parameter role. Two terms of
    the training objective that are not in the reading: label smoothing and a
    log-partition penalty. And one cap on the logits, which is part of the trained
    model and is therefore also part of the reading.

WHAT A POLICY IS NOT
    It is not an optimizer and it is not a schedule. Both are frozen in
    frozen/task_spec.json and this schema rejects any attempt to name them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

SCHEMA_ID = "oer-nanogpt-policy/v1"

# name -> (low, high, low_is_inclusive)
DECAY_BOUNDS = {
    "wd_embed": (0.0, 0.5, True),
    "wd_hidden": (0.0, 0.5, True),
    "wd_head": (0.0, 0.5, True),
    "wd_scalar": (0.0, 0.5, True),
}

OBJECTIVE_BOUNDS = {
    # Smoothing is a training-time term only; the reading is unsmoothed, so the whole
    # top of this range is a real and available way to make the reading worse.
    "label_smoothing": (0.0, 0.3, True),
    # The log-partition penalty. Small values steady the output scale; large ones
    # crush it.
    "z_loss": (0.0, 0.02, True),
    # The cap on the logits. Below the floor the distribution cannot become confident
    # at all; above the ceiling the cap is effectively absent.
    "logit_softcap": (2.0, 64.0, True),
}

POLICY_KEYS = {"schema", "decay", "objective", "notes"}


class Refusal(Exception):
    """A machine-readable refusal. `reason` is kebab-case and reaches reward.json."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def _number(where: str, key: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Refusal("policy-value-not-a-number",
                      f"{where}.{key} is {value!r}, which is not a number")
    value = float(value)
    if not math.isfinite(value):
        raise Refusal("policy-value-not-a-number", f"{where}.{key} is {value!r}, which is not finite")
    return value


def _bounded(where: str, table: dict, block) -> None:
    if not isinstance(block, dict):
        raise Refusal("policy-block-not-an-object",
                      f"{where} is {type(block).__name__}, not an object")
    unknown = sorted(set(block) - set(table))
    if unknown:
        raise Refusal("policy-key-unknown", f"{where} carries unknown key(s) {', '.join(unknown)}")
    missing = sorted(set(table) - set(block))
    if missing:
        raise Refusal("policy-key-missing", f"{where} is missing {', '.join(missing)}")
    for key, (low, high, low_inclusive) in table.items():
        value = _number(where, key, block[key])
        ok_low = value >= low if low_inclusive else value > low
        if not ok_low or value > high:
            edge = "[" if low_inclusive else "("
            raise Refusal("policy-value-out-of-bounds",
                          f"{where}.{key} is {value!r}, outside {edge}{low}, {high}]")


def validate(policy) -> dict:
    """Return the policy if it is well formed and in bounds. Otherwise raise Refusal."""
    if not isinstance(policy, dict):
        raise Refusal("policy-not-an-object",
                      f"the submission parsed to {type(policy).__name__}, not an object")
    unknown = sorted(set(policy) - POLICY_KEYS)
    if unknown:
        raise Refusal("policy-key-unknown", f"policy carries unknown key(s) {', '.join(unknown)}")
    for key in ("schema", "decay", "objective"):
        if key not in policy:
            raise Refusal("policy-key-missing", f"policy is missing {key}")
    if policy["schema"] != SCHEMA_ID:
        raise Refusal("policy-schema-unrecognised",
                      f"policy.schema is {policy['schema']!r}, expected {SCHEMA_ID!r}")
    _bounded("policy.decay", DECAY_BOUNDS, policy["decay"])
    _bounded("policy.objective", OBJECTIVE_BOUNDS, policy["objective"])
    return policy


def load(path) -> dict:
    """Read and validate a policy from disk. Every failure is a named Refusal."""
    path = Path(path)
    if not path.is_file():
        raise Refusal("submission-absent", f"no policy at {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise Refusal("policy-unreadable", f"{path} could not be read: {exc}") from exc
    if not raw.strip():
        raise Refusal("policy-unreadable", f"{path} is empty")
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise Refusal("policy-unreadable", f"{path} is not valid JSON: {exc}") from exc
    return validate(parsed)
