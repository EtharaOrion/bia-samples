#!/usr/bin/env python3
"""The update-rule schema, and the only place its bounds are written down.

BYTE-IDENTICAL on the agent surface and inside the verifier; tests/Dockerfile
asserts that with `cmp` at build time. A rule that `validate` accepts here is a rule
the verifier will train, so a submission never has to guess what will be accepted.

This module is a HYGIENE GATE and nothing else. It answers one question -- is this a
well-formed update rule inside the declared bounds -- and its answer is yes or no. It
never returns a quality, never returns a partial credit, and no number it produces
reaches the reward. A malformed rule is REFUSED with a reason at reward 0.0; a
well-formed one is trained and scored purely on the validation loss it produces.

THE BOUNDS ARE WIDE ON PURPOSE. Every one of them admits rules that train badly and
rules that diverge outright. They are here to keep the harness from being handed a
NaN or an infinity, not to fence the submission into a region where every point is
safe. A rule inside these bounds that blows the model up is a wrong answer, not a
refused one, and it scores 0.0 on its measured loss like any other wrong answer.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

SCHEMA_ID = "oer-nanogpt-updaterule/v1"

# The four parameter roles. Each is priced independently and each must be present:
# leaving one out is a refusal rather than a silent inheritance from another role.
ROLES = ("embed", "hidden", "head", "scalar")

# name -> (low, high, low_is_inclusive)
RULE_BOUNDS = {
    # The magnitude of the step, before the frozen envelope multiplies it. The range
    # spans four decades because the useful magnitude of a step depends entirely on
    # which law the other six numbers select: a sign-descent step of 0.3 and an
    # Adam step of 0.3 are not remotely the same object.
    "step_size": (0.0, 1.0, False),
    # Exponential decay of the first moment. 0.0 means no first moment at all, which
    # is a legal rule and not a degenerate one.
    "decay1": (0.0, 0.9999, True),
    # Exponential decay of the second moment. Only consulted when precond_power is
    # nonzero, and 0.0 there means the raw squared gradient with no memory.
    "decay2": (0.0, 0.99999, True),
    # The exponent the second moment is raised to before it divides. 0.5 is Adam's
    # root-mean-square preconditioner, 0.0 removes preconditioning entirely, and the
    # interior of the interval is a continuum of partial preconditioners.
    "precond_power": (0.0, 1.0, True),
    # How much of the direction is replaced by the sign of the first moment. 0.0 is
    # the magnitude-carrying direction, 1.0 is pure sign descent.
    "sign_mix": (0.0, 1.0, True),
    # Decoupled, applied to the parameter and not to the gradient.
    "weight_decay": (0.0, 1.0, True),
    # The stabiliser added to the preconditioner before it divides.
    "eps": (1e-16, 1e-3, True),
}

TOP_BOUNDS = {
    # Global gradient-norm clip, applied before any role updates. 0.0 disables it.
    "grad_clip": (0.0, 10.0, True),
}


class Refusal(Exception):
    """A machine-readable refusal. `reason` is kebab-case and reaches reward.json."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def _number(where: str, key: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Refusal("rule-value-not-a-number",
                      f"{where}.{key} is {value!r}, which is not a number")
    value = float(value)
    if not math.isfinite(value):
        raise Refusal("rule-value-not-a-number", f"{where}.{key} is {value!r}, which is not finite")
    return value


def _bounded(where: str, table: dict, block: dict) -> None:
    missing = sorted(set(table) - set(block))
    if missing:
        raise Refusal("rule-key-missing", f"{where} is missing {', '.join(missing)}")
    for key, (low, high, low_inclusive) in table.items():
        value = _number(where, key, block[key])
        ok_low = value >= low if low_inclusive else value > low
        if not ok_low or value > high:
            edge = "[" if low_inclusive else "("
            raise Refusal("rule-value-out-of-bounds",
                          f"{where}.{key} is {value!r}, outside {edge}{low}, {high}]")


def _exact_keys(where: str, allowed: set, block: dict) -> None:
    if not isinstance(block, dict):
        raise Refusal("rule-block-not-an-object", f"{where} is {type(block).__name__}, not an object")
    unknown = sorted(set(block) - allowed)
    if unknown:
        raise Refusal("rule-key-unknown", f"{where} carries unknown key(s) {', '.join(unknown)}")


def validate(rule) -> dict:
    """Return the rule if it is well formed and in bounds. Otherwise raise Refusal."""
    if not isinstance(rule, dict):
        raise Refusal("rule-not-an-object",
                      f"the submission parsed to {type(rule).__name__}, not an object")

    _exact_keys("rule", {"schema", "grad_clip", "rules", "notes"}, rule)

    for key in ("schema", "grad_clip", "rules"):
        if key not in rule:
            raise Refusal("rule-key-missing", f"rule is missing {key}")

    if rule["schema"] != SCHEMA_ID:
        raise Refusal("rule-schema-unrecognised",
                      f"rule.schema is {rule['schema']!r}, expected {SCHEMA_ID!r}")

    _bounded("rule", TOP_BOUNDS, rule)

    _exact_keys("rule.rules", set(ROLES), rule["rules"])
    missing = sorted(set(ROLES) - set(rule["rules"]))
    if missing:
        raise Refusal("rule-role-missing",
                      f"rule.rules is missing the role(s) {', '.join(missing)}; every role "
                      "must carry its own law, because a role left out would have to "
                      "inherit one and this schema never guesses")
    for role in ROLES:
        where = "rule.rules." + role
        _exact_keys(where, set(RULE_BOUNDS), rule["rules"][role])
        _bounded(where, RULE_BOUNDS, rule["rules"][role])

    return rule


def load(path) -> dict:
    """Read and validate an update rule from disk. Every failure is a named Refusal."""
    path = Path(path)
    if not path.is_file():
        raise Refusal("submission-absent", f"no update rule at {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise Refusal("rule-unreadable", f"{path} could not be read: {exc}") from exc
    if not raw.strip():
        raise Refusal("rule-unreadable", f"{path} is empty")
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise Refusal("rule-unreadable", f"{path} is not valid JSON: {exc}") from exc
    return validate(parsed)
