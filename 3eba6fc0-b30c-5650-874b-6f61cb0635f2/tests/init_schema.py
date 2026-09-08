#!/usr/bin/env python3
"""The initialisation schema, and the only place its bounds are written down.

BYTE-IDENTICAL on the agent surface and inside the verifier; tests/Dockerfile asserts
that with `cmp` at build time. An initialisation that `validate` accepts here is one
the verifier will train, so a submission never has to guess what will be accepted.

This module is a HYGIENE GATE and nothing else. It answers one question -- is this a
well-formed initialisation inside the declared bounds -- and its answer is yes or no.
It never returns a quality, never returns a partial credit, and no number it produces
reaches the reward. A malformed document is REFUSED with a reason at reward 0.0; a
well-formed one is trained and scored purely on the validation loss it produces.

THE BOUNDS ADMIT BAD ANSWERS ON PURPOSE. Every standard deviation may be 0.0, which
collapses that role to exactly zero, and may be as large as 1.0, which is fifty times
the conventional transformer initialisation and will not train. `norm_gain` may be
0.0, which zeroes the output of every RMSNorm and stops the network computing
anything at all. These are legal submissions and they are wrong answers, scored on the
loss they produce like any other.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

SCHEMA_ID = "oer-nanogpt-init/v1"

# The seven scaled roles. Every one must be present: a role left out would have to be
# initialised by some default the submission never chose.
SCALED_ROLES = ("embed", "attn_qkv", "attn_proj", "mlp_fc", "mlp_proj", "head", "bias")

# name -> (low, high, low_is_inclusive)
INIT_BOUNDS = {
    # The standard deviation each role's frozen unit draw is multiplied by. 0.0 is
    # legal and is the only way to say "start this role at exactly zero", which for
    # the output projection is a real and well-known choice rather than a degenerate
    # one.
    "embed_std": (0.0, 1.0, True),
    "attn_qkv_std": (0.0, 1.0, True),
    "attn_proj_std": (0.0, 1.0, True),
    "mlp_fc_std": (0.0, 1.0, True),
    "mlp_proj_std": (0.0, 1.0, True),
    "head_std": (0.0, 1.0, True),
    "bias_std": (0.0, 1.0, True),
    # The constant written into every RMSNorm gain. Not a scale on a draw: a gain is
    # not a random quantity, so this is the value itself.
    "norm_gain": (0.0, 4.0, True),
    # The exponent in the residual taper. attn_proj_std and mlp_proj_std are divided
    # by (2 * num_layers) ** this. 0.0 switches the taper off; 0.5 is the classic
    # 1/sqrt(2L). Exposed as a continuous exponent because the right amount of taper
    # at six layers is not obviously either endpoint.
    "residual_depth_power": (0.0, 1.0, True),
}


class Refusal(Exception):
    """A machine-readable refusal. `reason` is kebab-case and reaches reward.json."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def _number(where: str, key: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Refusal("init-value-not-a-number",
                      f"{where}.{key} is {value!r}, which is not a number")
    value = float(value)
    if not math.isfinite(value):
        raise Refusal("init-value-not-a-number",
                      f"{where}.{key} is {value!r}, which is not finite")
    return value


def _bounded(where: str, table: dict, block: dict) -> None:
    missing = sorted(set(table) - set(block))
    if missing:
        raise Refusal("init-key-missing", f"{where} is missing {', '.join(missing)}")
    for key, (low, high, low_inclusive) in table.items():
        value = _number(where, key, block[key])
        ok_low = value >= low if low_inclusive else value > low
        if not ok_low or value > high:
            edge = "[" if low_inclusive else "("
            raise Refusal("init-value-out-of-bounds",
                          f"{where}.{key} is {value!r}, outside {edge}{low}, {high}]")


def _exact_keys(where: str, allowed: set, block: dict) -> None:
    if not isinstance(block, dict):
        raise Refusal("init-block-not-an-object",
                      f"{where} is {type(block).__name__}, not an object")
    unknown = sorted(set(block) - allowed)
    if unknown:
        raise Refusal("init-key-unknown", f"{where} carries unknown key(s) {', '.join(unknown)}")


def validate(document) -> dict:
    """Return the document if it is well formed and in bounds. Otherwise raise Refusal."""
    if not isinstance(document, dict):
        raise Refusal("init-not-an-object",
                      f"the submission parsed to {type(document).__name__}, not an object")

    _exact_keys("submission", {"schema", "init", "notes"}, document)
    for key in ("schema", "init"):
        if key not in document:
            raise Refusal("init-key-missing", f"the submission is missing {key}")

    if document["schema"] != SCHEMA_ID:
        raise Refusal("init-schema-unrecognised",
                      f"schema is {document['schema']!r}, expected {SCHEMA_ID!r}")

    block = document["init"]
    _exact_keys("init", set(INIT_BOUNDS), block)
    missing = sorted(r + "_std" for r in SCALED_ROLES if r + "_std" not in block)
    if missing:
        raise Refusal("init-role-missing",
                      f"init is missing the role scale(s) {', '.join(missing)}; every role "
                      "must carry its own scale, because a role left out would have to "
                      "inherit one and this schema never guesses")
    _bounded("init", INIT_BOUNDS, block)
    return document


def load(path) -> dict:
    """Read and validate an initialisation from disk. Every failure is a named Refusal."""
    path = Path(path)
    if not path.is_file():
        raise Refusal("submission-absent", f"no initialisation at {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise Refusal("init-unreadable", f"{path} could not be read: {exc}") from exc
    if not raw.strip():
        raise Refusal("init-unreadable", f"{path} is empty")
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise Refusal("init-unreadable", f"{path} is not valid JSON: {exc}") from exc
    return validate(parsed)
