#!/usr/bin/env python3
"""The mixture schema, and the only place its bounds are written down.

BYTE-IDENTICAL on the agent surface and inside the verifier; tests/Dockerfile asserts
that with `cmp` at build time. A mixture that `validate` accepts here is a mixture the
verifier will train, so a submission never has to guess what will be accepted.

This module is a HYGIENE GATE and nothing else. It answers one question -- is this a
well-formed mixture over the declared shards -- and its answer is yes or no. It never
returns a quality, never returns a partial credit, and no number it produces reaches
the reward. A malformed mixture is REFUSED with a reason at reward 0.0; a well-formed
one is trained and scored purely on the validation loss it produces.

ON THE TOKEN BUDGET. There is no budget field in this schema and that is deliberate.
A weight vector says how the fixed 3072-block budget is DIVIDED, not how large it is,
so a submission has no way to ask for more tokens than the budget and the harness has
no code path that would grant it. The budget is enforced by the representation rather
than by a checker over a claim.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

SCHEMA_ID = "oer-nanogpt-mixture/v1"

# The shards a mixture may weight. Every one must be named: a shard left out would
# have to be given an implied weight, and this schema never implies one. Giving a
# shard 0.0 is how a submission says "none of this", explicitly.
SHARDS = ("shard_a", "shard_b", "shard_c", "shard_d", "shard_e", "shard_f")

ALLOWED_POLICIES = ("interleave", "sequential", "blocked")

# `block_group` is only consulted by the `blocked` policy but is required always, so
# that one document shape describes every policy. The divisors of 3072 keep the
# chunking of the 3072-block plan even.
ALLOWED_BLOCK_GROUPS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)

WEIGHT_BOUNDS = (0.0, 1.0, True)
SEED_BOUNDS = (0, 2147483647)


class Refusal(Exception):
    """A machine-readable refusal. `reason` is kebab-case and reaches reward.json."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def _number(where: str, key: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Refusal("mixture-value-not-a-number",
                      f"{where}.{key} is {value!r}, which is not a number")
    value = float(value)
    if not math.isfinite(value):
        raise Refusal("mixture-value-not-a-number",
                      f"{where}.{key} is {value!r}, which is not finite")
    return value


def _exact_keys(where: str, allowed: set, block: dict) -> None:
    if not isinstance(block, dict):
        raise Refusal("mixture-block-not-an-object",
                      f"{where} is {type(block).__name__}, not an object")
    unknown = sorted(set(block) - allowed)
    if unknown:
        raise Refusal("mixture-key-unknown",
                      f"{where} carries unknown key(s) {', '.join(unknown)}")


def validate(document) -> dict:
    """Return the document if it is well formed and in bounds. Otherwise raise Refusal."""
    if not isinstance(document, dict):
        raise Refusal("mixture-not-an-object",
                      f"the submission parsed to {type(document).__name__}, not an object")

    _exact_keys("submission", {"schema", "mixture", "order", "notes"}, document)
    for key in ("schema", "mixture", "order"):
        if key not in document:
            raise Refusal("mixture-key-missing", f"the submission is missing {key}")

    if document["schema"] != SCHEMA_ID:
        raise Refusal("mixture-schema-unrecognised",
                      f"schema is {document['schema']!r}, expected {SCHEMA_ID!r}")

    mixture = document["mixture"]
    _exact_keys("mixture", set(SHARDS), mixture)
    missing = sorted(set(SHARDS) - set(mixture))
    if missing:
        raise Refusal("mixture-shard-missing",
                      f"mixture is missing the shard(s) {', '.join(missing)}; every shard "
                      "must carry an explicit weight, and 0.0 is how a shard is excluded")
    low, high, inclusive = WEIGHT_BOUNDS
    total = 0.0
    for shard in SHARDS:
        value = _number("mixture", shard, mixture[shard])
        ok_low = value >= low if inclusive else value > low
        if not ok_low or value > high:
            raise Refusal("mixture-weight-out-of-bounds",
                          f"mixture.{shard} is {value!r}, outside [{low}, {high}]")
        total += value
    if total <= 0.0:
        raise Refusal("mixture-weights-sum-to-zero",
                      "every shard weight is 0.0, so the mixture names no tokens to train on")

    order = document["order"]
    _exact_keys("order", {"policy", "block_group", "seed"}, order)
    for key in ("policy", "block_group", "seed"):
        if key not in order:
            raise Refusal("mixture-key-missing", f"order is missing {key}")
    if order["policy"] not in ALLOWED_POLICIES:
        raise Refusal("mixture-order-policy-unknown",
                      f"order.policy is {order['policy']!r}, not one of {list(ALLOWED_POLICIES)}")
    group = order["block_group"]
    if isinstance(group, bool) or not isinstance(group, int):
        raise Refusal("mixture-block-group-not-allowed",
                      f"order.block_group is {group!r}, which is not an integer")
    if group not in ALLOWED_BLOCK_GROUPS:
        raise Refusal("mixture-block-group-not-allowed",
                      f"order.block_group is {group}, not one of {list(ALLOWED_BLOCK_GROUPS)}")
    seed = order["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise Refusal("mixture-seed-not-allowed",
                      f"order.seed is {seed!r}, which is not an integer")
    if not (SEED_BOUNDS[0] <= seed <= SEED_BOUNDS[1]):
        raise Refusal("mixture-seed-not-allowed",
                      f"order.seed is {seed}, outside [{SEED_BOUNDS[0]}, {SEED_BOUNDS[1]}]")

    return document


def load(path) -> dict:
    """Read and validate a mixture from disk. Every failure is a named Refusal."""
    path = Path(path)
    if not path.is_file():
        raise Refusal("submission-absent", f"no mixture at {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise Refusal("mixture-unreadable", f"{path} could not be read: {exc}") from exc
    if not raw.strip():
        raise Refusal("mixture-unreadable", f"{path} is empty")
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise Refusal("mixture-unreadable", f"{path} is not valid JSON: {exc}") from exc
    return validate(parsed)
