#!/usr/bin/env python3
"""The training-data-plan schema, and the only place its bounds are written down.

BYTE-IDENTICAL on the agent surface and inside the verifier; tests/Dockerfile asserts
that with `cmp` at build time. A plan that `validate` accepts here is a plan the
verifier will assemble and train, so a submission never has to guess what will be
accepted -- including whether its draws total the budget exactly, which is checked
here by the same arithmetic the harness assembles with.

This module is a HYGIENE GATE and nothing else. It answers one question -- is this a
well-formed plan that spends the frozen budget exactly -- and its answer is yes or
no. It never returns a quality, never returns a partial credit, and no number it
produces reaches the reward. A malformed plan is REFUSED with a machine-readable
reason at reward 0.0; a well-formed one is trained and scored purely on the
validation loss it produces.
"""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA_ID = "oer08-data-plan/v1"

# A plan is an ordered list of draws. The cap is generous -- the point of the task is
# the allocation, not the bookkeeping -- but it is finite so that assembly is bounded.
MAX_DRAWS = 32


class Refusal(Exception):
    """A machine-readable refusal. `reason` is kebab-case and reaches reward.json."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def _integer(where: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Refusal("plan-value-not-an-integer",
                      f"{where} is {value!r}, which is not an integer")
    return value


def validate(plan, spec) -> dict:
    """Return the plan if it is well formed and spends the budget exactly."""
    if not isinstance(plan, dict):
        raise Refusal("plan-not-an-object",
                      f"the submission parsed to {type(plan).__name__}, not an object")

    unknown = sorted(set(plan) - {"schema", "draws", "notes"})
    if unknown:
        raise Refusal("plan-key-unknown", f"plan carries unknown key(s) {', '.join(unknown)}")
    for key in ("schema", "draws"):
        if key not in plan:
            raise Refusal("plan-key-missing", f"plan is missing {key}")
    if plan["schema"] != SCHEMA_ID:
        raise Refusal("plan-schema-unrecognised",
                      f"plan.schema is {plan['schema']!r}, expected {SCHEMA_ID!r}")

    draws = plan["draws"]
    if not isinstance(draws, list) or not draws:
        raise Refusal("plan-draws-empty",
                      "plan.draws must be a non-empty ordered list of "
                      "{source, tokens[, offset]} objects")
    if len(draws) > MAX_DRAWS:
        raise Refusal("plan-too-many-draws",
                      f"plan.draws holds {len(draws)} draws, at most {MAX_DRAWS} are allowed")

    sources = set(spec["pool"]["sources"])
    granularity = int(spec["pool"]["draw_granularity"])
    budget = int(spec["budget"]["total_train_tokens"])

    total = 0
    for i, draw in enumerate(draws):
        where = f"plan.draws[{i}]"
        if not isinstance(draw, dict):
            raise Refusal("plan-draw-not-an-object",
                          f"{where} is {type(draw).__name__}, not an object")
        extra = sorted(set(draw) - {"source", "tokens", "offset"})
        if extra:
            raise Refusal("plan-key-unknown", f"{where} carries unknown key(s) {', '.join(extra)}")
        for key in ("source", "tokens"):
            if key not in draw:
                raise Refusal("plan-key-missing", f"{where} is missing {key}")
        if draw["source"] not in sources:
            raise Refusal("plan-source-not-in-pool",
                          f"{where}.source is {draw['source']!r}, not one of {sorted(sources)}")
        tokens = _integer(f"{where}.tokens", draw["tokens"])
        if tokens <= 0 or tokens % granularity != 0:
            raise Refusal("plan-tokens-not-a-granularity-multiple",
                          f"{where}.tokens is {tokens}, which is not a positive multiple "
                          f"of the draw granularity {granularity}")
        if "offset" in draw:
            offset = _integer(f"{where}.offset", draw["offset"])
            if offset < 0 or offset % granularity != 0:
                raise Refusal("plan-offset-not-a-granularity-multiple",
                              f"{where}.offset is {offset}, which is not a non-negative "
                              f"multiple of the draw granularity {granularity}")
        total += tokens

    if total != budget:
        raise Refusal("plan-budget-not-spent-exactly",
                      f"plan.draws total {total} tokens; the frozen budget is exactly "
                      f"{budget}. A plan must spend all of it and none beyond it -- "
                      f"under-spending would be training on less compute and "
                      f"over-spending on more, and neither is the task.")

    return plan


def load(path, spec) -> dict:
    """Read and validate a plan from disk. Every failure is a named Refusal."""
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
    return validate(parsed, spec)
