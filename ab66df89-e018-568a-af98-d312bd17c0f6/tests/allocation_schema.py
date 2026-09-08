#!/usr/bin/env python3
"""The compute-allocation schema, and the only place its bounds are written down.

BYTE-IDENTICAL on the agent surface and inside the verifier; tests/Dockerfile asserts
that with `cmp` at build time. An allocation that `validate` accepts here is an
allocation the verifier will train, so a submission never has to guess what will be
accepted -- including whether it fits the FLOP budget, which is checked here by the
same arithmetic the verifier charges it with.

This module is a HYGIENE GATE and nothing else. It answers one question -- is this a
well-formed allocation inside the declared budget -- and its answer is yes or no. It
never returns a quality, never returns a partial credit, and no number it produces
reaches the reward. An inadmissible allocation is REFUSED with a machine-readable
reason at reward 0.0; an admissible one is trained and scored purely on the
validation loss it produces.
"""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA_ID = "oer16-compute-allocation/v1"

# Gradient accumulation is bounded so that an allocation cannot express an optimizer
# step so large that the run degenerates into a handful of updates.
ALLOWED_GRAD_ACCUM = (1, 2, 4, 8, 16, 32)

# A floor on the run length. Below this the model has not left its initialisation and
# the measurement stops being about how compute was allocated.
MIN_MICRO_STEPS = 64


class Refusal(Exception):
    """A machine-readable refusal. `reason` is kebab-case and reaches reward.json."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def _integer(where: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Refusal("allocation-value-not-an-integer",
                      f"{where} is {value!r}, which is not an integer")
    return value


def validate(allocation, spec, harness) -> dict:
    """Return the allocation if it is well formed and inside the FLOP budget.

    `harness` is passed in rather than imported so that this module stays a pure
    schema: the FLOP arithmetic has exactly one definition, in harness.py, and this
    gate calls it rather than restating it.
    """
    if not isinstance(allocation, dict):
        raise Refusal("allocation-not-an-object",
                      f"the submission parsed to {type(allocation).__name__}, not an object")

    unknown = sorted(set(allocation) - {"schema", "model", "micro_steps", "grad_accum", "notes"})
    if unknown:
        raise Refusal("allocation-key-unknown",
                      f"allocation carries unknown key(s) {', '.join(unknown)}")
    for key in ("schema", "model", "micro_steps", "grad_accum"):
        if key not in allocation:
            raise Refusal("allocation-key-missing", f"allocation is missing {key}")
    if allocation["schema"] != SCHEMA_ID:
        raise Refusal("allocation-schema-unrecognised",
                      f"allocation.schema is {allocation['schema']!r}, expected {SCHEMA_ID!r}")

    model = allocation["model"]
    menu = sorted(spec["model_menu"])
    if not isinstance(model, str) or model not in spec["model_menu"]:
        raise Refusal("allocation-model-not-on-menu",
                      f"allocation.model is {model!r}, not one of {menu}")

    micro_steps = _integer("allocation.micro_steps", allocation["micro_steps"])
    accum = _integer("allocation.grad_accum", allocation["grad_accum"])

    if accum not in ALLOWED_GRAD_ACCUM:
        raise Refusal("allocation-grad-accum-not-allowed",
                      f"allocation.grad_accum is {accum}, not one of {list(ALLOWED_GRAD_ACCUM)}")
    if micro_steps < MIN_MICRO_STEPS:
        raise Refusal("allocation-micro-steps-below-floor",
                      f"allocation.micro_steps is {micro_steps}, below the floor "
                      f"{MIN_MICRO_STEPS}")
    if micro_steps % accum != 0:
        raise Refusal("allocation-micro-steps-not-a-multiple-of-grad-accum",
                      f"allocation.micro_steps {micro_steps} is not a multiple of "
                      f"grad_accum {accum}, so the last optimizer step would be short")

    per = harness.micro_batch_flops(spec, model)
    spent = micro_steps * per
    budget = int(spec["budget"]["flop_budget"])
    if spent > budget:
        raise Refusal("allocation-flop-budget-overspent",
                      f"allocation.model {model} costs {per} FLOPs per micro-batch and "
                      f"{micro_steps} micro-batches spend {spent}, above the budget "
                      f"{budget}. The most this model can buy is "
                      f"{harness.max_micro_steps(spec, model)} micro-batches.")

    return allocation


def load(path, spec, harness) -> dict:
    """Read and validate an allocation from disk. Every failure is a named Refusal."""
    path = Path(path)
    if not path.is_file():
        raise Refusal("submission-absent", f"no allocation at {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise Refusal("allocation-unreadable", f"{path} could not be read: {exc}") from exc
    if not raw.strip():
        raise Refusal("allocation-unreadable", f"{path} is empty")
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise Refusal("allocation-unreadable", f"{path} is not valid JSON: {exc}") from exc
    return validate(parsed, spec, harness)
