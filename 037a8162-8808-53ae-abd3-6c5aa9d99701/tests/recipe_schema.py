#!/usr/bin/env python3
"""The recipe schema, and the only place its bounds are written down.

BYTE-IDENTICAL on the agent surface and inside the verifier; tests/Dockerfile
asserts that with `cmp` at build time. A recipe that `validate` accepts here is a
recipe the verifier will train, so a submission never has to guess what will be
accepted.

This module is a HYGIENE GATE and nothing else. It answers one question -- is this
a well-formed recipe inside the declared bounds -- and its answer is yes or no. It
never returns a quality, never returns a partial credit, and no number it produces
reaches the reward. A malformed recipe is REFUSED with a reason at reward 0.0; a
well-formed one is trained and scored purely on the validation loss it produces.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

SCHEMA_ID = "oer-displacement-recipe/v1"

# grad_accum is FROZEN AT 1 in this slot and the schema admits nothing else.
#
# That is a measurement, not a preference. On this substrate and at this ceiling the
# accumulation axis is monotone -- accumulating 1 beats 2 beats 4 beats 8 beats 16 on
# final loss, and every ramp between them loses to constant 1 -- so offering it as a
# choice would add a dimension whose answer is already known. The recipe still has to
# declare it, because a recipe that does not say what it accumulates is not a complete
# recipe, and a value other than 1 is refused rather than silently coerced.
ALLOWED_GRAD_ACCUM = (1,)

ALLOWED_SHAPES = ("constant", "linear", "cosine", "wsd")

# name -> (low, high, low_is_inclusive)
OPTIMIZER_BOUNDS = {
    "lr_embed": (0.0, 0.1, False),
    "lr_hidden": (0.0, 0.1, False),
    "lr_head": (0.0, 0.1, False),
    "lr_scalar": (0.0, 0.1, False),
    "beta1": (0.0, 0.999, True),
    "beta2": (0.5, 0.99999, True),
    "eps": (1e-16, 1e-4, True),
    "weight_decay": (0.0, 1.0, True),
}

SCHEDULE_BOUNDS = {
    "warmup_frac": (0.0, 0.5, True),
    "final_frac": (0.0, 1.0, True),
    "stable_frac": (0.0, 1.0, True),
}

TOP_BOUNDS = {
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
        raise Refusal("recipe-value-not-a-number",
                      f"{where}.{key} is {value!r}, which is not a number")
    value = float(value)
    if not math.isfinite(value):
        raise Refusal("recipe-value-not-a-number", f"{where}.{key} is {value!r}, which is not finite")
    return value


def _bounded(where: str, table: dict, block: dict) -> None:
    missing = sorted(set(table) - set(block))
    if missing:
        raise Refusal("recipe-key-missing", f"{where} is missing {', '.join(missing)}")
    for key, (low, high, low_inclusive) in table.items():
        value = _number(where, key, block[key])
        ok_low = value >= low if low_inclusive else value > low
        if not ok_low or value > high:
            edge = "[" if low_inclusive else "("
            raise Refusal("recipe-value-out-of-bounds",
                          f"{where}.{key} is {value!r}, outside {edge}{low}, {high}]")


def _exact_keys(where: str, allowed: set, block: dict) -> None:
    if not isinstance(block, dict):
        raise Refusal("recipe-block-not-an-object", f"{where} is {type(block).__name__}, not an object")
    unknown = sorted(set(block) - allowed)
    if unknown:
        raise Refusal("recipe-key-unknown", f"{where} carries unknown key(s) {', '.join(unknown)}")


def validate(recipe) -> dict:
    """Return the recipe if it is well formed and in bounds. Otherwise raise Refusal."""
    if not isinstance(recipe, dict):
        raise Refusal("recipe-not-an-object",
                      f"the submission parsed to {type(recipe).__name__}, not an object")

    _exact_keys("recipe", {"schema", "grad_accum", "grad_clip", "optimizer", "schedule", "notes"}, recipe)

    for key in ("schema", "grad_accum", "grad_clip", "optimizer", "schedule"):
        if key not in recipe:
            raise Refusal("recipe-key-missing", f"recipe is missing {key}")

    if recipe["schema"] != SCHEMA_ID:
        raise Refusal("recipe-schema-unrecognised",
                      f"recipe.schema is {recipe['schema']!r}, expected {SCHEMA_ID!r}")

    accum = recipe["grad_accum"]
    if isinstance(accum, bool) or not isinstance(accum, int):
        raise Refusal("recipe-grad-accum-not-allowed",
                      f"recipe.grad_accum is {accum!r}, which is not an integer")
    if accum not in ALLOWED_GRAD_ACCUM:
        raise Refusal("recipe-grad-accum-not-allowed",
                      f"recipe.grad_accum is {accum}, not one of {list(ALLOWED_GRAD_ACCUM)}")

    _bounded("recipe", TOP_BOUNDS, recipe)

    _exact_keys("recipe.optimizer", set(OPTIMIZER_BOUNDS), recipe["optimizer"])
    _bounded("recipe.optimizer", OPTIMIZER_BOUNDS, recipe["optimizer"])

    _exact_keys("recipe.schedule", set(SCHEDULE_BOUNDS) | {"shape"}, recipe["schedule"])
    if "shape" not in recipe["schedule"]:
        raise Refusal("recipe-key-missing", "recipe.schedule is missing shape")
    if recipe["schedule"]["shape"] not in ALLOWED_SHAPES:
        raise Refusal("recipe-schedule-shape-unknown",
                      f"recipe.schedule.shape is {recipe['schedule']['shape']!r}, "
                      f"not one of {list(ALLOWED_SHAPES)}")
    _bounded("recipe.schedule", SCHEDULE_BOUNDS, recipe["schedule"])

    return recipe


def load(path) -> dict:
    """Read and validate a recipe from disk. Every failure is a named Refusal."""
    path = Path(path)
    if not path.is_file():
        raise Refusal("submission-absent", f"no recipe at {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise Refusal("recipe-unreadable", f"{path} could not be read: {exc}") from exc
    if not raw.strip():
        raise Refusal("recipe-unreadable", f"{path} is empty")
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise Refusal("recipe-unreadable", f"{path} is not valid JSON: {exc}") from exc
    return validate(parsed)
