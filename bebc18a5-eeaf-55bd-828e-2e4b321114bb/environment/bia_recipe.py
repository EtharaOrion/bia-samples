"""Recipe normalization, frozen-axis refusal and fingerprinting. Agent-visible copy.

A recipe is a document over a closed key set. Normalization fills defaults,
clamps every numeric field into its bound interval, drops keys the schema does
not name, and REFUSES any key naming a frozen axis. The refusal is recorded
rather than silently dropped, because a frozen-axis write that vanishes is a
frozen-axis write nobody is graded for.

The fingerprint is a digest over the normalized recipe, so two attempts that
differ only in a dropped unknown key carry the same fingerprint and cannot both
count toward a consolidation that is supposed to require distinct recipes.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

SCHEMA_PATH = pathlib.Path(__file__).resolve().parent / "recipe_schema.json"


def load_schema(path: pathlib.Path | None = None) -> dict:
    return json.loads((path or SCHEMA_PATH).read_text(encoding="utf-8"))


def _coerce(spec: dict, value, shape: dict, name: str):
    kind = spec["type"]
    if kind == "enum":
        return value if value in spec["values"] else spec["default"]
    if kind == "int":
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = spec["default"]
        if number is None:
            number = int(shape[spec["max_from_shape"]])
        low = int(spec["min"])
        high = spec["max"]
        if high is None:
            high = int(shape[spec["max_from_shape"]])
        return max(low, min(int(high), number))
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(spec["default"])
    if number != number:
        number = float(spec["default"])
    return max(float(spec["min"]), min(float(spec["max"]), number))


def normalize(raw, shape: dict, schema: dict | None = None) -> dict:
    """Return the normalized recipe plus what the raw document tried that it may not.

    `frozen_axis_writes` is the graded one: naming a frozen axis is a graded
    failure, not a typo. `unknown_keys` is recorded and is not graded, because a
    key the schema never named reaches nothing.
    """
    schema = schema or load_schema()
    fields = schema["fields"]
    frozen = set(schema["frozen_axis_keys"])
    document = raw if isinstance(raw, dict) else {}
    recipe = {}
    for name, spec in fields.items():
        recipe[name] = _coerce(spec, document.get(name, spec["default"]), shape, name)
    frozen_writes = sorted(key for key in document if key in frozen)
    unknown = sorted(key for key in document if key not in fields and key not in frozen)
    return {
        "recipe": recipe,
        "frozen_axis_writes": frozen_writes,
        "unknown_keys": unknown,
        "fingerprint": fingerprint(recipe),
    }


def fingerprint(recipe: dict) -> str:
    payload = json.dumps(recipe, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def eval_points(shape: dict, max_steps: int) -> list:
    stride = int(shape["eval_stride"])
    return [step for step in range(stride, int(max_steps) + 1, stride)]
