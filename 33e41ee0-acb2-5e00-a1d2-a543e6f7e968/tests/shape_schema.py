#!/usr/bin/env python3
"""The architecture-shape schema. BYTE-IDENTICAL on both surfaces.

tests/Dockerfile asserts that with `cmp` at build time. A shape that `validate`
accepts here is a shape the verifier will instantiate and train, so a submission
never has to guess what will be accepted.

THIS MODULE IS A HYGIENE GATE AND NOTHING ELSE. It answers one question -- is this
a well-formed shape inside the declared bounds -- and its answer is yes or no. It
never returns a quality, never returns a partial credit, and no number it produces
reaches the reward. A malformed or out-of-band shape is REFUSED with a reason at
reward 0.0; a well-formed one is trained and scored purely on the validation loss
it produces.

THE PARAMETER BAND IS NOT CHECKED HERE, AND THAT IS DELIBERATE.
`validate` checks the declared integers against the declared bounds. Whether the
shape lands inside the parameter band is decided by COUNTING THE PARAMETERS OF THE
INSTANTIATED MODEL -- see `count_parameters` -- rather than by trusting any formula
written down in this file or in a submission. `parameter_band_check` is the helper
that does it, and both surfaces call it on a real module.
"""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA_ID = "oer-nanogpt-shape/v1"

KEYS = ("num_layers", "model_dim", "head_dim", "mlp_ratio")


class Refusal(Exception):
    """A machine-readable refusal. `reason` is kebab-case and reaches reward.json."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def _int(where: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Refusal("shape-value-not-an-integer",
                      f"{where} is {value!r}, which is not an integer")
    return int(value)


def validate(doc, bounds: dict) -> dict:
    """Return the shape if it is well formed and inside the declared bounds."""
    if not isinstance(doc, dict):
        raise Refusal("shape-not-an-object",
                      f"the submission parsed to {type(doc).__name__}, not an object")

    unknown = sorted(set(doc) - ({"schema", "notes"} | set(KEYS)))
    if unknown:
        raise Refusal("shape-key-unknown", f"shape carries unknown key(s) {', '.join(unknown)}")

    if "schema" not in doc:
        raise Refusal("shape-key-missing", "shape is missing schema")
    if doc["schema"] != SCHEMA_ID:
        raise Refusal("shape-schema-unrecognised",
                      f"shape.schema is {doc['schema']!r}, expected {SCHEMA_ID!r}")
    for key in KEYS:
        if key not in doc:
            raise Refusal("shape-key-missing", f"shape is missing {key}")

    layers = _int("shape.num_layers", doc["num_layers"])
    dim = _int("shape.model_dim", doc["model_dim"])
    head = _int("shape.head_dim", doc["head_dim"])
    ratio = _int("shape.mlp_ratio", doc["mlp_ratio"])

    lo, hi = bounds["num_layers"]
    if not lo <= layers <= hi:
        raise Refusal("shape-value-out-of-bounds",
                      f"shape.num_layers is {layers}, outside [{lo}, {hi}]")
    lo, hi = bounds["model_dim"]
    if not lo <= dim <= hi:
        raise Refusal("shape-value-out-of-bounds",
                      f"shape.model_dim is {dim}, outside [{lo}, {hi}]")
    mult = int(bounds["model_dim_multiple_of"])
    if dim % mult:
        raise Refusal("shape-model-dim-not-a-multiple",
                      f"shape.model_dim is {dim}, which is not a multiple of {mult}")
    if head not in bounds["head_dim_allowed"]:
        raise Refusal("shape-head-dim-not-allowed",
                      f"shape.head_dim is {head}, not one of {list(bounds['head_dim_allowed'])}")
    if ratio not in bounds["mlp_ratio_allowed"]:
        raise Refusal("shape-mlp-ratio-not-allowed",
                      f"shape.mlp_ratio is {ratio}, not one of {list(bounds['mlp_ratio_allowed'])}")
    if dim % head:
        raise Refusal("shape-heads-not-integral",
                      f"shape.model_dim {dim} is not divisible by shape.head_dim {head}")
    heads = dim // head
    if heads < int(bounds["min_num_heads"]):
        raise Refusal("shape-too-few-heads",
                      f"model_dim // head_dim is {heads}, the floor is {bounds['min_num_heads']}")
    return doc


def load(path, bounds: dict) -> dict:
    """Read and validate a shape from disk. Every failure is a named Refusal."""
    path = Path(path)
    if not path.is_file():
        raise Refusal("submission-absent", f"no shape at {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise Refusal("shape-unreadable", f"{path} could not be read: {exc}") from exc
    if not raw.strip():
        raise Refusal("shape-unreadable", f"{path} is empty")
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise Refusal("shape-unreadable", f"{path} is not valid JSON: {exc}") from exc
    return validate(parsed, bounds)


def count_parameters(module) -> int:
    """The authority on a shape's size: the instantiated model's own tensors."""
    return sum(p.numel() for p in module.parameters())


def parameter_band_check(count: int, band: dict) -> None:
    """Raise a named Refusal when an instantiated shape is outside the band."""
    lo, hi = int(band["min"]), int(band["max"])
    if count < lo:
        raise Refusal("shape-under-parameter-band",
                      f"the instantiated shape holds {count} parameters, below the "
                      f"floor of {lo}. Spend the budget: an undersized model is not "
                      f"a cheaper answer, it is a different task.")
    if count > hi:
        raise Refusal("shape-over-parameter-band",
                      f"the instantiated shape holds {count} parameters, above the "
                      f"ceiling of {hi}.")
