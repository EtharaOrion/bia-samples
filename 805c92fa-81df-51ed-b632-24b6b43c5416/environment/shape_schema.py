#!/usr/bin/env python3
"""The decoder-shape schema, and the only place its bounds are written down.

BYTE-IDENTICAL on the agent surface and inside the verifier; tests/Dockerfile asserts
that with `cmp` at build time. A shape that `validate` accepts here is a shape the
verifier will build and train, so a submission never has to guess what will be
accepted.

This module is a HYGIENE GATE and nothing else. It answers one question -- is this a
well-formed shape whose block stack lands inside the declared parameter band -- and its
answer is yes or no. It never returns a quality, never returns a partial credit, and no
number it produces reaches the reward. A malformed or out-of-band shape is REFUSED with
a machine-readable reason at reward 0.0; an admissible one is built, trained under the
frozen recipe and scored purely on the validation loss it produces.

THE PARAMETER COUNT IS COMPUTED, NOT DECLARED.
    `non_embedding_parameters` below counts the tensors the frozen decoder will
    actually allocate, from the same arithmetic the module's constructors use. It is
    checked against the count of the REAL constructed module at verifier build time,
    so this function cannot drift away from the thing it claims to measure. Nothing a
    submission writes about its own parameter count is read.
"""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA_ID = "oer-nanogpt-shape/v1"

ALLOWED_NUM_LAYERS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16)
ALLOWED_MODEL_DIM = (256, 320, 384, 448, 512)
ALLOWED_HEAD_DIM = (32, 64, 128)
ALLOWED_MLP_RATIO = (2, 3, 4)

SHAPE_KEYS = {"num_layers", "model_dim", "head_dim", "mlp_ratio"}
DOC_KEYS = {"schema", "shape", "notes"}


class Refusal(Exception):
    """A machine-readable refusal. `reason` is kebab-case and reaches reward.json."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def non_embedding_parameters(num_layers: int, model_dim: int, head_dim: int, mlp_ratio: int) -> int:
    """Exactly what the frozen decoder allocates outside the embedding and the head.

    Per block:
      q, k, v, proj : four (model_dim x model_dim) matrices, each with a model_dim bias
      mlp fc        : (mlp_ratio * model_dim x model_dim) with a (mlp_ratio*model_dim) bias
      mlp proj      : (model_dim x mlp_ratio * model_dim) with a model_dim bias
      norm1, norm2  : two model_dim RMSNorm gain vectors
    Model level:
      norm1, norm2  : two more model_dim RMSNorm gain vectors
    """
    d, m = int(model_dim), int(mlp_ratio)
    attention = 4 * (d * d + d)
    mlp = (m * d * d + m * d) + (d * m * d + d)
    norms = 2 * d
    return int(num_layers) * (attention + mlp + norms) + 2 * d


def validate(document, budget: dict) -> dict:
    """Return the document if it is well formed and in band. Otherwise raise Refusal.

    `budget` is the frozen shape_budget block, passed in rather than restated here, so
    the band has exactly one declaration and this module cannot drift from it.
    """
    if not isinstance(document, dict):
        raise Refusal("shape-not-an-object",
                      f"the submission parsed to {type(document).__name__}, not an object")
    unknown = sorted(set(document) - DOC_KEYS)
    if unknown:
        raise Refusal("shape-key-unknown", f"the document carries unknown key(s) {', '.join(unknown)}")
    for key in ("schema", "shape"):
        if key not in document:
            raise Refusal("shape-key-missing", f"the document is missing {key}")
    if document["schema"] != SCHEMA_ID:
        raise Refusal("shape-schema-unrecognised",
                      f"schema is {document['schema']!r}, expected {SCHEMA_ID!r}")

    shape = document["shape"]
    if not isinstance(shape, dict):
        raise Refusal("shape-block-not-an-object",
                      f"shape is {type(shape).__name__}, not an object")
    unknown = sorted(set(shape) - SHAPE_KEYS)
    if unknown:
        raise Refusal("shape-key-unknown", f"shape carries unknown key(s) {', '.join(unknown)}")
    missing = sorted(SHAPE_KEYS - set(shape))
    if missing:
        raise Refusal("shape-key-missing", f"shape is missing {', '.join(missing)}")

    values = {}
    for key, allowed in (("num_layers", ALLOWED_NUM_LAYERS),
                         ("model_dim", ALLOWED_MODEL_DIM),
                         ("head_dim", ALLOWED_HEAD_DIM),
                         ("mlp_ratio", ALLOWED_MLP_RATIO)):
        value = shape[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise Refusal("shape-value-not-an-integer",
                          f"shape.{key} is {value!r}, which is not an integer")
        if value not in allowed:
            raise Refusal("shape-value-not-allowed",
                          f"shape.{key} is {value}, not one of {list(allowed)}")
        values[key] = int(value)

    if values["model_dim"] % values["head_dim"] != 0:
        raise Refusal("shape-heads-do-not-divide-width",
                      f"model_dim {values['model_dim']} is not a multiple of head_dim "
                      f"{values['head_dim']}, so the width does not split into whole heads")

    target = int(budget["non_embedding_parameters"])
    tolerance = float(budget["tolerance_frac"])
    count = non_embedding_parameters(**values)
    low, high = target * (1.0 - tolerance), target * (1.0 + tolerance)
    if count < low or count > high:
        raise Refusal("shape-parameter-budget-violated",
                      f"the block stack of this shape holds {count} non-embedding parameters, "
                      f"outside the band [{int(low)}, {int(high)}] around the frozen budget "
                      f"of {target}. The budget is not rescaled to fit a shape.")
    document = dict(document)
    document["_measured_non_embedding_parameters"] = count
    return document


def load(path, budget: dict) -> dict:
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
    return validate(parsed, budget)
