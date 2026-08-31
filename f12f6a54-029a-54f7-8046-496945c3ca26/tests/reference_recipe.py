"""GENERATED SECTION. DO NOT HAND-EDIT.

Source: solution/grounding.yaml

The HARNESS-OWNED reference curation recipe for slot OER-10, verifier side.

`tests/runner.py` trains this recipe as the reference floor arm, under the identical freeze,
inside the same run it grades a submission in. instruction.md binds `reference_floor_loss` as
"the reference recipe under the same freeze" and binds matching it to score exactly 1.0, so
the floor has to be a recipe the harness holds rather than a number anyone wrote down.

It lives under tests/ because Harbor assembles the agent surface from task.toml,
instruction.md and environment/ alone, so this file is absent from that surface by
construction rather than by a filter that could be misconfigured.

It and `solution/reference.py` are generated from the ONE `reference_recipe` block of
solution/grounding.yaml by solution/recompute.py, so the floor the verifier measures and the
recipe the oracle submits cannot drift apart into a floor the oracle cannot reach.
"""
from __future__ import annotations

BUCKET_WEIGHTS = {"code": 0.875, "dialogue": 0.5, "encyclopedic": 0.875, "legal": 0.375, "math": 1.0, "web": 0.75}
LEN_CENTER = 900.0
LEN_SPAN = 1400.0
DUP_SCALE = 6.0
W_SIGNAL = 3.0
W_BAND = 1.0
W_TAPER = 2.0

RECIPE_ID = "ref-derived-interaction"


def weight(doc):
    """The continuous retention weight this recipe assigns to one document."""
    base = BUCKET_WEIGHTS.get(doc["bucket"], 0.0)
    signal = (doc["quality_decile"] - doc["ppl_decile"] + 9) / 18.0
    band = 1.0 - abs(doc["len_tokens"] - LEN_CENTER) / LEN_SPAN
    band = 0.0 if band < 0.0 else (1.0 if band > 1.0 else band)
    taper = 1.0 - doc["dup_class"] / DUP_SCALE
    taper = 0.0 if taper < 0.0 else taper
    blend = (W_SIGNAL * signal + W_BAND * band + W_TAPER * taper) / (W_SIGNAL + W_BAND + W_TAPER)
    return base * blend
