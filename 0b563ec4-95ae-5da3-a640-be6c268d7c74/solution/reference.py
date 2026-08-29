"""GENERATED SECTION. DO NOT HAND-EDIT.

Source: solution/grounding.yaml

The reference curation recipe for slot OER-10. It is a submission, not a grader: the
harness copies this file alone into a fresh directory and executes it over the frozen
probe pool and the frozen raw pool. It reads no grader byte and reports no score.

The recipe is a DERIVED interaction rather than a gated bucket rule. Every pinned
published mixture throws documents away on threshold crossings; this one spends the
frozen token budget on a continuous interaction between document quality, model
surprisal, a length band and a duplication taper, so it retains partial mass exactly
where the pinned baselines retain none.
"""
from __future__ import annotations

BUCKET_WEIGHTS = {"code": 0.875, "dialogue": 0.5, "encyclopedic": 0.875, "legal": 0.375, "math": 1.0, "web": 0.75}
LEN_CENTER = 900.0
LEN_SPAN = 1400.0
DUP_SCALE = 6.0
W_SIGNAL = 3.0
W_BAND = 1.0
W_TAPER = 2.0

DECLARED_TOKEN_BUDGET = 419430400


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


def plan():
    """The curation plan, declared. The harness grades tokens as fed, never as declared."""
    return {
        "recipe_id": "ref-derived-interaction",
        "declared_token_budget": DECLARED_TOKEN_BUDGET,
        "form": "derived_interaction",
    }
