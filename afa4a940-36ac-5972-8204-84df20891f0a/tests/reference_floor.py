"""The harness-owned reference curation recipe: the lower end of the run-local ladder.

`task.toml` binds `ladder_reference_floor` to "the harness-owned reference recipe under the
identical freeze", and this file is that recipe. It lives on the VERIFIER surface, beside
the checkers, because the verifier has to be able to train it in every run; `solution/` is
the oracle surface and is not mounted into the verifier image, so a floor that depended on
it would be unreachable exactly when it is needed.

It is held out from the agent for the same reason every checker in this directory is: it is
the bar the submission is scored against. Nothing here is executed as a submission and
nothing here reads a submission byte. It carries the `derived_interaction` rule form that
`solution/grounding.yaml` declares under `reference_recipe`, so the floor this verifier
measures and the reference the oracle installs are the same recipe evaluated twice, once as
the bar and once as the arm.
"""
from __future__ import annotations

BUCKET_WEIGHTS = {"code": 0.875, "dialogue": 0.5, "encyclopedic": 0.875,
                  "legal": 0.375, "math": 1.0, "web": 0.75}
LEN_CENTER = 900.0
LEN_SPAN = 1400.0
DUP_SCALE = 6.0
W_SIGNAL = 3.0
W_BAND = 1.0
W_TAPER = 2.0


def weight(doc):
    base = BUCKET_WEIGHTS.get(doc["bucket"], 0.0)
    signal = (doc["quality_decile"] - doc["ppl_decile"] + 9) / 18.0
    band = 1.0 - abs(doc["len_tokens"] - LEN_CENTER) / LEN_SPAN
    band = 0.0 if band < 0.0 else (1.0 if band > 1.0 else band)
    taper = 1.0 - doc["dup_class"] / DUP_SCALE
    taper = 0.0 if taper < 0.0 else taper
    blend = (W_SIGNAL * signal + W_BAND * band + W_TAPER * taper) / (W_SIGNAL + W_BAND + W_TAPER)
    return base * blend
