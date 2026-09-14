#!/usr/bin/env python3
"""The interface your submission implements. Frozen: read it, do not modify it.

Write `/workspace/submission/recipe.py` exposing two names:

    weight(doc) -> float     the retention weight for one document, any real number.
                             The harness quantises it as clip(round(w * 8), 0, 8).
    plan()      -> dict      your declared curation plan. Declared, never graded as such:
                             the token budget is graded as the harness counted it fed.

`doc` is one row of `raw_pool.json` or `probe_pool.json`:

    {"id": "r000", "bucket": "web", "len_tokens": 704, "dup_class": 2,
     "quality_decile": 7, "ppl_decile": 3, "lang": "en"}

Buckets are exactly: web, code, math, encyclopedic, dialogue, legal.

Your module is executed ALONE, in a fresh temporary directory, as its own session leader,
under a five-key environment allowlist, with its process group killed afterwards. It cannot
read the grader, the fixtures, the reference or the held-out split, and nothing it prints
reaches the reward. Import only the standard library; there is no bundle path beside it.

The example below is a deliberately poor recipe, kept only so the interface has a runnable
shape. It is not a baseline and it is not a hint.
"""
from __future__ import annotations

BUCKETS = ("web", "code", "math", "encyclopedic", "dialogue", "legal")


def weight(doc):
    """Keep everything at equal weight. Poor, and offered only as a shape."""
    return 1.0 if doc["bucket"] in BUCKETS else 0.0


def plan():
    """Declare the plan. The harness grades tokens as fed, never as declared."""
    return {
        "recipe_id": "example-keep-everything",
        "declared_token_budget": 419430400,
        "form": "flat",
    }


if __name__ == "__main__":
    import json
    import sys

    docs = json.load(open(sys.argv[1], "r", encoding="utf-8"))["docs"]
    vector = [max(0, min(8, int(round(weight(doc) * 8.0)))) for doc in docs]
    print(json.dumps({"selection_vector": vector, "plan": plan()}))
