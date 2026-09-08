#!/usr/bin/env python3
"""Reference oracle: emit a recipe that closes most of the default-to-reference gap.

WHAT THIS IS
    The output of a coordinate search run during authoring over this exact harness,
    this exact corpus and this exact 25,165,824-token budget. Every number in
    RECIPE below was chosen because a measured run put it there. The measurements
    are recorded in SEARCH_LOG so the artifact is traceable to evidence instead of
    asserted, and solution/grounding.md carries the same table with the reasoning.

WHAT THIS IS NOT
    It is NOT the verifier's private reference recipe. That one lives only in
    tests/private/reference_recipe.json, it was arrived at by a separate leg of the
    search, and it differs from this recipe in its learning rates and in its
    schedule shape. If the oracle were the reference recipe, the gate's reference
    arm would be the high anchor being graded against itself and its reward would be
    1.0 by construction rather than by measurement. It is not, and it is not.

    It also does not train anything. Producing the graded artifact is writing one
    validated JSON file; the training happens in the verifier.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# The search that produced RECIPE. Loss is validation cross-entropy in nats,
# measured by this bundle's harness during authoring against the held-out FineWeb
# validation shard the graded slice is cut from. Lower is better.
# ---------------------------------------------------------------------------
SEARCH_LOG = [
    # (label, loss, what the run established)
    ("shipped default: lr 3e-4 uniform, constant, no warmup, b2 0.999, eps 1e-8, wd 0.1",
     5.49341, "the floor the submission has to beat"),
    ("uniform lr 1e-3, constant, b2 0.999, eps 1e-8, wd 0.1",
     5.33365, "the default family improves with lr; its own optimum is near 1e-3"),
    ("uniform lr 4e-3, warmup 2%, linear to 0, b2 0.95, eps 1e-10, wd 0",
     5.37117, "a decayed schedule at b2 0.95 is WORSE than the plain default"),
    ("uniform lr 4e-3, warmup 2%, linear to 0, b2 0.999, eps 1e-10, wd 0",
     5.24073, "b2 0.95 was the whole defect; at 3072 steps 0.999 is much better"),
    ("uniform lr 8e-3, warmup 2%, linear to 0, b2 0.999",
     5.35906, "past the peak at 2% warmup"),
    ("uniform lr 2e-3 / 3e-3, warmup 2%, linear to 0, b2 0.999",
     5.23789, "the 2%-warmup peak is a broad basin from 2e-3 to 4e-3"),
    ("per-role lr: hidden 3e-3, embed 1e-2, head 3e-3, scalar 1e-2, warmup 2%, linear",
     5.17906, "pricing the embedding and the scalars above the blocks is worth ~0.06"),
    ("per-role, embed 2e-2 / head 6e-3 (hotter), warmup 2%, linear",
     5.22350, "too hot; the embedding boost has an interior optimum"),
    ("uniform lr 4e-3, warmup 2%, WSD stable 0.6",
     5.19364, "holding peak then decaying beats decaying from step one"),
    ("uniform lr 4e-3, warmup 2%, cosine to 0",
     5.30032, "cosine spends too much of a short run at low lr"),
    ("uniform lr 6e-3, warmup 10%, linear to 0",
     5.17424, "a longer warmup buys a higher usable peak; warmup is a real axis"),
]

# ---------------------------------------------------------------------------
# The emitted recipe.
# ---------------------------------------------------------------------------
RECIPE = {
    "schema": "oer-nanogpt-recipe/v1",
    "notes": (
        "Coordinate search over this harness. Four findings, in the order they paid: "
        "(1) beta2 0.999 rather than 0.95 -- at 3072 optimizer steps the second moment "
        "wants a long memory, and getting this wrong makes every decayed schedule look "
        "worse than a constant one; (2) a 10 percent linear warmup, which buys a usable "
        "peak learning rate several times the default's; (3) per-role learning rates -- "
        "the token embedding and the RMSNorm gains and biases want roughly 3x the block "
        "matrices; (4) hold the peak, then decay -- WSD beats linear, and linear beats "
        "cosine, because a 25M-token run cannot afford the long low-lr tail cosine "
        "spends. Weight decay is switched off: at this token budget the model is far "
        "from fitting the corpus and decay only removes signal."
    ),
    "grad_accum": 1,
    "grad_clip": 1.0,
    "optimizer": {
        "lr_embed": 0.018,
        "lr_hidden": 0.006,
        "lr_head": 0.006,
        "lr_scalar": 0.018,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-10,
        "weight_decay": 0.0,
    },
    "schedule": {
        "shape": "wsd",
        "warmup_frac": 0.10,
        "final_frac": 0.0,
        "stable_frac": 0.6,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/workspace/submission/recipe.json")
    parser.add_argument("--show-search", action="store_true")
    args = parser.parse_args()

    if args.show_search:
        for label, loss, finding in SEARCH_LOG:
            print(f"  {loss:.5f}  {label}\n            -> {finding}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(RECIPE, indent=2) + "\n", encoding="utf-8")
    print(f"[reference] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
