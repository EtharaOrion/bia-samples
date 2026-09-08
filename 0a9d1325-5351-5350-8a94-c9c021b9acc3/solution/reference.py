#!/usr/bin/env python3
"""The OER-09 oracle: emit the curation chain the authoring search arrived at.

WHAT THIS IS AND IS NOT
    This file writes a filter chain. It does not train, it does not read the
    graded split, and it does not know either anchor's loss. The chain it writes
    is the OUTPUT of a search run during authoring against this exact register and
    this exact harness, and the search and its measurements are recorded in
    solution/grounding.md so the artifact is traceable to evidence rather than
    asserted.

    The chain below is NOT the verifier's private reference chain. The two were
    arrived at separately and differ in their thresholds, so the reference arm of
    the gate is never the reference chain being graded against itself.

HOW THE CHAIN WAS FOUND
    The register carries six computed features. The authoring pass grouped the
    pool by each feature in turn and found that no single feature separates the
    degraded blocks from the clean ones:

      * A block whose tokens were PERMUTED has, by construction, exactly the
        unigram statistics of the clean block it came from. Its distinct-token
        ratio, its top-1 share and its mean unigram log-probability are all
        indistinguishable from clean. Only the adjacent-pair term moves.
      * A block built by TILING a short span is locally as coherent as real text,
        so its adjacent-pair term sits inside the clean range. What collapses is
        the distinct-token ratio.
      * A block of UNIFORM ids collapses everything at once.

    So the chain needs at least two predicates from two different families, and
    the measured reward for one-feature chains is well under half.

THE STANDING CONSTRAINT
    The budget consumes 3072 blocks and the pool holds 5120. A chain that filters
    below 3072 is REFUSED with pool-underfilled, so the thresholds cannot simply
    be tightened until only the safest blocks survive. The chain below leaves 3672
    blocks standing, a margin of 600 over the budget.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CHAIN = {
    "schema": "oer-curation-filter/v1",
    "notes": (
        "Two feature families, because neither alone is sufficient. "
        "bigram_coherence >= -13.75 removes every block whose tokens were permuted "
        "(permutation leaves unigram statistics untouched and only the adjacent-pair "
        "term collapses) and every block of uniform ids. distinct_token_ratio >= 0.09 "
        "removes every block built by tiling a short span, which the bigram term "
        "cannot see because tiled real text is locally coherent. The upper bound at "
        "0.75 removes uniform-id blocks a second way and costs nothing on clean text, "
        "whose ratio never approaches it. Thresholds sit in the gap between the clean "
        "distribution and each degraded one rather than on a clean quantile, so the "
        "chain leaves 3672 blocks standing against a budget of 3072."
    ),
    "rules": [
        {"field": "bigram_coherence", "op": "ge", "value": -13.75},
        {"field": "distinct_token_ratio", "op": "ge", "value": 0.09},
        {"field": "distinct_token_ratio", "op": "le", "value": 0.75},
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(CHAIN, indent=2) + "\n", encoding="utf-8")
    print(f"[reference] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
