#!/usr/bin/env python3
"""Reference oracle: emit a decoder shape that closes most of the default-to-reference gap.

WHAT THIS IS
    The output of a search run during authoring over this exact harness, this exact
    corpus and this exact frozen budget. Every shape in SEARCH_LOG below was BUILT and
    TRAINED, and the loss beside it is the validation cross entropy that run produced on
    the split the verifier grades against. solution/grounding.md carries the same table
    with the reasoning.

WHAT THIS IS NOT
    It is NOT the verifier's private reference shape. That one lives only in
    tests/private/reference_shape.json, it is a different point of the same search, and
    it is a measurably better one. If the oracle were the reference shape, the gate's
    reference arm would be the high anchor being graded against itself and its reward
    would be 1.0 by construction rather than by measurement. It is not, and it is not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# (label, measured val loss on the graded split, num_layers, model_dim, head_dim,
#  mlp_ratio, non-embedding parameters as the constructed module counted them)
SEARCH_LOG = [
    ('L6_d384_m4', 5.97004, 6, 384, 64, 4, 10642944),
    ('L9_d320_m4', 5.97661, 9, 320, 64, 4, 11091520),
    ('L16_d256_m3', 5.99296, 16, 256, 64, 3, 10527232),
    ('L14_d256_m4', 5.99786, 14, 256, 64, 4, 11049984),
    ('L7_d384_m3', 6.01000, 7, 384, 64, 3, 10349568),
    ('L4_d512_m3', 6.02175, 4, 512, 64, 3, 10507264),
    ('L5_d512_m2', 6.02543, 5, 512, 64, 2, 10509824),
    ('L10_d320_m3', 6.03922, 10, 320, 64, 3, 10272640),
]

SHAPE = {
    "schema": "oer-nanogpt-shape/v1",
    "notes": "Search finding: the same block-stack budget buys more from depth than from width at the wide end of the band, so the shipped wide-and-shallow shape is beatable, and the gain is NOT monotone in depth -- past the middle of the band the narrower widths give it back and the deepest points measured are worse than the shipped default. This shape took the first half of that finding and went one step further into depth than the measured optimum.",
    "shape": {
        "num_layers": 9,
        "model_dim": 320,
        "head_dim": 64,
        "mlp_ratio": 4
    }
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/workspace/submission/shape.json")
    parser.add_argument("--show-search", action="store_true")
    args = parser.parse_args()
    if args.show_search:
        for label, loss, L, d, h, m, n in SEARCH_LOG:
            print(f"  {loss:.5f}  {label:<18} L={L} d={d} head={h} mlp={m}  {n} params")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(SHAPE, indent=2) + "\n", encoding="utf-8")
    print(f"[reference] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
