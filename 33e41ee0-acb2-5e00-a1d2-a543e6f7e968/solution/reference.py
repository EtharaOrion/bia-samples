#!/usr/bin/env python3
"""The OER-04 oracle: emit the architecture shape the authoring search arrived at.

WHAT THIS IS AND IS NOT
    This file writes four integers. It does not train, it does not read the graded
    split, and it does not know either anchor's loss. The shape it writes is the
    OUTPUT of a search run during authoring against this exact substrate, and that
    search and its measurements are recorded in solution/grounding.md.

    The shape below is NOT the verifier's private reference shape. The two differ,
    and they differ in a way that is recorded rather than hidden: the search behind
    this file explored depth, residual width and MLP ratio while holding head_dim at
    64, which is the conventional choice on this architecture. The private reference
    relaxed that and found a better head width. This oracle therefore closes most of
    the gap and does not reach the bar, which is the intended shape of this reward.

WHAT THE SEARCH FOUND
    The parameter band admits 42 shapes. Their losses are NOT monotone in depth:

      2 blocks at 448 wide   -- too shallow; the widest legal stream buys only two
                                blocks and two blocks cannot compose enough.
      14 blocks at 320 wide  -- deep and thin; better than the shallow end, worse
                                than the middle, and materially slower to train.
      6 blocks at 384 wide   -- the interior optimum on this budget.

    The MLP ratio moves with depth rather than against it: at 384 wide, ratio 4 with
    6 blocks beat ratio 3 with 7 blocks and ratio 2 with 9 blocks, so trading MLP
    width for depth at fixed parameters loses here.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

SHAPE = {
    "schema": "oer-nanogpt-shape/v1",
    "notes": (
        "Six blocks at a residual width of 384 with a 4x MLP, 49,326,720 parameters. "
        "The interior of the depth/width frontier: the widest legal stream buys only "
        "two blocks and loses badly, and fourteen blocks at 320 lose too and cost "
        "nearly twice the wall clock. Trading MLP width for depth at fixed parameters "
        "also loses -- ratio 3 at 7 blocks and ratio 2 at 9 blocks are both behind "
        "ratio 4 at 6. head_dim is left at 64, the conventional split for this "
        "architecture, which is the axis this search did not exhaust."
    ),
    "num_layers": 6,
    "model_dim": 384,
    "head_dim": 64,
    "mlp_ratio": 4,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(SHAPE, indent=2) + "\n", encoding="utf-8")
    print(f"[reference] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
