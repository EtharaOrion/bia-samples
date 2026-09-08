#!/usr/bin/env python3
"""Reference oracle for OER-07: emit a learning-rate schedule that closes most of the gap.

WHAT THIS IS
    The output of a search run during authoring over this exact harness, this exact
    corpus and this exact budget. Every number in DOCUMENT below was chosen because a
    measured run put it there. The measurements are in SEARCH_LOG so the artifact is
    traceable to evidence instead of asserted, and solution/grounding.md carries the
    same table with the reasoning.

WHAT THIS IS NOT
    It is NOT the verifier's private reference learning-rate schedule. That one lives only in
    tests/private/reference_schedule.json, it was arrived at on a separate leg of the
    search, and it differs from this document in its decay family and its warmup fraction -- the oracle decays linearly from a 10 percent warmup, the reference holds the peak first after a 20 percent warmup. If the oracle were the
    reference, the gate's reference arm would be the high anchor being graded against
    itself and its reward would be 1.0 by construction rather than by measurement. It
    is not, and it is not.

    It also does not train anything. Producing the graded artifact is writing one
    validated JSON file; the training happens in the verifier.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# The search that produced DOCUMENT. Loss is validation cross-entropy in nats on
# environment/data/devset_slice.bin, measured by this bundle's own harness during
# authoring, under the frozen budget. Lower is better. The graded split is a
# different slice, so the graded numbers sit at an offset from these; the ORDER is
# what carried the decisions.
# ---------------------------------------------------------------------------
SEARCH_LOG = [
    ("constant, no warmup, no decay -- the shipped default",
     5.30797, "the floor the submission has to beat; the frozen peaks train fine flat"),
    ("constant WITH a 10 percent warmup",
     5.17787, "warmup ALONE, with no decay whatever, is worth 0.130 nats"),
    ("exp, 10 percent warmup, decay_power 3, floor 0",
     5.17888, "a fast exponential spends too much of a short run near the floor"),
    ("inv_sqrt, 10 percent warmup, decay_power 8, floor 0",
     5.17294, "same defect as exp, from the other family"),
    ("poly, 10 percent warmup, decay_power 2, floor 0",
     5.16567, "a convex decay is worse than a straight line here"),
    ("cosine, 10 percent warmup, floor 0",
     5.11003, "cosine is better than the convex families and worse than linear"),
    ("linear, 10 percent warmup, floor 0",
     5.06949, "the straight line beats every curved family tried"),
    ("wsd, 5 percent warmup, stable 0.4, floor 0",
     5.06777, "holding the peak first beats decaying from step one, but not at a short warmup"),
    ("wsd, 10 percent warmup, stable 0.6, floor 0",
     5.03662, "a longer warmup and a longer hold are both worth real nats"),
    ("wsd, 20 percent warmup, stable 0.5, floor 0",
     5.02301, "the best point found: a fifth of the budget spent reaching the peak"),
    ("wsd, 30 percent warmup, stable 0.5, floor 0",
     5.02687, "past the warmup optimum, but only just -- the peak is broad"),
    ("wsd, 20 percent warmup, stable 0.3, floor 0",
     5.03545, "shortening the hold costs; the two knobs interact"),
]

# ---------------------------------------------------------------------------
# The emitted learning-rate schedule.
# ---------------------------------------------------------------------------
DOCUMENT = {
    "schema": "oer-nanogpt-schedule/v1",
    "notes": "Derived by a search over this harness at the frozen peaks. Two findings, in the order they paid: (1) warmup is the larger of the two axes -- a 10 percent linear ramp is worth 0.130 nats even with the envelope left flat afterwards, because the frozen peaks are set at the edge of what a cold start tolerates; (2) among the decay families a straight line beats cosine by 0.041 and beats the convex families (poly at power 2, exp, inv_sqrt) by about 0.10, because a 3072-step run cannot afford the long low-step-size tail those families spend. This is a strong answer and not the best one: holding the peak before decaying was measured better still.",
    "schedule": {
        "shape": "linear",
        "warmup_shape": "linear",
        "warmup_frac": 0.1,
        "warmup_power": 1.0,
        "stable_frac": 1.0,
        "final_frac": 0.0,
        "decay_power": 1.0
    }
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/workspace/submission/schedule.json")
    parser.add_argument("--show-search", action="store_true")
    args = parser.parse_args()

    if args.show_search:
        for label, loss, finding in SEARCH_LOG:
            print(f"  {loss:.5f}  {label}\n            -> {finding}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(DOCUMENT, indent=2) + "\n", encoding="utf-8")
    print(f"[reference] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
