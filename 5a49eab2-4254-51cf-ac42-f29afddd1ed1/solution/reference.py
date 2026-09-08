#!/usr/bin/env python3
"""Reference oracle for OER-05: emit a update rule that closes most of the gap.

WHAT THIS IS
    The output of a search run during authoring over this exact harness, this exact
    corpus and this exact budget. Every number in DOCUMENT below was chosen because a
    measured run put it there. The measurements are in SEARCH_LOG so the artifact is
    traceable to evidence instead of asserted, and solution/grounding.md carries the
    same table with the reasoning.

WHAT THIS IS NOT
    It is NOT the verifier's private reference update rule. That one lives only in
    tests/private/reference_rule.json, it was arrived at on a separate leg of the
    search, and it differs from this document in its second-moment decay -- the oracle runs the conventional 0.999, the reference runs 0.995, which was the last thing the search found and the single largest remaining gain. If the oracle were the
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
    ("heavy-ball momentum SGD, uniform 0.02",
     7.61175, "no preconditioner at all is nowhere near competitive"),
    ("heavy-ball momentum SGD, uniform 0.05",
     7.24541, "same, at a larger step"),
    ("heavy-ball momentum SGD, uniform 0.15",
     6.77797, "the best plain-momentum point found is still 1.6 nats behind Adam"),
    ("per-role, precond_power 0.35 instead of 0.5",
     5.53349, "the preconditioner exponent is sharply peaked at Adam's value"),
    ("per-role, precond_power 0.65 instead of 0.5",
     5.43934, "and it is peaked from the other side too"),
    ("sign momentum, uniform 1e-3",
     5.30835, "pure sign descent is a real optimizer here and still well behind"),
    ("per-role, first moment 0.95 instead of 0.9",
     5.17570, "lengthening the FIRST moment costs"),
    ("Adam point, uniform 6e-3",
     5.15298, "the uniform step size is already near its own optimum; there is no free win here"),
    ("Adam point, uniform 3e-3 -- the shipped default",
     5.13234, "the floor the submission has to beat"),
    ("per-role, sign_mix 0.5 on the block matrices",
     5.12992, "blending toward sign descent on hidden is worth almost nothing"),
    ("per-role, embed and scalar 4e-2, hidden and head 4e-3",
     5.07730, "past the peak: the embedding boost has an interior optimum"),
    ("per-role, embed and scalar 3.2e-2, hidden 5e-3, head 3e-3",
     5.06783, "also past it"),
    ("per-role, embed and scalar 1.2e-2, hidden and head 4e-3",
     5.06555, "splitting the roles apart is worth real nats"),
    ("per-role, embed and scalar 2.4e-2, hidden 6e-3, head 2e-3",
     5.06069, "pricing the head below hidden does not pay by itself"),
    ("per-role, embed and scalar 2.4e-2, hidden and head 4e-3",
     5.04580, "the interior optimum of the embedding boost, at six times the block step"),
    ("per-role at 2.4e-2, plus sign_mix 0.35 on embed and scalar",
     5.05079, "sign blending does not help the embedding either"),
    ("per-role at 2.4e-2, plus decoupled weight decay 0.05",
     5.03646, "a little decay pays at this budget"),
    ("per-role at 2.4e-2, second moment 0.995 instead of 0.999",
     5.01871, "the best point found: SHORTENING the second moment is worth 0.027 on top of everything else"),
]

# ---------------------------------------------------------------------------
# The emitted update rule.
# ---------------------------------------------------------------------------
DOCUMENT = {
    "schema": "oer-nanogpt-updaterule/v1",
    "notes": "Derived by a coordinate search over this harness. Three findings, in the order they paid: (1) the preconditioner exponent is sharply peaked at 0.5 and moving it in EITHER direction costs 0.3 nats or more, so the Adam-shaped part of the law is not where the room is; (2) neither is the uniform step size, which is already near its own optimum at the default's 3e-3; (3) the room is in the roles -- the token embedding and the scalars want about six times the step of the block matrices, and 2.4e-2 against 4e-3 is the interior optimum of that ratio, worth 0.087 nats. Sign blending, a longer first moment and heavy-ball momentum were all measured and all cost. This is a strong answer and not the best one: the second moment's decay was found to be a further axis afterwards.",
    "grad_clip": 1.0,
    "rules": {
        "embed": {
            "decay1": 0.9,
            "decay2": 0.999,
            "precond_power": 0.5,
            "sign_mix": 0.0,
            "weight_decay": 0.0,
            "eps": 1e-10,
            "step_size": 0.024
        },
        "hidden": {
            "decay1": 0.9,
            "decay2": 0.999,
            "precond_power": 0.5,
            "sign_mix": 0.0,
            "weight_decay": 0.0,
            "eps": 1e-10,
            "step_size": 0.004
        },
        "head": {
            "decay1": 0.9,
            "decay2": 0.999,
            "precond_power": 0.5,
            "sign_mix": 0.0,
            "weight_decay": 0.0,
            "eps": 1e-10,
            "step_size": 0.004
        },
        "scalar": {
            "decay1": 0.9,
            "decay2": 0.999,
            "precond_power": 0.5,
            "sign_mix": 0.0,
            "weight_decay": 0.0,
            "eps": 1e-10,
            "step_size": 0.024
        }
    }
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/workspace/submission/update_rule.json")
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
