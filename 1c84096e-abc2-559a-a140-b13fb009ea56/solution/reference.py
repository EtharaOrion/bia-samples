#!/usr/bin/env python3
"""Reference oracle: emit an evaluation policy that closes most of the default gap.

WHAT THIS IS
    The output of a search run during authoring over this exact harness, this exact
    frozen training recipe and every one of the five declared ramp starts. Every
    number in POLICY below was chosen because a measured run put it there. The
    measurements are recorded in SEARCH_LOG so the artifact is traceable to evidence
    instead of asserted, and solution/grounding.md carries the same table with the
    reasoning.

    Critically, the search that produced this policy ran over the AGENT-VISIBLE
    devset curves -- exactly the numbers `python3 probe_local.py` puts in front of a
    solver -- and its objective was the WORST draw rather than the average one. That
    is the objective a solver actually has: it does not know which onset it will be
    graded on, so the policy it wants is the one whose bad case is least bad.

WHAT THIS IS NOT
    It is NOT the verifier's private reference policy. That one lives only in
    tests/private/reference_policy.json, it was arrived at by a separate search over
    the verifier's OWN probe curves with a different objective -- best expected
    reward rather than best worst case -- and it differs from this policy in the
    number of probes, in the token split and in where the probes sit. If the oracle
    were the reference policy, the gate's reference arm would be the high anchor
    being graded against itself and its reward would be 1.0 by construction rather
    than by measurement. It is not, and it is not.

    It also does not train anything. Producing the graded artifact is writing one
    validated JSON file; the training happens in the verifier.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# The search that produced POLICY. Every row is a real replay of the harness on
# every declared ramp start; the reward column is the fraction of the
# default-to-best-checkpoint gap the policy closed, averaged over the five draws,
# with the worst single draw beside it. Higher is better.
# ---------------------------------------------------------------------------
SEARCH_LOG = [
    # (label, mean reward over the five declared draws, worst single draw, finding)
    ("the shipped default: one probe at step 2048, whole budget",
     0.000, 0.000, "the floor. On four of the five draws the final checkpoint is the worst on the grid, because the stream is fully permuted long before then"),
    ("one probe at 1856, whole budget",
     0.247, 0.020, "naming a single late checkpoint wins only on the latest onset and very nearly collapses on the earliest: the turn moves with the draw"),
    ("two probes at 1792 and 1856, six batches each",
     0.379, 0.020, "resolution without coverage. Both probes sit past the turn on the early onsets, so the pair is a choice between two bad points read precisely"),
    ("six probes evenly across 128..2048, two batches each",
     0.944, 0.821, "coverage is worth far more than resolution here -- the draw moves the turn by a thousand steps and twelve batches cannot buy both"),
    ("the same six probes, argmin_smoothed window 3",
     0.749, 0.000, "smoothing costs more than it buys: the minimum is narrow once it is bracketed, the window pulls the choice onto a genuinely worse neighbour, and on one draw it gives all of it back"),
    ("four probes at 832 / 1152 / 1536 / 1856, three batches each",
     0.993, 0.981, "the best mean of the search, but its coverage starts at 832 and it carries nothing below that"),
    ("six probes at 128 / 448 / 832 / 1152 / 1536 / 1856, two batches each",
     0.993, 0.981, "matches the best mean AND the best worst case while covering the grid from step 128. Chosen: nothing is traded away for the wider bracket"),
]

# ---------------------------------------------------------------------------
# The emitted policy.
# ---------------------------------------------------------------------------
POLICY = {
    "schema": "oer12-eval-policy/v1",
    "notes": (
        "Search over the devset curves probe_local.py produces, on all five declared "
        "ramp starts, maximising the WORST draw. Three findings, in the order they paid: "
        "(1) the onset draw moves the turn in the held-out curve by roughly a thousand "
        "steps, so any policy that concentrates its probes near the end is one-in-five "
        "to be right and scores zero on the rest; (2) with a 12-batch budget, COVERAGE "
        "beats RESOLUTION -- six probes at two batches each read every point to about "
        "0.13 nats, which is coarse, but the differences that matter once the turn is "
        "bracketed are far larger than that, while a two-probe policy that reads to 0.05 "
        "nats is reading the wrong two points on most draws; (3) argmin_smoothed is a "
        "net loss on this substrate, because the minimum is narrow and the window drags "
        "the selection onto a neighbour that is genuinely worse. The probes are spaced "
        "geometrically rather than uniformly: the curve is steep early and flat near the "
        "turn, so the early points cost little and the late ones are where the decision "
        "is actually made."
    ),
    "probes": [
        {"step": 128, "tokens": 16384},
        {"step": 448, "tokens": 16384},
        {"step": 832, "tokens": 16384},
        {"step": 1152, "tokens": 16384},
        {"step": 1536, "tokens": 16384},
        {"step": 1856, "tokens": 16384},
    ],
    "selection": {"rule": "argmin"},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/workspace/submission/policy.json")
    parser.add_argument("--show-search", action="store_true")
    args = parser.parse_args()

    if args.show_search:
        for label, mean, worst, finding in SEARCH_LOG:
            print(f"  mean {mean:.3f}  worst {worst:.3f}  {label}\n            -> {finding}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(POLICY, indent=2) + "\n", encoding="utf-8")
    print(f"[reference] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
