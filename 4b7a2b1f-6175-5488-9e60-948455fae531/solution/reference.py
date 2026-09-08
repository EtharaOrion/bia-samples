#!/usr/bin/env python3
"""Reference oracle: emit a data plan that closes most of the default-to-reference gap.

WHAT THIS IS
    The output of a survey run during authoring over this exact harness, this exact
    pool and this exact 8,388,608-token budget: every source measured alone at the
    full budget, then the orderings and splits that survey implied. Every number in
    SURVEY_LOG below was measured, not asserted, and every one of them is
    reproducible on the agent surface with `python3 probe_local.py --solo` and a
    handful of candidate plans. solution/grounding.md carries the same table with the
    reasoning.

WHAT THIS IS NOT
    It is NOT the verifier's private reference plan. That one lives only in
    tests/private/reference_plan.json. Both plans reached the same conclusion about
    WHICH sources are worth budget -- the three clean ones, and nothing else -- and
    then differed on the ORDER to feed them in and on which one takes the odd draw.
    They are 0.034 nats apart, which on this substrate is the difference between a
    reward of 1.0 and a reward of about 0.98. If the oracle were the reference plan,
    the gate's reference arm would be the high anchor being graded against itself and
    its reward would be 1.0 by construction rather than by measurement. It is not,
    and it is not.

    It also does not train anything at solve time. Producing the graded artifact is
    writing one validated JSON file; the training happens in the verifier.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# The survey that produced PLAN. Loss is validation cross-entropy in nats on the
# held-out split, measured by this bundle's harness during authoring. Lower is
# better. Every row spends the identical 8,388,608 token budget.
# ---------------------------------------------------------------------------
SURVEY_LOG = [
    # (label, loss, what the run established)
    ("the shipped default: equal eighths of every source, manifest order",
     7.3881, "the floor. Two of the eight sources are catastrophic and dragging them "
             "along costs more than everything else put together"),
    ("solo clean-beta (whole budget, source cycled twice)",
     5.8912, "the best single source. All three clean sources land within 0.03 of "
             "each other, so which clean source is not the question"),
    ("solo clean-gamma",
     5.8916, "confirms it: the clean three are interchangeable measured alone"),
    ("solo clean-alpha",
     5.9145, "same band"),
    ("solo spliced-theta (alternating clean and permuted windows)",
     6.3784, "half-damaged text is worth roughly half a nat less than clean text"),
    ("solo scrambled-delta (256-token windows permuted)",
     7.6163, "block-permuted text is worse than the default mixture is, on its own"),
    ("solo scrambled-epsilon",
     7.6213, "the two scrambled sources are equivalent"),
    ("solo looped-zeta (a 16384-token passage repeated)",
     11.9536, "catastrophic. Fluent text with almost no information per token after "
              "the first period"),
    ("solo looped-eta (a 65536-token passage repeated)",
     12.8742, "worse still. These two are the whole reason the default is so bad"),
    ("clean 3/4 then spliced-theta 1/4",
     5.9311, "adding the least-damaged source to a clean plan still costs, because "
             "the budget it takes was worth more spent on clean text"),
    ("spliced-theta 1/4 first, then clean 3/4",
     6.0138, "and putting it early does not rescue it"),
    ("scrambled-delta 1/4 first, then clean 3/4",
     6.4965, "scrambled material is expensive wherever it goes"),
    ("clean 3/4 then scrambled-delta 1/4",
     7.4137, "but it is far worse LAST. Order is a real axis: the same tokens in the "
             "same proportion cost 0.92 nats more when they are what the run ends on"),
    ("clean-beta alone for the whole budget",
     5.8899, "one clean source cycled twice"),
    ("all three clean sources interleaved in six draws",
     5.8966, "chopping the budget finer buys nothing"),
    ("clean thirds in order alpha, beta, gamma",
     5.8382, "three clean sources beat one clean source cycled: the third pass over "
             "fresh text beats a second pass over the same text. CHOSEN"),
    ("clean thirds in order alpha, gamma, beta",
     5.8363, "reordering the clean three moves the number by about 0.03, which is the "
             "same size as the gap between the sources measured alone"),
    ("clean thirds in order beta, gamma, alpha",
     5.8038, "the best ordering the survey found. The margin over the chosen plan is "
             "0.034 nats -- real, but inside the range where a second seed could "
             "reorder the top three"),
]

# ---------------------------------------------------------------------------
# The emitted plan.
# ---------------------------------------------------------------------------
PLAN = {
    "schema": "oer08-data-plan/v1",
    "notes": (
        "Source-by-source survey with probe_local.py --solo, then orderings. Three "
        "findings, in the order they paid: (1) DROP THE DAMAGED FIVE. The two looped "
        "sources measure 11.95 and 12.87 alone against 5.89 for clean text, and the "
        "shipped default spends a quarter of the budget on them; simply omitting all "
        "five damaged sources is worth about 1.5 nats and is most of the available "
        "gap. (2) SPREAD ACROSS ALL THREE CLEAN SOURCES rather than cycling one. The "
        "budget is 8388608 tokens against 4194304 per source, so a single-source plan "
        "sees everything twice; three sources at a third each see three different "
        "texts once and a bit, and that is worth 0.05. (3) ORDER MATTERS AND IT IS "
        "NOT SUBTLE. The same tokens in the same proportion cost 0.92 nats more when "
        "the damaged material is what the run ends on rather than what it starts "
        "with, so whatever a plan does spend on weaker material belongs early. This "
        "plan spends nothing on weaker material at all, and orders the clean three "
        "alpha, beta, gamma with the odd draw on the last."
    ),
    "draws": [
        {"source": "clean-alpha", "tokens": 2793472, "offset": 0},
        {"source": "clean-beta", "tokens": 2793472, "offset": 0},
        {"source": "clean-gamma", "tokens": 2801664, "offset": 0},
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/workspace/submission/plan.json")
    parser.add_argument("--show-survey", action="store_true")
    args = parser.parse_args()

    if args.show_survey:
        for label, loss, finding in SURVEY_LOG:
            print(f"  {loss:8.4f}  {label}\n            -> {finding}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(PLAN, indent=2) + "\n", encoding="utf-8")
    print(f"[reference] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
