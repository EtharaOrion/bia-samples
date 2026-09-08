#!/usr/bin/env python3
"""Reference oracle for OER-11: emit a initialisation that closes most of the gap.

WHAT THIS IS
    The output of a search run during authoring over this exact harness, this exact
    corpus and this exact budget. Every number in DOCUMENT below was chosen because a
    measured run put it there. The measurements are in SEARCH_LOG so the artifact is
    traceable to evidence instead of asserted, and solution/grounding.md carries the
    same table with the reasoning.

WHAT THIS IS NOT
    It is NOT the verifier's private reference initialisation. That one lives only in
    tests/private/reference_init.json, it was arrived at on a separate leg of the
    search, and it differs from this document in the two residual-writing projections -- the oracle leaves them at the same 0.02 as everything else, the reference tapers them to 0.008. If the oracle were the
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
    ("FAMILY B, on the framework default: taper the projections to 0.006",
     5.10601, "on the framework base the projection taper is nearly inert -- the embedding is the problem"),
    ("FAMILY B, on the framework default: taper the projections to 0.011",
     5.08986, "same conclusion from the other side"),
    ("FAMILY B, the framework's own initialisation -- the shipped default",
     5.08238, "the floor; nn.Embedding's unit normal is fifty times the rest of the network"),
    ("FAMILY B, on the framework default: taper the projections to 0.008",
     5.07217, "0.010 nats for the best taper, against 0.133 for fixing the embedding: the two are not comparable"),
    ("FAMILY B, on the framework default: embedding to 0.014 and projections to 0.008",
     4.95960, "fixing the embedding is where essentially all of the framework gap lives"),
    ("FAMILY A, from a flat 0.02: output projection to 0.0",
     5.31025, "the standard zero-init-the-head trick is the single worst move in this space"),
    ("FAMILY A: output projection to 0.005",
     5.22347, "the head is the most sensitive role and it wants to be LARGE"),
    ("FAMILY A: output projection to 0.01",
     5.16010, "still losing; nothing below 0.02 pays on the head"),
    ("FAMILY A: embedding to 0.10",
     5.13267, "the embedding is monotone the other way too"),
    ("FAMILY A: embedding to 0.05",
     5.10488, "already past the optimum at two and a half times"),
    ("FAMILY A: flat 0.06 on every weight role",
     5.02346, "uniformly larger is worse"),
    ("FAMILY A: flat 0.014 on every weight role",
     4.99565, "and uniformly smaller is worse too, so 0.02 is not an accident"),
    ("FAMILY A: residual_depth_power 0.5, the textbook 1/sqrt(2L) taper",
     4.97758, "the standard depth taper OVERSHOOTS at six layers and loses 0.028"),
    ("FAMILY A: norm_gain 0.8",
     4.96427, "the RMSNorm gains want to start at one"),
    ("FAMILY A: q, k, v and the MLP input to 0.015",
     4.96277, "the block matrices want 0.02 as well"),
    ("FAMILY A: embedding to 0.01",
     4.95639, "shrinking the embedding does not pay either"),
    ("FAMILY A, flat 0.02 everywhere -- what GPT-2 uses",
     4.94938, "88 percent of the framework gap, from one number applied everywhere"),
    ("FAMILY A: embedding to 0.014",
     4.94711, "inside the measurement floor of flat 0.02; not a real gain"),
    ("FAMILY A: the two residual projections to 0.008, a divisor of 2.5",
     4.93091, "the best point found: a MILDER taper than the textbook one, and the optimum is sharp"),
]

# ---------------------------------------------------------------------------
# The emitted initialisation.
# ---------------------------------------------------------------------------
DOCUMENT = {
    "schema": "oer-nanogpt-init/v1",
    "notes": "Derived by a search over this harness. The finding that carries it is that the framework's default is bad in exactly one place and enormously so: nn.Embedding initialises to a unit normal, fifty times the scale the rest of the network uses, and correcting the embedding alone accounts for essentially the whole gap (5.08238 to 4.95960 with the projections already tapered). Once every role is at a conventional 0.02 the obvious refinements were measured and all of them cost -- zeroing the output projection loses 0.36, shrinking it to 0.005 loses 0.27, enlarging the embedding to 0.05 loses 0.16, and the textbook 1/sqrt(2L) residual taper loses 0.028. This is a strong answer and not the best one: a milder taper of the residual projections than the textbook prescribes was measured better afterwards.",
    "init": {
        "embed_std": 0.02,
        "attn_qkv_std": 0.02,
        "attn_proj_std": 0.02,
        "mlp_fc_std": 0.02,
        "mlp_proj_std": 0.02,
        "head_std": 0.02,
        "bias_std": 0.0,
        "norm_gain": 1.0,
        "residual_depth_power": 0.0
    }
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/workspace/submission/init.json")
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
