#!/usr/bin/env python3
"""Reference oracle: emit a policy that closes most of the default-to-reference gap.

WHAT THIS IS
    The output of a search run during authoring over this exact harness, this exact
    corpus and this exact 25,165,824-token budget. Every policy in SEARCH_LOG below was
    TRAINED, and the loss beside it is the plain validation cross entropy that run
    produced on the split the verifier grades against. solution/grounding.md carries the
    same table with the reasoning.

WHAT THIS IS NOT
    It is NOT the verifier's private reference policy. That one lives only in
    tests/private/reference_policy.json, it is a different point of the same search, and
    it is a measurably better one. If the oracle were the reference policy, the gate's
    reference arm would be the high anchor being graded against itself and its reward
    would be 1.0 by construction rather than by measurement. It is not, and it is not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# (label, measured val loss on the graded split, logit_softcap, label_smoothing,
#  z_loss, [wd_embed, wd_hidden, wd_head, wd_scalar])
SEARCH_LOG = [
    ('bare_z1e3', 5.06324, 15.0, 0.0, 0.001, [0.0, 0.0, 0.0, 0.0]),
    ('hidden_decay', 5.06698, 15.0, 0.0, 0.0, [0.0, 0.05, 0.0, 0.0]),
    ('bare_cap64', 5.06824, 64.0, 0.0, 0.0, [0.0, 0.0, 0.0, 0.0]),
    ('smooth002', 5.06883, 15.0, 0.02, 0.0, [0.0, 0.0, 0.0, 0.0]),
    ('bare_cap30', 5.06894, 30.0, 0.0, 0.0, [0.0, 0.0, 0.0, 0.0]),
    ('best_guess', 5.07167, 30.0, 0.0, 0.001, [0.0, 0.0, 0.0, 0.0]),
    ('bare_z1e4', 5.07408, 15.0, 0.0, 0.0001, [0.0, 0.0, 0.0, 0.0]),
    ('decay_only', 5.07579, 15.0, 0.0, 0.0, [0.1, 0.1, 0.1, 0.1]),
    ('bare', 5.07829, 15.0, 0.0, 0.0, [0.0, 0.0, 0.0, 0.0]),
    ('best_guess_c', 5.08455, 22.0, 0.0, 0.0003, [0.0, 0.0, 0.0, 0.0]),
    ('bare_cap8', 5.10022, 8.0, 0.0, 0.0, [0.0, 0.0, 0.0, 0.0]),
    ('bare_z5e3', 5.12548, 15.0, 0.0, 0.005, [0.0, 0.0, 0.0, 0.0]),
    ('shipped_default', 5.13324, 15.0, 0.1, 0.0, [0.1, 0.1, 0.1, 0.1]),
    ('smoothing_only', 5.13471, 15.0, 0.1, 0.0, [0.0, 0.0, 0.0, 0.0]),
    ('bare_cap4', 5.99377, 4.0, 0.0, 0.0, [0.0, 0.0, 0.0, 0.0]),
]

POLICY = {
    "schema": "oer-nanogpt-policy/v1",
    "notes": "Search finding: the two regularisers the shipped default inherited from convention both cost real nats here. Label smoothing appears in the training objective and not in the reading, so it buys nothing and is charged in full; a uniform weight decay removes signal at a budget where the model is nowhere near fitting the corpus. The control worth actually tuning is the one the default left at its transcribed upstream value.",
    "decay": {
        "wd_embed": 0.0,
        "wd_hidden": 0.0,
        "wd_head": 0.0,
        "wd_scalar": 0.0
    },
    "objective": {
        "label_smoothing": 0.0,
        "z_loss": 0.0,
        "logit_softcap": 30.0
    }
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/workspace/submission/policy.json")
    parser.add_argument("--show-search", action="store_true")
    args = parser.parse_args()
    if args.show_search:
        for label, loss, cap, sm, z, wd in SEARCH_LOG:
            print(f"  {loss:.5f}  {label:<18} cap={cap} smoothing={sm} z={z} wd={wd}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(POLICY, indent=2) + "\n", encoding="utf-8")
    print(f"[reference] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
