#!/usr/bin/env python3
"""Reference oracle for OER-10: emit a data mixture that closes most of the gap.

WHAT THIS IS
    The output of a search run during authoring over this exact harness, this exact
    corpus and this exact budget. Every number in DOCUMENT below was chosen because a
    measured run put it there. The measurements are in SEARCH_LOG so the artifact is
    traceable to evidence instead of asserted, and solution/grounding.md carries the
    same table with the reasoning.

WHAT THIS IS NOT
    It is NOT the verifier's private reference data mixture. That one lives only in
    tests/private/reference_mixture.json, it was arrived at on a separate leg of the
    search, and it differs from this document in its ordering policy -- the reference interleaves the chosen blocks evenly across the whole run, the oracle feeds them in permuted chunks of 64. If the oracle were the
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
    ("equal weight on all six shards, interleaved -- the shipped default",
     5.49905, "the floor; a third of the budget goes to shards that damage the run"),
    ("shard_d and shard_e alone",
     7.68379, "the two damaged shards are not merely weak, they are far worse than nothing"),
    ("shard_a alone, so 3072 blocks drawn from 1023 -- three passes",
     5.30848, "concentration is a trap: three passes over clean data is worse than one pass over a mixture containing junk"),
    ("shard_a, shard_b, shard_c plus shard_f at 0.5",
     5.16582, "the half-reordered shard costs real nats even at half weight"),
    ("shard_a, shard_b, shard_c, fed one shard at a time",
     5.06660, "the right blocks in the wrong order gives most but not all of the value back"),
    ("shard_a, shard_b, shard_c, permuted chunks of 64",
     5.03957, "chunk-permuted ordering recovers most of the gap to interleaving"),
    ("shard_a, shard_b, shard_c, evenly interleaved",
     5.02281, "the best point found: three clean shards, each spent almost exactly once, stationary over the run"),
    ("shard_a, shard_b, shard_c, permuted chunks of 512",
     5.01954, "the clean-shard mixture is a plateau: three orderings of it lie within 0.003 nats"),
    ("shard_a, shard_b, shard_c, permuted chunks of 8",
     5.02089, "fine-grained permutation is indistinguishable from interleaving"),
    ("shard_a and shard_b only, so 1536 blocks each against 1023",
     5.07851, "two clean shards is worse than three: the repetition costs more than the third shard's tokens are worth"),
    ("shard_a, shard_b, shard_c plus shard_e at 0.15",
     5.06401, "even a sixth of the duplicated shard costs 0.041"),
    ("shard_a, shard_b, shard_c plus shard_f at 0.15",
     5.08536, "and a sixth of the half-reordered shard costs 0.063"),
    ("shard_a, shard_b, shard_c plus shard_f at 0.35",
     5.11480, "the cost of shard_f is roughly linear in its weight"),
]

# ---------------------------------------------------------------------------
# The emitted data mixture.
# ---------------------------------------------------------------------------
DOCUMENT = {
    "schema": "oer-nanogpt-mixture/v1",
    "notes": "Derived by measuring each shard's marginal value on this harness. The weights follow from two measurements: the two fully-touched shards are catastrophic on their own (7.68 against the default's 5.50) and the half-touched shard still costs 0.14 nats at half weight, while concentrating on a single clean shard costs more than the junk did because 3072 blocks against 1023 available is three passes. That leaves the three untouched shards at equal weight, which spends the budget almost exactly once over. The ordering here is chunk-permutation at a group of 64, which was measured to recover most -- but not all -- of the value that even interleaving gives; a stationary mixture was measured better still.",
    "mixture": {
        "shard_a": 1.0,
        "shard_b": 1.0,
        "shard_c": 1.0,
        "shard_d": 0.0,
        "shard_e": 0.0,
        "shard_f": 0.0
    },
    "order": {
        "policy": "blocked",
        "block_group": 64,
        "seed": 7
    }
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/workspace/submission/mixture.json")
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
