#!/usr/bin/env python3
"""The reference oracle for slot OER-22. It emits one retention selection.

HOW THIS SELECTION WAS DERIVED, so a reader can redo it rather than trust it.

The frozen run holds its learning rate CONSTANT after a 5 percent warmup and never
decays it. A run in that regime does not settle onto a minimum; after the first third it
wanders around one, and each snapshot is a different sample of the same neighbourhood.
That is the regime in which averaging iterates is worth something, and it is why the
shipped default -- keep the last snapshot, throw the other fifteen away -- leaves value
on the table even though it is what almost every training script does.

The derivation ran entirely on the AGENT-VISIBLE devset, which is all a solver has:

  1. Score all sixteen snapshots singly. The curve falls steeply to about snapshot 9 and
     is then noisy and roughly flat, moving by less between neighbours than it moves
     between two evaluations of the same neighbourhood. Snapshot 16 is not reliably the
     best single snapshot; it is merely the last one.
  2. Score contiguous windows of every length up to the storage budget of 6. Every window
     of 2 or more beats every single snapshot. The gain is not a tie-break: it is several
     times the spacing between adjacent singles.
  3. Score windows that skip -- stride 2, 3, 4, 5 -- against contiguous ones of the same
     size. Skipping LOSES here, and that is worth stating because it is the opposite of
     what the usual "average decorrelated iterates" intuition predicts. The reason is
     that this run has not plateaued far enough back: a stride-3 window reaches into
     snapshots that are still genuinely worse, and averaging in a worse model costs more
     than the extra decorrelation buys.
  4. Score weight shapes over the best windows: uniform, linear, quadratic, square-root
     and geometric decay in both directions. Rising weights beat falling ones, which is
     the same finding as step 3 from the other side -- later snapshots deserve more mass,
     just not all of it.

This file installs the best selection step 4 found ON THE DEVSET inside the family a
solver reaches for first: the uniform average of a contiguous tail, with the window
length chosen by measurement rather than assumed. That is snapshots 12 through 16.

It is NOT the verifier's private reference. The bar was derived from the held-out split
and from the wider weight family, and it differs from this file in both its window and
its weighting, so the reference arm of the gate is never the bar being graded against
itself. This selection is expected to land close to the bar but below it, and the exact
distance is a measurement the verifier makes, not a number written here.

Nothing in this file reports a metric. The graded number is computed by the verifier from
its own snapshots on a split this container has no path to, and nothing printed here can
move it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = "oer22-selection/v1"

# Snapshots 12..16 of the frozen run, averaged uniformly. Five of the six retention slots
# the storage budget allows; the sixth is deliberately left unspent, because step 2 above
# measured window 12..16 as better on the devset than window 11..16 in this weight family.
KEEP = [12, 13, 14, 15, 16]
WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.2]

NOTES = (
    "Uniform average of the last five snapshots of the frozen run. Derived on the "
    "agent-visible devset by scoring every single snapshot, then every contiguous window "
    "up to the storage budget, then the same windows with stride, then a family of weight "
    "shapes. Contiguous beat strided, and window 12..16 was the best uniform window "
    "measured. The unspent sixth retention slot is a measurement, not an oversight."
)


def build() -> dict:
    assert len(KEEP) == len(WEIGHTS)
    assert abs(sum(WEIGHTS) - 1.0) < 1e-9
    return {"schema": SCHEMA, "notes": NOTES, "keep": list(KEEP), "weights": list(WEIGHTS)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")
    print(f"selection written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
