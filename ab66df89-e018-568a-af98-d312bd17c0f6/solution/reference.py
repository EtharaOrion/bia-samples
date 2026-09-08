#!/usr/bin/env python3
"""Reference oracle: emit a compute allocation that closes most of the default gap.

WHAT THIS IS
    The output of a sweep run during authoring over this exact harness, this exact
    corpus and this exact FLOP budget: every menu entry at its own full budget with
    grad_accum 1, plus half-budget and large-batch variants to price the other two
    axes. Every number in SWEEP_LOG below was measured, not asserted, and
    solution/grounding.md carries the same table with the reasoning.

WHAT THIS IS NOT
    It is NOT the verifier's private reference allocation. That one lives only in
    tests/private/reference_allocation.json and names a DIFFERENT menu entry. The two
    sit on a flat stretch of the frontier where the difference between them is a
    couple of ten-thousandths of a nat, which is exactly why an authoring sweep and a
    solver's sweep can land on either: the finding they share -- go small, spend
    everything, keep grad_accum at 1 -- is robust, and which of the two smallest
    entries wins is not. If the oracle were the reference allocation, the gate's
    reference arm would be the high anchor being graded against itself and its reward
    would be 1.0 by construction rather than by measurement. It is not, and it is not.

    It also does not train anything at solve time. Producing the graded artifact is
    writing one validated JSON file; the training happens in the verifier.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# The sweep that produced ALLOCATION. Loss is validation cross-entropy in nats on
# the held-out slice, measured by this bundle's harness during authoring. Every row
# spends the same FLOP budget unless the row says otherwise. Lower is better.
# ---------------------------------------------------------------------------
SWEEP_LOG = [
    # (label, val loss, what the run established)
    ("m06x640, 616 micro-steps, grad_accum 8 -- the shipped default",
     None, "the floor: the largest affordable model with a large optimizer batch. "
           "The verifier measures this on every run; no literal for it is carried here"),
    ("m06x640, 616 micro-steps, grad_accum 1",
     6.0271, "dropping the batch back to one micro-batch per step is worth a lot on "
             "its own, before any question of model size is asked"),
    ("m08x512, 736 micro-steps, grad_accum 1",
     5.9658, "smaller and deeper beats wider at the same budget"),
    ("m12x384, 896 micro-steps, grad_accum 1",
     5.8853, "still improving as capacity comes down and tokens go up"),
    ("m06x384, 1256 micro-steps, grad_accum 1",
     5.6692, "the trend is not a local wobble; it is the whole frontier"),
    ("m06x384, 624 micro-steps, grad_accum 1 -- half the budget",
     6.1039, "spending only half the budget costs more than any model-size mistake "
             "on this menu. Spend all of it"),
    ("m06x384, 1256 micro-steps, grad_accum 4",
     6.1055, "a four-fold optimizer batch costs about as much as throwing away half "
             "the compute. grad_accum is not a free efficiency knob here"),
    ("m04x256, 2376 micro-steps, grad_accum 1 -- exactly one epoch of the corpus",
     5.3706, "the corpus runs out here, and the frontier has not turned"),
    ("m03x192, 3512 micro-steps, grad_accum 1 -- 1.47 epochs",
     5.2742, "past one epoch. Repeated tokens still buy more than the capacity they "
             "were traded for. CHOSEN"),
    ("m02x128, 5728 micro-steps, grad_accum 1 -- 2.40 epochs",
     5.2740, "the smallest entry, and the frontier is flat between it and m03x192 -- "
             "two ten-thousandths of a nat apart, well inside run-to-run noise. The "
             "sweep takes m03x192, which reaches the same place at 1.47 epochs "
             "instead of 2.40 and so leans less on repetition"),
]

# ---------------------------------------------------------------------------
# The emitted allocation.
# ---------------------------------------------------------------------------
ALLOCATION = {
    "schema": "oer16-compute-allocation/v1",
    "notes": (
        "Full-menu sweep at full budget, plus half-budget and large-batch controls. "
        "Three findings, in the order they paid: (1) SPEND ALL OF IT -- half the "
        "budget costs 0.43 nats, more than any model-size error available on this "
        "menu; (2) grad_accum 1 -- a four-fold optimizer batch costs 0.44 nats, about "
        "the same as discarding half the compute, so the efficiency instinct is "
        "actively expensive at this token count; (3) GO SMALL -- the frontier is "
        "monotone from m06x640 down to m03x192, 0.75 nats end to end, and it does not "
        "turn at one epoch of the corpus. m03x192 spends 1.47 epochs and matches the "
        "smallest entry, which needs 2.40; both are far better than any model large "
        "enough to look like the obvious choice. The lesson is that at this budget "
        "the models on the menu are all over-parameterised for the tokens they can "
        "afford, so capacity traded for tokens keeps paying until the corpus itself "
        "starts running out."
    ),
    "model": "m03x192",
    "micro_steps": 3512,
    "grad_accum": 1,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/workspace/submission/allocation.json")
    parser.add_argument("--show-sweep", action="store_true")
    args = parser.parse_args()

    if args.show_sweep:
        for label, loss, finding in SWEEP_LOG:
            shown = "measured on the grading run" if loss is None else f"{loss:.4f}"
            print(f"  {shown:>26}  {label}\n            -> {finding}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ALLOCATION, indent=2) + "\n", encoding="utf-8")
    print(f"[reference] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
