#!/usr/bin/env python3
"""The reference update rule the live checkers accept end to end. Held out.

Design, stated so a reader can check that it is not a corpus entry wearing a
different name. Three things it does that no entry in `tests/corpus/entries.json`
does, and all three move the DELTA pattern the probe records rather than the
source text:

1. **Sign-agreement gating.** A coordinate whose momentum and current gradient
   disagree in sign is damped rather than followed. No corpus entry conditions
   its step on the agreement of two of its own quantities.
2. **Median-magnitude normalization over the block.** The step is divided by the
   median absolute gated magnitude across the block, not by an L2 norm, not by
   an RMS and not per-coordinate by a second moment. A median is a rank
   statistic, so it responds to the shape of the block rather than to its
   energy, and that shows up in the recorded deltas as a different distribution
   across coordinates at the same step.
3. **A schedule-free trailing average.** Parameters are pulled toward an
   internal anchor with a 1/(t+1) weight rather than being displaced directly,
   so the recorded delta at step t is a fraction of the anchor's own movement
   and shrinks with t at a rate no corpus entry reproduces.

Determinism: pure function of `(params, grads, state)`. No clock, no random
source, no environment read, no import beyond the standard library, so
`tests/probe.py` reproduces its transcript exactly on every pass.
"""

from __future__ import annotations


class SignGatedMedianAnchorRule:
    """Sign-gated, median-normalized, schedule-free anchored update."""

    def __init__(self, shape, hyper) -> None:
        self.width = int(shape[0])
        self.lr = float(hyper.get("lr", 0.06))
        self.beta = float(hyper.get("beta", 0.7))
        self.damp = float(hyper.get("damp", 0.25))
        self.eps = float(hyper.get("eps", 1e-8))

    def step(self, params, grads, state):
        count = int(state.get("count", 0)) + 1
        momentum = state.get("momentum")
        if momentum is None:
            momentum = [0.0] * len(params)
        anchor = state.get("anchor")
        if anchor is None:
            anchor = list(params)

        momentum = [
            self.beta * m + (1.0 - self.beta) * g for m, g in zip(momentum, grads)
        ]

        # Sign-agreement gate. Disagreement is damped, never followed.
        gated = [
            m if m * g > 0.0 else self.damp * m for m, g in zip(momentum, grads)
        ]

        # Median absolute magnitude across the block. A rank statistic, so the
        # reduction order is fixed by a sort over magnitudes and never by the
        # order the block happened to be built in.
        magnitudes = sorted(abs(value) for value in gated)
        middle = magnitudes[len(magnitudes) // 2] if magnitudes else 0.0
        scale = middle + self.eps

        # The anchor moves; the parameters trail it. The schedule is internal
        # and needs no step budget, which is what makes it schedule-free.
        rate = self.lr * (1.0 + 1.0 / float(count))
        anchor = [a - rate * (value / scale) for a, value in zip(anchor, gated)]
        weight = 1.0 / float(count + 1)
        params = [(1.0 - weight) * p + weight * a for p, a in zip(params, anchor)]

        return params, {"momentum": momentum, "anchor": anchor, "count": count}


def build_update_rule(shape, hyper):
    """The entry point the frozen loop and the verifier's probe both call."""
    return SignGatedMedianAnchorRule(shape, hyper or {})
