#!/usr/bin/env python3
"""The one file the agent edits.

The rule shipped here is a plain momentum step. It works, it reaches the target,
and it is BEHAVIOURALLY PRESENT IN THE PINNED CORPUS, so submitting it unchanged
scores zero with the reason `behaviour-matches-corpus-entry`. It is here to be
the working starting point the instruction promises, not the answer.

Contract for this file, enforced by the frozen loop and by the verifier's probe:

- `build_update_rule(shape, hyper)` returns an object with a `step` method.
- `step(params, grads, state)` returns `(new_params, new_state)`.
- The rule sees gradients from exactly one forward-backward pass per step. It
  must not request another.
- The rule may keep any internal state it likes across steps, and may schedule
  its own hyperparameters however it likes.
- The rule must be deterministic given `(params, grads, state)`. The verifier
  drives it twice over identical probe inputs and requires an identical
  behavioural signature; a rule that consults a clock, a random source or an
  environment variable will not reproduce and scores
  `probe-digest-nondeterministic`.
"""

from __future__ import annotations


class MomentumRule:
    """Heavy-ball momentum. Present in the corpus; replace its behaviour."""

    def __init__(self, shape, hyper) -> None:
        self.width = int(shape[0])
        self.lr = float(hyper.get("lr", 0.05))
        self.beta = float(hyper.get("beta", 0.9))

    def step(self, params, grads, state):
        buffer = state.get("buffer")
        if buffer is None:
            buffer = [0.0] * len(params)
        buffer = [self.beta * b + g for b, g in zip(buffer, grads)]
        params = [p - self.lr * b for p, b in zip(params, buffer)]
        return params, {"buffer": buffer}


def build_update_rule(shape, hyper):
    """The one entry point the frozen loop and the verifier's probe both call."""
    return MomentumRule(shape, hyper or {})
