"""Private reference estimator for bia slot S06.

This is the oracle. It is not the only route to a score and it is not claimed to
be optimal. It is one estimator that treats the injected process as the
structured, lossy, deterministic corruption it actually is rather than as
independent zero mean noise.

The load bearing observation is that the erasure and the two additive components
interact. On an erased coordinate the true gradient contributes nothing at all,
so whatever is read there is the drift term plus, on a spike step, the spike
term, and nothing else. The erased support is therefore a clean two column
regression in the drift direction and the spike direction, both of which the
fixture publishes for every step. Solving it recovers the two additive
coefficients directly instead of inferring them from a guess at the gradient
norm, so the reconstruction does not degrade as those components grow.

What it does, per parameter tensor, at every step:

  1. solves the least squares problem restricted to the erased coordinates for
     the drift coefficient and, on a frozen spike step, the spike coefficient
  2. subtracts both fitted components from the whole observation, which removes
     them to the accuracy of that fit rather than attenuating them
  3. undoes the block erasure rescale on the coordinates the mask kept, and
     writes an honest zero on the coordinates it erased
  4. inverts the rank one amplification exactly, which averaging can never do
     because that component is a deterministic function of the gradient itself

Step 3 is the measured part and it is the counterintuitive one. The obvious move
is to restore an erased coordinate from a staleness buffer, since a rescaled zero
is not an unbiased sample of a coordinate nobody read. Measured against this
frozen substrate that move is actively harmful: filling erased coordinates from a
buffer of the last kept observation reached the target loss at step 950 on seed
0, exactly where the identity baseline reached it, while writing zero there
reached it at step 850, against a ceiling of 800 for an arm handed the
uncorrupted gradient. The reason is that the frozen optimizer already carries a
momentum estimate for every coordinate, so an unwritten coordinate coasts on it,
whereas a buffered value overwrites that estimate with a stale minibatch reading
held for the whole erasure window. Erasure is lossy and no estimator recovers it;
what an estimator can do is decline to make it worse.

The restriction to the erased support is written as a multiply by the erasure
indicator rather than as an index gather, and every fitted coefficient stays on
device as a zero dimensional tensor, so the estimator adds no host device
synchronization and no dynamically shaped allocation to the training step.
"""

from __future__ import annotations

import os
import sys

import torch

_FIXTURES = os.environ.get("BIA_FIXTURES")
if _FIXTURES and _FIXTURES not in sys.path:
    sys.path.insert(0, _FIXTURES)
else:
    _guess = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "environment", "fixtures")
    if os.path.isdir(_guess) and _guess not in sys.path:
        sys.path.insert(0, _guess)

import noise_process as npx  # noqa: E402

RIDGE = 1e-12


class StructuredEstimator:
    """Inverts the frozen corruption to the extent one observation per step allows."""

    def __init__(self, meta):
        self.meta = meta
        self.proc = npx.NoiseProcess(torch)
        self.support = {}
        self.amp = npx.BETA / (1.0 + npx.BETA)
        self.keep = 1.0 - npx.P_DROP

    def _support(self, name, st, step):
        """Keep mask and erasure indicator, cached over the frozen hold window."""
        key = int(step) // npx.BLOCK_HOLD
        held = self.support.get(name)
        if held is not None and held[0] == key:
            return held[1], held[2]
        mask = st.mask(int(step))
        erased = 1.0 - mask
        self.support[name] = (key, mask, erased)
        return mask, erased

    def estimate(self, step, observed):
        out = {}
        t = int(step)
        is_spike = npx.spike_step(t)
        for name, obs in observed.items():
            flat = obs.reshape(-1).to(torch.float32)
            st = self.proc.state(name, flat.numel(), flat.device)
            mask, erased = self._support(name, st, t)
            d = st.drift_dir(t)
            de = d * erased
            if is_spike:
                q = st.q
                qe = q * erased
                a11 = de @ d
                a12 = de @ q
                a22 = qe @ q
                b1 = de @ flat
                b2 = qe @ flat
                det = a11 * a22 - a12 * a12 + RIDGE
                cd = (b1 * a22 - b2 * a12) / det
                cq = (a11 * b2 - a12 * b1) / det
                y = flat - cd * d - cq * q
            else:
                cd = (de @ flat) / (de @ d + RIDGE)
                y = flat - cd * d
            full = y * (self.keep * mask)
            ghat = full - self.amp * (full @ st.u) * st.u
            out[name] = ghat.reshape(obs.shape)
        return out


def build_estimator(meta):
    return StructuredEstimator(meta)
