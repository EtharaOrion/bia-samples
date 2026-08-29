"""Starter estimator for bia slot S06. It works, and it is deliberately suboptimal.

This is the reflex answer to a noisy gradient: smooth it with an exponential
moving average and clip the result. Copy this file to submission/estimator.py to
have a running submission from minute one, then improve it. Improving it is the
objective; shipping it unchanged is not a solution, it is the starting line.

Contract a submission must satisfy:

  build_estimator(meta) returns an object exposing estimate(step, observed).
  meta carries tensor_ids, shapes, numels, max_steps, device, point and seed.
  observed is a dict mapping tensor id to the corrupted gradient observation for
  that step, one observation per step and no more.
  estimate returns a dict with the same keys and the same shapes, holding the
  update direction the frozen optimizer will apply.

The estimator never sees the model, the data, the loss or the true gradient. One
forward and one backward pass happen per step and the harness counts them.
"""

from __future__ import annotations

import torch

EMA_BETA = 0.9
CLIP_MULTIPLE = 3.0


class EmaEstimator:
    """Exponential moving average with a running norm gate."""

    def __init__(self, meta):
        self.meta = meta
        self.avg = {}
        self.norm_ref = {}

    def estimate(self, step, observed):
        out = {}
        for name, obs in observed.items():
            flat = obs.reshape(-1).to(torch.float32)
            n = float(flat.norm())
            ref = self.norm_ref.get(name)
            if ref is None:
                ref = n
            if n > CLIP_MULTIPLE * ref and n > 0.0:
                flat = flat * (CLIP_MULTIPLE * ref / n)
                n = CLIP_MULTIPLE * ref
            self.norm_ref[name] = 0.9 * ref + 0.1 * n
            prev = self.avg.get(name)
            cur = flat if prev is None else (EMA_BETA * prev + (1.0 - EMA_BETA) * flat)
            self.avg[name] = cur
            out[name] = cur.reshape(obs.shape)
        return out


def build_estimator(meta):
    return EmaEstimator(meta)
