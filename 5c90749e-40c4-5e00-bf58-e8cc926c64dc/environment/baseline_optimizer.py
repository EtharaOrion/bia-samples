"""The published baseline recipe, agent visible.

This is the anchor the score is measured against. Copy it into
`submission/optimizer.py` to reproduce the baseline step count, then improve on
it. It is decoupled AdamW with linear warmup and cosine decay, which is the
recipe the baseline anchor in `environment/frozen_manifest.json` was bound
from.
"""

from __future__ import annotations

import math

import torch


def build_optimizer(params, **kwargs):
    tensors = [p for _, p in params]
    decay = [p for p in tensors if p.dim() >= 2]
    plain = [p for p in tensors if p.dim() < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": plain, "weight_decay": 0.0},
        ],
        lr=3e-3,
        betas=(0.9, 0.95),
        eps=1e-10,
    )


def build_schedule(total_steps, **kwargs):
    warmup = max(1, int(0.05 * total_steps))
    floor = 0.10

    def sched(step):
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))

    return sched
