"""The frozen published recipe.

This is the optimizer configuration under which the published result was
reported: decoupled AdamW on every parameter, a short linear warmup into a
cosine decay, and global gradient-norm clipping. It is harness owned. The
submission never sees it as a writable surface and never runs it.

It drives both anchors:

  anchor_target    this recipe on the un-ablated model, which reproduces the
                   published result at the scaled operating point
  anchor_baseline  this recipe on the ablated model, which is exactly what a
                   naive port of the published recipe produces once the
                   component that carried the gain is gone

The two anchors are measured by the harness in the same process, on the same
device, on the same frozen data order, so the reward denominator is a measured
quantity rather than a number an author typed.
"""

from __future__ import annotations

import math

import torch

PUBLISHED = {
    "optimizer": "AdamW",
    "lr": 1.0e-3,
    "betas": (0.9, 0.95),
    "eps": 1.0e-8,
    "weight_decay": 0.1,
    "warmup_frac": 0.05,
    "final_lr_frac": 0.1,
    "grad_clip": 1.0,
}


class ReferenceRecipe:
    """Frozen. Never exposed as an agent-writable surface."""

    def __init__(self, matrices, others, cfg):
        self.params = [p for _, p in matrices] + [p for _, p in others]
        self.opt = torch.optim.AdamW(
            self.params,
            lr=PUBLISHED["lr"],
            betas=PUBLISHED["betas"],
            eps=PUBLISHED["eps"],
            weight_decay=PUBLISHED["weight_decay"],
        )
        self.cfg = cfg

    def _lr_at(self, step, total):
        warm = max(1, int(PUBLISHED["warmup_frac"] * total))
        if step < warm:
            return PUBLISHED["lr"] * (step + 1) / warm
        prog = (step - warm) / max(1, total - warm)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
        floor = PUBLISHED["final_lr_frac"]
        return PUBLISHED["lr"] * (floor + (1.0 - floor) * cos)

    def step(self, step, total):
        torch.nn.utils.clip_grad_norm_(self.params, PUBLISHED["grad_clip"])
        lr = self._lr_at(step, total)
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.opt.step()


def factory(matrices, others, cfg):
    return ReferenceRecipe(matrices, others, cfg)
