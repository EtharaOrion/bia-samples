"""Private reference submission for bia S09 multi-objective-frontier.

This is the oracle. It is never visible to the agent.

The design intent it encodes is the one the slot is built around: a frontier is
a set, not a point, so the submission does not tune one optimizer well, it
proposes a family of optimizers whose members sit at deliberately different
places on the frozen loss versus density tradeoff surface.

The base rule is decoupled-moment AdamW. The free axis carries one addition:
a proximal soft-threshold applied after the base step to the frozen weight
matrices, with strength lam. lam = 0 recovers a pure loss-seeking point.
Large lam drives exact zeros into the weight matrices, which the scale-invariant
density objective rewards and the loss objective punishes. Sweeping lam walks
the declared tradeoff surface, and the six proposed points are chosen to spread
along it rather than to cluster at either end.
"""

from __future__ import annotations

import math

import torch

FRONTIER_POINTS = 6

def _is_weight_matrix(p) -> bool:
    return p.dim() == 2


class ProximalAdamW(torch.optim.Optimizer):
    """AdamW with a per-group proximal soft-threshold on the parameter itself.

    The soft-threshold is the proximal operator of an L1 penalty, applied after
    the base step with threshold lam * lr. It creates exact zeros rather than
    merely small values, which is what a scale-invariant density objective can
    actually see.
    """

    def __init__(self, groups, lr=0.02, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, lam=0.0)
        super().__init__(groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            lam = group.get("lam", 0.0)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["t"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                st["t"] += 1
                t = st["t"]
                m, v = st["m"], st["v"]
                m.mul_(b1).add_(g, alpha=1.0 - b1)
                v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                mhat = m / (1.0 - b1**t)
                vhat = v / (1.0 - b2**t)
                if wd:
                    p.mul_(1.0 - lr * wd)
                p.add_(mhat / (vhat.sqrt() + eps), alpha=-lr)
                if lam > 0.0:
                    thr = lam * lr
                    p.copy_(torch.sign(p) * torch.clamp(p.abs() - thr, min=0.0))
        return loss


def propose_frontier():
    """Exactly FRONTIER_POINTS optimizer configurations, spread along the surface.

    lam is swept geometrically because the density response to L1 strength is
    roughly logarithmic: a linear sweep would pile five of the six points into
    the dense corner and collapse most of the hypervolume.
    """
    return [
        {"lam": 0.00, "lr_mult": 1.00},
        {"lam": 0.03, "lr_mult": 1.00},
        {"lam": 0.08, "lr_mult": 1.00},
        {"lam": 0.15, "lr_mult": 1.00},
        {"lam": 0.25, "lr_mult": 0.95},
        {"lam": 0.40, "lr_mult": 0.90},
    ]


def build_optimizer(named_params, point, lr, **kwargs):
    lam = float(point["lam"])
    lr_eff = float(lr) * float(point.get("lr_mult", 1.0))
    matrices, other = [], []
    for _name, p in named_params:
        (matrices if _is_weight_matrix(p) else other).append(p)
    groups = [
        {"params": matrices, "lam": lam},
        {"params": other, "lam": 0.0},
    ]
    return ProximalAdamW(groups, lr=lr_eff, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
