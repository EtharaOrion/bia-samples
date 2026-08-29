"""Starter update rule for BIA S02. Copy this to submission/update_rule.py and improve it.

This is a deliberately plain AdamW-class rule. It ignores everything specific
about the frozen schedule: it has no answer to the very short warmup, no answer
to the restart bump at forty percent, and no answer to a tail that floors at a
quarter of peak instead of annealing to zero. It is a starting point, not a
target.

Contract your replacement must keep:

  build_update_rule(param_groups) -> torch.optim.Optimizer

`param_groups` is the frozen group list. Each entry is a dict with "name" in
{embed, hidden_matrix, head, vector}, "params", and a placeholder "lr". Keep the
"name" key on every group you pass to the optimizer: the runner writes the
frozen learning rate into the group that carries it.

Three rules the verifier enforces on whatever you write here:

1. The update applied to a parameter must be exactly proportional to the "lr"
   the runner wrote into that group. The verifier calls your rule twice from
   identical state at two different learning rates and checks the two parameter
   deltas scale linearly.
2. Your rule must never write "lr" into a param group. The runner reads it back
   after every step.
3. Your rule is never told the total step count and must not try to obtain it.
   No environment reads, no file reads, no imports of the runner or the
   schedule.
"""

from __future__ import annotations

import torch


class StarterAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=0.0, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
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
                m, v, t = st["m"], st["v"], st["t"]
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                mhat = m / (1.0 - beta1 ** t)
                vhat = v / (1.0 - beta2 ** t)
                direction = mhat / (vhat.sqrt() + eps)
                if wd:
                    direction = direction + wd * p
                p.add_(direction, alpha=-lr)
        return loss


def build_update_rule(param_groups):
    return StarterAdamW(param_groups)
