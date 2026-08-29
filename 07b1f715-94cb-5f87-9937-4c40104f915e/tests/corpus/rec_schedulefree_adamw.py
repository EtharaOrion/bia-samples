"""Fingerprint corpus entry: schedule-free AdamW, the published interpolation
between an averaged iterate and an extrapolated one. Present because it is the
first published recipe an agent reaches for when a schedule is imposed from
outside. Not executable evidence.
"""

import torch


class ScheduleFreeAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=2.5e-3, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.0, warmup_steps=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        warmup_steps=warmup_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["z"] = p.detach().clone()
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                k = state["step"]
                z, v = state["z"], state["v"]
                v.mul_(b2).addcmul_(grad, grad, value=1 - b2)
                denom = (v / (1 - b2 ** k)).sqrt().add_(group["eps"])
                warm = min(1.0, k / max(1, group["warmup_steps"])) if group["warmup_steps"] else 1.0
                lr = group["lr"] * warm
                if group["weight_decay"]:
                    grad = grad.add(p, alpha=group["weight_decay"])
                z.addcdiv_(grad, denom, value=-lr)
                ckp = 1.0 / k
                p.mul_(1 - ckp).add_(z, alpha=ckp)
                y = p.mul(1 - b1).add_(z, alpha=b1)
                p.copy_(p.mul(1 - ckp).add_(y, alpha=ckp))
