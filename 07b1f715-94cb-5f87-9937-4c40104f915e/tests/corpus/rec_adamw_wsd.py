"""Fingerprint corpus entry: AdamW with a co-designed warmup-stable-decay
schedule. Present so a verbatim port is detected before any accelerator time is
spent. Not executable evidence.
"""

import torch


class AdamWWSD(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1,
                 warmup=0.1, decay_start=0.8, total_steps=1000):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        warmup=warmup, decay_start=decay_start, total_steps=total_steps)
        super().__init__(params, defaults)

    def factor(self, t, group):
        frac = t / group["total_steps"]
        if frac < group["warmup"]:
            return frac / group["warmup"]
        if frac < group["decay_start"]:
            return 1.0
        return max(0.0, (1.0 - frac) / (1.0 - group["decay_start"]))

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
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.mul_(b1).add_(grad, alpha=1 - b1)
                exp_avg_sq.mul_(b2).addcmul_(grad, grad, value=1 - b2)
                bc1 = 1 - b1 ** state["step"]
                bc2 = 1 - b2 ** state["step"]
                denom = (exp_avg_sq / bc2).sqrt().add_(group["eps"])
                lr = group["lr"] * self.factor(state["step"], group)
                p.mul_(1 - lr * group["weight_decay"])
                p.addcdiv_(exp_avg / bc1, denom, value=-lr)
