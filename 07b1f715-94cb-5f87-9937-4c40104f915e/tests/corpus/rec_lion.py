"""Fingerprint corpus entry: Lion, sign-momentum update with its published
decoupled decay and linear decay schedule. Not executable evidence.
"""

import torch


class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0, total_steps=1000):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, total_steps=total_steps)
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
                    state["exp_avg"] = torch.zeros_like(p)
                    state["step"] = 0
                state["step"] += 1
                lr = group["lr"] * max(0.0, 1.0 - state["step"] / group["total_steps"])
                p.mul_(1 - lr * group["weight_decay"])
                exp_avg = state["exp_avg"]
                update = exp_avg.mul(b1).add(grad, alpha=1 - b1).sign_()
                p.add_(update, alpha=-lr)
                exp_avg.mul_(b2).add_(grad, alpha=1 - b2)
