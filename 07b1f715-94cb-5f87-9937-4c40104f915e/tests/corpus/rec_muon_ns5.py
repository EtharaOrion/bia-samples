"""Fingerprint corpus entry: canonical Muon, orthogonalized momentum with a
co-designed cosine cooldown. Present so a verbatim port of this recipe is
detected before any accelerator time is spent. Not executable evidence.
"""

import math

import torch


def zeropower_via_newtonschulz5(G, steps=5):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, total_steps=1000):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, total_steps=total_steps)
        super().__init__(params, defaults)

    def cooldown(self, t, total):
        return 0.5 * (1.0 + math.cos(math.pi * t / total))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    state["t"] = 0
                state["t"] += 1
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                g = zeropower_via_newtonschulz5(g)
                scale = max(1, p.size(0) / p.size(1)) ** 0.5
                lr = group["lr"] * self.cooldown(state["t"], group["total_steps"])
                p.add_(g, alpha=-lr * scale)
