"""Excluded recipe: evolved sign momentum, published as Lion."""
import torch


class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "exp_avg" not in st:
                    st["exp_avg"] = torch.zeros_like(p)
                m = st["exp_avg"]
                p.mul_(1 - group["lr"] * group["weight_decay"])
                update = m.mul(b1).add(p.grad, alpha=1 - b1).sign_()
                p.add_(update, alpha=-group["lr"])
                m.mul_(b2).add_(p.grad, alpha=1 - b2)
