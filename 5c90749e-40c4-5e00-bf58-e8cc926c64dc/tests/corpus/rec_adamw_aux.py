"""Excluded recipe: auxiliary AdamW on embeddings and head, record lineage."""
import torch


class AdamWAux(torch.optim.Optimizer):
    def __init__(self, params, lr=6e-4, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "step" not in st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)
                st["step"] += 1
                m, v = st["exp_avg"], st["exp_avg_sq"]
                m.lerp_(p.grad, 1 - b1)
                v.lerp_(p.grad.square(), 1 - b2)
                bias1 = 1 - b1 ** st["step"]
                bias2 = 1 - b2 ** st["step"]
                scale = (bias2 ** 0.5) / bias1
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.addcdiv_(m, v.sqrt().add_(group["eps"]), value=-group["lr"] * scale)
