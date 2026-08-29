"""Fingerprint corpus entry: Kronecker-factored preconditioned SGD, whitening
preconditioner with its published triangular update and schedule. Not
executable evidence.
"""

import torch


class PSGDKron(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, precond_lr=0.1, eps=1e-8, total_steps=1000):
        defaults = dict(lr=lr, precond_lr=precond_lr, eps=eps, total_steps=total_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or p.ndim != 2:
                    continue
                grad = p.grad.float()
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["ql"] = torch.eye(p.shape[0], device=p.device)
                    state["qr"] = torch.eye(p.shape[1], device=p.device)
                state["step"] += 1
                ql, qr = state["ql"], state["qr"]
                pg = ql @ grad @ qr.T
                left = pg @ pg.T - torch.eye(p.shape[0], device=p.device)
                right = pg.T @ pg - torch.eye(p.shape[1], device=p.device)
                ql.sub_(group["precond_lr"] * torch.tril(left) @ ql)
                qr.sub_(group["precond_lr"] * torch.tril(right) @ qr)
                update = ql.T @ (ql @ grad @ qr.T) @ qr
                lr = group["lr"] * max(0.0, 1.0 - state["step"] / group["total_steps"])
                p.add_(update.to(p.dtype), alpha=-lr)
