"""Fingerprint corpus entry: SOAP/Shampoo-class preconditioner, eigenbasis
rotation with Adam statistics in the rotated frame and its co-designed
schedule. Not executable evidence.
"""

import torch


class SoapLite(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-3, betas=(0.95, 0.95), eps=1e-8,
                 precondition_frequency=10, total_steps=1000):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        precondition_frequency=precondition_frequency, total_steps=total_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None or p.ndim != 2:
                    continue
                grad = p.grad.float()
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["gg_left"] = torch.zeros(p.shape[0], p.shape[0], device=p.device)
                    state["gg_right"] = torch.zeros(p.shape[1], p.shape[1], device=p.device)
                    state["q_left"] = torch.eye(p.shape[0], device=p.device)
                    state["q_right"] = torch.eye(p.shape[1], device=p.device)
                    state["m"] = torch.zeros_like(grad)
                    state["v"] = torch.zeros_like(grad)
                state["step"] += 1
                state["gg_left"].mul_(b2).add_(grad @ grad.T, alpha=1 - b2)
                state["gg_right"].mul_(b2).add_(grad.T @ grad, alpha=1 - b2)
                if state["step"] % group["precondition_frequency"] == 0:
                    state["q_left"] = torch.linalg.eigh(state["gg_left"])[1]
                    state["q_right"] = torch.linalg.eigh(state["gg_right"])[1]
                rotated = state["q_left"].T @ grad @ state["q_right"]
                state["m"].mul_(b1).add_(rotated, alpha=1 - b1)
                state["v"].mul_(b2).addcmul_(rotated, rotated, value=1 - b2)
                step_rot = state["m"] / (state["v"].sqrt() + group["eps"])
                update = state["q_left"] @ step_rot @ state["q_right"].T
                lr = group["lr"] * (1.0 - state["step"] / group["total_steps"])
                p.add_(update.to(p.dtype), alpha=-lr)
