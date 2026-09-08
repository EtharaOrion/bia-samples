from __future__ import annotations

import math

import torch

ALGORITHMS = ("sgd", "momentum", "nesterov", "adam", "adamw", "lion", "ortho_momentum")
SCHEDULES = ("constant", "linear_decay", "cosine", "wsd", "inverse_sqrt")

def lr_at(recipe: dict, step: int, total: int) -> float:
    base = float(recipe["lr"])
    floor = float(recipe["final_lr_frac"])
    warm = max(1, int(round(float(recipe["warmup_frac"]) * total)))
    if float(recipe["warmup_frac"]) > 0.0 and step < warm:
        return base * (step + 1) / warm
    schedule = recipe["schedule"]
    progress = 0.0 if total <= 1 else min(1.0, max(0.0, step / float(total - 1)))
    if schedule == "constant":
        factor = 1.0
    elif schedule == "linear_decay":
        factor = 1.0 - progress
    elif schedule == "cosine":
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    elif schedule == "wsd":
        hold = 1.0 - float(recipe["decay_frac"])
        factor = 1.0 if progress <= hold else (1.0 - progress) / max(1e-9, 1.0 - hold)
    else:
        factor = 1.0 / math.sqrt(1.0 + step)
    return base * (floor + (1.0 - floor) * max(0.0, factor))

def _orthogonalize(matrix: torch.Tensor, steps: int) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(int(steps)):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.T
    return x.to(matrix.dtype)

class RecipeOptimizer:

    def __init__(self, groups, recipe: dict):
        self.groups = groups
        self.recipe = recipe
        self.algorithm = recipe["optimizer"]
        self.state = {}
        for group in groups:
            for param in group["params"]:
                self.state[id(param)] = {
                    "m": torch.zeros_like(param),
                    "v": torch.zeros_like(param),
                    "t": 0,
                }

    def clip_(self) -> float:
        clip = float(self.recipe["grad_clip"])
        params = [p for group in self.groups for p in group["params"] if p.grad is not None]
        total = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in params))
        if float(total) > clip:
            scale = clip / (float(total) + 1e-12)
            for param in params:
                param.grad.detach().mul_(scale)
        return float(total)

    @torch.no_grad()
    def step(self, lr: float) -> None:
        recipe = self.recipe
        b1, b2 = float(recipe["beta1"]), float(recipe["beta2"])
        eps, decay = float(recipe["eps"]), float(recipe["weight_decay"])
        for group in self.groups:
            group_lr = lr * float(group["lr_mult"])
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                slot = self.state[id(param)]
                slot["t"] += 1
                if self.algorithm == "sgd":
                    update = grad
                elif self.algorithm == "momentum":
                    slot["m"].mul_(b1).add_(grad)
                    update = slot["m"]
                elif self.algorithm == "nesterov":
                    slot["m"].mul_(b1).add_(grad)
                    update = grad + b1 * slot["m"]
                elif self.algorithm in ("adam", "adamw"):
                    slot["m"].mul_(b1).add_(grad, alpha=1.0 - b1)
                    slot["v"].mul_(b2).addcmul_(grad, grad, value=1.0 - b2)
                    bias1 = 1.0 - b1 ** slot["t"]
                    bias2 = 1.0 - b2 ** slot["t"]
                    update = (slot["m"] / bias1) / ((slot["v"] / bias2).sqrt() + eps)
                    if self.algorithm == "adam" and decay > 0.0:
                        update = update + decay * param
                elif self.algorithm == "lion":
                    blended = b1 * slot["m"] + (1.0 - b1) * grad
                    update = torch.sign(blended)
                    slot["m"].mul_(b2).add_(grad, alpha=1.0 - b2)
                elif self.algorithm == "ortho_momentum":
                    slot["m"].mul_(b1).add_(grad)
                    if slot["m"].ndim == 2 and min(slot["m"].shape) > 1:
                        update = _orthogonalize(slot["m"], recipe["ns_steps"])
                    else:
                        update = slot["m"]
                else:
                    raise ValueError("algorithm outside the closed set: " + str(self.algorithm))
                if decay > 0.0 and self.algorithm in ("adamw", "lion"):
                    param.mul_(1.0 - group_lr * decay)
                param.add_(update, alpha=-group_lr)
