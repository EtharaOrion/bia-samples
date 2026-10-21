"""Track-3 optimizer: short-memory launch, compressed cooldown, refinement tail."""

import torch


def _zeropower(g):
    x = g.bfloat16()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(12):
        xx = x @ x.mT
        x = 2.0 * x + (-1.5 * xx + 0.5 * (xx @ xx)) @ x
    return x.mT if transposed else x


@torch.compile(dynamic=False)
def _direction(grad, momentum, beta):
    momentum.lerp_(grad, 1.0 - beta)
    u = _zeropower(grad.lerp(momentum, beta))
    return u * max(1.0, grad.shape[0] / grad.shape[1]) ** 0.5


class Track3Optimizer(torch.optim.Optimizer):
    owns_schedule = True

    def __init__(self, params, lr, weight_decay=0.05):
        params = sorted(list(params), key=lambda p: tuple(p.shape), reverse=True)
        super().__init__(params, dict(lr=lr, base_lr=lr,
                                      weight_decay=weight_decay))

    def set_progress(self, step, max_steps):
        horizon = max_steps * (6.0 / 7.0)
        cooldown_start = 0.30 * horizon
        if step < cooldown_start:
            compressed = 1.0
        elif step < horizon:
            compressed = (horizon - step) / (horizon - cooldown_start)
        else:
            compressed = 0.0

        # Keep refining the blocks while the frozen auxiliary AdamW weights settle.
        tail = 0.08 * max(0.0, (max_steps - step) / (max_steps - horizon))
        eta = max(compressed, tail)
        for group in self.param_groups:
            group["lr"] = group["base_lr"] * eta

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            early = int(group.get("harness_step", 0)) < 125
            beta = 0.85 if early else 0.95
            step_lr = group["lr"] * (0.90 if early else 1.0)
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["momentum"] = torch.zeros_like(p)
                update = _direction(p.grad, state["momentum"], beta)
                p.mul_(1.0 - step_lr * group["weight_decay"])
                p.add_(update, alpha=-step_lr)
        return loss


def build_optimizer(params, lr=0.02, **kwargs):
    return Track3Optimizer(params, lr=lr)
