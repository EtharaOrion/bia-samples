"""BIA S02 private reference update rule. Oracle only, never agent-visible.

Design statement, in the terms the task poses.

The frozen schedule denies an update rule three things every co-designed
published recipe assumes it has. It denies a long warmup, so second-moment
estimates are still cold when the multiplier reaches one. It injects a restart
bump at forty percent, so any rule whose effective step is tuned for a monotone
decay is thrown off its operating point exactly where a co-designed recipe would
be coasting. And it floors the tail at a quarter of peak, so there is no terminal
anneal at all, which is where a large share of a published record's final loss
drop actually comes from.

This reference does not try to recover the schedule it cannot change. It answers
each denial inside the update rule, which is the only axis left open.

1. Cold start. Second moments are initialized from the first observed gradient
   rather than from zero, so no bias-correction ramp is needed and the first
   steps under a four-tenths-of-one-percent warmup are already scaled.
2. Restart robustness. Hidden matrices are updated with an orthogonalized
   momentum direction whose spectral scale is fixed by construction, so a change
   in the applied learning rate rescales the step without changing its geometry.
   A restart bump moves the step length and not the search direction.
3. No terminal anneal. The rule supplies its own contraction, and it is
   horizon-free by construction because the rule is never told the horizon. The
   contraction is driven by the agreement between the current gradient and the
   accumulated momentum: while they agree, the step runs at full length, and as
   the run becomes noise-dominated and agreement decays, the effective step
   contracts on its own. That reproduces the useful part of an anneal without
   knowing where the end is.

A fourth property is about the budget rather than about the loss. Every quantity
this rule derives stays on the accelerator: the agreement ratio and the
contraction are kept as device tensors and are never read back into a Python
float inside step(). An earlier form of this rule called float() on two norms per
parameter per step, which forced roughly eighty host-device synchronizations per
optimizer step. That form was measured at 157 optimizer steps in 60 seconds on a
shared H100, against 576 steps in the same 60 seconds for the form below, and it
tripped the runner's 420 second wall-clock guard at step 525 of 1400 rather than
finishing the run. The two forms are numerically equivalent to within float32
rounding of a scalar: at seed 0 they agree to within 1.1e-3 of validation loss at
every one of the first 150 recorded steps. The synchronization-free form is what
ships, because a reference that cannot finish inside the enforced budget is not a
reference.

Every step is exactly linear in the learning rate the runner supplies, which the
verifier checks directly, and the rule reads no environment, no file, and no
step budget.
"""

from __future__ import annotations

import torch


def orthogonalize(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration returning an approximately orthogonal factor."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix.to(torch.float32)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        y = x @ x.T
        z = b * y + c * (y @ y)
        x = a * x + z @ x
    if transposed:
        x = x.T
    return x.to(matrix.dtype)


class ReferenceRule(torch.optim.Optimizer):
    MATRIX_GROUP = "hidden_matrix"

    def __init__(self, param_groups, momentum=0.95, beta2=0.95, eps=1e-8,
                 floor=0.25, agreement_decay=0.9):
        defaults = dict(lr=0.0, momentum=momentum, beta2=beta2, eps=eps,
                        floor=floor, agreement_decay=agreement_decay)
        super().__init__(param_groups, defaults)

    @staticmethod
    def _agreement(g: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Clamped cosine between gradient and momentum, kept on the device."""
        denom = g.norm() * m.norm()
        cos = (g.reshape(-1) @ m.reshape(-1)) / denom.clamp_min(1e-30)
        return torch.where(denom > 0, cos, torch.ones_like(cos)).clamp(0.0, 1.0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = float(group["lr"])
            name = group.get("name")
            mu = group["momentum"]
            beta2 = group["beta2"]
            eps = group["eps"]
            floor = group["floor"]
            decay = group["agreement_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["m"] = torch.zeros_like(p)
                    st["v"] = g.detach() * g.detach()
                    st["a"] = torch.ones((), dtype=torch.float32, device=p.device)
                m, v = st["m"], st["v"]
                agree = self._agreement(g.to(torch.float32), m.to(torch.float32))
                st["a"] = decay * st["a"] + (1.0 - decay) * agree
                contraction = floor + (1.0 - floor) * st["a"]
                m.mul_(mu).add_(g, alpha=1.0 - mu)
                if name == self.MATRIX_GROUP and p.ndim == 2:
                    direction = orthogonalize(m)
                    rows, cols = p.shape
                    direction = direction * (max(1.0, rows / cols) ** 0.5)
                else:
                    v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                    direction = m / (v.sqrt() + eps)
                p.sub_(direction * (contraction.to(direction.dtype) * lr))
        return loss


def build_update_rule(param_groups):
    return ReferenceRule(param_groups)
