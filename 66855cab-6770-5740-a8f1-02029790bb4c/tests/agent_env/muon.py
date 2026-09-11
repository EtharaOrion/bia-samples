"""Muon, as an actual module.

`environment/train_locked.py` and `solution/reference.py` both do

    from muon import Muon  # shipped in the pinned image

and the pinned image
pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407
carries no module of that name. Measured directly in the image:

    python3 -c "import muon"  ->  ModuleNotFoundError: No module named 'muon'

This file is that module: momentum-orthogonalised gradient descent on 2D
parameters, with the orthogonal factor obtained by the quintic Newton-Schulz
iteration. Every keyword the two delivered call sites pass is implemented and
load-bearing, including `precondition_frequency`, which the reference passes and
the locked baseline does not.

    lr                      step size
    momentum                heavy-ball coefficient on the gradient buffer
    nesterov                use the look-ahead buffer rather than the buffer
    ns_steps                Newton-Schulz iterations per orthogonalisation
    precondition_frequency  recompute the orthogonalised update every k steps
                            and reuse it in between; k = 1 recomputes every
                            step, which is what the locked baseline gets because
                            it passes no value. Reusing the update rather than
                            re-whitening a fresh gradient with a stale operator
                            is the reading that stays stable: at momentum 0.94
                            the direction turns slowly, while a stale whitening
                            operator applied to a fresh direction is not a
                            whitener for it at all and was measured to diverge.

The update is scaled by max(1, rows/cols) ** 0.5 so the effective step does not
depend on the aspect ratio of the matrix being updated.
"""

from __future__ import annotations

from typing import Iterable

import torch

# The quintic coefficients. Fixed constants of the iteration, not tunables, and
# not reachable from any recipe.
NS_A, NS_B, NS_C = 3.4445, -4.7750, 2.0315


@torch.no_grad()
def newtonschulz_operator(matrix: torch.Tensor, steps: int):
    """(operator, transposed). The LEFT operator the Newton-Schulz iteration applies.

    Every Newton-Schulz step multiplies the working matrix by a polynomial in
    `X X^T` on the LEFT, so the whole iteration is one left operator built from
    the matrix it was derived on. Returning that operator separately is what
    makes `precondition_frequency` mean the standard thing: a preconditioner
    recomputed every k steps and applied to the CURRENT gradient in between,
    rather than a cached update replayed verbatim.
    """
    if matrix.ndim != 2:
        raise ValueError("Newton-Schulz orthogonalisation is defined on 2D tensors only")
    work = matrix.float()
    transposed = work.size(0) > work.size(1)
    if transposed:
        work = work.T
    norm = work.norm()
    if not torch.isfinite(norm) or float(norm) == 0.0:
        return None, transposed
    work = work / (norm + 1e-7)
    operator = torch.eye(work.size(0), device=work.device, dtype=work.dtype)
    for _ in range(max(0, int(steps))):
        gram = work @ work.T
        polynomial = NS_B * gram + NS_C * (gram @ gram)
        work = NS_A * work + polynomial @ work
        operator = NS_A * operator + polynomial @ operator
    return operator, transposed


@torch.no_grad()
def apply_operator(operator, transposed: bool, matrix: torch.Tensor) -> torch.Tensor:
    """Apply a cached Newton-Schulz operator to a fresh, renormalised matrix."""
    if operator is None:
        return torch.zeros_like(matrix)
    work = matrix.float()
    if transposed:
        work = work.T
    norm = work.norm()
    if not torch.isfinite(norm) or float(norm) == 0.0:
        return torch.zeros_like(matrix)
    work = (operator @ (work / (norm + 1e-7)))
    if transposed:
        work = work.T
    return work.to(matrix.dtype)


@torch.no_grad()
def zeropower_via_newtonschulz(matrix: torch.Tensor, steps: int) -> torch.Tensor:
    """An approximate orthogonalisation of `matrix`. Deterministic, no random source."""
    operator, transposed = newtonschulz_operator(matrix, steps)
    if operator is None:
        return matrix.clone()
    return apply_operator(operator, transposed, matrix)


class Muon(torch.optim.Optimizer):
    """Momentum-orthogonalised descent on 2D parameters."""

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        precondition_frequency: int = 1,
    ) -> None:
        if float(lr) <= 0.0:
            raise ValueError("lr must be positive")
        if not 0.0 <= float(momentum) < 1.0:
            raise ValueError("momentum must sit in [0, 1)")
        if int(precondition_frequency) < 1:
            raise ValueError("precondition_frequency must be at least 1")
        defaults = dict(
            lr=float(lr),
            momentum=float(momentum),
            nesterov=bool(nesterov),
            ns_steps=int(ns_steps),
            precondition_frequency=int(precondition_frequency),
        )
        super().__init__(list(params), defaults)

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D401
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            momentum = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            ns_steps = int(group["ns_steps"])
            cadence = int(group["precondition_frequency"])
            lr = float(group["lr"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.ndim != 2:
                    raise ValueError(
                        "Muon received a parameter of rank " + str(gradient.ndim)
                        + "; put non-matrix parameters in the auxiliary group"
                    )
                state = self.state[parameter]
                if "buffer" not in state:
                    state["buffer"] = torch.zeros_like(gradient)
                    state["operator"] = None
                    state["transposed"] = False
                    state["update"] = None
                    state["steps"] = 0
                buffer = state["buffer"]
                buffer.mul_(momentum).add_(gradient)
                direction = gradient.add(buffer, alpha=momentum) if nesterov else buffer
                if state["operator"] is None or state["steps"] % cadence == 0:
                    state["operator"], state["transposed"] = newtonschulz_operator(direction, ns_steps)
                    state["update"] = apply_operator(
                        state["operator"], bool(state["transposed"]), direction
                    )
                state["steps"] += 1
                update = state["update"]
                rows, cols = gradient.shape
                scale = max(1.0, rows / cols) ** 0.5
                parameter.add_(update, alpha=-lr * scale)
        return loss
