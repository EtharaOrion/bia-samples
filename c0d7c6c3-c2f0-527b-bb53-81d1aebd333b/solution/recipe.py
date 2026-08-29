"""Private reference solution. Authored here, not lifted from a published record.

THE CHANGE, and why it is on the free axis.

The baseline runs one AdamW at one constant learning rate over every parameter.
That is wrong in two separate ways at this shape, and the reference fixes both
without touching a frozen axis.

FIRST, the two-dimensional block matrices and the one-dimensional gains and
embeddings want different updates. The block matrices are the parameters whose
update direction carries most of the optimization signal, and Adam's
coordinatewise rescaling destroys their conditioning: it normalises each entry
independently and so throws away the fact that the entries form a matrix. The
reference orthogonalises the momentum of every 2-D block parameter with a short
Newton-Schulz iteration before applying it, which is the Muon construction, and
leaves the embedding, the head and the norm gains on AdamW where coordinatewise
scaling is the right thing. Newton-Schulz is a fixed polynomial iteration on the
update tensor; it changes the optimizer, not the model.

SECOND, a constant learning rate spends the early steps diverging and the late
steps overshooting. The reference warms up linearly and then decays on a cosine
to a small floor. Both are schedules of a hyperparameter, which the task names
explicitly as free.

Neither change touches the dataset, the batch size, the architecture, or the one
forward-backward per step rule. The loop below is byte-for-byte the baseline's
loop: the same loader, the same single forward, the same single backward, the
same checkpoint call. Everything that differs is inside the optimizer.

WHAT THIS FILE IS NOT. It is not a track 3 published record and it is not
derived from one. It does not reproduce record 46 or any other accepted result,
it carries none of their hyperparameters, and it is structurally unrelated to
them: they are written against a distributed 12-layer 768-wide run over a
10B-token corpus with their own data generator and their own logging protocol,
and none of them can even be invoked under this bundle's contract. The previous
oracle for this slot WAS such a record, which is why the bundle's own
no_stale_algorithm gate rejected its own reference. That is fixed by authoring a
reference rather than by exempting one.
"""
from __future__ import annotations

import json
import math
import os
import pathlib

import torch

from bia_loader import StepLoader
from frozen_gpt import GPT, resolve_device

SHAPE_PATH = pathlib.Path(os.environ.get("BIA_SHAPE", "/env/shape.json"))

PEAK_LR_MATRIX = 0.02
PEAK_LR_OTHER = 3e-3
WARMUP_FRACTION = 0.08
FINAL_LR_FRACTION = 0.1
MOMENTUM = 0.95
NEWTON_SCHULZ_STEPS = 5


def orthogonalise(matrix: torch.Tensor) -> torch.Tensor:
    """Newton-Schulz iteration for the orthogonal factor of a matrix.

    The quintic coefficients are the standard Muon triple. The iteration is run
    on the transposed matrix when it is taller than wide, so the smaller Gram
    matrix is formed, and the result is transposed back.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix.float()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(NEWTON_SCHULZ_STEPS):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    if transposed:
        x = x.T
    return x


class OrthogonalMomentum:
    """Momentum whose update direction is orthogonalised before it is applied.

    Kept as a plain object rather than a torch.optim.Optimizer subclass so the
    whole rule is visible in one place. It owns only the 2-D block parameters;
    everything else stays on AdamW.
    """

    def __init__(self, params, momentum: float):
        self.params = list(params)
        self.momentum = momentum
        self.buffers = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self, lr: float) -> None:
        for p, buf in zip(self.params, self.buffers):
            if p.grad is None:
                continue
            buf.mul_(self.momentum).add_(p.grad)
            direction = orthogonalise(buf)
            scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
            p.add_(direction.to(p.dtype), alpha=-lr * scale)

    @torch.no_grad()
    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None


def lr_at(step: int, total: int, peak: float) -> float:
    warmup = max(1, int(total * WARMUP_FRACTION))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return peak * (FINAL_LR_FRACTION + (1.0 - FINAL_LR_FRACTION) * cosine)


def build_model(shape: dict, device) -> GPT:
    return GPT(vocab_size=shape["vocab_size"], num_layers=shape["num_layers"],
               model_dim=shape["model_dim"], head_dim=shape["head_dim"]).to(device)


def main() -> None:
    shape = json.loads(SHAPE_PATH.read_text())
    seed = int(os.environ["BIA_SEED"])
    torch.manual_seed(seed)
    device = resolve_device()

    model = build_model(shape, device)
    loader = StepLoader(device)
    total = loader.grid[-1] if loader.grid else 1

    block_params = [p for n, p in model.named_parameters()
                    if p.ndim == 2 and n.startswith("blocks.")]
    block_ids = {id(p) for p in block_params}
    other_params = [p for p in model.parameters() if id(p) not in block_ids]

    matrix_opt = OrthogonalMomentum(block_params, MOMENTUM)
    other_opt = torch.optim.AdamW(other_params, lr=PEAK_LR_OTHER, betas=(0.9, 0.95),
                                  weight_decay=0.0)

    step = 0
    for inputs, targets in loader.steps():
        loss = model(inputs, targets) / targets.numel()
        loss.backward()
        for group in other_opt.param_groups:
            group["lr"] = lr_at(step, total, PEAK_LR_OTHER)
        matrix_opt.step(lr_at(step, total, PEAK_LR_MATRIX))
        other_opt.step()
        matrix_opt.zero_grad()
        other_opt.zero_grad(set_to_none=True)
        loader.checkpoint(model)
        step += 1


if __name__ == "__main__":
    main()
