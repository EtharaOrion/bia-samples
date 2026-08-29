"""Private reference solution. Authored here, not lifted from a published record.

THE CHANGE, and why it is on the free axis.

The shipped baseline runs stochastic gradient descent with momentum at one
constant learning rate over every parameter. That is wrong in three separate
ways at this shape, and the reference fixes all three without touching a frozen
axis.

FIRST, the gradient scale differs by orders of magnitude between the embedding,
the output head and the transformer body, so a single learning rate is either
too small for one of them or divergent for another. The reference rescales each
parameter's update by a running estimate of its own gradient second moment,
which is the Adam construction, and so removes the shared scale entirely.

SECOND, momentum without bias correction spends the first few dozen steps
applying an update built from an almost empty buffer. The reference corrects
both moment estimates for their initialization bias, so the first steps are
full-sized rather than damped.

THIRD, a constant learning rate cannot both move quickly early and settle late.
The reference warms up linearly over a short prefix and then decays on a cosine
to a small floor.

None of the three touches the dataset, the batch size, the architecture, or the
one forward-backward per step rule. The loop below is the baseline's loop: the
same loader, the same single forward, the same single backward, the same
checkpoint call. Everything that differs is inside the optimizer.

WHAT THIS FILE IS NOT. It is not a track 3 published record and it is not
derived from one. The previous oracle for this slot WAS such a record, and it
could not run under this bundle's contract at all: it took --seed while the
grader invoked --steps, it required eight GPUs through torchrun, it read a
10B-token corpus that ships nowhere in the bundle, and it never read the seed
variable the bundle's own checker looked for. The bundle's own fingerprint gate
rejected it on sight. That is fixed by authoring a reference rather than by
exempting one.
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

PEAK_LR = 4e-3
BETA_FAST = 0.9
BETA_SLOW = 0.95
EPSILON = 1e-8
WARMUP_FRACTION = 0.06
FINAL_LR_FRACTION = 0.05


class BiasCorrectedMoments:
    """Per-parameter second-moment rescaling with both moments bias corrected.

    Written out rather than taken from torch.optim so the whole rule is visible
    in one place, which is what the task asks a submission to do.
    """

    def __init__(self, params, beta_fast: float, beta_slow: float, eps: float):
        self.params = list(params)
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.eps = eps
        self.first = [torch.zeros_like(p) for p in self.params]
        self.second = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    @torch.no_grad()
    def step(self, lr: float) -> None:
        self.t += 1
        correct_fast = 1.0 - self.beta_fast ** self.t
        correct_slow = 1.0 - self.beta_slow ** self.t
        for p, m, v in zip(self.params, self.first, self.second):
            if p.grad is None:
                continue
            m.mul_(self.beta_fast).add_(p.grad, alpha=1.0 - self.beta_fast)
            v.mul_(self.beta_slow).addcmul_(p.grad, p.grad, value=1.0 - self.beta_slow)
            direction = (m / correct_fast) / ((v / correct_slow).sqrt() + self.eps)
            p.add_(direction, alpha=-lr)

    @torch.no_grad()
    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None


def lr_at(step: int, total: int) -> float:
    warmup = max(1, int(total * WARMUP_FRACTION))
    if step < warmup:
        return PEAK_LR * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return PEAK_LR * (FINAL_LR_FRACTION + (1.0 - FINAL_LR_FRACTION) * cosine)


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
    opt = BiasCorrectedMoments(model.parameters(), BETA_FAST, BETA_SLOW, EPSILON)

    step = 0
    for inputs, targets in loader.steps():
        loss = model(inputs, targets) / targets.numel()
        loss.backward()
        opt.step(lr_at(step, total))
        opt.zero_grad()
        loader.checkpoint(model)
        step += 1


if __name__ == "__main__":
    main()
