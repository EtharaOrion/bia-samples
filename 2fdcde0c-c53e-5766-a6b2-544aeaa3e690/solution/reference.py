"""Reference solution for slot OER-03. Private. Never on the agent surface.

The change against `environment/train_locked.py` is entirely inside the free
axes: a different second-order preconditioner cadence, a different auxiliary
group split, a warmup-free schedule with a trapezoidal warmdown, and a scaled
initialization. Dataset, batch size, architecture and the one-forward-backward-
pass-per-step rule are untouched, which is what keeps the run gradable at all.

The two properties this reference exists to demonstrate are properties of the
GRADING, not of the recipe:

  * the schedule runs to the full max_schedule_steps while the sustained crossing
    lands well before the end, so a stretch of schedule sits past the graded point
    and buys exactly nothing;
  * the script never decides when it crossed and never reports a crossing, so
    there is no readout for the verifier to diverge from.

SCHEDULE_LENGTH follows the re-scaled operating point declared in
environment/operating_point.json: the bound point's 3000 steps of a 124M model at
a batch of 480x1024 could not be graded inside any budget in this corpus, and the
verifier re-runs this script itself, so the graded run paid that cost in full.
The crossing step this reaches is not asserted anywhere in this bundle; the
verifier measures it, and it measures both ends of the reward scale the same way.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, "/app/env")

import torch  # noqa: E402

import bia_harness  # noqa: E402
from bia_loader import build_frozen_model, frozen_batches  # noqa: E402

# --------------------------------------------------------------------------
# FREE AXES. Every constant below is inside what the task leaves free.
# --------------------------------------------------------------------------
# Every constant here is inside the free axes, and every one of them was re-derived by
# measurement at the re-scaled operating point. The values this file used to carry -- a
# trapezoidal schedule with a 0.62 flat section, matrix lr 0.062, auxiliary lr 0.0041,
# init_std 0.014 -- were tuned against the bound 124M point. Measured at the point this
# bundle now executes they are WORSE than the handed baseline, not better: a flat section
# helps a long schedule and costs a short one, and 0.0041 leaves the auxiliary group, which
# holds most of the parameters here, badly underdriven.
SCHEDULE_LENGTH = 400
PEAK_LR_MATRIX = 0.05
PEAK_LR_AUX = 0.022
MOMENTUM = 0.95
NESTEROV = True
NS_STEPS = 6
PRECONDITION_EVERY = 1
WARMDOWN_FRACTION = 1.0
INIT_STD = 0.018


def schedule_scale(step: int) -> float:
    """Linear decay to zero. A schedule, and therefore inside the free axes."""
    return max(0.0, 1.0 - (step / (SCHEDULE_LENGTH * WARMDOWN_FRACTION)))


def build_optimizers(model):
    from muon import Muon  # shipped in the pinned image

    matrices, auxiliary = [], []
    for name, parameter in model.named_parameters():
        if parameter.ndim == 2 and "embed" not in name and "head" not in name:
            matrices.append(parameter)
        else:
            auxiliary.append(parameter)
    muon = Muon(
        matrices,
        lr=PEAK_LR_MATRIX,
        momentum=MOMENTUM,
        nesterov=NESTEROV,
        ns_steps=NS_STEPS,
        precondition_frequency=PRECONDITION_EVERY,
    )
    adamw = torch.optim.AdamW(auxiliary, lr=PEAK_LR_AUX, betas=(0.9, 0.96), weight_decay=0.0)
    for optimizer in (muon, adamw):
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
    return [muon, adamw]


def main() -> None:
    run_dir = pathlib.Path.cwd()
    harness = bia_harness.Harness(run_dir=run_dir, schedule_length=SCHEDULE_LENGTH)

    model = build_frozen_model(init_std=INIT_STD).cuda()
    optimizers = build_optimizers(model)
    batches = frozen_batches()

    for step in range(1, SCHEDULE_LENGTH + 1):
        scale = schedule_scale(step)
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * scale

        # Exactly one forward-backward pass per optimizer step. Frozen axis.
        inputs, targets = next(batches)
        loss = model(inputs, targets)
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        harness.step(model, step)

    # Deliberately reports NO crossing. The verifier recomputes the graded step
    # from its own evaluations, and a reference that also claimed one would be
    # asserting a number it has no way to compute inside this container.


if __name__ == "__main__":
    main()
