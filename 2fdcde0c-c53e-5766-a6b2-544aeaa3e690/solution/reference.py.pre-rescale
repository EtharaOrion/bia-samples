"""Reference solution for slot OER-03. Private. Never on the agent surface.

The change against `environment/train_locked.py` is entirely inside the free
axes: a different second-order preconditioner cadence, a different auxiliary
group split, a warmup-free schedule with a trapezoidal warmdown, and a scaled
initialization. Dataset, batch size, architecture and the one-forward-backward-
pass-per-step rule are untouched, which is what keeps the run gradable at all.

The two properties this reference exists to demonstrate are properties of the
GRADING, not of the recipe:

  * the schedule is set to 3000 steps while the sustained crossing lands at 2600,
    so 400 steps of schedule sit past the graded point and buy exactly nothing;
  * the script never decides when it crossed and never reports a crossing, so
    there is no readout for the verifier to diverge from.

Its measured behaviour is recorded in solution/grounding.yaml under `golden`, and
that record is an authored deterministic fixture rather than a measured rollout.
The gap is declared: gap-oer-03-golden-trajectory-derived-not-measured.
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
SCHEDULE_LENGTH = 3000
PEAK_LR_MATRIX = 0.062
PEAK_LR_AUX = 0.0041
MOMENTUM = 0.94
NESTEROV = True
NS_STEPS = 6
PRECONDITION_EVERY = 2
FLAT_FRACTION = 0.62
INIT_STD = 0.014


def schedule_scale(step: int) -> float:
    """Trapezoidal: flat, then linear warmdown to zero over the tail."""
    flat = int(SCHEDULE_LENGTH * FLAT_FRACTION)
    if step <= flat:
        return 1.0
    tail = max(1, SCHEDULE_LENGTH - flat)
    return max(0.0, 1.0 - ((step - flat) / tail))


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
