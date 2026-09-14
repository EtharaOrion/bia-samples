"""The locked baseline. It reaches the target loss. Your job is to reach it sooner.

This is the reference instantiation of the upstream track-3 baseline: tuned Muon
on the hidden matrices with an auxiliary AdamW group on the embeddings, scalars
and head, on a linear-decay-to-zero schedule over 3250 optimizer steps. It is the
anchor the score's denominator comes from.

Copy this file to /app/submission.py and change it. Every line below that touches
the optimizer, its hyperparameters, its schedules or the initialization is yours.
Every line that touches the dataset, the batch size, the architecture, or the
one-forward-backward-pass-per-step rule is frozen; moving one produces a run the
grader refuses with the reason `frozen-axis-moved` rather than a lower score.

Note what this script does NOT do. It never decides when it has crossed. It never
computes the number that scores it. It calls harness.step(model, i) once per
optimizer step and lets the pinned harness snapshot the parameters on its own
cadence. The verifier evaluates those snapshots itself, unsmoothed, on a held-out
split that is not in this container.
"""

from __future__ import annotations

import pathlib

import torch

import bia_harness
from bia_loader import frozen_batches, build_frozen_model

# --------------------------------------------------------------------------
# FREE. Everything in this block is yours.
# --------------------------------------------------------------------------
SCHEDULE_LENGTH = 3250
PEAK_LR_MATRIX = 0.05
PEAK_LR_AUX = 0.0036
MOMENTUM = 0.95
NESTEROV = True
NS_STEPS = 5
WARMDOWN_FRACTION = 1.0
INIT_STD = 0.02


def learning_rate(step: int) -> float:
    """Linear decay to zero. A schedule, and therefore yours to replace."""
    remaining = max(0.0, 1.0 - (step / (SCHEDULE_LENGTH * WARMDOWN_FRACTION)))
    return remaining


def build_optimizers(model):
    """Tuned Muon on the hidden matrices, auxiliary AdamW on everything else."""
    from muon import Muon  # shipped in the pinned image

    # Partitioned by NAME, not by `p not in matrix_params`. That membership test
    # evaluates `tensor == tensor` elementwise and then takes bool() of the
    # result, so it raises
    #   RuntimeError: The size of tensor a (256) must match the size of tensor b
    #   (50304) at non-singleton dimension 0
    # the first time it compares a block matrix against the embedding. It raised
    # at the bound shape too: this script could never reach one optimizer step,
    # at any operating point, and it is the script the instruction tells you to
    # copy. The two groups below are the same two groups it meant.
    matrix_names = {n for n, p in model.named_parameters()
                    if p.ndim == 2 and "embed" not in n and "head" not in n}
    matrix_params = [p for n, p in model.named_parameters() if n in matrix_names]
    aux_params = [p for n, p in model.named_parameters() if n not in matrix_names]
    muon = Muon(matrix_params, lr=PEAK_LR_MATRIX, momentum=MOMENTUM, nesterov=NESTEROV, ns_steps=NS_STEPS)
    adamw = torch.optim.AdamW(aux_params, lr=PEAK_LR_AUX, betas=(0.9, 0.95), weight_decay=0.0)
    return [muon, adamw]


# --------------------------------------------------------------------------
# FROZEN. Editing anything below produces a refused run, not a faster one.
# --------------------------------------------------------------------------
def main() -> None:
    run_dir = pathlib.Path.cwd()
    harness = bia_harness.Harness(run_dir=run_dir, schedule_length=SCHEDULE_LENGTH)

    model = build_frozen_model(init_std=INIT_STD).cuda()
    optimizers = build_optimizers(model)
    batches = frozen_batches()

    for step in range(1, SCHEDULE_LENGTH + 1):
        scale = learning_rate(step)
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = group.get("initial_lr", group["lr"]) * scale

        # Exactly one forward-backward pass per optimizer step. Frozen by
        # upstream rule; a second pass here is a refused run.
        inputs, targets = next(batches)
        loss = model(inputs, targets)
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        harness.step(model, step)

    # Optional and never graded. Reported values are compared with the verifier's
    # own recomputation, and a divergence is graded rather than absorbed, so
    # reporting nothing is always safe.
    harness.report(final_train_loss=float(loss.detach()))


if __name__ == "__main__":
    main()
