"""The HANDED baseline, re-run by the verifier to MEASURE the low end of the scale.

VERIFIER-OWNED. This is not a submission and it is never mounted into the agent
environment. It is submission-SHAPED on purpose: `tests/anchors.py` launches it
through the same `tests/runner.py` that launches a submission, into a
verifier-owned run directory, and the crossing it produces is read back through
the same `evaluate_ledger` and the same `checkers.first_sustained_below`. That is
what makes the bar and the submission commensurable -- they are measured by one
chain, not by two that are asserted to agree.

WHAT RECIPE THIS IS. Exactly the free-axis constants `environment/train_locked.py`
hands the agent: tuned Muon on the hidden matrices, an auxiliary AdamW group on
the embeddings, scalars and head, linear decay to zero, init_std 0.02. That file
is what the instruction tells the agent to copy and change, so its crossing step
is what the environment gives away for free, and a submission scores above zero
only by crossing sooner than it.

The one thing that is NOT copied from it is `SCHEDULE_LENGTH`. The handed script
carries a schedule proportional to the bound operating point; here the schedule
is the full `max_schedule_steps` the re-scaled point declares, so the handed
recipe is given every step the envelope allows to cross in. A bar measured on a
short schedule would be a bar that is easy for the wrong reason.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, "/app/env")

import torch  # noqa: E402

import bia_corpus  # noqa: E402
import bia_harness  # noqa: E402
from bia_loader import build_frozen_model, frozen_batches  # noqa: E402

POINT = bia_corpus.operating_point()

# The free axes, verbatim from environment/train_locked.py.
PEAK_LR_MATRIX = 0.05
PEAK_LR_AUX = 0.0036
MOMENTUM = 0.95
NESTEROV = True
NS_STEPS = 5
WARMDOWN_FRACTION = 1.0
INIT_STD = 0.02

SCHEDULE_LENGTH = int(POINT["max_schedule_steps"])


def learning_rate(step: int) -> float:
    return max(0.0, 1.0 - (step / (SCHEDULE_LENGTH * WARMDOWN_FRACTION)))


def build_optimizers(model):
    from muon import Muon

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
    muon = Muon(matrix_params, lr=PEAK_LR_MATRIX, momentum=MOMENTUM,
                nesterov=NESTEROV, ns_steps=NS_STEPS)
    adamw = torch.optim.AdamW(aux_params, lr=PEAK_LR_AUX, betas=(0.9, 0.95), weight_decay=0.0)
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
        scale = learning_rate(step)
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * scale

        inputs, targets = next(batches)
        loss = model(inputs, targets)
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        harness.step(model, step)


if __name__ == "__main__":
    main()
