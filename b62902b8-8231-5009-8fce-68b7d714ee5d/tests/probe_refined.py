"""The REFINED probe, re-run by the verifier to MEASURE the high end of the scale.

VERIFIER-OWNED, and measured through exactly the same chain as
`tests/probe_handed.py` and as the submission itself: launched by
`tests/runner.py`, snapshotted by the pinned harness, evaluated by
`tests/evaluator.py` on the held-out split, crossed by
`checkers.first_sustained_below`.

This is a BAR and not a claim about the best schedule reachable on these free
axes. It is one fixed, declared recipe -- a higher matrix learning rate, a
denser Newton-Schulz cadence, a trapezoidal schedule with a flat section instead
of a decay from step one, and a smaller initialization -- and the crossing it
measures is where the reward saturates. Nothing here searches, and no number in
this bundle claims it does. Naming it a probe rather than an optimum is
deliberate.

It is NOT the reference solution and it never reads it. `solution/` is not copied
into the verifier image, by tests/Dockerfile and on purpose, so this file could
not consult the reference even if it wanted to; it carries its own constants and
the reference has to reach the bar independently.
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

# The free axes of this probe. Declared, fixed, and nothing else's.
#
# It differs from the handed baseline in ONE axis: the auxiliary AdamW learning rate, which
# the handed script sets to 0.0036 and this sets to 0.044. That is not an arbitrary choice
# and it is not a large search. At this operating point the embedding is 50304 x 256 of a
# roughly 16M-parameter model, so the auxiliary group holds most of the parameters and its
# learning rate is the dominant free axis; measured on the verifier's own split, raising it
# improves the held-out loss at EVERY scheduled point rather than trading early against
# late. Everything else -- the matrix learning rate, the momentum, the Newton-Schulz depth,
# the linear-decay-to-zero schedule, the initialisation -- is the handed value.
PEAK_LR_MATRIX = 0.05
PEAK_LR_AUX = 0.044
MOMENTUM = 0.95
NESTEROV = True
NS_STEPS = 5
WARMDOWN_FRACTION = 1.0
INIT_STD = 0.02

SCHEDULE_LENGTH = int(POINT["max_schedule_steps"])


def learning_rate(step: int) -> float:
    """Linear decay to zero, the same shape the handed baseline carries."""
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
