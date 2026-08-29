"""Verifier-owned evaluation. Runs outside the submission's process.

The submission trains and the provided loader writes checkpoints. This module
loads those checkpoints and computes the validation loss itself, on a split that
never enters the training environment. That placement is the whole measurement
argument: anything produced inside a process the submission authored can be
authored too, so the graded quantity is produced here instead. The previous
verifier for this slot read the graded loss out of the submission's standard
output, which is why a five-line script that printed a descending curve scored
full marks.

Three reconciliations decide whether a checkpoint counts, and each corresponds
to one frozen axis.

ON THE VERIFIER'S GRID. Only checkpoints at the verifier's own step milestones
are considered. The cadence is a constant of the bundle, not a submission
choice, so a submission cannot gain by being evaluated more often or at a moment
of its choosing.

CORROBORATED BY THE RUN LEDGER. The ledger the provided loader wrote must affirm
that this milestone was reached at that step index, with exactly the frozen
batch size at every step, and with exactly one forward pass per step. The
condition is phrased as the ledger must affirm rather than the ledger must not
contradict, so an absent ledger drops the run rather than waiving the check.

LOADABLE INTO THE FROZEN ARCHITECTURE. The state dict must fit the verifier's
own model definition under strict=True, which rejects a changed depth, width,
head dimension or vocabulary structurally rather than trusting a declaration.

Residual limitation, recorded rather than papered over: the ledger is written
inside the submission's own container, so a submission that reimplements the
loader can forge it. Making it unforgeable needs process isolation at the mount
layer that this bundle does not own. What the ledger buys is that every ordinary
route, skipping the loader, renaming a checkpoint, enlarging the batch, or
taking extra forward-backward passes per step, is caught deterministically. The
remaining forgery route is a rubric obligation, not a claim of impossibility.
"""
from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass

VAL_SHARD = pathlib.Path(os.environ["BIA_VAL_SHARD"])
STEP_GRID: tuple[int, ...] = tuple(
    sorted(json.loads(pathlib.Path(os.environ["BIA_STEP_GRID"]).read_text())))
SHAPE = json.loads(pathlib.Path(os.environ["BIA_SHAPE"]).read_text())
BATCH_TOKENS = int(SHAPE["batch_tokens_per_step"])


@dataclass(frozen=True)
class RunMeasurement:
    losses: dict[int, float]
    checkpoint_steps: tuple[int, ...]
    ledger_present: bool
    frozen_batch_ok: bool
    one_forward_per_step_ok: bool
    reason: str


def read_ledger(run_dir: pathlib.Path) -> dict[int, dict] | None:
    p = run_dir / "run_ledger.json"
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(raw, dict):
        return None
    out: dict[int, dict] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            return None
        try:
            out[int(key)] = value
        except (TypeError, ValueError):
            return None
    return out


def milestone_corroborated(step: int, ledger: dict[int, dict] | None) -> bool:
    if not ledger or step not in ledger:
        return False
    row = ledger[step]
    return row.get("step") == step and row.get("tokens_served") == step * BATCH_TOKENS


def batch_size_frozen(ledger: dict[int, dict] | None) -> bool:
    """Every recorded milestone served exactly the frozen batch, never more."""
    if not ledger:
        return False
    return all(row.get("batch_tokens_per_step") == BATCH_TOKENS
               for row in ledger.values())


def one_forward_per_step(ledger: dict[int, dict] | None) -> bool:
    """Exactly one forward pass per optimizer step, at every milestone.

    The frozen rule of the task. A submission accumulating several
    forward-backward passes into one optimizer step is running a larger batch
    than the frozen one, which is why this is checked rather than assumed.
    """
    if not ledger:
        return False
    return all(row.get("forward_calls") == step for step, row in ledger.items())


def checkpoint_steps(run_dir: pathlib.Path) -> list[int]:
    found = []
    for p in run_dir.glob("step_*.pt"):
        stem = p.stem.removeprefix("step_")
        if stem.isdigit():
            found.append(int(stem))
    return sorted(s for s in found if s in STEP_GRID)


def evaluate_checkpoint(path: pathlib.Path, model, val_inputs, val_targets,
                        micro_batch: int) -> float | None:
    import torch

    try:
        from frozen_gpt import resolve_device
        state = torch.load(path, map_location=resolve_device(), weights_only=True)
    except Exception:  # noqa: BLE001
        return None
    try:
        model.load_state_dict(state, strict=True)
    except Exception:  # noqa: BLE001
        return None
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(val_inputs), micro_batch):
            total += float(model(val_inputs[i:i + micro_batch],
                                 val_targets[i:i + micro_batch]))
    loss = total / val_targets.numel()
    if loss != loss or loss in (float("inf"), float("-inf")):
        return None
    return loss


def evaluate_run(run_dir: pathlib.Path, micro_batch: int = 32) -> RunMeasurement:
    run_dir = pathlib.Path(run_dir)
    ledger = read_ledger(run_dir)
    steps = checkpoint_steps(run_dir)
    if not steps:
        return RunMeasurement({}, (), ledger is not None, False, False,
                              "no-checkpoints-on-the-verifier-grid")
    if ledger is None:
        return RunMeasurement({}, tuple(steps), False, False, False,
                              "run-ledger-absent")

    frozen_batch_ok = batch_size_frozen(ledger)
    forwards_ok = one_forward_per_step(ledger)

    from frozen_arch import build, load_val_tokens

    model = build()
    val_inputs, val_targets = load_val_tokens(VAL_SHARD)
    losses: dict[int, float] = {}
    for step in steps:
        if not milestone_corroborated(step, ledger):
            continue
        loss = evaluate_checkpoint(run_dir / f"step_{step}.pt", model,
                                   val_inputs, val_targets, micro_batch)
        if loss is not None:
            losses[step] = loss
    return RunMeasurement(losses, tuple(sorted(losses)), True, frozen_batch_ok,
                          forwards_ok, "measured" if losses else "no-evaluable-checkpoints")
