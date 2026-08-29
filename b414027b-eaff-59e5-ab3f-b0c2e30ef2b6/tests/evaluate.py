"""Verifier-owned evaluation. Runs outside the submission's process.

Ported from a prior voided attempt, re-keyed from optimizer steps to training
tokens.

The submission trains and the provided loader writes checkpoints. This module
loads them and computes the loss itself, on a validation split that never enters
the agent environment. That placement is the whole security argument: anything
executed inside a process the agent authored can be forged, so the graded
quantity is produced here instead.

Loading also enforces the frozen architecture for free. The state dict must fit
the verifier's own model definition, so a submission that changed layer count,
width or vocabulary fails to load rather than being trusted to have left them
alone.

Two additions over the ported original, both consequences of tokens being the
graded axis rather than steps.

First, only checkpoints sitting on the verifier's predeclared token grid are
considered. The grid is a constant here, not a submission choice, which is what
makes the cadence verifier-owned rather than agent-expressible.

Second, each checkpoint must be corroborated by the loader's own token ledger.
The ledger is written by the provided loader as it serves batches, so a
checkpoint claiming a token count the loader never reached is dropped.
"""
from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass

VAL_SHARD = pathlib.Path(os.environ["BIA_VAL_SHARD"])
CHECKPOINT_DIR = pathlib.Path(os.environ.get("BIA_CHECKPOINT_DIR", "/submission/checkpoints"))

# The verifier-owned evaluation cadence. Fixed here, identical for every
# submission and every seed, and not expressible by the agent. Values are
# training-token counts. Populated from the frozen scaled shape when the
# operating point is measured; empty until then, which fails closed because an
# empty grid yields no evaluable checkpoints.
EXPECTED_TOKEN_GRID: tuple[int, ...] = tuple(json.loads(pathlib.Path(os.environ["BIA_TOKEN_GRID"]).read_text()))


@dataclass(frozen=True)
class Evaluation:
    training_tokens: int
    loss: float | None
    error: str | None


def checkpoint_tokens(run_dir: pathlib.Path) -> list[int]:
    """Token counts for which a checkpoint file exists, restricted to the grid."""
    found = []
    for p in run_dir.glob("tokens_*.pt"):
        stem = p.stem.removeprefix("tokens_")
        if stem.isdigit():
            found.append(int(stem))
    if not EXPECTED_TOKEN_GRID:
        return []
    return sorted(t for t in found if t in EXPECTED_TOKEN_GRID)


def read_token_ledger(run_dir: pathlib.Path) -> dict[int, int] | None:
    """The provided loader's own record of tokens served at each checkpoint.

    Returns None when absent or unreadable, which drops the whole run rather
    than trusting checkpoint filenames alone.
    """
    p = run_dir / "token_ledger.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, dict):
        return None
    out: dict[int, int] = {}
    for k, v in data.items():
        try:
            out[int(k)] = int(v)
        except (TypeError, ValueError):
            return None
    return out


def ledger_corroborates(tokens: int, ledger: dict[int, int] | None) -> bool:
    """A checkpoint counts only if the loader recorded serving that many tokens.

    Residual limitation, recorded rather than papered over: the ledger is written
    inside the submission's own container, so a determined submission could forge
    it. Making it unforgeable needs process isolation this bundle does not own.
    What the ledger does buy is that the ordinary paths to an understated count,
    naming a checkpoint with a smaller number or skipping the loader entirely,
    are both caught deterministically. The remaining forgery route is covered by
    a rubric rather than claimed as structurally impossible.
    """
    if ledger is None:
        return False
    return ledger.get(tokens) == tokens


def build_frozen_model():
    """Instantiate the architecture the task froze.

    Lives in the verifier image so the agent cannot redefine it.
    """
    from frozen_arch import build

    return build()


def load_validation_batch():
    from frozen_arch import load_val_tokens

    return load_val_tokens(VAL_SHARD)


def evaluate_checkpoint(path: pathlib.Path, model, val_inputs, val_targets, micro_batch: int) -> Evaluation:
    import torch

    tokens = int(path.stem.removeprefix("tokens_"))
    try:
        from frozen_gpt import resolve_device
        state = torch.load(path, map_location=resolve_device(), weights_only=True)
    except Exception as exc:  # noqa: BLE001
        return Evaluation(tokens, None, f"checkpoint-unreadable: {type(exc).__name__}")

    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:  # noqa: BLE001
        return Evaluation(tokens, None, f"architecture-mismatch: {type(exc).__name__}")

    model.eval()
    total = 0.0
    n_tokens = val_targets.numel()
    with torch.no_grad():
        for i in range(0, len(val_inputs), micro_batch):
            total += float(model(val_inputs[i:i + micro_batch],
                                 val_targets[i:i + micro_batch]))
    loss = total / n_tokens
    if not (loss == loss) or loss in (float("inf"), float("-inf")):
        return Evaluation(tokens, None, "non-finite-loss")
    return Evaluation(tokens, loss, None)


def evaluate_run(run_dir: pathlib.Path, micro_batch: int = 64) -> dict[int, float]:
    """Every on-grid checkpoint in one run, keyed by training tokens.

    Omission is deliberate: an absent measurement must never become a pass, and
    the crossing search intersects grids across seeds, so a point that failed to
    evaluate on any seed drops out of consideration entirely.
    """
    ledger = read_token_ledger(run_dir)
    tokens_list = checkpoint_tokens(run_dir)
    if not tokens_list:
        return {}
    model = build_frozen_model()
    val_inputs, val_targets = load_validation_batch()
    out: dict[int, float] = {}
    for tokens in tokens_list:
        if not ledger_corroborates(tokens, ledger):
            continue
        result = evaluate_checkpoint(run_dir / f"tokens_{tokens}.pt", model,
                                     val_inputs, val_targets, micro_batch)
        if result.loss is not None:
            out[tokens] = result.loss
    return out
