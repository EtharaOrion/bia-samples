"""The provided step loader. Draw every training step through it.

It owns four things the submission does not, and each one is a frozen axis of
this task made mechanical rather than requested.

BATCH SIZE. The number of tokens served per optimizer step is read from the
bound shape and is not a parameter of this class. A submission cannot enlarge
its effective batch by asking for more, and every step it takes is recorded with
the token count it actually received.

STEP COUNT. Steps are counted here, not reported by the submission. The graded
axis of this task is the number of optimizer steps taken to reach the target, so
the count is taken by the harness that serves them.

CHECKPOINT CADENCE. Checkpoints are written at the verifier's own step
milestones, read from the mounted grid. Placement is therefore not a submission
choice, which is what makes the evaluation cadence verifier-owned and what makes
a per-run stopping decision visible when it happens.

THE RUN LEDGER. Every checkpoint records the step index it was written at, the
tokens served up to that point, and the number of forward passes the frozen
model has performed. The verifier reconciles all three. A ledger written inside
the submission's process is forgeable by a submission that reimplements this
file; that residue is a named rubric obligation. What the ledger forecloses
deterministically is every ordinary route: skipping the loader, renaming a
checkpoint, enlarging the batch, or taking extra forward-backward passes per
step.
"""
from __future__ import annotations

import json
import os
import pathlib

CHECKPOINT_DIR = pathlib.Path(os.environ.get("BIA_CHECKPOINT_DIR", "./checkpoints"))
STEP_GRID_PATH = pathlib.Path(os.environ.get("BIA_STEP_GRID", "/env/step_grid.json"))
SHAPE_PATH = pathlib.Path(os.environ.get("BIA_SHAPE", "/env/shape.json"))
TRAIN_SHARDS = os.environ.get("BIA_TRAIN_SHARDS", "/data/train_*.bin")


def step_grid() -> list[int]:
    return sorted(int(x) for x in json.loads(STEP_GRID_PATH.read_text()))


def shape() -> dict:
    return json.loads(SHAPE_PATH.read_text())


class StepLoader:
    def __init__(self, device):
        self.shape = shape()
        self.batch_tokens = int(self.shape["batch_tokens_per_step"])
        self.seq_len = int(self.shape["seq_len"])
        self.device = device
        self.grid = step_grid()
        self.step = 0
        self.tokens_served = 0
        self._next = 0
        self._ledger: dict[str, dict] = {}
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    def _write_ledger(self) -> None:
        (CHECKPOINT_DIR / "run_ledger.json").write_text(
            json.dumps(self._ledger, indent=2, sort_keys=True) + "\n")

    def checkpoint(self, model) -> None:
        """Save at every milestone the step count has reached or passed."""
        import torch

        while self._next < len(self.grid) and self.step >= self.grid[self._next]:
            milestone = self.grid[self._next]
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"step_{milestone}.pt")
            self._ledger[str(milestone)] = {
                "step": self.step,
                "tokens_served": self.tokens_served,
                "batch_tokens_per_step": self.batch_tokens,
                "forward_calls": int(getattr(model, "forward_calls", -1)),
            }
            self._write_ledger()
            self._next += 1

    def steps(self):
        """Yield one frozen-size batch per optimizer step, up to the last milestone."""
        from bia_data import token_stream

        budget = self.grid[-1] if self.grid else 0
        stream = token_stream(TRAIN_SHARDS, self.batch_tokens, self.seq_len, self.device)
        while self.step < budget:
            inputs, targets = next(stream)
            self.step += 1
            self.tokens_served += self.batch_tokens
            yield inputs, targets
