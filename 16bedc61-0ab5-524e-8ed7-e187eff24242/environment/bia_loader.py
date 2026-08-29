"""The provided data loader. Draw every training token through it.

It does three jobs the submission does not do for itself.

It serves batches from the frozen dataset in the frozen order. The order does not
depend on the seed, because the dataset and its ordering are frozen; the seed
affects initialization and any stochastic choice the optimizer makes, not which
tokens arrive when.

It counts training tokens. The count is the graded quantity, so it is taken here
rather than from anything the submission reports.

It writes checkpoints at the verifier's token milestones and records each one in
a ledger. Checkpoint placement is therefore not a submission choice, which is
what makes the evaluation cadence verifier-owned.

Batch size is yours. Pass whatever you want to `batches()`, and change it between
calls if you want to schedule it. The loader serves exactly that many tokens per
batch and counts them.
"""
from __future__ import annotations

import json
import os
import pathlib

CHECKPOINT_DIR = pathlib.Path(os.environ.get("BIA_CHECKPOINT_DIR", "./checkpoints"))
TOKEN_GRID_PATH = pathlib.Path(os.environ.get("BIA_TOKEN_GRID", "/env/token_grid.json"))


def token_grid() -> list[int]:
    """The verifier-owned milestones at which checkpoints are written."""
    return sorted(int(x) for x in json.loads(TOKEN_GRID_PATH.read_text()))


class Loader:
    def __init__(self, shard_glob: str, seq_len: int):
        self.shard_glob = shard_glob
        self.seq_len = seq_len
        self.tokens_served = 0
        self._grid = token_grid()
        self._next = 0
        self._ledger: dict[int, int] = {}
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    def _write_ledger(self) -> None:
        (CHECKPOINT_DIR / "token_ledger.json").write_text(
            json.dumps({str(k): v for k, v in sorted(self._ledger.items())}, indent=2) + "\n"
        )

    def maybe_checkpoint(self, model) -> None:
        """Save at every milestone the served count has reached or passed."""
        import torch

        while self._next < len(self._grid) and self.tokens_served >= self._grid[self._next]:
            milestone = self._grid[self._next]
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"tokens_{milestone}.pt")
            self._ledger[milestone] = milestone
            self._write_ledger()
            self._next += 1

    def batches(self, batch_size_tokens: int):
        """Yield (inputs, targets) of exactly batch_size_tokens, counting as it goes.

        Stops when the last milestone is passed, so the token budget is bounded by
        the verifier's grid rather than by anything the submission decides.
        """
        from bia_data_impl import raw_token_stream  # frozen, ships in the image

        budget = self._grid[-1] if self._grid else 0
        for inputs, targets in raw_token_stream(self.shard_glob, batch_size_tokens, self.seq_len):
            self.tokens_served += batch_size_tokens
            yield inputs, targets
            if self.tokens_served >= budget:
                return
