"""Frozen dataset access. The token stream and its order are not yours to change.

Shards are read in sorted filename order and consumed strictly in order, so the
sequence of tokens a run sees is a property of the dataset rather than of the
submission or of the seed. The seed governs initialization and any stochastic
choice the optimizer makes; it never governs which tokens arrive when.
"""
from __future__ import annotations

import glob
import pathlib

import numpy as np
import torch


def shard_paths(shard_glob: str) -> list[pathlib.Path]:
    found = sorted(pathlib.Path(p) for p in glob.glob(shard_glob))
    if not found:
        raise FileNotFoundError(f"no dataset shard matches {shard_glob!r}")
    return found


def token_stream(shard_glob: str, batch_tokens: int, seq_len: int, device: torch.device):
    """Yield (inputs, targets) of exactly batch_tokens, in frozen order, forever."""
    if batch_tokens % seq_len:
        raise ValueError("batch_tokens must be a whole number of sequences")
    rows = batch_tokens // seq_len
    paths = shard_paths(shard_glob)
    while True:
        for path in paths:
            tokens = np.fromfile(path, dtype=np.uint16)
            usable = ((len(tokens) - 1) // batch_tokens) * batch_tokens
            for start in range(0, usable, batch_tokens):
                buf = torch.from_numpy(
                    tokens[start:start + batch_tokens + 1].astype("int64"))
                yield (buf[:-1].view(rows, seq_len).to(device),
                       buf[1:].view(rows, seq_len).to(device))
