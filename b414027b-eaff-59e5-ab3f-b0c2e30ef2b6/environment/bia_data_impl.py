"""The frozen token stream. Ships in the image; never authored by the submission.

Shards are flat uint16 token files. Order is fixed by shard name then position,
and does not depend on the seed, because the dataset and its ordering are frozen
while initialization and optimizer stochasticity are not.
"""
from __future__ import annotations

import glob
import pathlib

import numpy as np
import torch

from frozen_gpt import resolve_device


def _load(pattern: str) -> np.ndarray:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no shards matched {pattern!r}")
    return np.concatenate([np.fromfile(p, dtype=np.uint16) for p in paths])


def raw_token_stream(pattern: str, batch_size_tokens: int, seq_len: int):
    """Yield (inputs, targets), each exactly batch_size_tokens tokens.

    batch_size_tokens is the submission's choice and may change between calls,
    which is what makes the batch-size axis free. It must be a multiple of
    seq_len so every batch is a whole number of sequences.
    """
    if batch_size_tokens % seq_len != 0:
        raise ValueError(f"batch_size_tokens {batch_size_tokens} must be a multiple of seq_len {seq_len}")
    tokens = _load(pattern)
    rows = batch_size_tokens // seq_len
    pos = 0
    while pos + batch_size_tokens + 1 <= len(tokens):
        buf = torch.from_numpy(tokens[pos:pos + batch_size_tokens + 1].astype(np.int64))
        dev = resolve_device()
        yield buf[:-1].view(rows, seq_len).to(dev), buf[1:].view(rows, seq_len).to(dev)
        pos += batch_size_tokens
