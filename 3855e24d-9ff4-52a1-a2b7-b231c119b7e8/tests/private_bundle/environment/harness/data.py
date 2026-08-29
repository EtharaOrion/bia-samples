"""Frozen data pipeline for bia slot S06.

The corpus is a real text corpus shipped as a frozen fixture at
environment/fixtures/corpus.bin. Its provenance recipe is recorded in
solution/grounding.yaml. Byte level tokenization means no tokenizer artifact is
needed and the vocabulary is exactly 256 symbols.

The split, the batch order and the validation batches are all frozen. A
submission never touches this file.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import torch

VAL_FRACTION = 0.10
BATCH_STREAM_SEED = 907


def corpus_path(bundle_root: str) -> str:
    return os.path.join(bundle_root, "environment", "fixtures", "corpus.bin")


def load_corpus(bundle_root: str, smoke: bool):
    """Load the frozen corpus. Smoke mode uses a frozen prefix of the same bytes."""
    path = corpus_path(bundle_root)
    with open(path, "rb") as fh:
        blob = fh.read()
    digest = hashlib.sha256(blob).hexdigest()
    if smoke:
        blob = blob[:131072]
    arr = np.frombuffer(blob, dtype=np.uint8)
    n_val = int(len(arr) * VAL_FRACTION)
    train = arr[: len(arr) - n_val]
    val = arr[len(arr) - n_val:]
    return train, val, digest


def _offsets(label: str, count: int, high: int) -> np.ndarray:
    key = int.from_bytes(hashlib.sha256(("%d|%s" % (BATCH_STREAM_SEED, label)).encode("ascii")).digest()[:8], "little")
    raw = np.random.Philox(key).random_raw(int(count)).astype(np.uint64)
    return (raw % np.uint64(max(high, 1))).astype(np.int64)


def batch_stream(split: np.ndarray, label: str, n_batches: int, batch_size: int, block_size: int, device):
    """Frozen batch order. Deterministic in the label alone, never in wall clock."""
    high = len(split) - block_size - 1
    offs = _offsets(label, n_batches * batch_size, high)
    for i in range(n_batches):
        sel = offs[i * batch_size:(i + 1) * batch_size]
        x = np.stack([split[o:o + block_size] for o in sel]).astype(np.int64)
        y = np.stack([split[o + 1:o + 1 + block_size] for o in sel]).astype(np.int64)
        yield (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))


def data_digest(train: np.ndarray, val: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(b"%d|%d|" % (len(train), len(val)))
    h.update(train[:4096].tobytes())
    h.update(val[:4096].tobytes())
    return h.hexdigest()
