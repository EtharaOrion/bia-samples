"""BIA S02 frozen data path. Harness-owned. Not a free axis.

Real GPT-2 token shards. Two profiles read the same code path:

  full   pinned FineWeb-10B shards staged into the image at build time, verified
         by sha256 against the digests recorded below
  smoke  the small real slices of those same shards vendored at
         environment/fixtures/, used by the CPU proof path

The token order is frozen and identical across seeds. Seed selects the model
initialization draw and nothing else, so seed-to-seed spread measures the thing
it is supposed to measure.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import torch

HEADER_MAGIC = 20240520
HEADER_INTS = 256

PINNED_SHARDS = {
    "fineweb_train_000001.bin": "771fa4a99b9fe0946ffb6e848b4ba5c6a9b0fe87860ebf03bc2c1c7e45f8178e",
    "fineweb_val_000000.bin": "5b95c8e0966f0861685b307b23dc5ae42b228ef74b28cb499784ae021f201640",
}

FIXTURE_SHARDS = {
    "smoke_train.bin": "a81a534750858bb96848ad8f2f726066dc8afd9356ea49f5b8a71f486ac6c7ab",
    "smoke_val.bin": "fbc35229e0d9e78b155b0b228ce34619c68d6d7305d0634dbbf0b38e88f641ee",
}


def read_shard(path: str) -> np.ndarray:
    header = np.fromfile(path, dtype=np.int32, count=HEADER_INTS)
    if int(header[0]) != HEADER_MAGIC:
        raise ValueError("bad shard magic in %s" % path)
    ntok = int(header[2])
    tokens = np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_INTS * 4)
    return tokens[:ntok]


def digest_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_shard(path: str, expected: str) -> str:
    got = digest_of(path)
    if expected is not None and got != expected:
        raise ValueError("shard digest mismatch for %s: %s" % (path, got))
    return got


class FrozenLoader:
    """Deterministic sequential batches. No shuffling, no curriculum, no seed."""

    def __init__(self, path: str, batch_sequences: int, seq_len: int, expected_digest=None):
        self.path = path
        self.digest = verify_shard(path, expected_digest)
        self.tokens = read_shard(path)
        self.batch_sequences = batch_sequences
        self.seq_len = seq_len
        self.stride = batch_sequences * seq_len
        self.cursor = 0
        need = self.stride + 1
        if len(self.tokens) < need:
            raise ValueError("shard %s too small for one batch" % path)

    def reset(self):
        self.cursor = 0

    def next_batch(self, device):
        span = self.stride + 1
        if self.cursor + span > len(self.tokens):
            self.cursor = 0
        window = np.asarray(self.tokens[self.cursor:self.cursor + span], dtype=np.int64)
        self.cursor += self.stride
        flat = torch.from_numpy(window)
        x = flat[:-1].view(self.batch_sequences, self.seq_len).to(device)
        y = flat[1:].view(self.batch_sequences, self.seq_len).to(device)
        return x, y


def resolve_shards(profile: str, data_dir: str, fixture_dir: str):
    """Return (train_path, train_digest, val_path, val_digest) for a profile."""
    if profile == "full":
        train = os.path.join(data_dir, "fineweb_train_000001.bin")
        val = os.path.join(data_dir, "fineweb_val_000000.bin")
        return (train, PINNED_SHARDS["fineweb_train_000001.bin"],
                val, PINNED_SHARDS["fineweb_val_000000.bin"])
    train = os.path.join(fixture_dir, "smoke_train.bin")
    val = os.path.join(fixture_dir, "smoke_val.bin")
    return (train, FIXTURE_SHARDS["smoke_train.bin"],
            val, FIXTURE_SHARDS["smoke_val.bin"])
