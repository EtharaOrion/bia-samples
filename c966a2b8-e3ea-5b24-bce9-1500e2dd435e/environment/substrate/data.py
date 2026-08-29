"""Frozen byte-level corpus reader.

The corpus is real text: enwik8, a byte prefix of a 2006 English Wikipedia XML
dump. Tokens are raw bytes, so the vocabulary is exactly 256 and the parameter
mass sits in the transformer blocks rather than in an embedding table, which is
where the ablation under study actually bites.

Batch order is frozen. Every arm of every attempt sees the identical sequence of
training windows, so a difference between two arms is a difference in the update
rule and never a difference in the data the rule saw.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import torch


class FrozenByteCorpus:
    def __init__(self, path: str, cfg: dict):
        if not os.path.exists(path):
            raise FileNotFoundError(f"corpus_absent:{path}")
        self.path = path
        self.cfg = cfg
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        need = cfg["train_region_bytes"] + cfg["val_region_bytes"]
        if self.mm.shape[0] < need:
            raise ValueError(
                f"corpus_too_small:have={self.mm.shape[0]}:need={need}"
            )
        self.train_end = cfg["train_region_bytes"]
        self.val_end = self.train_end + cfg["val_region_bytes"]
        self._build_order()

    def _build_order(self):
        cfg = self.cfg
        rng = np.random.default_rng(cfg["seed"])
        span = cfg["seq_len"] + 1
        n_draws = cfg["steps"] * cfg["batch_size"]
        self.train_offsets = rng.integers(
            0, self.train_end - span, size=n_draws, dtype=np.int64
        )
        vrng = np.random.default_rng(cfg["seed"] + 1)
        self.val_offsets = vrng.integers(
            self.train_end, self.val_end - span, size=cfg["eval_batches"] * cfg["batch_size"], dtype=np.int64
        )

    def order_digest(self) -> str:
        h = hashlib.sha256()
        h.update(self.train_offsets.tobytes())
        h.update(self.val_offsets.tobytes())
        return h.hexdigest()

    def corpus_digest(self, probe_bytes: int = 1 << 20) -> str:
        """Digest of a frozen prefix plus the total length. Hashing 100 MB on
        every arm would be pure overhead, and the prefix plus length pins the
        artifact well enough to detect a swapped corpus."""
        h = hashlib.sha256()
        h.update(str(self.mm.shape[0]).encode())
        h.update(bytes(self.mm[:probe_bytes]))
        return h.hexdigest()

    def _gather(self, offsets, device):
        cfg = self.cfg
        span = cfg["seq_len"] + 1
        rows = np.stack([np.asarray(self.mm[o : o + span]) for o in offsets])
        t = torch.from_numpy(rows.astype(np.int64))
        x = t[:, :-1].contiguous().to(device)
        y = t[:, 1:].contiguous().to(device)
        return x, y

    def train_batch(self, step: int, device):
        cfg = self.cfg
        b = cfg["batch_size"]
        lo = step * b
        return self._gather(self.train_offsets[lo : lo + b], device)

    def val_batch(self, index: int, device):
        cfg = self.cfg
        b = cfg["batch_size"]
        lo = index * b
        return self._gather(self.val_offsets[lo : lo + b], device)
