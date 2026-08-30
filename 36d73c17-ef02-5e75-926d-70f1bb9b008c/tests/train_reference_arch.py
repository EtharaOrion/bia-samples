"""The verifier-owned architecture and validation loader. Never the submission's copy.

`tests/held_out_eval.py` imports `GPT`, `FROZEN` and `ValidationLoader` from this module
by name. In the delivered bundle no such module existed anywhere in the tree, in the
verifier image, or in the repository, so `evaluate_snapshot` raised
`ModuleNotFoundError` on its first call on every host, shard present or absent, and the
verifier could not grade a successful run at all. This file is that missing backend.

Three obligations shape it:

  * The architecture here is the LOCKED one, and it is the verifier's own copy. It is
    transcribed from `environment/train.py`, which is the locked substrate the agent is
    handed, so a submission cannot move the graded architecture by editing its own file.
    A state dict that does not fit this module is refused rather than adapted.
  * It builds from a shape MAPPING rather than a module literal, because the harness
    dictates the operating point and the same evaluator must serve both bound points.
  * The validation stream is a fixed, ascending, non-shuffled walk of the frozen shard.
    Nothing here averages, blends or filters a loss: the graded readout is one raw
    cross-entropy per snapshot and the no-smoothing checker is a fact about this code.

It also materialises the demonstration substrate, deterministically and offline. That
corpus is generated from two integer tables and one bound seed, never downloaded and
never sampled from a clock, so two runs over frozen bytes produce byte-identical shards.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

# The bound frozen axes, byte for byte what `environment/train.py` and
# `solution/reference.py` carry as their `FROZEN` literal.
FROZEN: Dict[str, object] = {
    "dataset": "fineweb10B",
    "sequence_length": 1024,
    "batch_size": 512,
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "vocab_size": 50257,
    "passes_per_step": 1,
}


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.n_head = int(cfg["n_head"])
        self.n_embd = int(cfg["n_embd"])
        self.qkv = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        shape = (b, t, self.n_head, c // self.n_head)
        q, k, v = (z.view(*shape).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(b, t, c))


class Block(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.ln1 = nn.LayerNorm(int(cfg["n_embd"]))
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(int(cfg["n_embd"]))
        self.mlp = nn.Sequential(
            nn.Linear(int(cfg["n_embd"]), 4 * int(cfg["n_embd"]), bias=False),
            nn.GELU(),
            nn.Linear(4 * int(cfg["n_embd"]), int(cfg["n_embd"]), bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    """The locked architecture. Widening it is a frozen-axis move, not a free choice."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.wte = nn.Embedding(int(cfg["vocab_size"]), int(cfg["n_embd"]))
        self.wpe = nn.Embedding(int(cfg["sequence_length"]), int(cfg["n_embd"]))
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(int(cfg["n_layer"])))
        self.lnf = nn.LayerNorm(int(cfg["n_embd"]))
        self.head = nn.Linear(int(cfg["n_embd"]), int(cfg["vocab_size"]), bias=False)
        self.head.weight = self.wte.weight

    def forward(self, idx, targets=None):
        pos = torch.arange(idx.shape[1], device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.lnf(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
        )
        return logits, loss


class ValidationLoader:
    """A fixed ascending walk of the frozen validation shard. No shuffle, no sampling.

    The walk is a pure function of (path, batch_size, sequence_length, token budget), so
    every snapshot in a run is scored against exactly the same held-out batches in
    exactly the same order. Making it deterministic is what lets the graded readout be
    compared across steps and seeds at all.
    """

    def __init__(self, path: str, batch_size: int, seq_len: int, tokens: int):
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        raw = numpy.memmap(path, dtype="uint16", mode="r")
        budget = min(int(tokens), int(raw.shape[0]))
        self.tokens = torch.from_numpy(numpy.asarray(raw[:budget]).astype("int64"))
        self.windows = max(0, (int(self.tokens.numel()) - 1) // self.seq_len)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for start in range(0, self.windows, self.batch_size):
            index = range(start, min(start + self.batch_size, self.windows))
            x = torch.stack([self.tokens[i * self.seq_len: i * self.seq_len + self.seq_len]
                             for i in index])
            y = torch.stack([self.tokens[i * self.seq_len + 1: i * self.seq_len + 1 + self.seq_len]
                             for i in index])
            yield x, y


# --------------------------------------------------------------------------- #
# The demonstration substrate. Generated, never fetched.
# --------------------------------------------------------------------------- #

CORPUS_SCHEMA = "forge.oer04_demonstration_corpus/v1"


def _tables(vocab: int, table_seed: int):
    draw = numpy.random.default_rng(table_seed)
    return (draw.integers(0, vocab, size=vocab, dtype=numpy.int64),
            draw.integers(0, vocab, size=vocab, dtype=numpy.int64))


def _stream(tokens: int, vocab: int, lag: int, noise: float, table_seed: int,
            stream_seed: int) -> numpy.ndarray:
    """`x[t+1] = (a[x[t]] + b[x[t-lag]]) mod V` with probability `1 - noise`.

    The lag term is what makes the sequence unlearnable from a bigram table alone, so
    the attention blocks carry graded signal rather than the embedding alone. The noise
    term puts a floor under the achievable loss, so a run cannot drive the loss to zero
    and the target is a bar rather than an asymptote.
    """
    a, b = _tables(vocab, table_seed)
    rng = numpy.random.default_rng(stream_seed)
    out = numpy.zeros(tokens, dtype=numpy.uint16)
    out[: lag + 1] = rng.integers(0, vocab, size=lag + 1, dtype=numpy.int64)
    corrupt = rng.random(tokens) < noise
    fallback = rng.integers(0, vocab, size=tokens, dtype=numpy.int64)
    for t in range(lag, tokens - 1):
        out[t + 1] = fallback[t] if corrupt[t] else (a[int(out[t])] + b[int(out[t - lag])]) % vocab
    return out


def materialize_demonstration_corpus(root: Path, spec: dict) -> dict:
    """Write `train.bin` and `val.bin` under `root`, or reuse a byte-identical pair.

    Returns the digest record. The digests are recomputed from the bytes on disk every
    time, so a shard that moved between runs is rebuilt rather than trusted.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    vocab = int(spec["vocab_size"])
    lag = int(spec["lag"])
    noise = float(spec["noise"])
    table_seed = int(spec["table_seed"])
    record = {"schema": CORPUS_SCHEMA, "spec": dict(spec), "shards": {}}
    for split, tokens, stream_seed in (
        ("train", int(spec["train_tokens"]), int(spec["train_stream_seed"])),
        ("val", int(spec["val_tokens"]), int(spec["val_stream_seed"])),
    ):
        path = root / (split + ".bin")
        want_bytes = tokens * 2
        if not (path.is_file() and path.stat().st_size == want_bytes):
            _stream(tokens, vocab, lag, noise, table_seed, stream_seed).tofile(path)
        record["shards"][split] = {
            "path": str(path),
            "tokens": tokens,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (root / "corpus.json").write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return record


def corpus_is_present(root: Path, spec: dict) -> bool:
    root = Path(root)
    for split, tokens in (("train", int(spec["train_tokens"])), ("val", int(spec["val_tokens"]))):
        path = root / (split + ".bin")
        if not path.is_file() or path.stat().st_size != tokens * 2:
            return False
    return True


def shard_is_present(path: str) -> bool:
    return os.path.isfile(path)
