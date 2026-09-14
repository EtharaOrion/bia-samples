"""The frozen data path and the frozen architecture. Read it; do not change it.

Both functions here are frozen axes. They are shipped as source so you can see
exactly what the training loop consumes, not so you can tune them. The verifier
re-reads the digests in environment/shape.json from its own copy at grading time,
so a change here produces a refused run with the reason `frozen-axis-moved`.

The validation split is deliberately absent from this module and from this
container. The number that scores you is computed by the verifier on a held-out
split you cannot see, from the parameter snapshots the pinned harness wrote. That
is why there is no way to tune against the graded readout: there is nothing here
to tune against.
"""

from __future__ import annotations

import json
import pathlib
from typing import Iterator, Tuple

HERE = pathlib.Path(__file__).resolve().parent
SHAPE = json.loads((HERE / "shape.json").read_text(encoding="utf-8"))

import bia_corpus  # noqa: E402  the one declaration of the executed operating point

POINT = bia_corpus.operating_point()

# Every one of these used to be a literal here AND a literal in gpt.py's
# GPTConfig defaults AND a literal in tests/evaluator.py's expectations. Three
# restatements of one shape is how a re-scale moves two of them and not the
# third. They are now read from environment/operating_point.json, which is the
# same document gpt.py resolves its config against, so the loader, the model and
# the verifier's evaluator cannot disagree about what was trained.
BATCH_SIZE = int(POINT["batch_size"])
SEQUENCE_LENGTH = int(POINT["seq_len"])
VOCAB_SIZE = int(POINT["vocab_size"])
N_LAYER = int(POINT["n_layer"])
N_HEAD = int(POINT["n_head"])
N_EMBD = int(POINT["n_embd"])

# The frozen train shards. Ordered by preference and NOT collapsed to one path:
# the shard is staged into the image at build time at the first entry, and a
# runtime that bind-mounts the upstream directory instead lands at the second.
# Whichever exists is the same upstream FineWeb10B bytes.
TRAIN_SHARD_ROOTS = (
    pathlib.Path("/data/fineweb/train"),
    pathlib.Path("/workspace/data/fineweb10B"),
)

# TRAIN shards only, by name. The previous glob was "*.bin", and the upstream
# directory that the second root above points at carries fineweb_val_000000.bin
# beside the train shards -- which is the shard the verifier's held-out split is
# sliced out of. A "*.bin" glob there would have trained the graded model on the
# graded split. The glob is bound in operating_point.json and is matched here.
SHARD_GLOB = str(POINT["shard_glob"])

# Kept as a name because tests/evaluator.py and this module's own callers refer
# to it; it resolves to whichever root actually carries shards.
TRAIN_SHARD = TRAIN_SHARD_ROOTS[0]


def train_shards() -> list:
    """Every frozen train shard, from the first root that carries any."""
    for root in TRAIN_SHARD_ROOTS:
        files = sorted(root.glob(SHARD_GLOB))
        if files:
            return files
    raise FileNotFoundError(
        "no frozen training shard matching " + SHARD_GLOB + " under any of "
        + ", ".join(str(root) for root in TRAIN_SHARD_ROOTS)
    )


def frozen_batches() -> Iterator[Tuple["object", "object"]]:
    """The one frozen data stream. Batch size and shard order are both frozen."""
    import torch

    files = train_shards()
    while True:
        for path in files:
            tokens = torch.from_numpy(_read_shard(path))
            usable = (tokens.numel() - 1) // (BATCH_SIZE * SEQUENCE_LENGTH)
            for index in range(usable):
                start = index * BATCH_SIZE * SEQUENCE_LENGTH
                stop = start + BATCH_SIZE * SEQUENCE_LENGTH
                inputs = tokens[start:stop].view(BATCH_SIZE, SEQUENCE_LENGTH)
                targets = tokens[start + 1 : stop + 1].view(BATCH_SIZE, SEQUENCE_LENGTH)
                yield inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)


def _read_shard(path: pathlib.Path):
    import numpy

    with path.open("rb") as handle:
        header = numpy.frombuffer(handle.read(1024), dtype=numpy.int32)
        count = int(header[2])
        return numpy.frombuffer(handle.read(count * 2), dtype=numpy.uint16).astype(numpy.int64)


def build_frozen_model(init_std: float = 0.02):
    """The frozen architecture. Only the initialization standard deviation is free."""
    from gpt import GPT, GPTConfig  # shipped in the pinned image

    config = GPTConfig(
        block_size=SEQUENCE_LENGTH,
        vocab_size=VOCAB_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
    )
    return GPT(config, init_std=init_std)


def executed_point() -> dict:
    """What this loader actually resolved. For a reader, and for the build check."""
    return {
        "operating_point": str(POINT["name"]),
        "batch_size": BATCH_SIZE,
        "seq_len": SEQUENCE_LENGTH,
        "vocab_size": VOCAB_SIZE,
        "n_layer": N_LAYER,
        "n_head": N_HEAD,
        "n_embd": N_EMBD,
        "tokens_per_step": BATCH_SIZE * SEQUENCE_LENGTH,
        "shard_glob": SHARD_GLOB,
    }
