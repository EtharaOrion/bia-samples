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

WHAT CHANGED, AND WHY. This module used to say `from gpt import GPT, GPTConfig
# shipped in the pinned image` and used to read a shard at /data/fineweb/train
that no byte of this bundle and no layer of the pinned image ever carried. Both
claims were false, so no run of any submission could reach a single optimizer
step. The modules now ship in the bundle beside this file and are COPYed into
both images, and the executed shape and corpus come from
environment/operating_point.json. BATCH_SIZE still comes from
environment/shape.json and is still the bound 480, and there is still exactly
one forward-backward pass per optimizer step: neither numeric frozen axis moves.
"""

from __future__ import annotations

import json
import pathlib
from typing import Iterator, Tuple

import bia_corpus

SHAPE = json.loads((pathlib.Path(__file__).resolve().parent / "shape.json").read_text(encoding="utf-8"))

# The executed operating point. `bound` is the 124M point task.toml describes and
# is retained and still selectable; `demonstration` is the point this bundle can
# actually execute from its own closure. Selection is a bound byte, never a
# constant hidden in a module.
bia_corpus.pin_determinism()
POINT = bia_corpus.operating_point()

# FROZEN AXIS, unchanged and read from environment/shape.json exactly as before.
BATCH_SIZE = int(SHAPE["batch_size"])

# Architecture and sequence length, resolved at the executed point.
SEQUENCE_LENGTH = int(POINT["seq_len"])
VOCAB_SIZE = int(POINT["vocab_size"])
N_LAYER = int(POINT["n_layer"])
N_HEAD = int(POINT["n_head"])
N_EMBD = int(POINT["n_embd"])

TRAIN_SHARD = pathlib.Path(str(POINT["train_shard_dir"]))


def frozen_batches() -> Iterator[Tuple["object", "object"]]:
    """The one frozen data stream. Batch size and shard order are both frozen."""
    import torch

    bia_corpus.ensure_train_shard(POINT)
    files = sorted(TRAIN_SHARD.glob("*.bin"))
    if not files:
        raise FileNotFoundError("frozen training shard is absent at " + str(TRAIN_SHARD))
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
    from gpt import GPT, GPTConfig  # environment/gpt.py, COPYed into both images

    config = GPTConfig(
        block_size=SEQUENCE_LENGTH,
        vocab_size=VOCAB_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
    )
    return GPT(config, init_std=init_std)
