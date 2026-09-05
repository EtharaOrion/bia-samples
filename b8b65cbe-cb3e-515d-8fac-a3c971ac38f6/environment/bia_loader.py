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

SHAPE = json.loads((pathlib.Path(__file__).resolve().parent / "shape.json").read_text(encoding="utf-8"))

BATCH_SIZE = int(SHAPE["batch_size"])
SEQUENCE_LENGTH = 1024
VOCAB_SIZE = 50304
N_LAYER = 12
N_HEAD = 6
N_EMBD = 768

TRAIN_SHARD = pathlib.Path("/data/fineweb/train")


def frozen_batches() -> Iterator[Tuple["object", "object"]]:
    """The one frozen data stream. Batch size and shard order are both frozen."""
    import torch

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
    from gpt import GPT, GPTConfig  # shipped in the pinned image

    config = GPTConfig(
        block_size=SEQUENCE_LENGTH,
        vocab_size=VOCAB_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
    )
    return GPT(config, init_std=init_std)
