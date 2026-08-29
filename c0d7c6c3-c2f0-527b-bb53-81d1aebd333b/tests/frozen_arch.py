"""Verifier-side frozen architecture and withheld split loading.

The shape comes from the same file the training image ships, so the verifier and
the submission cannot disagree about what is frozen. The class comes from the
verifier's own copy of frozen_gpt, which is what makes strict=True loading an
enforcement of the frozen architecture rather than a courtesy.
"""
from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import torch

from frozen_gpt import GPT, resolve_device

SHAPE = json.loads(pathlib.Path(os.environ["BIA_SHAPE"]).read_text())


def build():
    return GPT(vocab_size=SHAPE["vocab_size"], num_layers=SHAPE["num_layers"],
               model_dim=SHAPE["model_dim"], head_dim=SHAPE["head_dim"]).to(resolve_device())


def load_val_tokens(shard: pathlib.Path):
    seq = SHAPE["seq_len"]
    tokens = np.fromfile(shard, dtype=np.uint16)
    n = ((len(tokens) - 1) // seq) * seq
    buf = torch.from_numpy(tokens[:n + 1].astype("int64"))
    device = resolve_device()
    return buf[:-1].view(-1, seq).to(device), buf[1:].view(-1, seq).to(device)
