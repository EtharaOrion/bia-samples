"""Verifier-side frozen architecture. Bound.

Shape comes from the same shape.json the training image ships, so the verifier
and the submission cannot disagree about what "frozen" means.
"""
from __future__ import annotations
import json, os, pathlib
import numpy as np
import torch
from frozen_gpt import GPT, resolve_device

SHAPE_BOUND = True
SHAPE = json.loads(pathlib.Path(os.environ["BIA_SHAPE"]).read_text())

def build():
    m = GPT(vocab_size=SHAPE["vocab_size"], num_layers=SHAPE["num_layers"],
            model_dim=SHAPE["model_dim"], head_dim=SHAPE["head_dim"])
    return m.to(resolve_device())

def load_val_tokens(shard: pathlib.Path):
    seq = SHAPE["seq_len"]
    tok = np.fromfile(shard, dtype=np.uint16)
    n = ((len(tok) - 1) // seq) * seq
    buf = torch.from_numpy(tok[:n + 1].astype("int64"))
    d = resolve_device()
    return buf[:-1].view(-1, seq).to(d), buf[1:].view(-1, seq).to(d)
