"""The frozen training core, mirrored here so you can iterate locally.

This is the model, the optimizer, the corpus mix and the budget the graded run uses. It
is FROZEN: the graded run executes the verifier's own copy of this code with the bound
values, and editing this file changes nothing about how you are scored. It is here so
that you can measure your generator against a held-out split you build yourself before
you submit.

What is free is the generator you write. What is frozen is everything below.

  model      the canonical modded-nanogpt decoder declared in nanogpt_substrate.json:
             vocab_size 50304, num_layers 12, model_dim 768, head_dim 128, num_heads 6,
             seq_len 1024
  optimizer  AdamW on the embedding, the output projection and the one-dimensional
             parameters; Muon on the hidden block matrices
  budget     524288 tokens per step, exactly one forward and one backward pass per step,
             frozen_train_steps steps
  corpus     each step draws 131072 tokens from a shard built by tokenizing YOUR corpus
             and 393216 tokens from the FineWeb10B train shards, so your documents own
             one quarter of every frozen batch and nothing else
  metric     exp(-val_loss) on a held-out FineWeb validation split, higher better

The graded run is 3250 steps. That is a full nanoGPT training run and it is not something
you can execute inside one attempt, which is why `--steps` exists here: a short local run
tells you whether your corpus helps or hurts, and it is telemetry rather than the score.
The number the verifier grades is measured by the verifier on a split absent from this
container.

Usage:
    python3 trainer.py --corpus corpus.jsonl --eval my_own_split.jsonl --steps 50
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW

HERE = Path(__file__).resolve().parent
SUBSTRATE = HERE / "nanogpt_substrate.json"

FROZEN_TRAIN_STEPS = 3250
FROZEN_BATCH_TOKENS_PER_STEP = 524288
FROZEN_SYNTHETIC_TOKENS_PER_STEP = 131072
FROZEN_FINEWEB_TOKENS_PER_STEP = 393216
FROZEN_OPTIMIZER = "adamw-muon"
FROZEN_MUON_LR = 0.025
FROZEN_MUON_WEIGHT_DECAY = 0.05
FROZEN_ADAMW_EMBED_LR = 0.7
FROZEN_ADAMW_HEAD_LR = 0.004
FROZEN_ADAMW_SCALAR_LR = 0.015
FROZEN_ADAMW_WEIGHT_DECAY = 0.001
MAX_CORPUS_DOCUMENTS = 2000
MAX_DOCUMENT_CHARS = 65536

SHARD_MAGIC = 20240520
SHARD_VERSION = 1
SHARD_HEADER_INTS = 256

FINEWEB_TRAIN_ROOT = Path(os.environ.get("OER17_FINEWEB_ROOT", "/workspace/data/fineweb10B"))
FINEWEB_TRAIN_GLOB = "fineweb_train_*.bin"


def substrate() -> dict:
    with SUBSTRATE.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_documents(path, cap=MAX_CORPUS_DOCUMENTS):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict) or "text" not in row:
                continue
            text = str(row["text"])
            if not text or len(text) > MAX_DOCUMENT_CHARS:
                continue
            rows.append({"text": text})
            if len(rows) >= cap:
                break
    return rows


def encode_documents(documents, enc, vocab_size):
    eot = enc.eot_token
    stream = []
    usable = 0
    for row in documents:
        ids = enc.encode_ordinary(str(row.get("text", "")))
        if not ids:
            continue
        stream.append(eot)
        stream.extend(token for token in ids if 0 <= token < vocab_size)
        usable += 1
    return stream, usable


def write_shard(tokens, path):
    array = numpy.asarray(tokens, dtype=numpy.uint16)
    header = numpy.zeros(SHARD_HEADER_INTS, dtype=numpy.int32)
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = int(array.size)
    with Path(path).open("wb") as handle:
        handle.write(header.tobytes())
        handle.write(array.tobytes())
    return int(array.size)


def read_shard(path):
    with Path(path).open("rb") as handle:
        header = numpy.frombuffer(handle.read(SHARD_HEADER_INTS * 4), dtype=numpy.int32)
        if int(header[0]) != SHARD_MAGIC or int(header[1]) != SHARD_VERSION:
            raise ValueError("shard container mismatch at " + str(path))
        count = int(header[2])
        return numpy.frombuffer(handle.read(2 * count), dtype=numpy.uint16)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))


class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))


class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int = 128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=0.12, is_causal=True
        ).transpose(1, 2)
        return self.proj(y.contiguous().view(B, T, self.num_heads * self.head_dim))


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = Linear(dim, 4 * dim)
        self.proj = Linear(4 * dim, dim)

    def forward(self, x: Tensor):
        return self.proj(self.fc(x).relu().square())


class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, head_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim, head_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15 ** 2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")


def zeropower_via_newtonschulz5(G):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(5):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, momentum=0.95):
        super().__init__(list(params), dict(lr=lr, weight_decay=weight_decay, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.lerp_(p.grad, 1 - group["momentum"])
                update = p.grad.lerp_(buf, group["momentum"])
                update = zeropower_via_newtonschulz5(update)
                update = update * max(1, p.size(-2) / p.size(-1)) ** 0.5
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])


def build_model(architecture):
    model = GPT(
        vocab_size=int(architecture["vocab_size"]),
        num_layers=int(architecture["num_layers"]),
        model_dim=int(architecture["model_dim"]),
        head_dim=int(architecture["head_dim"]),
    )
    for name, parameter in model.named_parameters():
        w = parameter.data
        if name.endswith("weight"):
            if "proj" in name:
                w.zero_()
            elif "embed" in name:
                w.normal_()
            else:
                w.normal_(std=0.33 ** 0.5 / w.size(-1) ** 0.5)
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)
        else:
            raise ValueError("uninitialized parameter: " + name)
    return model


def build_optimizers(model):
    adamw = AdamW(
        [
            dict(params=[model.embed.weight], lr=FROZEN_ADAMW_EMBED_LR),
            dict(params=[model.proj.weight], lr=FROZEN_ADAMW_HEAD_LR),
            dict(params=[p for p in model.parameters() if p.ndim < 2], lr=FROZEN_ADAMW_SCALAR_LR),
        ],
        betas=(0.8, 0.95),
        eps=1e-10,
        weight_decay=FROZEN_ADAMW_WEIGHT_DECAY,
    )
    muon = Muon(
        [p for p in model.blocks.parameters() if p.ndim >= 2],
        lr=FROZEN_MUON_LR,
        weight_decay=FROZEN_MUON_WEIGHT_DECAY,
    )
    optimizers = [adamw, muon]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return optimizers


def set_hparams(optimizers, step, budget, cooldown_frac=0.7):
    progress = step / budget
    eta = 1.0 if progress < 1 - cooldown_frac else (1 - progress) / cooldown_frac
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * eta


def batch_stream(synthetic, fineweb_files, seq_len):
    synthetic_pos = 0
    file_index = 0
    fineweb = read_shard(fineweb_files[0])
    fineweb_pos = 0
    while True:
        if synthetic_pos + FROZEN_SYNTHETIC_TOKENS_PER_STEP + 1 >= len(synthetic):
            synthetic_pos = 0
        if fineweb_pos + FROZEN_FINEWEB_TOKENS_PER_STEP + 1 >= len(fineweb):
            file_index = (file_index + 1) % len(fineweb_files)
            fineweb = read_shard(fineweb_files[file_index])
            fineweb_pos = 0
        left = synthetic[synthetic_pos:synthetic_pos + FROZEN_SYNTHETIC_TOKENS_PER_STEP + 1]
        right = fineweb[fineweb_pos:fineweb_pos + FROZEN_FINEWEB_TOKENS_PER_STEP + 1]
        synthetic_pos += FROZEN_SYNTHETIC_TOKENS_PER_STEP
        fineweb_pos += FROZEN_FINEWEB_TOKENS_PER_STEP
        inputs = numpy.concatenate([left[:-1], right[:-1]]).astype(numpy.int64)
        targets = numpy.concatenate([left[1:], right[1:]]).astype(numpy.int64)
        yield inputs.reshape(-1, seq_len), targets.reshape(-1, seq_len)


def held_out_loss(model, documents, enc, architecture, device):
    seq_len = int(architecture["seq_len"])
    stream, _ = encode_documents(documents, enc, int(architecture["vocab_size"]))
    usable = (len(stream) - 1) // seq_len * seq_len
    if usable <= 0:
        raise ValueError("the evaluation split tokenized to fewer than one sequence")
    inputs = torch.tensor(stream[:usable], dtype=torch.int32, device=device).view(-1, seq_len)
    targets = torch.tensor(stream[1:usable + 1], dtype=torch.int64, device=device).view(-1, seq_len)
    total = 0.0
    model.eval()
    with torch.no_grad():
        for index in range(inputs.size(0)):
            total += float(model(inputs[index:index + 1], targets[index:index + 1]))
    model.train()
    return total / float(usable)


def main() -> int:
    parser = argparse.ArgumentParser(description="local mirror of the frozen training core")
    parser.add_argument("--corpus", required=True, help="jsonl your generator produced")
    parser.add_argument("--eval", required=True, help="jsonl you built yourself to measure against")
    parser.add_argument("--steps", type=int, default=50, help="local iteration budget; the graded run uses 3250")
    args = parser.parse_args()

    architecture = substrate()["architecture"]
    enc = tiktoken.get_encoding("gpt2")
    documents = load_documents(args.corpus)
    stream, usable = encode_documents(documents, enc, int(architecture["vocab_size"]))
    if usable <= 0:
        print(json.dumps({"corpus_documents": len(documents), "usable_documents": 0, "score": None}, sort_keys=True))
        return 0

    workspace = Path("./.trainer-shards")
    workspace.mkdir(exist_ok=True)
    synthetic_tokens = write_shard(stream, workspace / "synthetic_train_000000.bin")
    fineweb_files = sorted(FINEWEB_TRAIN_ROOT.glob(FINEWEB_TRAIN_GLOB))
    if not fineweb_files:
        raise FileNotFoundError("no FineWeb10B train shard under " + str(FINEWEB_TRAIN_ROOT))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(architecture).to(device)
    optimizers = build_optimizers(model)
    batches = batch_stream(read_shard(workspace / "synthetic_train_000000.bin"), fineweb_files, int(architecture["seq_len"]))

    budget = int(args.steps)
    for step in range(budget):
        set_hparams(optimizers, step, budget)
        inputs, targets = next(batches)
        inputs = torch.from_numpy(inputs).to(device=device, dtype=torch.int32)
        targets = torch.from_numpy(targets).to(device=device, dtype=torch.int64)
        loss = model(inputs, targets)
        loss.backward()
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

    val_loss = held_out_loss(model, load_documents(args.eval, cap=10 ** 9), enc, architecture, device)
    print(
        json.dumps(
            {
                "corpus_documents": len(documents),
                "usable_documents": usable,
                "synthetic_tokens": synthetic_tokens,
                "steps": budget,
                "tokens_fed": budget * FROZEN_BATCH_TOKENS_PER_STEP,
                "val_loss": val_loss,
                "score": math.exp(-val_loss),
                "note": "telemetry from a short local run; the graded number is the verifier's own measurement at 3250 steps",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
