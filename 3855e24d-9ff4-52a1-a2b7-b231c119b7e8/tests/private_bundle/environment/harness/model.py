"""Frozen model architecture for bia slot S06.

The architecture is part of the frozen substrate. A submission never edits this
file: the verifier records a substrate digest over the resolved configuration at
every logged step and rejects a run whose digest moved.
"""

from __future__ import annotations

import hashlib
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 256


class Config:
    """Resolved frozen architecture and schedule for one operating point."""

    def __init__(self, **kw):
        self.n_layer = kw["n_layer"]
        self.n_head = kw["n_head"]
        self.d_model = kw["d_model"]
        self.block_size = kw["block_size"]
        self.batch_size = kw["batch_size"]
        self.max_steps = kw["max_steps"]
        self.eval_every = kw["eval_every"]
        self.eval_batches = kw["eval_batches"]
        self.ref_step = kw["ref_step"]
        self.target_fraction = kw["target_fraction"]
        self.lr = kw["lr"]
        self.warmup = kw["warmup"]
        self.min_lr_fraction = kw["min_lr_fraction"]
        self.weight_decay = kw["weight_decay"]
        self.beta1 = kw["beta1"]
        self.beta2 = kw["beta2"]
        self.eps = kw["eps"]
        self.grad_clip = kw["grad_clip"]
        self.seeds = tuple(kw["seeds"])
        self.point = kw["point"]

    def as_dict(self):
        return {
            "point": self.point,
            "vocab_size": VOCAB_SIZE,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "d_model": self.d_model,
            "block_size": self.block_size,
            "batch_size": self.batch_size,
            "max_steps": self.max_steps,
            "eval_every": self.eval_every,
            "eval_batches": self.eval_batches,
            "ref_step": self.ref_step,
            "target_fraction": self.target_fraction,
            "lr": self.lr,
            "warmup": self.warmup,
            "min_lr_fraction": self.min_lr_fraction,
            "weight_decay": self.weight_decay,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "grad_clip": self.grad_clip,
            "seeds": list(self.seeds),
        }

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode("ascii")).hexdigest()


SCALED = Config(
    point="scaled-h100-7m2",
    n_layer=6,
    n_head=6,
    d_model=384,
    block_size=256,
    batch_size=32,
    max_steps=1500,
    eval_every=25,
    eval_batches=8,
    ref_step=1000,
    target_fraction=0.60,
    lr=6e-4,
    warmup=100,
    min_lr_fraction=0.1,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    eps=1e-8,
    grad_clip=1.0,
    seeds=(0, 1),
)

SMOKE = Config(
    point="smoke-cpu",
    n_layer=2,
    n_head=2,
    d_model=64,
    block_size=64,
    batch_size=8,
    max_steps=60,
    eval_every=10,
    eval_batches=2,
    ref_step=40,
    target_fraction=0.60,
    lr=6e-4,
    warmup=5,
    min_lr_fraction=0.1,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    eps=1e-8,
    grad_clip=1.0,
    seeds=(0,),
)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head = cfg.n_head
        self.d_model = cfg.d_model
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.out = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        b, t, c = x.shape
        h = self.ln1(x)
        q, k, v = self.attn(h).split(c, dim=2)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(b, t, c)
        x = x + self.proj(a)
        h = self.ln2(x)
        x = x + self.out(F.gelu(self.fc(h)))
        return x


class TinyGPT(nn.Module):
    """Frozen decoder only byte level transformer."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(VOCAB_SIZE, cfg.d_model)
        self.pos = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.lnf = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, VOCAB_SIZE, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, idx, targets):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.lnf(x)
        logits = self.head(x)
        return F.cross_entropy(logits.view(-1, VOCAB_SIZE).float(), targets.reshape(-1))


def build_model(cfg: Config, seed: int, device):
    """Frozen initialization. Seed fixes the weights; a submission cannot change this."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1_000_003 * (seed + 1) + 17)
    model = TinyGPT(cfg)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            if p.dim() >= 2:
                std = 0.02 if "tok" in name or "pos" in name else (0.02 / math.sqrt(2 * cfg.n_layer))
                p.copy_(torch.empty(p.shape).normal_(mean=0.0, std=std, generator=gen))
            else:
                p.zero_()
        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln2.weight.fill_(1.0)
        model.lnf.weight.fill_(1.0)
    return model.to(device)


def lr_at(cfg: Config, step: int) -> float:
    """Frozen learning rate schedule. Part of the substrate, not of the estimator."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / max(cfg.warmup, 1)
    span = max(cfg.max_steps - cfg.warmup, 1)
    frac = min(max((step - cfg.warmup) / span, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * frac))
    return cfg.lr * (cfg.min_lr_fraction + (1.0 - cfg.min_lr_fraction) * cos)
