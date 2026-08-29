"""Frozen model. Two construction modes and nothing else.

norm_mode "rmsnorm" is the published, un-ablated architecture: pre-normalization
RMSNorm on both residual branch inputs plus a final norm before the head.

norm_mode "none" is the ablated architecture the graded submission must train:
the identical module tree with every normalization layer replaced by an
identity, so the residual stream is never rescaled to unit root mean square at
any point in the forward pass, and the state dict carries no normalization
parameter.

The agent never constructs a model. The harness does, and the mode it used is
written into the run record before the first step.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, cfg: dict, norm_mode: str):
        super().__init__()
        d = cfg["d_model"]
        self.n_head = cfg["n_head"]
        self.head_dim = d // cfg["n_head"]
        if norm_mode == "rmsnorm":
            self.norm_attn = nn.RMSNorm(d)
            self.norm_mlp = nn.RMSNorm(d)
        elif norm_mode == "none":
            self.norm_attn = nn.Identity()
            self.norm_mlp = nn.Identity()
        else:
            raise ValueError(f"unknown_norm_mode:{norm_mode}")
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        h = cfg["mlp_mult"] * d
        self.fc = nn.Linear(d, h, bias=False)
        self.out = nn.Linear(h, d, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        h = self.norm_attn(x)
        qkv = self.qkv(h).view(b, t, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(b, t, d)
        x = x + self.proj(a)
        h = self.norm_mlp(x)
        x = x + self.out(F.relu(self.fc(h)).square())
        return x


class GPT(nn.Module):
    def __init__(self, cfg: dict, norm_mode: str):
        super().__init__()
        self.cfg = dict(cfg)
        self.norm_mode = norm_mode
        d = cfg["d_model"]
        self.embed = nn.Embedding(cfg["vocab_size"], d)
        self.pos = nn.Embedding(cfg["seq_len"], d)
        self.blocks = nn.ModuleList([Block(cfg, norm_mode) for _ in range(cfg["n_layer"])])
        if norm_mode == "rmsnorm":
            self.norm_final = nn.RMSNorm(d)
        else:
            self.norm_final = nn.Identity()
        self.head = nn.Linear(d, cfg["vocab_size"], bias=False)
        self._init(cfg)

    def _init(self, cfg: dict):
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos.weight, mean=0.0, std=0.02)
        scale = 1.0 / math.sqrt(2.0 * cfg["n_layer"])
        for blk in self.blocks:
            nn.init.normal_(blk.qkv.weight, mean=0.0, std=0.02)
            nn.init.normal_(blk.fc.weight, mean=0.0, std=0.02)
            nn.init.normal_(blk.proj.weight, mean=0.0, std=0.02 * scale)
            nn.init.normal_(blk.out.weight, mean=0.0, std=0.02 * scale)
        # Zero initialized head. The untrained loss is therefore exactly
        # ln(vocab_size), which gives the harness a defined ceiling to fall back
        # on when the naive port of the published recipe diverges.
        nn.init.zeros_(self.head.weight)

    def forward(self, x, y):
        b, t = x.shape
        pos = torch.arange(t, device=x.device)
        h = self.embed(x) + self.pos(pos)[None, :, :]
        for blk in self.blocks:
            h = blk(h)
        h = self.norm_final(h)
        logits = self.head(h)
        loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.reshape(-1))
        return loss

    def param_groups(self):
        """The split the submission receives. Matrices are the rank two weights
        inside the transformer blocks, which are the surface every published
        optimizer record acts on. The two embedding tables and the output head
        are handed over separately so the submission may treat them differently.
        The head is not a block matrix and is zero initialized, so grouping it
        with the block matrices would hand a scale-relative rule a parameter
        whose scale is exactly zero."""
        matrices, others = [], []
        for name, p in self.named_parameters():
            if p.ndim == 2 and not name.startswith(("embed", "pos", "head")):
                matrices.append((name, p))
            else:
                others.append((name, p))
        return matrices, others


def norm_parameter_names(model: nn.Module):
    """Live-state read: the names in the constructed model's state dict that
    belong to a normalization module. On the ablated build this must be empty."""
    owned = set()
    for mod_name, mod in model.named_modules():
        if type(mod).__name__ in (
            "LayerNorm",
            "RMSNorm",
            "GroupNorm",
            "BatchNorm1d",
            "BatchNorm2d",
            "BatchNorm3d",
            "SyncBatchNorm",
            "InstanceNorm1d",
            "InstanceNorm2d",
            "InstanceNorm3d",
            "LocalResponseNorm",
        ):
            for pname, _ in mod.named_parameters(recurse=False):
                owned.add(f"{mod_name}.{pname}" if mod_name else pname)
            for bname, _ in mod.named_buffers(recurse=False):
                owned.add(f"{mod_name}.{bname}" if mod_name else bname)
    return sorted(owned)
