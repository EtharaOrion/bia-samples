"""Frozen substrate for BIA slot S04, memory-budget architecture search.

This module owns everything the task freezes: the corpus, the split rule, the
tokenizer, the model family, the optimizer, the schedule, the data order and the
evaluation protocol. The only thing it does not own is the architecture
allocation, which arrives as a JSON config and is the whole free surface.

The same functions serve the full profile on one H100 and the smoke profile on
CPU. Nothing branches on profile except the frozen constants themselves, so a
smoke run exercises the identical code path at tiny scale.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sysconfig
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE_ROOT = os.environ.get("BIA_BUNDLE", os.path.dirname(os.path.dirname(HERE)))
FROZEN_DIR = os.environ.get("BIA_FROZEN_DIR", os.path.join(os.path.dirname(HERE), "frozen"))

ARCH_KEYS = ("n_layer", "d_model", "head_dim", "n_head", "n_kv_head", "d_ff")


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------

def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: str) -> Any:
    with open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def arch_digest(arch: Dict[str, int]) -> str:
    return sha256_hex(canonical_json({k: int(arch[k]) for k in ARCH_KEYS}).encode("utf-8"))


def frozen_recipe(path: str = None) -> Dict[str, Any]:
    return load_json(path or os.path.join(FROZEN_DIR, "frozen_recipe.json"))


def arch_space(path: str = None) -> Dict[str, Any]:
    return load_json(path or os.path.join(FROZEN_DIR, "arch_space.json"))


def frozen_tree_digest(frozen_dir: str = None) -> str:
    """Digest of every frozen recipe file, in sorted relative-path order.

    The runner records this at the head of the log and the verifier recomputes it
    from its own copy of the same tree, so an edit to the frozen surface inside
    the solving container shows up as a digest that does not exist on both sides.
    """
    root = frozen_dir or FROZEN_DIR
    parts: List[str] = []
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(files):
            if name.endswith(".pyc"):
                continue
            full = os.path.join(base, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            with open(full, "rb") as handle:
                parts.append(rel + "\0" + sha256_hex(handle.read()))
    return sha256_hex("\n".join(parts).encode("utf-8"))


# ----------------------------------------------------------------------------
# corpus: real Python source text, frozen order, frozen split
# ----------------------------------------------------------------------------

def corpus_file_list(recipe: Dict[str, Any]) -> List[str]:
    """Sorted list of real Python source files the corpus is built from.

    The source is the interpreter's own standard library inside the pinned image.
    It is real text rather than a synthetic stand-in, it is present without any
    network access, and the ordering rule below is total, so two builds of the
    same image produce the same bytes.
    """
    excluded = set(recipe["corpus"]["exclude_dir_names"])
    root = sysconfig.get_paths()["stdlib"]
    found: List[str] = []
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in excluded)
        for name in sorted(files):
            if name.endswith(".py"):
                found.append(os.path.join(base, name))
    found.sort(key=lambda p: os.path.relpath(p, root).replace(os.sep, "/"))
    return found


def build_corpus(recipe: Dict[str, Any], corpus_bytes: int) -> bytes:
    sep = bytes.fromhex(recipe["corpus"]["separator_hex"])
    chunks: List[bytes] = []
    total = 0
    for path in corpus_file_list(recipe):
        try:
            with open(path, "rb") as handle:
                blob = handle.read()
        except OSError:
            continue
        chunks.append(blob)
        chunks.append(sep)
        total += len(blob) + len(sep)
        if total >= corpus_bytes:
            break
    data = b"".join(chunks)[:corpus_bytes]
    return data


def split_corpus(data: bytes, val_frac: float) -> Dict[str, Any]:
    """Frozen split rule: the final val_frac of the corpus is the held-out tail.

    Returned ranges are half-open byte offsets into the corpus, so the verifier
    can assert disjointness without holding the bytes.
    """
    n = len(data)
    val_len = int(n * val_frac)
    boundary = n - val_len
    train = data[:boundary]
    val = data[boundary:]
    return {
        "n_bytes": n,
        "train_range": [0, boundary],
        "val_range": [boundary, n],
        "train_digest": sha256_hex(train),
        "val_digest": sha256_hex(val),
        "train": train,
        "val": val,
    }


# ----------------------------------------------------------------------------
# frozen data order
# ----------------------------------------------------------------------------

def train_starts(n_train: int, seq_len: int, batch: int, steps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    high = max(1, n_train - seq_len - 1)
    return rng.integers(0, high, size=(steps, batch), dtype=np.int64)


def val_starts(n_val: int, seq_len: int, batch: int, eval_batches: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    high = max(1, n_val - seq_len - 1)
    return rng.integers(0, high, size=(eval_batches, batch), dtype=np.int64)


def order_digest(starts: np.ndarray) -> str:
    return sha256_hex(np.ascontiguousarray(starts, dtype=np.int64).tobytes())


def gather(tokens: np.ndarray, starts: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = starts[:, None] + np.arange(seq_len + 1, dtype=np.int64)[None, :]
    win = tokens[idx]
    return win[:, :-1], win[:, 1:]


# ----------------------------------------------------------------------------
# the model family: the only thing the submission changes
# ----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (x * self.weight.float()).to(dtype)


def rope_tables(head_dim: int, seq_len: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    inv = 1.0 / (10000.0 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    ang = pos[:, None] * inv[None, :]
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    b, h, t, d = x.shape
    x = x.float().view(b, h, t, d // 2, 2)
    x0 = x[..., 0]
    x1 = x[..., 1]
    c = cos[:t].view(1, 1, t, d // 2)
    s = sin[:t].view(1, 1, t, d // 2)
    out = torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1)
    return out.view(b, h, t, d)


class Attention(nn.Module):
    def __init__(self, cfg: Dict[str, int]):
        super().__init__()
        d = cfg["d_model"]
        self.n_head = cfg["n_head"]
        self.n_kv_head = cfg["n_kv_head"]
        self.head_dim = cfg["head_dim"]
        self.rep = self.n_head // self.n_kv_head
        self.q_proj = nn.Linear(d, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, d, bias=False)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin).to(x.dtype)
        k = apply_rope(k, cos, sin).to(x.dtype)
        if self.rep > 1:
            k = k.repeat_interleave(self.rep, dim=1)
            v = v.repeat_interleave(self.rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(b, t, self.n_head * self.head_dim)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: Dict[str, int]):
        super().__init__()
        d = cfg["d_model"]
        h = cfg["d_ff"]
        self.gate_proj = nn.Linear(d, h, bias=False)
        self.up_proj = nn.Linear(d, h, bias=False)
        self.down_proj = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: Dict[str, int]):
        super().__init__()
        self.norm1 = RMSNorm(cfg["d_model"])
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg["d_model"])
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class TinyLM(nn.Module):
    def __init__(self, cfg: Dict[str, int], vocab_size: int):
        super().__init__()
        self.cfg = dict(cfg)
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, cfg["d_model"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])])
        self.norm_f = RMSNorm(cfg["d_model"])
        self.lm_head = nn.Linear(cfg["d_model"], vocab_size, bias=False)

    def forward(self, idx, cos, sin):
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.lm_head(self.norm_f(x))


def init_model(model: TinyLM, std: float, seed: int) -> None:
    """Frozen initialization rule, applied identically to every allocation."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    depth = max(1, model.cfg["n_layer"])
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() < 2:
                param.fill_(1.0)
                continue
            flat = torch.empty(param.shape, dtype=torch.float32)
            flat.normal_(mean=0.0, std=std, generator=gen)
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                flat.mul_(1.0 / (2.0 * depth) ** 0.5)
            param.copy_(flat.to(param.dtype))


def live_param_count(model: nn.Module) -> int:
    """Counted off the instantiated module, never off the formula."""
    return int(sum(p.numel() for p in model.parameters()))


# ----------------------------------------------------------------------------
# frozen optimizer and schedule
# ----------------------------------------------------------------------------

def build_optimizer(model: nn.Module, recipe: Dict[str, Any]):
    spec = recipe["optimizer"]
    decay, no_decay = [], []
    for _, param in model.named_parameters():
        (decay if param.dim() >= 2 else no_decay).append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": spec["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=spec["lr"],
        betas=(spec["beta1"], spec["beta2"]),
        eps=spec["eps"],
    )


def lr_at(step: int, profile: Dict[str, Any], peak: float) -> float:
    warm = profile["warmup_steps"]
    total = profile["steps"]
    floor = peak * profile["final_lr_frac"]
    if step < warm:
        return peak * (step + 1) / warm
    import math

    frac = (step - warm) / max(1, total - warm)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * min(1.0, frac)))


def schedule_digest(profile: Dict[str, Any], recipe: Dict[str, Any]) -> str:
    peak = recipe["optimizer"]["lr"]
    vals = [round(lr_at(s, profile, peak), 12) for s in range(profile["steps"])]
    return sha256_hex(canonical_json(vals).encode("utf-8"))


# ----------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------

def validate_arch(arch: Dict[str, Any], space: Dict[str, Any], profile_name: str) -> List[str]:
    """Every reason the allocation is outside the frozen search space."""
    rules = space["profiles"][profile_name]
    problems: List[str] = []
    for key in ARCH_KEYS:
        if key not in arch:
            problems.append("missing_key_" + key)
    if problems:
        return problems
    for key in ARCH_KEYS:
        value = arch[key]
        if not isinstance(value, int) or isinstance(value, bool):
            problems.append("non_integer_" + key)
    if problems:
        return problems
    for key in ARCH_KEYS:
        rule = rules[key]
        value = arch[key]
        if "choices" in rule and value not in rule["choices"]:
            problems.append("not_an_allowed_choice_" + key)
        if "min" in rule and value < rule["min"]:
            problems.append("below_min_" + key)
        if "max" in rule and value > rule["max"]:
            problems.append("above_max_" + key)
        if "multiple_of" in rule and value % rule["multiple_of"] != 0:
            problems.append("not_a_multiple_" + key)
    if arch["n_kv_head"] > arch["n_head"]:
        problems.append("n_kv_head_exceeds_n_head")
    elif arch["n_head"] % arch["n_kv_head"] != 0:
        problems.append("n_kv_head_does_not_divide_n_head")
    if arch["head_dim"] % 2 != 0:
        problems.append("odd_head_dim_breaks_rotary")
    return problems


def param_band(profile: Dict[str, Any]) -> Tuple[int, int]:
    budget = profile["param_budget"]
    tol = profile["param_tolerance"]
    return int(round(budget * (1.0 - tol))), int(round(budget * (1.0 + tol)))


# ----------------------------------------------------------------------------
# hash-chained telemetry
# ----------------------------------------------------------------------------

class Chain:
    def __init__(self, key: str, path: str):
        self.key = key.encode("utf-8")
        self.path = path
        self.prev = "genesis"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w"):
            pass

    def append(self, record: Dict[str, Any]) -> None:
        body = {k: v for k, v in record.items() if k != "chain"}
        mac = hmac.new(
            self.key,
            (self.prev + canonical_json(body)).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        record = dict(body)
        record["chain"] = mac
        self.prev = mac
        with open(self.path, "a") as handle:
            handle.write(canonical_json(record) + "\n")


def verify_chain(records: List[Dict[str, Any]], key: str) -> Tuple[bool, str]:
    prev = "genesis"
    for rec in records:
        body = {k: v for k, v in rec.items() if k != "chain"}
        mac = hmac.new(
            key.encode("utf-8"),
            (prev + canonical_json(body)).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if rec.get("chain") != mac:
            return False, "chain_break_at_index_" + str(records.index(rec))
        prev = mac
    return True, ""
