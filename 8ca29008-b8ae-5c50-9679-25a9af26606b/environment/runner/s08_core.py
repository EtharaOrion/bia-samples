"""S08 frozen core: operating point, synthetic source, model, batching and evaluation.

This module is the single source of truth for every frozen axis of the task. The
private fixture generator and the harness runner both import it, so the checkpoint
that ships and the run that is graded are produced by identical code.

Nothing in this file is free for a submission to change. The submission owns only
the recovery of the optimizer, which lives behind the `recover` entry point the
runner imports from the submission tree.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, asdict, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CORE_VERSION = "s08-core-1"

# ---------------------------------------------------------------------------
# Operating points
# ---------------------------------------------------------------------------
# "scaled" is the graded operating point. It is sized against the per-attempt
# budget of 7.2 minutes of single-H100 time that the environment spec derives
# from 6 session hours divided by 50 attempts.
#
# "smoke" is the identical code path at a tiny CPU-only scale. It exists so the
# whole pipeline, including fixture generation, poisoned resume, control run,
# submission run and grading, can be executed end to end without an accelerator.

PROFILES = {
    "scaled": dict(
        profile="scaled",
        vocab_size=1024,
        seq_len=256,
        d_model=128,
        n_layer=3,
        n_head=4,
        d_ff=512,
        batch_sequences=32,
        n_contexts=4096,
        coarse=4,
        support=8,
        dirichlet_alpha=0.35,
        streams=4096,
        stream_len=2048,
        val_streams=128,
        corpus_seed=20260821,
        init_seed=20260822,
        pretrain_steps=400,
        pretrain_warmup=100,
        pretrain_lr=3.0e-3,
        pretrain_wd=0.01,
        max_steps=6000,
        eval_every=50,
        eval_batches=16,
        seeds=(0, 1),
        probe_batches_max=16,
        target_fraction=0.5,
        full_reward_fraction=0.20,
    ),
    "smoke": dict(
        profile="smoke",
        vocab_size=128,
        seq_len=64,
        d_model=64,
        n_layer=2,
        n_head=4,
        d_ff=256,
        batch_sequences=16,
        n_contexts=256,
        coarse=2,
        support=4,
        dirichlet_alpha=0.35,
        streams=512,
        stream_len=512,
        val_streams=64,
        corpus_seed=20260821,
        init_seed=20260822,
        pretrain_steps=80,
        pretrain_warmup=20,
        pretrain_lr=3.0e-3,
        pretrain_wd=0.01,
        max_steps=400,
        eval_every=20,
        eval_batches=8,
        seeds=(0, 1),
        probe_batches_max=8,
        target_fraction=0.5,
        full_reward_fraction=0.20,
    ),
}

# Keys whose value the telemetry records on every step. A run whose records do
# not all carry these exact values violated the frozen contract.
FROZEN_KEYS = (
    "vocab_size",
    "seq_len",
    "d_model",
    "n_layer",
    "n_head",
    "d_ff",
    "batch_sequences",
    "fwd_bwd_per_step",
)


@dataclass(frozen=True)
class Cfg:
    profile: str
    vocab_size: int
    seq_len: int
    d_model: int
    n_layer: int
    n_head: int
    d_ff: int
    batch_sequences: int
    n_contexts: int
    coarse: int
    support: int
    dirichlet_alpha: float
    streams: int
    stream_len: int
    val_streams: int
    corpus_seed: int
    init_seed: int
    pretrain_steps: int
    pretrain_warmup: int
    pretrain_lr: float
    pretrain_wd: float
    max_steps: int
    eval_every: int
    eval_batches: int
    seeds: tuple
    probe_batches_max: int
    target_fraction: float
    full_reward_fraction: float

    @property
    def tokens_per_step(self) -> int:
        return self.batch_sequences * self.seq_len


def load_cfg(profile: str) -> Cfg:
    if profile not in PROFILES:
        raise KeyError(f"unknown profile {profile!r}, expected one of {sorted(PROFILES)}")
    d = dict(PROFILES[profile])
    d["seeds"] = tuple(d["seeds"])
    return Cfg(**d)


def frozen_facts(cfg: Cfg) -> dict:
    return {
        "vocab_size": cfg.vocab_size,
        "seq_len": cfg.seq_len,
        "d_model": cfg.d_model,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "d_ff": cfg.d_ff,
        "batch_sequences": cfg.batch_sequences,
        "fwd_bwd_per_step": 1,
    }


def public_cfg(cfg: Cfg, checkpoint_step: int, target_loss: float, bayes_loss: float) -> dict:
    """The read-only mapping the runner hands to `recover`. Every value here is
    also present in the agent-visible fixture manifest."""
    d = frozen_facts(cfg)
    d.update(
        {
            "profile": cfg.profile,
            "max_steps": cfg.max_steps,
            "eval_every": cfg.eval_every,
            "tokens_per_step": cfg.tokens_per_step,
            "probe_batches_max": cfg.probe_batches_max,
            "checkpoint_step": checkpoint_step,
            "target_loss": target_loss,
            "source_entropy_nats": bayes_loss,
        }
    )
    return d


# ---------------------------------------------------------------------------
# Synthetic source
# ---------------------------------------------------------------------------
# A second-order hashed Markov source. The generating conditional is known in
# closed form, so the Bayes-optimal cross entropy of the validation shard is
# computed exactly rather than estimated. That exact number is what anchors the
# target loss, which removes the need for an unmeasured hand-picked constant.


def context_ids(cfg: Cfg, prev2, prev1):
    """The context a token is drawn from: the whole previous token plus a coarse
    bucket of the token before it. The second-order dependence is real, and it is
    learnable from a realistic token count rather than requiring a random pair hash
    to be memorized."""
    return (prev1.astype(np.int64) * cfg.coarse + (prev2.astype(np.int64) % cfg.coarse)) % cfg.n_contexts


def build_source(cfg: Cfg):
    rng = np.random.default_rng(cfg.corpus_seed)
    V, K, S = cfg.vocab_size, cfg.n_contexts, cfg.support
    support = np.empty((K, S), dtype=np.int64)
    for k in range(K):
        support[k] = rng.choice(V, size=S, replace=False)
    weights = rng.dirichlet(np.full(S, cfg.dirichlet_alpha), size=K)
    return support, weights


def _sample(cfg: Cfg, src, n_streams: int, length: int, seed: int) -> np.ndarray:
    support, weights = src
    rng = np.random.default_rng(seed)
    K, S = cfg.n_contexts, cfg.support
    cum = np.cumsum(weights, axis=1)
    cum[:, -1] = 1.0
    out = np.empty((n_streams, length), dtype=np.uint16)
    t2 = rng.integers(0, cfg.vocab_size, size=n_streams, dtype=np.int64)
    t1 = rng.integers(0, cfg.vocab_size, size=n_streams, dtype=np.int64)
    out[:, 0] = t2
    out[:, 1] = t1
    for i in range(2, length):
        ctx = context_ids(cfg, t2, t1)
        u = rng.random(n_streams)
        j = (u[:, None] > cum[ctx]).sum(axis=1)
        np.clip(j, 0, S - 1, out=j)
        nxt = support[ctx, j]
        out[:, i] = nxt
        t2, t1 = t1, nxt
    return out


def bayes_nats(cfg: Cfg, src, arr: np.ndarray) -> float:
    """Exact Bayes-optimal cross entropy in nats over the positions the model is
    scored on, which are the positions that carry the full second-order context."""
    support, weights = src
    toks = arr.astype(np.int64)
    prev2 = toks[:, :-2]
    prev1 = toks[:, 1:-1]
    tgt = toks[:, 2:]
    ctx = context_ids(cfg, prev2, prev1)
    sup = support[ctx]
    match = sup == tgt[..., None]
    idx = np.argmax(match, axis=-1)
    found = match.any(axis=-1)
    p = np.take_along_axis(weights[ctx], idx[..., None], axis=-1)[..., 0]
    p = np.where(found, p, 1e-12)
    return float(-np.log(p).mean())


def materialize_corpus(cfg: Cfg, data_dir: str) -> dict:
    """Write train.bin, val.bin and meta.json into data_dir if absent. Returns meta."""
    os.makedirs(data_dir, exist_ok=True)
    meta_path = os.path.join(data_dir, "meta.json")
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    if os.path.exists(meta_path) and os.path.exists(train_path) and os.path.exists(val_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("profile") == cfg.profile and meta.get("core_version") == CORE_VERSION:
            return meta
    src = build_source(cfg)
    train = _sample(cfg, src, cfg.streams, cfg.stream_len, cfg.corpus_seed + 1)
    val = _sample(cfg, src, cfg.val_streams, cfg.stream_len, cfg.corpus_seed + 2)
    train.tofile(train_path)
    val.tofile(val_path)
    meta = {
        "core_version": CORE_VERSION,
        "profile": cfg.profile,
        "train_shape": list(train.shape),
        "val_shape": list(val.shape),
        "train_sha256": sha256_file(train_path),
        "val_sha256": sha256_file(val_path),
        "bayes_loss_nats": bayes_nats(cfg, src, val),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    return meta


def load_corpus(cfg: Cfg, data_dir: str):
    meta = materialize_corpus(cfg, data_dir)
    train = np.fromfile(os.path.join(data_dir, "train.bin"), dtype=np.uint16)
    val = np.fromfile(os.path.join(data_dir, "val.bin"), dtype=np.uint16)
    train = train.reshape(tuple(meta["train_shape"]))
    val = val.reshape(tuple(meta["val_shape"]))
    return train, val, meta


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Batching. The data order is frozen: it is a pure function of the step index and
# the run seed, so a submission cannot reorder or reselect data.
# ---------------------------------------------------------------------------


def get_batch(arr: np.ndarray, cfg: Cfg, step: int, seed: int, device) -> tuple:
    rng = np.random.default_rng((cfg.corpus_seed * 1000003 + seed * 7919 + step) % (2**62))
    n_streams, length = arr.shape
    s = rng.integers(0, n_streams, size=cfg.batch_sequences)
    o = rng.integers(0, length - cfg.seq_len - 1, size=cfg.batch_sequences)
    x = np.stack([arr[si, oi : oi + cfg.seq_len] for si, oi in zip(s, o)]).astype(np.int64)
    y = np.stack([arr[si, oi + 1 : oi + 1 + cfg.seq_len] for si, oi in zip(s, o)]).astype(np.int64)
    return (
        torch.from_numpy(x).to(device, non_blocking=True),
        torch.from_numpy(y).to(device, non_blocking=True),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Block(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        b, t, c = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(c, dim=2)
        q = q.view(b, t, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.d_head).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(b, t, c)
        x = x + self.proj(a)
        h = self.ln2(x)
        x = x + self.fc2(F.gelu(self.fc1(h)))
        return x


class TinyLM(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.lnf = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, x):
        b, t = x.shape
        p = torch.arange(t, device=x.device)
        h = self.tok(x) + self.pos(p)[None]
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.lnf(h))


# The first predicted position of a window does not carry the full second-order
# context, so it is excluded from both the model loss and the Bayes anchor.
LOSS_SKIP = 1


def lm_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    lg = logits[:, LOSS_SKIP:, :]
    tg = y[:, LOSS_SKIP:]
    return F.cross_entropy(lg.reshape(-1, lg.size(-1)), tg.reshape(-1))


def build_model(cfg: Cfg, device) -> TinyLM:
    g = torch.Generator(device="cpu").manual_seed(cfg.init_seed)
    model = TinyLM(cfg)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() >= 2:
                std = 0.02 if "tok" in name or "pos" in name else (1.0 / math.sqrt(p.shape[-1]))
                p.copy_(torch.empty(p.shape).normal_(0.0, std, generator=g))
            else:
                p.zero_()
        for n, b in model.named_buffers():
            pass
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.fill_(1.0)
                m.bias.zero_()
    return model.to(device)


@torch.no_grad()
def evaluate(model: TinyLM, val: np.ndarray, cfg: Cfg, device) -> float:
    model.eval()
    total = 0.0
    for i in range(cfg.eval_batches):
        x, y = get_batch(val, cfg, step=i, seed=987654, device=device)
        total += float(lm_loss(model(x), y))
    model.train()
    return total / cfg.eval_batches


def weight_digest(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().to("cpu", torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


def optimizer_fingerprint(opt) -> dict:
    """A compact, comparable read of live optimizer state. The runner records this
    for the poisoned checkpoint and again for whatever `recover` returned, which
    is the live-state read the EFFECT checker traces to."""
    sd = opt.state_dict()
    per_index = {}
    for idx in sorted(sd["state"], key=lambda k: int(k)):
        st = sd["state"][idx]
        entry = {}
        for key in ("exp_avg", "exp_avg_sq"):
            if key in st and torch.is_tensor(st[key]):
                t = st[key].detach().to("cpu", torch.float64)
                entry[key + "_rms"] = float(torch.sqrt((t * t).mean()))
        if "step" in st:
            s = st["step"]
            entry["step"] = float(s) if not torch.is_tensor(s) else float(s.item())
        per_index[str(idx)] = entry
    groups = [
        {
            "lr": float(g.get("lr", float("nan"))),
            "weight_decay": float(g.get("weight_decay", float("nan"))),
            "betas": [float(b) for b in g.get("betas", (float("nan"), float("nan")))],
            "eps": float(g.get("eps", float("nan"))),
        }
        for g in sd["param_groups"]
    ]
    return {"state": per_index, "param_groups": groups}


def fingerprint_from_state(state: dict) -> dict:
    """Same read shape as optimizer_fingerprint, taken from a raw saved state dict."""
    per_index = {}
    for idx in sorted(state["state"], key=lambda k: int(k)):
        st = state["state"][idx]
        entry = {}
        for key in ("exp_avg", "exp_avg_sq"):
            if key in st and torch.is_tensor(st[key]):
                t = st[key].detach().to("cpu", torch.float64)
                entry[key + "_rms"] = float(torch.sqrt((t * t).mean()))
        if "step" in st:
            s = st["step"]
            entry["step"] = float(s) if not torch.is_tensor(s) else float(s.item())
        per_index[str(idx)] = entry
    groups = [
        {
            "lr": float(g.get("lr", float("nan"))),
            "weight_decay": float(g.get("weight_decay", float("nan"))),
            "betas": [float(b) for b in g.get("betas", (float("nan"), float("nan")))],
            "eps": float(g.get("eps", float("nan"))),
        }
        for g in state["param_groups"]
    ]
    return {"state": per_index, "param_groups": groups}
