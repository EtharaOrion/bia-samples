"""BIA S01 frozen substrate.

This module owns every axis the task freezes: the data generating process, the
batch contract, the architecture, and the weight initialization. A submission
never imports from here to change any of it; the runner imports it to build the
substrate and to observe it while training.

The same functions run at both declared scales. `full` is the graded scaled
operating point on one H100. `smoke` is the CPU-only proof path that exercises
the identical call graph at tiny size. Nothing in this file branches on scale
except by reading the resolved config dict.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TELEMETRY_SCHEMA = "bia.s01/telemetry/v1"

# ---------------------------------------------------------------------------
# Scale configurations. Frozen bytes. A submission cannot reach these.
# ---------------------------------------------------------------------------

SCALES: Dict[str, dict] = {
    "full": dict(
        scale="full",
        # data generating process. vocab**order is the context table row count,
        # and that row count rather than the token count sets how many optimizer
        # steps the descent needs, so it is sized against the attempt budget.
        vocab=256,
        order=2,
        support=8,
        alpha=0.4,
        corpus_seed=20260821,
        train_seqs=262144,
        val_seqs=1024,
        seq_len=256,
        # architecture
        d_model=384,
        n_layer=6,
        n_head=6,
        d_ff=1536,
        # batch contract
        seqs_per_forward=256,
        forward_backward_per_step=1,
        # schedule of the measurement, not of the optimizer
        total_steps=700,
        eval_every=10,
        eval_batches=2,
        # grading anchors
        target_closure=0.50,
        baseline_steps=310,
        target_steps=248,
        sig_margin=0.002,
        min_seeds=2,
        recon_tol=0.05,
    ),
    "smoke": dict(
        scale="smoke",
        vocab=64,
        order=2,
        support=4,
        alpha=0.4,
        corpus_seed=20260821,
        train_seqs=2048,
        val_seqs=256,
        seq_len=64,
        d_model=64,
        n_layer=2,
        n_head=2,
        d_ff=256,
        seqs_per_forward=16,
        forward_backward_per_step=1,
        total_steps=1200,
        eval_every=10,
        eval_batches=2,
        target_closure=0.50,
        baseline_steps=560,
        target_steps=448,
        sig_margin=0.005,
        min_seeds=2,
        recon_tol=1e-06,
    ),
}


def resolve_scale(name: str) -> dict:
    if name not in SCALES:
        raise SystemExit(f"unknown scale {name!r}, expected one of {sorted(SCALES)}")
    cfg = dict(SCALES[name])
    cfg["tokens_per_step"] = cfg["seqs_per_forward"] * cfg["seq_len"]
    return cfg


def resolve_device(requested: str) -> torch.device:
    """Never touches CUDA unless explicitly asked for it.

    `cpu` is honoured without any driver query at all, which is what keeps the
    smoke path free of CUDA context creation.
    """
    requested = (requested or "auto").lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Frozen data generating process
# ---------------------------------------------------------------------------


def _source_tables(cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Sparse order-k categorical source, fully determined by the frozen seed."""
    v, k, s = cfg["vocab"], cfg["order"], cfg["support"]
    n_ctx = v ** k
    rng = np.random.default_rng(cfg["corpus_seed"])
    choices = np.empty((n_ctx, s), dtype=np.int32)
    block = 16384
    for lo in range(0, n_ctx, block):
        hi = min(lo + block, n_ctx)
        keys = rng.random((hi - lo, v), dtype=np.float64)
        part = np.argpartition(keys, s, axis=1)[:, :s]
        choices[lo:hi] = np.sort(part, axis=1).astype(np.int32)
    weights = rng.gamma(cfg["alpha"], size=(n_ctx, s))
    probs = weights / weights.sum(axis=1, keepdims=True)
    return choices, probs


def _ctx_index(window: np.ndarray, vocab: int) -> np.ndarray:
    """window is (n, order) with the oldest token first."""
    idx = np.zeros(window.shape[0], dtype=np.int64)
    for col in range(window.shape[1]):
        idx = idx * vocab + window[:, col].astype(np.int64)
    return idx


def _sample_block(cfg, choices, probs, n_seq, rng) -> np.ndarray:
    v, k, L = cfg["vocab"], cfg["order"], cfg["seq_len"]
    cum = probs.cumsum(axis=1)
    chunk = max(1, min(n_seq, 65536))
    parts = []
    for lo in range(0, n_seq, chunk):
        n = min(chunk, n_seq - lo)
        out = np.empty((n, L), dtype=np.int64)
        out[:, :k] = rng.integers(0, v, size=(n, k))
        for t in range(k, L):
            ctx = _ctx_index(out[:, t - k:t], v)
            u = rng.random(n)
            j = (cum[ctx] < u[:, None]).sum(axis=1)
            np.clip(j, 0, cfg["support"] - 1, out=j)
            out[:, t] = choices[ctx, j]
        parts.append(out.astype(np.int32))
    return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]


def _true_nll(cfg, choices, probs, block: np.ndarray) -> float:
    """Cross entropy of the true source on this block, over graded positions."""
    k, v = cfg["order"], cfg["vocab"]
    tot, n = 0.0, 0
    for t in range(k, cfg["seq_len"]):
        ctx = _ctx_index(block[:, t - k:t], v)
        tgt = block[:, t]
        rows = choices[ctx]
        pr = (probs[ctx] * (rows == tgt[:, None])).sum(axis=1)
        pr = np.maximum(pr, 1e-300)
        tot += float(-np.log(pr).sum())
        n += pr.shape[0]
    return tot / n


def _marginal_entropy(cfg, block: np.ndarray) -> float:
    k = cfg["order"]
    tgt = block[:, k:].reshape(-1)
    counts = np.bincount(tgt, minlength=cfg["vocab"]).astype(np.float64)
    f = counts / counts.sum()
    nz = f[f > 0]
    return float(-(nz * np.log(nz)).sum())


class Corpus:
    def __init__(self, cfg: dict):
        choices, probs = _source_tables(cfg)
        rng = np.random.default_rng(cfg["corpus_seed"] + 1)
        self.train = _sample_block(cfg, choices, probs, cfg["train_seqs"], rng)
        self.val = _sample_block(cfg, choices, probs, cfg["val_seqs"], rng)
        self.cond_loss = _true_nll(cfg, choices, probs, self.val)
        self.uni_loss = _marginal_entropy(cfg, self.val)
        self.target_loss = self.uni_loss - cfg["target_closure"] * (self.uni_loss - self.cond_loss)
        h = hashlib.sha256()
        h.update(self.train.astype("<i4").tobytes())
        h.update(self.val.astype("<i4").tobytes())
        self.digest = h.hexdigest()

    def anchors(self) -> dict:
        return {
            "val_conditional_loss": round(self.cond_loss, 6),
            "val_marginal_loss": round(self.uni_loss, 6),
            "target_loss": round(self.target_loss, 6),
            "corpus_digest": self.digest,
        }


def build_corpus(cfg: dict, cache_dir: str | None = None) -> Corpus:
    """Deterministic. The cache is a speed convenience and never a source of truth."""
    if not cache_dir:
        return Corpus(cfg)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"corpus_{cfg['scale']}_{cfg['corpus_seed']}.npz")
    if os.path.exists(path):
        try:
            z = np.load(path)
            c = Corpus.__new__(Corpus)
            c.train, c.val = z["train"], z["val"]
            c.cond_loss = float(z["cond"])
            c.uni_loss = float(z["uni"])
            c.target_loss = float(z["target"])
            c.digest = str(z["digest"])
            return c
        except Exception:
            pass
    c = Corpus(cfg)
    np.savez(path, train=c.train, val=c.val, cond=c.cond_loss, uni=c.uni_loss,
             target=c.target_loss, digest=c.digest)
    return c


# ---------------------------------------------------------------------------
# Frozen architecture
# ---------------------------------------------------------------------------


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, h = cfg["d_model"], cfg["n_head"]
        self.n_head = h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, cfg["d_ff"], bias=False)
        self.fc2 = nn.Linear(cfg["d_ff"], d, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        y = self.ln1(x)
        q, k, v = self.qkv(y).split(d, dim=2)
        shape = (b, t, self.n_head, d // self.n_head)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(b, t, d)
        x = x + self.proj(a)
        y = self.ln2(x)
        x = x + self.fc2(F.relu(self.fc1(y)) ** 2)
        return x


class FrozenGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.cfg = cfg
        self.tok = nn.Embedding(cfg["vocab"], d)
        self.pos = nn.Embedding(cfg["seq_len"], d)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg["vocab"], bias=False)

    def forward(self, idx):
        b, t = idx.shape
        p = torch.arange(t, device=idx.device)
        x = self.tok(idx) + self.pos(p)[None, :, :]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.lnf(x))


def build_model(cfg: dict) -> FrozenGPT:
    return FrozenGPT(cfg)


def frozen_init(model: FrozenGPT, init_seed: int) -> None:
    """Harness-owned initialization. A submission cannot reach or replace it.

    The walk is over sorted parameter names from a single CPU generator, so the
    resulting weights depend on the seed and on nothing else, including device,
    thread count and torch global RNG state.
    """
    g = torch.Generator(device="cpu").manual_seed(int(init_seed))
    n_layer = model.cfg["n_layer"]
    with torch.no_grad():
        for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
            if name.endswith("ln1.weight") or name.endswith("ln2.weight") or name == "lnf.weight":
                p.copy_(torch.ones_like(p))
            elif name.endswith(".bias"):
                p.copy_(torch.zeros_like(p))
            elif name.endswith("proj.weight") or name.endswith("fc2.weight"):
                std = 0.02 / math.sqrt(2.0 * n_layer)
                p.copy_(torch.normal(0.0, std, size=tuple(p.shape), generator=g))
            elif name == "head.weight":
                p.copy_(torch.zeros_like(p))
            else:
                p.copy_(torch.normal(0.0, 0.02, size=tuple(p.shape), generator=g))


def arch_signature(model: FrozenGPT) -> str:
    items = [(n, list(p.shape)) for n, p in sorted(model.named_parameters(), key=lambda kv: kv[0])]
    payload = json.dumps(items, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def init_digest(model: FrozenGPT) -> str:
    h = hashlib.sha256()
    with torch.no_grad():
        for n, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
            h.update(n.encode())
            h.update(p.detach().to("cpu", torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


def expected_init_digest(cfg: dict, seed: int) -> str:
    """Recomputed by the grader on CPU. Never reads anything the agent wrote."""
    m = build_model(cfg)
    frozen_init(m, seed)
    return init_digest(m)


# ---------------------------------------------------------------------------
# Frozen training loop, the only place a submission is invoked
# ---------------------------------------------------------------------------


class WriteGuard:
    """Records writes performed by submitted code across the Python write surfaces.

    An earlier revision patched `builtins.open` alone, which submitted code
    bypassed by descending to `os.open` and `os.write`. The guard now patches
    every documented Python-level write entry point rather than the one most
    submissions happen to use.

    Destinations under `allow` are runtime scratch the harness itself hands to
    the process, for instance the torch inductor cache. They are excluded so
    the control does not fire on its own harness. Nothing in that scratch is
    read as evidence by anything, and the verifier root that contains it is
    created per grading pass, so the exclusion carries no smuggling route.

    Bounded claim: this records writes issued through the Python names in
    `_TARGETS`. A write issued through a raw syscall, for instance through
    ctypes, is outside what this object can see, and is left to the
    verifier-side before-and-after diff of the monitored roots.
    """

    _TARGETS = (
        ("builtins", "open", "open"),
        ("io", "open", "open"),
        ("os", "open", "osopen"),
        ("os", "write", "oswrite"),
        ("os", "rename", "arg1"),
        ("os", "replace", "arg1"),
        ("os", "link", "arg1"),
        ("os", "symlink", "arg1"),
        ("os", "remove", "arg0"),
        ("os", "unlink", "arg0"),
        ("os", "rmdir", "arg0"),
        ("os", "mkdir", "arg0"),
        ("os", "makedirs", "arg0"),
        ("os", "truncate", "arg0"),
        ("shutil", "copyfile", "arg1"),
        ("shutil", "copy", "arg1"),
        ("shutil", "copy2", "arg1"),
        ("shutil", "move", "arg1"),
        ("shutil", "rmtree", "arg0"),
        ("pathlib.Path", "open", "open"),
        ("pathlib.Path", "write_text", "arg0"),
        ("pathlib.Path", "write_bytes", "arg0"),
        ("pathlib.Path", "mkdir", "arg0"),
        ("pathlib.Path", "touch", "arg0"),
        ("pathlib.Path", "unlink", "arg0"),
        ("pathlib.Path", "rename", "arg0"),
        ("pathlib.Path", "replace", "arg0"),
    )

    _WRITE_FLAGS = ("O_WRONLY", "O_RDWR", "O_CREAT", "O_APPEND", "O_TRUNC")

    def __init__(self, allow=None):
        self.seen: List[str] = []
        self.allow = [os.path.abspath(a) for a in (allow or []) if a]
        self._saved: List[tuple] = []
        self._depth = 0
        self._tracked_fds = {}

    @staticmethod
    def _resolve(dotted: str):
        import importlib
        head, _, tail = dotted.partition(".")
        obj = importlib.import_module(head)
        return getattr(obj, tail) if tail else obj

    def _allowed(self, target) -> bool:
        try:
            path = os.path.abspath(os.fspath(target))
        except TypeError:
            return False
        return any(path == a or path.startswith(a + os.sep) for a in self.allow)

    def _record(self, label, target):
        if self._allowed(target):
            return
        self.seen.append("%s:%s" % (label, target))

    def _wrap(self, real, label, kind):
        if kind == "open":
            def guarded(file, mode="r", *a, **kw):
                if any(c in str(mode) for c in ("w", "a", "x", "+")):
                    self._record(label, file)
                return real(file, mode, *a, **kw)
        elif kind == "osopen":
            mask = 0
            for name in self._WRITE_FLAGS:
                mask |= getattr(os, name, 0)

            def guarded(path, flags, *a, **kw):
                fd = real(path, flags, *a, **kw)
                if int(flags) & mask:
                    self._record(label, path)
                    if not self._allowed(path):
                        self._tracked_fds[fd] = str(path)
                return fd
        elif kind == "oswrite":
            def guarded(fd, data, *a, **kw):
                if fd in self._tracked_fds:
                    self._record(label, self._tracked_fds[fd])
                return real(fd, data, *a, **kw)
        elif kind == "arg1":
            def guarded(*a, **kw):
                self._record(label, a[1] if len(a) > 1 else "")
                return real(*a, **kw)
        else:
            def guarded(*a, **kw):
                self._record(label, a[0] if a else "")
                return real(*a, **kw)
        return guarded

    def __enter__(self):
        self._depth += 1
        if self._depth > 1:
            return self
        for dotted, attr, kind in self._TARGETS:
            try:
                holder = self._resolve(dotted)
                real = getattr(holder, attr)
            except (ImportError, AttributeError):
                continue
            self._saved.append((holder, attr, real))
            try:
                setattr(holder, attr, self._wrap(real, "%s.%s" % (dotted, attr), kind))
            except (AttributeError, TypeError):
                self._saved.pop()
        return self

    def __exit__(self, *exc):
        self._depth -= 1
        if self._depth > 0:
            return False
        while self._saved:
            holder, attr, real = self._saved.pop()
            try:
                setattr(holder, attr, real)
            except (AttributeError, TypeError):
                pass
        return False


def guard_allowlist() -> List[str]:
    raw = os.environ.get("BIA_GUARD_ALLOW", "")
    return [part for part in raw.split(os.pathsep) if part]


class CallCounter:
    """Counts model forward and backward traversals inside one step window.

    The counters exist so `forward_backward_per_step` is a measured quantity
    rather than a constant the runner asserts. They are installed by the runner
    around the model it owns, so extra passes taken inside a submitted
    optimizer step are visible to the checker that bounds them.
    """

    def __init__(self):
        self.forward = 0
        self.backward = 0
        self._model = None
        self._real_forward = None

    def attach(self, model):
        self._model = model
        self._real_forward = model.forward

        def counting_forward(*a, **kw):
            self.forward += 1
            out = self._real_forward(*a, **kw)
            if isinstance(out, torch.Tensor) and out.requires_grad:
                out.register_hook(self._tick_backward)
            return out

        model.forward = counting_forward
        return self

    def _tick_backward(self, grad):
        self.backward += 1
        return grad

    def detach(self):
        if self._model is not None and self._real_forward is not None:
            try:
                del self._model.forward
            except AttributeError:
                self._model.forward = self._real_forward
        self._model = None

    def reset(self):
        self.forward = 0
        self.backward = 0


def _loss(logits, targets, order):
    lg = logits[:, order - 1:-1, :]
    tg = targets[:, order:]
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1))


@torch.no_grad()
def evaluate(model, cfg, val, device) -> float:
    model.eval()
    n = cfg["eval_batches"] * cfg["seqs_per_forward"]
    n = min(n, val.shape[0])
    total, count = 0.0, 0
    for i in range(0, n, cfg["seqs_per_forward"]):
        chunk = val[i:i + cfg["seqs_per_forward"]]
        x = torch.from_numpy(chunk).to(device, torch.long)
        logits = model(x)
        total += float(_loss(logits, x, cfg["order"])) * chunk.shape[0]
        count += chunk.shape[0]
    model.train()
    return total / max(count, 1)


def train_one_seed(
    cfg: dict,
    seed: int,
    corpus: Corpus,
    build_optimizer: Callable,
    build_schedule: Callable,
    device: torch.device,
    submission_digest: str,
    mode: str,
    emit: Callable[[dict], None],
    on_eval: Callable[[int, "FrozenGPT"], None] | None = None,
) -> List[Tuple[int, float]]:
    """Runs one seed end to end and emits one telemetry record per evaluation.

    `on_eval` is the verifier's checkpoint hook. It is called with the live
    model at step zero and at every evaluation point, which is what lets the
    verifier recompute the graded loss series itself from parameters rather
    than read a loss the training process reported about itself.
    """
    model = build_model(cfg)
    frozen_init(model, seed)
    sig = arch_signature(model)
    idig = init_digest(model)
    model.to(device)
    model.train()

    named = [(n, p) for n, p in sorted(model.named_parameters(), key=lambda kv: kv[0])]
    counter = CallCounter().attach(model)
    guard = WriteGuard(allow=guard_allowlist())
    with guard:
        opt = build_optimizer(named)
        sched = build_schedule(cfg["total_steps"])
    if on_eval is not None:
        # Taken after the submission has had its one chance to touch the
        # parameters, so a rule that perturbs the frozen initialization inside
        # build_optimizer is visible to the digest the verifier recomputes.
        on_eval(0, model)
    if not isinstance(opt, torch.optim.Optimizer):
        raise SystemExit("build_optimizer did not return a torch.optim.Optimizer")
    if not callable(sched):
        raise SystemExit("build_schedule did not return a callable")
    for group in opt.param_groups:
        group.setdefault("initial_lr", group.get("lr", 0.0))

    rng = np.random.default_rng(1000 + seed)
    order = rng.permutation(corpus.train.shape[0])
    cursor = 0
    bs = cfg["seqs_per_forward"]
    probe = named[0][1]
    curve: List[Tuple[int, float]] = []
    use_amp = device.type == "cuda"

    for step in range(1, cfg["total_steps"] + 1):
        if cursor + bs > order.shape[0]:
            order = rng.permutation(corpus.train.shape[0])
            cursor = 0
        batch = corpus.train[order[cursor:cursor + bs]]
        cursor += bs
        x = torch.from_numpy(batch).to(device, torch.long)

        with guard:
            mult = float(sched(step - 1))
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * mult
        applied_lr = float(opt.param_groups[0]["lr"])

        opt.zero_grad(set_to_none=True)
        counter.reset()
        if use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = _loss(logits, x, cfg["order"])
        else:
            logits = model(x)
            loss = _loss(logits, x, cfg["order"])
        observed_vocab = int(logits.shape[-1])
        loss.backward()
        before = probe.detach().clone()
        with guard:
            opt.step()
        forward_calls = int(counter.forward)
        backward_calls = int(counter.backward)
        delta = float((probe.detach() - before).abs().sum())
        train_loss = float(loss.detach())

        if step % cfg["eval_every"] == 0 or step == cfg["total_steps"]:
            vl = evaluate(model, cfg, corpus.val, device)
            if on_eval is not None:
                on_eval(step, model)
            curve.append((step, vl))
            emit({
                "schema": TELEMETRY_SCHEMA,
                "mode": mode,
                "scale": cfg["scale"],
                "seed": int(seed),
                "step": int(step),
                "val_loss": round(vl, 6),
                "train_loss": round(train_loss, 6),
                "lr": applied_lr,
                "param_delta_l1": delta,
                "tokens_per_step": int(x.shape[0]) * int(x.shape[1]),
                "sequences_per_forward": int(x.shape[0]),
                "sequence_length": int(x.shape[1]),
                "forward_backward_per_step": forward_calls,
                "forward_calls_this_step": forward_calls,
                "backward_calls_this_step": backward_calls,
                "vocab_size": observed_vocab,
                "arch_signature": sig,
                "init_digest": idig,
                "corpus_digest": corpus.digest,
                "target_loss": round(corpus.target_loss, 6),
                "submission_digest": submission_digest,
                "submission_writes": sorted(set(guard.seen)),
                "eval_weights_source": "live_parameters",
                "early_stop": False,
            })
    counter.detach()
    return curve
