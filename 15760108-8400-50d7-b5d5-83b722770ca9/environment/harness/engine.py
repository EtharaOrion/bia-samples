"""Frozen measurement engine for bia S09 multi-objective-frontier.

Everything in this file is harness-owned and frozen. The agent supplies only
`propose_frontier()` and `build_optimizer()` through `submission/frontier.py`.

The two objectives, the tradeoff surface, the normalization, the hypervolume
reference point, the data source, the architecture, the token budget and the
per-run wall-clock cap are all fixed here and in the frozen spec fixture.

This module runs identically at graded scale (one H100) and at smoke scale
(CPU only). The only difference is which frozen spec fixture is loaded.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SPEC_FIELDS = (
    "spec_id",
    "vocab_size",
    "successors",
    "uniform_floor",
    "source_seed",
    "train_tokens",
    "val_tokens",
    "seq_len",
    "batch_sequences",
    "max_steps",
    "per_run_seconds",
    "shared_overhead_seconds",
    "n_layer",
    "n_head",
    "d_model",
    "frontier_points",
    "run_seed",
    "base_lr",
    "density_tau",
    "density_ideal",
    "loss_offset_ideal",
    "density_scope",
    "reference_point_source",
    "force_cpu",
)


@dataclass(frozen=True)
class FrozenSpec:
    spec_id: str
    vocab_size: int
    successors: int
    uniform_floor: float
    source_seed: int
    train_tokens: int
    val_tokens: int
    seq_len: int
    batch_sequences: int
    max_steps: int
    per_run_seconds: float
    shared_overhead_seconds: float
    n_layer: int
    n_head: int
    d_model: int
    frontier_points: int
    run_seed: int
    base_lr: float
    density_tau: float
    density_ideal: float
    loss_offset_ideal: float
    density_scope: str
    reference_point_source: str
    force_cpu: bool


def load_spec(path: str | os.PathLike) -> FrozenSpec:
    raw = json.loads(pathlib.Path(path).read_text())
    missing = [k for k in SPEC_FIELDS if k not in raw]
    if missing:
        raise ValueError(f"frozen spec is missing fields: {missing}")
    extra = [k for k in raw if k not in SPEC_FIELDS]
    if extra:
        raise ValueError(f"frozen spec carries unknown fields: {extra}")
    return FrozenSpec(**{k: raw[k] for k in SPEC_FIELDS})


def spec_digest(spec: FrozenSpec) -> str:
    blob = json.dumps(asdict(spec), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# The frozen data source: a first-order Markov chain with a uniform floor.
# Its entropy rate is computable in closed form, which is what anchors the
# loss axis of the normalization without any measured reference run.
# ---------------------------------------------------------------------------


def build_source(spec: FrozenSpec):
    rng = np.random.default_rng(spec.source_seed)
    v, k = spec.vocab_size, spec.successors
    succ = np.empty((v, k), dtype=np.int64)
    for i in range(v):
        succ[i] = rng.choice(v, size=k, replace=False)
    weights = rng.gamma(shape=1.0, scale=1.0, size=(v, k))
    q = weights / weights.sum(axis=1, keepdims=True)
    return succ, q


def source_entropy_rate(spec: FrozenSpec, succ: np.ndarray, q: np.ndarray) -> float:
    """Exact entropy rate in nats of the frozen chain, by power iteration."""
    v, k = succ.shape
    eps = spec.uniform_floor
    floor = eps / v
    p_succ = (1.0 - eps) * q + floor
    row_h = -(p_succ * np.log(p_succ)).sum(axis=1) - (v - k) * floor * math.log(floor)
    pi = np.full(v, 1.0 / v)
    for _ in range(400):
        mass = (pi[:, None] * (1.0 - eps) * q).ravel()
        nxt = np.bincount(succ.ravel(), weights=mass, minlength=v) + eps / v
        nxt /= nxt.sum()
        if np.abs(nxt - pi).max() < 1e-13:
            pi = nxt
            break
        pi = nxt
    return float((pi * row_h).sum())


def generate_tokens(spec: FrozenSpec, succ: np.ndarray, q: np.ndarray, n_tokens: int, seed: int):
    rng = np.random.default_rng(seed)
    v, k = succ.shape
    eps = spec.uniform_floor
    cum = np.cumsum(q, axis=1)
    out = np.empty(n_tokens, dtype=np.int64)
    jump = rng.random(n_tokens)
    pick = rng.random(n_tokens)
    unif = rng.integers(0, v, size=n_tokens)
    state = 0
    for t in range(n_tokens):
        if jump[t] < eps:
            state = int(unif[t])
        else:
            j = int(np.searchsorted(cum[state], pick[t] * cum[state, -1]))
            if j >= k:
                j = k - 1
            state = int(succ[state, j])
        out[t] = state
    return out


def data_digest(train: np.ndarray, val: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(train.tobytes())
    h.update(b"|")
    h.update(val.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# The frozen architecture. Only the four block matrices per layer carry the
# density objective; embeddings and the tied head are excluded by declaration.
# ---------------------------------------------------------------------------


def rmsnorm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.fc1 = nn.Linear(d_model, 4 * d_model, bias=False)
        self.fc2 = nn.Linear(4 * d_model, d_model, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        h = rmsnorm(x)
        qkv = self.qkv(h).view(b, t, 3, self.n_head, d // self.n_head)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        h = rmsnorm(x)
        x = x + self.fc2(F.relu(self.fc1(h)).pow(2))
        return x


class TinyGPT(nn.Module):
    def __init__(self, spec: FrozenSpec):
        super().__init__()
        self.tok = nn.Embedding(spec.vocab_size, spec.d_model)
        self.pos = nn.Embedding(spec.seq_len, spec.d_model)
        self.blocks = nn.ModuleList(Block(spec.d_model, spec.n_head) for _ in range(spec.n_layer))
        self.head = nn.Linear(spec.d_model, spec.vocab_size, bias=False)
        self.head.weight = self.tok.weight
        self.fwd_calls = 0
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=1.0 / math.sqrt(m.embedding_dim))
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=1.0 / math.sqrt(m.in_features))

    def forward(self, idx):
        self.fwd_calls += 1
        b, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))[None]
        for blk in self.blocks:
            x = blk(x)
        return self.head(rmsnorm(x))


def weight_matrices(model: nn.Module):
    """Every two-dimensional parameter tensor: both embeddings and, per layer,
    qkv, proj, fc1 and fc2. The head is tied to the token embedding, so
    named_parameters yields that tensor exactly once and it is counted once."""
    return [(name, p) for name, p in model.named_parameters() if p.dim() == 2]


def architecture_signature(spec: FrozenSpec, model: nn.Module) -> str:
    shapes = [f"{n}:{tuple(p.shape)}" for n, p in sorted(model.named_parameters(), key=lambda kv: kv[0])]
    blob = "|".join(shapes) + f"||L{spec.n_layer}H{spec.n_head}D{spec.d_model}T{spec.seq_len}V{spec.vocab_size}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# The two objectives.
# ---------------------------------------------------------------------------


@torch.no_grad()
def measure_val_loss(model: nn.Module, val: torch.Tensor, spec: FrozenSpec, device) -> float:
    model.eval()
    t = spec.seq_len
    usable = (val.numel() - 1) // t
    if usable < spec.batch_sequences:
        raise ValueError("frozen spec val_tokens is too small for one validation batch")
    n_batches = max(1, min(8, usable // spec.batch_sequences))
    total, count = 0.0, 0
    for b in range(n_batches):
        idxs = []
        for s in range(spec.batch_sequences):
            start = ((b * spec.batch_sequences) + s) * t
            idxs.append(start)
        x = torch.stack([val[i : i + t] for i in idxs]).to(device)
        y = torch.stack([val[i + 1 : i + 1 + t] for i in idxs]).to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
        total += float(loss) * y.numel()
        count += y.numel()
    model.train()
    return total / count


@torch.no_grad()
def measure_density(model: nn.Module, spec: FrozenSpec) -> float:
    """Scale-invariant retained-weight fraction over every frozen weight matrix.

    An entry is retained when |w| >= density_tau * rms(W) for its own tensor.
    The per-tensor relative threshold is what makes this objective immune to a
    uniform rescaling of the weights, which the residual RMS-normalized trunk
    would otherwise leave almost free.
    """
    kept, total = 0, 0
    for _, p in weight_matrices(model):
        w = p.detach().float()
        rms = float(w.pow(2).mean().sqrt())
        if rms <= 0.0:
            total += w.numel()
            continue
        kept += int((w.abs() >= spec.density_tau * rms).sum())
        total += w.numel()
    return kept / max(1, total)


# ---------------------------------------------------------------------------
# Normalization and hypervolume. Both are pinned here and nowhere else.
# ---------------------------------------------------------------------------


def untrained_reference(spec: FrozenSpec, val: torch.Tensor, device) -> dict:
    """Diagnostic measurement of the untrained frozen model.

    This does NOT set the normalization. The hypervolume reference point is
    analytic and is computed in `anchors`. This measurement is recorded so a
    reader can see that the analytic reference coordinates describe the actual
    untrained model, and so the per-run EFFECT evidence has a run-independent
    companion. It is a pure function of the frozen bytes and no agent code has
    run when it is taken.
    """
    torch.manual_seed(spec.run_seed)
    model = TinyGPT(spec).to(device)
    return {
        "val_loss": measure_val_loss(model, val, spec, device),
        "density": measure_density(model, spec),
        "architecture_signature": architecture_signature(spec, model),
    }


def dense_matrix_density(tau: float) -> float:
    """Retained fraction of a dense, unstructured (Gaussian) weight matrix.

    P(|w| >= tau * rms(W)) = erfc(tau / sqrt(2)). This is the analytic density
    of a matrix carrying no sparsity structure at all, so it is the natural
    reference coordinate on the density axis and it needs no measurement.
    """
    return math.erfc(tau / math.sqrt(2.0))


def anchors(spec: FrozenSpec, entropy_rate: float, reference: dict) -> dict:
    """The pinned normalization.

    Reference point, in raw objective space, is (ln(vocab_size), erfc(tau/sqrt2)):
    the uniform next-token predictor on the loss axis and a dense unstructured
    weight matrix on the density axis, which together describe a model that has
    learned nothing and compressed nothing. Ideal point is (entropy_rate +
    loss_offset_ideal, density_ideal): the information-theoretic floor of the
    frozen source plus a declared margin, and the declared sparse ideal. Both
    coordinates of both points are closed-form functions of the frozen bytes and
    no measurement enters either. Each objective is mapped to [0, 1] by the
    reference-to-ideal distance it covers, clipped, exactly as the cross-family
    scoring form in requirements/bia-environment-spec.md lines 82 to 85 states.
    In that normalized maximization space the hypervolume reference point is
    (0.0, 0.0) and the ideal point is (1.0, 1.0), so hypervolume is already on
    [0, 1] and the clamp is a guard rather than a rescaling.
    """
    loss_ref = math.log(spec.vocab_size)
    loss_ideal = entropy_rate + spec.loss_offset_ideal
    if loss_ref <= loss_ideal + 0.10:
        raise ValueError(
            f"degenerate loss axis: reference {loss_ref} is not above ideal {loss_ideal}"
        )
    density_ref = dense_matrix_density(spec.density_tau)
    if density_ref <= spec.density_ideal + 0.10:
        raise ValueError(
            f"degenerate density axis: reference {density_ref} is not above ideal {spec.density_ideal}"
        )
    return {
        "loss_ref": loss_ref,
        "loss_ideal": loss_ideal,
        "density_ref": density_ref,
        "density_ideal": spec.density_ideal,
        "entropy_rate": entropy_rate,
        "reference_point_source": spec.reference_point_source,
    }


def normalize_point(val_loss: float, density: float, anc: dict) -> tuple:
    u1 = (anc["loss_ref"] - val_loss) / (anc["loss_ref"] - anc["loss_ideal"])
    u2 = (anc["density_ref"] - density) / (anc["density_ref"] - anc["density_ideal"])
    return (min(max(u1, 0.0), 1.0), min(max(u2, 0.0), 1.0))


def pareto_front(points):
    """Maximization Pareto front over (u1, u2)."""
    front = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            if q[0] >= p[0] and q[1] >= p[1] and (q[0] > p[0] or q[1] > p[1]):
                dominated = True
                break
        if not dominated:
            front.append(p)
    return sorted(set(front), key=lambda p: (-p[0], -p[1]))


def hypervolume_sweep(points) -> float:
    """2D hypervolume, maximization, reference point (0.0, 0.0).

    Sweep by descending u1, accumulating the strip each newly dominating u2 adds.
    """
    hv, best_u2 = 0.0, 0.0
    for u1, u2 in sorted(points, key=lambda p: (-p[0], -p[1])):
        if u2 > best_u2:
            hv += u1 * (u2 - best_u2)
            best_u2 = u2
    return hv


def score_frontier(raw_points, anc: dict) -> dict:
    normalized = [normalize_point(l, d, anc) for l, d in raw_points]
    front = pareto_front(normalized)
    raw = hypervolume_sweep(normalized)
    return {
        "normalized_points": [list(p) for p in normalized],
        "pareto_front": [list(p) for p in front],
        "hypervolume_raw": raw,
        "hypervolume_front": hypervolume_sweep(front),
        "score": min(max(raw, 0.0), 1.0),
    }


# ---------------------------------------------------------------------------
# The frozen training loop. One forward and one backward per step, always.
# ---------------------------------------------------------------------------


def select_device(spec: FrozenSpec) -> torch.device:
    if spec.force_cpu or os.environ.get("BIA_S09_FORCE_CPU") == "1":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_point(spec, point_index, point, build_optimizer, train, val, device, anc):
    torch.manual_seed(spec.run_seed + point_index)
    np.random.seed(spec.run_seed + point_index)
    model = TinyGPT(spec).to(device)
    arch_sig = architecture_signature(spec, model)

    pre = {
        "val_loss": measure_val_loss(model, val, spec, device),
        "density": measure_density(model, spec),
    }
    model.fwd_calls = 0

    named = [(n, p) for n, p in model.named_parameters()]
    opt = build_optimizer(named, point, spec.base_lr)
    if not isinstance(opt, torch.optim.Optimizer):
        raise TypeError(f"build_optimizer returned {type(opt).__name__}, not a torch.optim.Optimizer")

    bwd_calls = {"n": 0}

    def count_backward(*_args, **_kwargs):
        bwd_calls["n"] += 1

    t = spec.seq_len
    n_windows = train.numel() - t - 1
    gen = torch.Generator().manual_seed(spec.run_seed * 1000 + point_index)

    t_start = time.time()
    deadline = t_start + spec.per_run_seconds
    steps_taken = 0
    stop_reason = "max_steps"
    loss_first, loss_last = None, None

    model.train()
    for _step in range(spec.max_steps):
        starts = torch.randint(0, n_windows, (spec.batch_sequences,), generator=gen)
        x = torch.stack([train[s : s + t] for s in starts.tolist()]).to(device)
        y = torch.stack([train[s + 1 : s + 1 + t] for s in starts.tolist()]).to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.register_hook(count_backward)
        loss.backward()
        opt.step()
        steps_taken += 1
        lv = float(loss.detach())
        if loss_first is None:
            loss_first = lv
        loss_last = lv
        if time.time() >= deadline:
            stop_reason = "per_run_deadline"
            break

    elapsed = time.time() - t_start
    fwd_train = model.fwd_calls
    post = {
        "val_loss": measure_val_loss(model, val, spec, device),
        "density": measure_density(model, spec),
    }
    u1, u2 = normalize_point(post["val_loss"], post["density"], anc)

    return {
        "point_index": point_index,
        "order_index": point_index,
        "point_spec_repr": repr(point)[:512],
        "architecture_signature": arch_sig,
        "max_steps": spec.max_steps,
        "steps_taken": steps_taken,
        "stop_reason": stop_reason,
        "forward_calls": fwd_train,
        "backward_calls": bwd_calls["n"],
        "train_loss_first": loss_first,
        "train_loss_last": loss_last,
        "pre": pre,
        "post": post,
        "objectives": {"val_loss": post["val_loss"], "density": post["density"]},
        "normalized": [u1, u2],
        "t_start": t_start,
        "t_end": time.time(),
        "elapsed_seconds": elapsed,
        "device": str(device),
    }
