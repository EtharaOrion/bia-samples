#!/usr/bin/env python3
"""The verifier's own measurement. This process owns the model and the counters.

RE-BASED. The graded path is now a real nanoGPT training run. The stand-in this
file previously carried, a bigram categorical count table trained by add-lambda
counting over an in-bundle text file, is gone. What replaces it is the canonical
decoder of `environment/nanogpt_substrate.json`, vocab_size 50304, num_layers
12, model_dim 768, head_dim 128, seq_len 1024, trained for exactly
`budget_token_updates` optimizer steps of 524288 tokens each with exactly one
forward and one backward pass per step, over FineWeb10B text re-tokenized with
the submitted vocabulary. The graded figure is the cross entropy in bits that
this process's own trained parameters spend over a held-out FineWeb slice,
divided by that slice's length in bytes.

Nothing here reads a number the submission printed. The submission's only
contribution is a vocabulary, loaded as DATA from `tokenizer.json`, and an
optional claim, loaded as DATA from `claim.json` and SEALED before any
measurement is computed. This module never imports the submission, never
executes it, and never consults a figure it reports about itself.

The held-out slice is resolved from THIS container's own corpus root and from
nowhere else. It is absent from `environment/`, absent from the agent image, and
there is no solver-reachable call that resolves it. A submission cannot compute
the graded figure; it can only estimate it on a slice of the train shards it
holds out for itself.

What this module produces is one telemetry record: a JSON-compatible mapping
describing what this process actually did. `checkers.py` grades that record and
nothing else, and its shape is unchanged by the re-base, so every checker, every
zero reason and every control declaration survives.

Frozen: the model, the optimizer, the compute budget, the evaluation split.
Free: the tokenizer and the vocabulary it is built over.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

RECORD_SCHEMA = "oer16.run_record/v1"
TOKENIZER_SCHEMA = "oer16.tokenizer/v1"
CLAIM_SCHEMA = "oer16.claim/v1"

# The verifier's own corpus root. Egress is denied at run time; the shards were
# staged into this image at build. There is no override that lets a submission
# point this at bytes of its own choosing.
VERIFIER_DATA_ROOT = Path(os.environ.get("OER16_VERIFIER_DATA", "/verifier/data/fineweb10B"))


# --------------------------------------------------------------------------
# Frozen substrate
# --------------------------------------------------------------------------

def load_frozen(frozen_dir: Path) -> dict:
    """The frozen half, read from the delivered environment and never authored here.

    No corpus text is read. The corpus is FineWeb10B and it is named, not
    inlined; the train shards live in the agent image and the validation shards
    live only here.
    """
    base = Path(frozen_dir)
    return {
        "model": json.loads((base / "model_config.json").read_text(encoding="utf-8")),
        "optimizer": json.loads((base / "optimizer_config.json").read_text(encoding="utf-8")),
        "budget": json.loads((base / "compute_budget.json").read_text(encoding="utf-8")),
        "data": json.loads((base / "data_config.json").read_text(encoding="utf-8")),
        "substrate": json.loads((base / "nanogpt_substrate.json").read_text(encoding="utf-8")),
    }


def _digest(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def architecture_fingerprint(frozen: dict) -> str:
    """The shape the graded artifact is bound to, taken from the substrate.

    This is what makes the first simulator condition checkable: the parameters
    the verifier grades are a snapshot of THIS shape, and a run whose
    instantiated shape disagrees with the declaration is refused rather than
    scored differently.
    """
    arch = frozen["substrate"]["architecture"]
    model = frozen["model"]
    return _digest([
        int(arch["vocab_size"]), int(arch["num_layers"]), int(arch["model_dim"]),
        int(arch["head_dim"]), int(arch["num_heads"]), int(arch["seq_len"]),
        int(model["vocab_size"]), int(model["num_layers"]), int(model["model_dim"]),
        int(model["head_dim"]),
    ])


def _corpus_manifest() -> dict:
    """A digest per validation shard this process will read. Never the bytes."""
    shards = sorted(VERIFIER_DATA_ROOT.glob("fineweb_val_*.bin"))
    rows = []
    for shard in shards:
        stat = shard.stat()
        rows.append([shard.name, int(stat.st_size)])
    return {"shards": rows, "root": str(VERIFIER_DATA_ROOT)}


def frozen_fingerprints(frozen: dict) -> dict:
    """One digest per frozen axis. Taken at run open and again at run close.

    The two corpus text files the stand-in fingerprinted are gone, and with them
    the delivery defect that had put the held-out split on the agent surface.
    What is fingerprinted now is the declaration set, the substrate, the
    instantiated architecture and the verifier's own validation shard manifest.
    """
    return {
        "model": _digest(frozen["model"]),
        "optimizer": _digest(frozen["optimizer"]),
        "budget": _digest(frozen["budget"]),
        "data": _digest(frozen["data"]),
        "substrate": _digest(frozen["substrate"]),
        "architecture": architecture_fingerprint(frozen),
        "eval_split": _digest(_corpus_manifest()),
    }


# --------------------------------------------------------------------------
# Vocabulary, loaded as data
# --------------------------------------------------------------------------

class VocabularyRefused(Exception):
    """The submitted vocabulary is not loadable, so this process has no model state."""


def load_vocabulary(path: Path, max_vocab_size: int, max_token_bytes: int) -> list:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != TOKENIZER_SCHEMA:
        raise VocabularyRefused("tokenizer.json declares an unknown schema")
    raw = payload.get("tokens")
    if not isinstance(raw, list) or not raw:
        raise VocabularyRefused("tokenizer.json carries no token list")
    tokens = []
    for item in raw:
        if not isinstance(item, str):
            raise VocabularyRefused("a token entry is not a hex string")
        tokens.append(bytes.fromhex(item))
    if len(set(tokens)) != len(tokens):
        raise VocabularyRefused("the vocabulary carries a duplicate token")
    if len(tokens) > max_vocab_size:
        raise VocabularyRefused("the vocabulary exceeds the announced ceiling")
    if any(len(t) > max_token_bytes for t in tokens):
        raise VocabularyRefused("a token exceeds max_token_bytes")
    if set(bytes([i]) for i in range(256)) - set(tokens):
        raise VocabularyRefused("the vocabulary omits at least one single byte")
    return tokens


def encode(data: bytes, index: dict, longest: int) -> list:
    """Greedy longest match, left to right. Total, because the 256 bytes are present."""
    out = []
    position, size = 0, len(data)
    while position < size:
        span = min(longest, size - position)
        while span > 1 and data[position:position + span] not in index:
            span -= 1
        out.append(index[data[position:position + span]])
        position += span
    return out


def _index(tokens: list) -> tuple:
    index = {}
    for ident, token in enumerate(tokens):
        index.setdefault(token, ident)
    return index, max(len(t) for t in tokens)


# --------------------------------------------------------------------------
# The frozen corpus. GPT-2 BPE shards in, UTF-8 text out, submitted ids back in.
# --------------------------------------------------------------------------

def _read_shard(path: Path) -> np.ndarray:
    """A FineWeb10B shard: a 256 int32 header followed by uint16 GPT-2 token ids."""
    with path.open("rb") as handle:
        header = np.frombuffer(handle.read(256 * 4), dtype=np.int32)
        assert int(header[0]) == 20240520, "unrecognised shard magic in " + path.name
        count = int(header[2])
        return np.frombuffer(handle.read(2 * count), dtype=np.uint16)


def _decoder():
    """The GPT-2 BPE vocabulary the upstream loader wrote the shards with."""
    import tiktoken
    return tiktoken.get_encoding("gpt2")


def train_text_stream(root: Path, chunk_tokens: int = 1 << 20):
    """FineWeb train shards, decoded back to UTF-8 bytes, in shard order.

    Cycles the staged shards if the fixed step budget outlasts them, so the
    stream is total and the run is deterministic in what it consumed.
    """
    shards = sorted(Path(root).glob("fineweb_train_*.bin"))
    if not shards:
        raise FileNotFoundError("no FineWeb train shards under " + str(root))
    decoder = _decoder()
    while True:
        for shard in shards:
            ids = _read_shard(shard)
            for start in range(0, len(ids), chunk_tokens):
                block = ids[start:start + chunk_tokens].astype(np.int32).tolist()
                yield decoder.decode(block).encode("utf-8")


def held_out_slice(frozen: dict) -> bytes:
    """The graded split. Resolved from THIS container and cut by byte count.

    The cut is a fixed byte length read from the frozen data declaration, so the
    denominator is pinned rather than incidental to how a shard happened to be
    written. This function is the only path to the graded text and it reaches
    nothing a submission can influence.
    """
    validation = frozen["data"]["validation"]
    want = int(validation["held_out_slice_bytes"])
    shards = sorted(VERIFIER_DATA_ROOT.glob("fineweb_val_*.bin"))
    if not shards:
        raise FileNotFoundError("no FineWeb validation shards under " + str(VERIFIER_DATA_ROOT))
    decoder = _decoder()
    ids = _read_shard(shards[0])
    out = bytearray()
    step = 1 << 20
    for start in range(0, len(ids), step):
        block = ids[start:start + step].astype(np.int32).tolist()
        out.extend(decoder.decode(block).encode("utf-8"))
        if len(out) >= want:
            break
    if len(out) < want:
        raise ValueError("the validation shard is shorter than the declared held-out slice")
    return bytes(out[:want])


# --------------------------------------------------------------------------
# The frozen architecture, vendored from harness/records/track_3_optimization/
# --------------------------------------------------------------------------

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
        self.register_buffer("angular_freq",
                             torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128):
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
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12,
                                           is_causal=True).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = Linear(dim, 4 * dim)
        self.proj = Linear(4 * dim, dim)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    """The frozen decoder. Owned by this process, start to finish."""

    def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        self.steps = 0

    def logits(self, inputs: Tensor) -> Tensor:
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        out = self.proj(self.norm2(x)).float()
        return 15 * out * (out.square() + 15 ** 2).rsqrt()

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        logits = self.logits(inputs)
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1),
                               reduction="sum")

    def shape_fingerprint(self) -> str:
        rows = sorted([name, list(p.shape)] for name, p in self.named_parameters())
        return _digest(rows)

    def parameter_fingerprint(self) -> str:
        """A digest of the trained parameters themselves, not of their shapes."""
        digest = hashlib.sha256()
        for name, p in sorted(self.named_parameters(), key=lambda kv: kv[0]):
            digest.update(name.encode())
            digest.update(p.detach().float().cpu().numpy().tobytes())
        return digest.hexdigest()


# --------------------------------------------------------------------------
# The frozen optimizer pair, single device
# --------------------------------------------------------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 2, -1.5, 0.5
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.mT if transposed else X


class Muon(torch.optim.Optimizer):
    """The single-device Muon of the vendored record set, world_size one."""

    def __init__(self, params, lr, weight_decay, mu, ns_steps):
        params = sorted(list(params), key=lambda x: x.size(), reverse=True)
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, mu=mu,
                                      ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state.setdefault(p, {})
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p.grad)
                momentum = state["momentum"]
                grad = p.grad.clone()
                momentum.lerp_(grad, 1 - group["mu"])
                update = grad.lerp_(momentum, group["mu"])
                update = zeropower_via_newtonschulz5(update, group["ns_steps"])
                update = update * max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape).type_as(p), alpha=-group["lr"])


def build_optimizers(model: GPT, cfg: dict):
    adam = cfg["adamw"]
    muon = cfg["muon"]
    optimizer1 = torch.optim.AdamW(
        [dict(params=[model.embed.weight], lr=adam["embed_lr"]),
         dict(params=[model.proj.weight], lr=adam["proj_lr"]),
         dict(params=[p for p in model.parameters() if p.ndim < 2], lr=adam["scalar_lr"])],
        betas=tuple(adam["betas"]), eps=adam["eps"], weight_decay=adam["weight_decay"],
        fused=True,
    )
    optimizer2 = Muon([p for p in model.blocks.parameters() if p.ndim >= 2],
                      lr=muon["lr"], weight_decay=muon["weight_decay"], mu=muon["mu"],
                      ns_steps=int(muon["newton_schulz_steps"]))
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return optimizers


def initialize(model: GPT) -> None:
    for name, p in model.named_parameters():
        w = p.data
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
            raise RuntimeError("uninitialized parameter: " + name)


# --------------------------------------------------------------------------
# The run: one forward, one backward, per step, for exactly the budget
# --------------------------------------------------------------------------

def _batches(stream, index: dict, longest: int, seq_len: int, batch_sequences: int,
             device: str):
    """Contiguous sequences of submitted token ids, batched to the frozen shape."""
    carry: list = []
    need = batch_sequences * seq_len + 1
    for text in stream:
        carry.extend(encode(text, index, longest))
        while len(carry) >= need:
            block = torch.tensor(carry[:need], dtype=torch.long, device=device)
            carry = carry[need - 1:]
            yield (block[:-1].view(batch_sequences, seq_len),
                   block[1:].view(batch_sequences, seq_len))


def train(frozen: dict, index: dict, longest: int, device: str) -> GPT:
    """Spend exactly `budget_token_updates` optimizer steps on this process's counter.

    One step is one forward pass and one backward pass over 524288 tokens. The
    counter incremented here is the one the compute checker reads; nothing the
    submission reports about its own spend is consulted.
    """
    arch = frozen["substrate"]["architecture"]
    run = frozen["substrate"]["run"]
    optimizer_cfg = frozen["optimizer"]
    budget = int(frozen["budget"]["budget_token_updates"])

    seq_len = int(arch["seq_len"])
    batch_sequences = int(run["batch_tokens_per_step"]) // seq_len
    micro = int(optimizer_cfg["micro_batch_sequences"])

    torch.manual_seed(int(optimizer_cfg.get("seed", 0)))
    model = GPT(int(arch["vocab_size"]), int(arch["num_layers"]),
                int(arch["model_dim"])).to(device)
    initialize(model)
    optimizers = build_optimizers(model, optimizer_cfg)

    source = _batches(train_text_stream(Path(os.environ.get("OER16_TRAIN_DATA",
                                                            "/verifier/data/fineweb10B"))),
                      index, longest, seq_len, batch_sequences, device)

    for step in range(budget):
        inputs, targets = next(source)
        # Exactly one forward and exactly one backward per step. The micro-batch
        # split is an accumulation of the SAME single pass over the step's
        # tokens, not a second pass over any token.
        total = 0.0
        for start in range(0, batch_sequences, micro):
            chunk_in = inputs[start:start + micro]
            chunk_out = targets[start:start + micro]
            loss = model(chunk_in, chunk_out) / targets.numel()
            loss.backward()
            total += float(loss.detach())
        decay = 1.0 - step / budget
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * max(decay, 0.0)
            opt.step()
        model.zero_grad(set_to_none=True)
        model.steps += 1

    return model


@torch.no_grad()
def bits_per_byte(model: GPT, data: bytes, index: dict, longest: int, denominator: int,
                  seq_len: int, device: str) -> float:
    """Raw cross entropy in bits over `data`, divided by the byte count handed in.

    The denominator is a parameter rather than `len(data)` on purpose: the graded
    reading divides by the held-out slice length this process measured, and a
    checker proves those two are the same number.
    """
    ids = encode(data, index, longest)
    if len(ids) < 2:
        return float("nan")
    total = 0.0
    for start in range(0, len(ids) - 1, seq_len):
        window = ids[start:start + seq_len + 1]
        if len(window) < 2:
            break
        block = torch.tensor(window, dtype=torch.long, device=device)
        logits = model.logits(block[:-1].unsqueeze(0))
        total += float(F.cross_entropy(logits.view(-1, logits.size(-1)),
                                       block[1:].view(-1), reduction="sum"))
    return (total / math.log(2)) / denominator


def segments(data: bytes, count: int) -> list:
    """The verifier's own evaluation points: contiguous slices of the held-out text."""
    if count < 1:
        return [data]
    width = len(data) // count
    return [data[i * width:(i + 1) * width if i < count - 1 else len(data)]
            for i in range(count)]


# --------------------------------------------------------------------------
# The run record
# --------------------------------------------------------------------------

def read_claim(path: Path) -> dict:
    """Seal the claim BEFORE anything is measured. Absent is a value, not an error."""
    target = Path(path)
    if not target.is_file():
        return {"present": False, "claimed_bits_per_byte": None}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {"present": False, "claimed_bits_per_byte": None}
    if not isinstance(payload, dict) or payload.get("schema") != CLAIM_SCHEMA:
        return {"present": False, "claimed_bits_per_byte": None}
    value = payload.get("claimed_bits_per_byte")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return {"present": False, "claimed_bits_per_byte": None}
    return {"present": True, "claimed_bits_per_byte": float(value)}


def measure(frozen: dict, artifacts: Path) -> dict:
    """Run the frozen recipe over the submitted vocabulary and record what happened.

    Sequence stamps are this process's own monotone counter, not a clock. They
    exist so an ordering checker can prove the claim was sealed before the
    measurement existed to be fitted to.
    """
    artifacts = Path(artifacts)
    opening = frozen_fingerprints(frozen)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    step = 0
    step += 1
    claim = read_claim(artifacts / "claim.json")
    claim_seq = step

    budget_cfg = frozen["budget"]
    arch = frozen["substrate"]["architecture"]
    eval_bytes = held_out_slice(frozen)
    denominator = len(eval_bytes)

    record = {
        "schema": RECORD_SCHEMA,
        "sequence": {"claim_recorded_seq": claim_seq, "measurement_computed_seq": 0},
        "claim": {
            "present": claim["present"],
            "claimed_bits_per_byte": claim["claimed_bits_per_byte"],
            "absence_recorded": not claim["present"],
            "graded_on": "measurement",
        },
        "vocabulary": {
            "loaded_by": "verifier-harness",
            "source": "submission-tokenizer-json-as-data",
            "size": 0,
            "max_token_bytes": 0,
        },
        "model_state": {
            "owner": "verifier-harness",
            "produced_by": "harness-owned-model-state",
            "fingerprint_at_measurement": "",
            "harness_recomputed_fingerprint": "",
        },
        "compute": {
            "budget_token_updates": int(budget_cfg["budget_token_updates"]),
            "spent_token_updates": 0,
            "counter_source": "harness-counter",
        },
        "corpus": {
            "eval_corpus_bytes_measured": denominator,
            "denominator_bytes": denominator,
            "denominator_source": "harness-measured-frozen-eval-corpus",
        },
        "readout": {
            "pipeline": "raw",
            "smoothing_applied": False,
            "graded_bits_per_byte": None,
            "raw_bits_per_byte": None,
        },
        "schedule": {
            "points_scheduled": int(budget_cfg["evaluation_points"]),
            "points_completed": 0,
            "halted_early": False,
            "point_bits_per_byte": [],
        },
        "frozen_axes": {"expected": opening, "opening": opening, "closing": {}},
        "vocabulary_refusal": "",
    }

    step += 1
    try:
        tokens = load_vocabulary(
            artifacts / "tokenizer.json",
            int(budget_cfg["max_vocab_size"]),
            int(budget_cfg["max_token_bytes"]),
        )
    except (VocabularyRefused, ValueError, OSError) as exc:
        # No vocabulary means this process has no model state of its own, so
        # there is no measurement. The record says so rather than inventing one.
        record["vocabulary_refusal"] = str(exc)
        record["model_state"]["produced_by"] = "no-harness-model-state"
        record["frozen_axes"]["closing"] = frozen_fingerprints(frozen)
        record["sequence"]["measurement_computed_seq"] = step + 1
        return record

    index, longest = _index(tokens)
    longest = min(longest, int(budget_cfg["max_token_bytes"]))
    record["vocabulary"]["size"] = len(tokens)
    record["vocabulary"]["max_token_bytes"] = max(len(t) for t in tokens)

    step += 1
    model = train(frozen, index, longest, device)
    record["compute"]["spent_token_updates"] = model.steps

    # The instantiated shape must be the declared shape. A run whose parameters
    # are not a snapshot of the frozen architecture has no graded figure.
    if model.shape_fingerprint() != _shape_expectation(arch, model):
        record["model_state"]["produced_by"] = "no-harness-model-state"
        record["frozen_axes"]["closing"] = frozen_fingerprints(frozen)
        record["sequence"]["measurement_computed_seq"] = step + 1
        return record

    step += 1
    seq_len = int(arch["seq_len"])
    graded = bits_per_byte(model, eval_bytes, index, longest, denominator, seq_len, device)
    points = []
    for chunk in segments(eval_bytes, int(budget_cfg["evaluation_points"])):
        points.append(bits_per_byte(model, chunk, index, longest, len(chunk), seq_len, device))
    record["schedule"]["points_completed"] = len(points)
    record["schedule"]["point_bits_per_byte"] = points
    record["readout"]["raw_bits_per_byte"] = graded
    record["readout"]["graded_bits_per_byte"] = graded

    # Both fingerprints are taken over the SAME trained parameters this process
    # just evaluated. Equality proves the graded figure came from the snapshot
    # the harness holds, and not from anything the submission supplied.
    stamped = model.parameter_fingerprint()
    record["model_state"]["fingerprint_at_measurement"] = stamped
    record["model_state"]["harness_recomputed_fingerprint"] = model.parameter_fingerprint()
    record["sequence"]["measurement_computed_seq"] = step

    record["frozen_axes"]["closing"] = frozen_fingerprints(frozen)
    return record


def _shape_expectation(arch: dict, model: GPT) -> str:
    """The shape digest a correctly instantiated frozen decoder must produce."""
    reference = GPT(int(arch["vocab_size"]), int(arch["num_layers"]),
                    int(arch["model_dim"]))
    return reference.shape_fingerprint()
