#!/usr/bin/env python3
"""The frozen harness: encoder, decoder, optimizer, step counter, evaluator.

Everything in this file is a FROZEN axis. A submission never edits it, and the
verifier runs its own copy from the delivery unit rather than the one a run may
have left in a workspace. The only free axis is the *vocabulary construction*,
which reaches this harness as a list of byte strings.

The model here is the canonical nanoGPT decoder declared in
`environment/nanogpt_substrate.json`: vocab_size 50304, num_layers 12,
model_dim 768, head_dim 128, num_heads 6, seq_len 1024, trained on FineWeb10B
shards at 524288 tokens per step with exactly one forward and one backward pass
per step. Nothing in this file stands in for that decoder. There is no count
table, no closed-form cost model and no tick clock: the graded quantity is read
off real parameters that a real gradient step moved.

Four properties this file exists to hold, all of them readable here rather than
asserted elsewhere.

1. The denominator of bits per byte is the byte count of the held-out slice the
   verifier hands in, and never a token count. A vocabulary that shrinks the
   token count therefore shrinks nothing on the denominator. Token counts are
   recorded for information and are never divided by.

2. Compute is counted as OPTIMIZER STEPS the training loop actually consumed, by
   a counter this file owns. One step is one gradient computation over 524288
   tokens. A submission may request a halt; the halt is recorded and the run is
   marked as not having spent the budget. Nothing a submission prints or writes
   reaches the counter.

3. Evaluation reads the parameters this harness holds at the scheduled step
   count. There is no path by which a submission hands back a state, a
   checkpoint or a number. The graded reading is raw: no average, no EMA, no
   filter of any kind is applied to the bits this file computes.

4. The evaluation text is NOT resolved from anywhere inside this container. It
   is passed in by the caller. Called without it, `run` trains and returns
   telemetry whose graded reading is absent, because the split that produces the
   grade is held by the verifier and the solver cannot see it.

No clock, no network and no locale-dependent operation is used anywhere below.
The one entropy source is the initialisation seed, which is a frozen manifest
field, so two runs over the same bytes on the same device produce the same
telemetry.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = Path(__file__).resolve().parent
MANIFEST_PATH = BASE / "manifest.json"
SUBSTRATE_PATH = BASE / "nanogpt_substrate.json"

# Encoder constant. Frozen: it is part of "the encoder".
MAX_TOKEN_LEN = 16

# The fixed context token every evaluation sequence opens with, so that every
# token drawn from the held-out slice is predicted from something and the bits
# cover the whole slice. Byte 0x00 is entry 0 of every normalised vocabulary.
PREFIX_ID = 0


def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def substrate() -> dict:
    return json.loads(SUBSTRATE_PATH.read_text(encoding="utf-8"))


def substrate_digest() -> str:
    """A digest over the canonical substrate declaration exactly as delivered."""
    return hashlib.sha256(SUBSTRATE_PATH.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# The vocabulary and the encoder. Both frozen.
# ---------------------------------------------------------------------------
def normalise_vocabulary(entries, budget: int) -> list:
    """The 256 single bytes first, then the submission's entries, deduped in order.

    Single bytes are not optional: without them the encoder would not be total
    over arbitrary input, and a partial encoder makes bits per byte undefined on
    the bytes it cannot express. They occupy budget like anything else, and the
    budget is the decoder's own vocab_size because the embedding table has
    exactly that many rows.
    """
    vocab = [bytes([i]) for i in range(256)]
    seen = set(vocab)
    for raw in entries or []:
        if isinstance(raw, str):
            item = raw.encode("utf-8")
        else:
            item = bytes(raw)
        if len(item) < 2 or len(item) > MAX_TOKEN_LEN or item in seen:
            continue
        seen.add(item)
        vocab.append(item)
        if len(vocab) >= budget:
            break
    return vocab


def build_index(vocab) -> tuple:
    index = {}
    longest = 1
    for ident, token in enumerate(vocab):
        index[token] = ident
        longest = max(longest, len(token))
    return index, longest


def encode(vocab, data: bytes) -> list:
    """Greedy longest match. The encoder is frozen.

    A submission that wanted a cleverer segmentation gets none: the free axis is
    which strings are in the vocabulary, not how the vocabulary is applied.
    """
    index, longest = build_index(vocab)
    out = []
    position = 0
    size = len(data)
    while position < size:
        span = min(longest, size - position)
        while span > 1:
            ident = index.get(data[position:position + span])
            if ident is not None:
                out.append(ident)
                position += span
                break
            span -= 1
        else:
            out.append(index[data[position:position + 1]])
            position += 1
    return out


# ---------------------------------------------------------------------------
# The corpus. FineWeb10B shards, decoded back to bytes, then re-encoded under the
# submission's vocabulary. Decoding is exact: the shards are GPT-2 token ids.
# ---------------------------------------------------------------------------
def shard_root() -> Path:
    spec = manifest()["corpus"]
    return Path(os.environ.get("OER15_FINEWEB_ROOT", spec["container_train_path"]))


def train_shards() -> list:
    rows = sorted(shard_root().glob("fineweb_train_*.bin"))
    if not rows:
        raise SystemExit(
            "no FineWeb10B training shards under " + str(shard_root())
            + "; run environment/build_corpus.py to stage them"
        )
    return rows


def read_shard(path: Path):
    """The upstream .bin layout: a 1024-int32 header, then uint16 GPT-2 ids."""
    import numpy as np

    with path.open("rb") as handle:
        header = np.frombuffer(handle.read(256 * 4), dtype=np.int32)
        if int(header[0]) != 20240520:
            raise SystemExit("shard " + path.name + " does not carry the upstream magic")
        count = int(header[2])
        handle.seek(1024)
        return np.frombuffer(handle.read(count * 2), dtype=np.uint16)


_DECODER = {}


def gpt2_decoder():
    """The GPT-2 byte-pair decoder, used only to recover the original bytes."""
    if "codec" not in _DECODER:
        import tiktoken

        _DECODER["codec"] = tiktoken.get_encoding("gpt2")
    return _DECODER["codec"]


def shard_bytes(path: Path, limit_tokens: int = 0) -> bytes:
    ids = read_shard(path)
    if limit_tokens:
        ids = ids[:limit_tokens]
    return gpt2_decoder().decode_bytes([int(value) for value in ids])


def fit_bytes() -> bytes:
    """The bytes handed to a submission's build_vocab. One step of tokens, decoded."""
    spec = manifest()["tokenizer"]
    return shard_bytes(train_shards()[0], int(spec["fit_tokens"]))


# ---------------------------------------------------------------------------
# The decoder. Shapes are read from the substrate declaration and never authored
# here, so a shape that drifts from the canonical operating point is a refused
# run rather than a different grade.
# ---------------------------------------------------------------------------
def rmsnorm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class Rotary(nn.Module):
    def __init__(self, head_dim: int, seq_len: int):
        super().__init__()
        angular = (1.0 / 1024.0) ** torch.linspace(0.0, 1.0, head_dim // 4)
        frequency = torch.cat([angular, angular.new_zeros(head_dim // 4)])
        position = torch.arange(seq_len, dtype=torch.float32)
        theta = torch.outer(position, frequency)
        self.register_buffer("cos", theta.cos(), persistent=False)
        self.register_buffer("sin", theta.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cos = self.cos[None, : x.size(-3), None, :]
        sin = self.sin[None, : x.size(-3), None, :]
        left, right = x.float().chunk(2, dim=-1)
        return torch.cat(
            [left * cos + right * sin, right * cos - left * sin], dim=-1
        ).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, head_dim: int, seq_len: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        width = num_heads * head_dim
        self.qkv = nn.Linear(model_dim, 3 * width, bias=False)
        self.rotary = Rotary(head_dim, seq_len)
        self.proj = nn.Linear(width, model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = self.rotary(rmsnorm(q)), self.rotary(rmsnorm(k))
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        return self.proj(y.transpose(1, 2).reshape(batch, length, -1))


class MLP(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        self.up = nn.Linear(model_dim, 4 * model_dim, bias=False)
        self.down = nn.Linear(4 * model_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.relu(self.up(x)).square())


class Block(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, head_dim: int, seq_len: int):
        super().__init__()
        self.attention = CausalSelfAttention(model_dim, num_heads, head_dim, seq_len)
        self.mlp = MLP(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(rmsnorm(x))
        return x + self.mlp(rmsnorm(x))


class GPT(nn.Module):
    """The canonical 12-layer 768-dim decoder of the frozen substrate."""

    def __init__(self, architecture: dict):
        super().__init__()
        self.architecture = dict(architecture)
        vocab = int(architecture["vocab_size"])
        model_dim = int(architecture["model_dim"])
        self.embed = nn.Embedding(vocab, model_dim)
        self.blocks = nn.ModuleList(
            Block(
                model_dim,
                int(architecture["num_heads"]),
                int(architecture["head_dim"]),
                int(architecture["seq_len"]),
            )
            for _ in range(int(architecture["num_layers"]))
        )
        self.head = nn.Linear(model_dim, vocab, bias=False)
        self.head.weight.detach().zero_()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = rmsnorm(self.embed(idx))
        for block in self.blocks:
            x = block(x)
        return self.head(rmsnorm(x))

    def parameter_shapes(self) -> dict:
        return {name: list(tensor.shape) for name, tensor in sorted(self.named_parameters())}

    def parameter_count(self) -> int:
        return int(sum(tensor.numel() for tensor in self.parameters()))

    def snapshot_digest(self) -> str:
        """A digest over the live parameter bytes, in a fixed name order.

        This is a digest of WEIGHTS, not of a summary of weights. Two runs whose
        gradients differ anywhere produce different digests here, so a checker
        can prove the graded reading came off parameters that a gradient moved.
        """
        digest = hashlib.sha256()
        for name, tensor in sorted(self.named_parameters()):
            digest.update(name.encode("ascii"))
            digest.update(tensor.detach().to(torch.float32).cpu().numpy().tobytes())
        return digest.hexdigest()


def architecture_digest(architecture: dict) -> str:
    payload = json.dumps(
        {key: architecture[key] for key in sorted(architecture) if not key.startswith("_")},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# The training loop and the evaluator.
# ---------------------------------------------------------------------------
def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _schedule(spec: dict, budget: int) -> list:
    points = []
    for fraction in spec["progress_fractions"]:
        steps = int(round(budget * fraction))
        points.append({"steps": steps, "role": "progress", "charged_to_budget": True})
    points.append({"steps": budget, "role": "bound", "charged_to_budget": True})
    for extra in spec["sustain_offsets"]:
        points.append(
            {"steps": budget + int(extra), "role": "sustain", "charged_to_budget": False}
        )
    ordered, seen = [], set()
    for row in sorted(points, key=lambda item: (item["steps"], item["role"])):
        if row["steps"] in seen:
            continue
        seen.add(row["steps"])
        ordered.append(row)
    return ordered


class TokenStream:
    """The training token stream: shards decoded to bytes, re-encoded under the vocabulary.

    Shards are consumed one at a time and the encoded ids of the shard in hand
    are held in memory, so the stream costs one shard of working set no matter
    how many steps are taken. A vocabulary that expresses the same bytes in
    fewer tokens therefore covers MORE corpus bytes inside the same step budget,
    which is the effect the objective rewards.
    """

    def __init__(self, vocab, shards):
        self.vocab = vocab
        self.shards = list(shards)
        self.cursor = 0
        self.buffer = []
        self.offset = 0
        self.bytes_consumed = 0
        self.tokens_consumed = 0

    def _refill(self) -> None:
        path = self.shards[self.cursor % len(self.shards)]
        self.cursor += 1
        data = shard_bytes(path)
        self.bytes_consumed += len(data)
        self.buffer = encode(self.vocab, data)
        self.offset = 0

    def take(self, count: int) -> list:
        out = []
        while len(out) < count:
            if self.offset >= len(self.buffer):
                self._refill()
            span = min(count - len(out), len(self.buffer) - self.offset)
            out.extend(self.buffer[self.offset:self.offset + span])
            self.offset += span
        self.tokens_consumed += count
        return out


def _optimizer(model: nn.Module, spec: dict):
    decay, plain = [], []
    for _, tensor in sorted(model.named_parameters()):
        (decay if tensor.dim() >= 2 else plain).append(tensor)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": plain, "weight_decay": 0.0},
        ],
        lr=0.0018,
        betas=(0.9, 0.95),
        fused=torch.cuda.is_available(),
    )


def _learning_rate(step: int, budget: int) -> float:
    warmup = 128
    if step < warmup:
        return (step + 1) / warmup
    remaining = max(0, budget - step)
    return max(0.0, remaining / max(1, budget - warmup))


@torch.no_grad()
def bits_of(model: GPT, vocab, data: bytes, seq_len: int, device) -> dict:
    """Raw bits the decoder assigns to one byte string. No filter, no clipping.

    Every token drawn from the slice is predicted from a context, because the
    sequence opens with a fixed prefix token, so the bits cover the whole slice
    and the denominator can stay the slice's byte count.
    """
    tokens = [PREFIX_ID] + encode(vocab, data)
    total = 0.0
    scored = 0
    stride = seq_len // 2
    was_training = model.training
    model.eval()
    start = 0
    already = 1
    while already < len(tokens):
        window = tokens[start:start + seq_len]
        if len(window) < 2:
            break
        idx = torch.tensor(window, dtype=torch.long, device=device)[None, :]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(idx)
        logprobs = F.log_softmax(logits.float(), dim=-1)[0, :-1, :]
        targets = idx[0, 1:]
        first = already - start - 1
        if first < 0:
            first = 0
        picked = logprobs[first:, :].gather(1, targets[first:, None]).squeeze(1)
        total += float(-picked.sum().item())
        scored += int(picked.numel())
        already = start + len(window)
        if already >= len(tokens):
            break
        start += stride
    model.train(was_training)
    return {
        "bits": total / math.log(2.0),
        "tokens": len(tokens) - 1,
        "scored_tokens": scored,
        "denominator_bytes": len(data),
        "bits_per_byte": (total / math.log(2.0)) / len(data),
    }


def run(entries, eval_bytes=None, eval_source="", halt_at=None, reported=None,
        snapshot_path=None) -> dict:
    """Train the canonical decoder under the frozen step counter, then evaluate.

    `eval_bytes` is the held-out slice. It is passed IN. This file resolves no
    evaluation split of its own, because the split that produces the grade is
    the verifier's and is absent from every agent-visible container. Called
    without it, this returns telemetry whose graded reading is absent.

    `halt_at` is the submission's request to stop early. It is honoured, because
    refusing it would hide the behaviour instead of grading it, and it is
    recorded so a run that took it is graded as not having established the
    metric.

    `reported` is whatever the submission said about itself. It is carried into
    the telemetry untouched and is never read by anything that computes a graded
    number; it exists so a checker can prove the graded number diverges from it.
    """
    spec = manifest()
    canonical = substrate()
    architecture = dict(spec["architecture"])
    if architecture != {
        key: value for key, value in canonical["architecture"].items() if not key.startswith("_")
    }:
        raise SystemExit("the manifest architecture does not match nanogpt_substrate.json")

    budget = int(spec["run"]["compute_budget_steps"])
    seq_len = int(architecture["seq_len"])
    per_step = int(spec["run"]["batch_tokens_per_step"])
    sequences = per_step // seq_len
    micro = int(os.environ.get("OER15_MICRO_BATCH", "8"))
    device = _device()

    torch.manual_seed(int(spec["run"]["init_seed"]))
    submitted = submission_digest(entries)
    vocab = normalise_vocabulary(entries, int(spec["tokenizer"]["vocab_budget"]))
    vocab_digest = hashlib.sha256(
        json.dumps([token.hex() for token in vocab], separators=(",", ":")).encode("ascii")
    ).hexdigest()

    model = GPT(architecture).to(device)
    optimizer = _optimizer(model, spec)
    stream = TokenStream(vocab, train_shards())

    points = _schedule(spec["evaluation_schedule"], budget)
    ceiling = points[-1]["steps"]
    stop_at = ceiling if halt_at is None else min(int(halt_at), ceiling)

    records = []
    consumed = 0
    charged = 0
    forward_passes = 0
    backward_passes = 0
    halted = False
    for row in points:
        target = row["steps"]
        while consumed < target and consumed < stop_at:
            for group in optimizer.param_groups:
                group["lr"] = 0.0018 * _learning_rate(consumed, budget)
            optimizer.zero_grad(set_to_none=True)
            for _ in range(sequences // micro):
                chunk = stream.take(micro * seq_len + 1)
                idx = torch.tensor(chunk[:-1], dtype=torch.long, device=device).view(micro, seq_len)
                targets = torch.tensor(chunk[1:], dtype=torch.long, device=device).view(micro, seq_len)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(idx)
                loss = F.cross_entropy(
                    logits.float().view(-1, logits.size(-1)), targets.reshape(-1)
                ) / (sequences // micro)
                loss.backward()
            optimizer.step()
            # ONE gradient per step. The accumulation micro-batches above compute
            # exactly one gradient between two optimizer steps, which is what the
            # substrate's one-forward-one-backward-pass-per-step rule fixes, and
            # the micro-batch count is recorded rather than folded away.
            forward_passes += 1
            backward_passes += 1
            consumed += 1
            if row["charged_to_budget"] or consumed <= budget:
                charged = min(consumed, budget)
        if consumed < target:
            halted = True
            break
        if eval_bytes is None:
            continue
        reading = bits_of(model, vocab, eval_bytes, seq_len, device)
        records.append(
            {
                "steps": target,
                "role": row["role"],
                "charged_to_budget": bool(row["charged_to_budget"]),
                "bits": reading["bits"],
                "denominator_bytes": reading["denominator_bytes"],
                "bits_per_byte": reading["bits_per_byte"],
                "eval_tokens": reading["tokens"],
                "snapshot_digest": model.snapshot_digest(),
            }
        )

    if snapshot_path is not None:
        Path(snapshot_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), snapshot_path)

    bound = next((row for row in records if row["role"] == "bound"), None)
    return {
        "schema": "oer15.telemetry/v2",
        "frozen": {
            "compute_budget_steps": budget,
            "architecture": architecture,
            "architecture_digest": architecture_digest(architecture),
            "substrate_digest": substrate_digest(),
            "batch_tokens_per_step": per_step,
            "forward_passes_per_step": 1,
            "backward_passes_per_step": 1,
            "vocab_budget": int(spec["tokenizer"]["vocab_budget"]),
            "train_shards": [path.name for path in train_shards()],
        },
        "model": {
            "parameter_count": model.parameter_count(),
            "parameter_shapes": model.parameter_shapes(),
            "snapshot_digest": model.snapshot_digest(),
            "initialised_from_seed": int(spec["run"]["init_seed"]),
        },
        "vocabulary": {
            "size": len(vocab),
            "digest": vocab_digest,
            "submitted_digest": submitted,
            "train_bytes_covered": stream.bytes_consumed,
            "train_tokens_consumed": stream.tokens_consumed,
            "eval_tokens": None if bound is None else bound["eval_tokens"],
        },
        "compute": {
            "budget_steps": budget,
            "charged_steps": charged,
            "consumed_steps": consumed,
            "micro_batches_per_step": sequences // micro,
            "forward_passes": forward_passes,
            "backward_passes": backward_passes,
            "halt_requested_at": None if halt_at is None else int(halt_at),
            "halted_before_bound_point": bool(halted or bound is None),
        },
        "denominator": {
            "bytes": None if bound is None else bound["denominator_bytes"],
            "source": "held-out-fineweb-slice-bytes",
            "slice_source": str(eval_source or ""),
            "digest": "" if eval_bytes is None else hashlib.sha256(eval_bytes).hexdigest(),
        },
        "readout": {
            "filter": "none",
            "weights_origin": "harness-owned-parameters-at-bound-step",
            "graded_role": "bound" if bound is not None else "absent",
            "eval_split_resolved_by": "caller" if eval_bytes is not None else "nobody",
        },
        "eval_points": records,
        "graded": None
        if bound is None
        else {
            "steps": bound["steps"],
            "bits": bound["bits"],
            "denominator_bytes": bound["denominator_bytes"],
            "bits_per_byte": bound["bits_per_byte"],
            "snapshot_digest": bound["snapshot_digest"],
        },
        "reported_by_submission": dict(reported or {}),
    }


def submission_digest(entries) -> str:
    """A digest over exactly what the submission returned, before normalisation.

    Kept separate from the normalised vocabulary digest so a checker can prove
    the graded run used the list the submission handed back, rather than proving
    only that the harness digested its own output.
    """
    rows = []
    for raw in entries or []:
        item = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
        rows.append(item.hex())
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode("ascii")
    ).hexdigest()
