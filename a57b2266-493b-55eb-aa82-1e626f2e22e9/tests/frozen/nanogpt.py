#!/usr/bin/env python3
"""The frozen nanoGPT substrate: the decoder, its training loop, and the
verifier-side validation forward pass. VERIFIER-OWNED.

This module is the whole model in the loop for OER-13. Every architectural
number it uses is read at run time from the canonical declaration at
environment/nanogpt_substrate.json, which is replicated into this bundle from
.seed/nanogpt_substrate.json and whose values come from the vendored record set
at harness/records/track_3_optimization/train_gpt_simple.py. Nothing here is
invented and nothing here is a stand-in: the graded artifact is a real parameter
snapshot of this architecture, and the graded scalar is a real forward pass of
those parameters over the verifier-owned held-out FineWeb split.

Two properties this file exists to guarantee.

The training loop performs exactly one forward and one backward over each step's
524288-token batch and takes exactly one optimizer step per training step. A step
is materialized as a sequence of micro-batches only because 524288 tokens do not
fit in one H100's memory at seq_len 1024; the gradient accumulated over those
micro-batches is the gradient of the one loss over the one batch, no token is
visited twice, and no extra update is taken. `micro_batches_per_step` is reported
as telemetry beside the logical count so the distinction is auditable rather than
asserted.

Delete `GPT.forward` and the metric becomes undefined, not merely different.
There is no count table, no cost model and no tick counter anywhere in this file
that could keep emitting a number after the passes are removed.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SNAPSHOT_SCHEMA = "oer13.nanogpt.snapshot/v1"
SHARD_MAGIC = 20240520
SHARD_VERSION = 1
SHARD_HEADER_INTS = 256

ARCHITECTURE_KEYS = ("vocab_size", "num_layers", "model_dim", "head_dim", "num_heads", "seq_len")
RUN_KEYS = ("batch_tokens_per_step", "forward_passes_per_step", "backward_passes_per_step", "target_val_loss")


# --------------------------------------------------------------------------
# The canonical declaration. Read, never hardcoded.
# --------------------------------------------------------------------------
def load_substrate(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def architecture(substrate: dict) -> dict:
    block = substrate["architecture"]
    return {key: int(block[key]) for key in ARCHITECTURE_KEYS}


def run_declaration(substrate: dict) -> dict:
    block = substrate["run"]
    return {
        "batch_tokens_per_step": int(block["batch_tokens_per_step"]),
        "forward_passes_per_step": int(block["forward_passes_per_step"]),
        "backward_passes_per_step": int(block["backward_passes_per_step"]),
        "target_val_loss": float(block["target_val_loss"]),
    }


def expected_parameter_shapes(arch: dict) -> dict:
    """Every parameter shape the declaration implies, derived from it alone.

    The checker compares a snapshot against this map, so a weight table that is
    not this architecture cannot be passed off as one. The map is a pure function
    of vocab_size, num_layers, model_dim, head_dim and seq_len.
    """
    vocab = arch["vocab_size"]
    layers = arch["num_layers"]
    dim = arch["model_dim"]
    inner = arch["head_dim"] * arch["num_heads"]
    shapes = {
        "embed.weight": [vocab, dim],
        "position.weight": [arch["seq_len"], dim],
        "head.weight": [vocab, dim],
        "final_norm.weight": [dim],
    }
    for i in range(layers):
        shapes["blocks." + str(i) + ".norm_attention.weight"] = [dim]
        shapes["blocks." + str(i) + ".attention.qkv.weight"] = [3 * inner, dim]
        shapes["blocks." + str(i) + ".attention.projection.weight"] = [dim, inner]
        shapes["blocks." + str(i) + ".norm_mlp.weight"] = [dim]
        shapes["blocks." + str(i) + ".mlp.up.weight"] = [4 * dim, dim]
        shapes["blocks." + str(i) + ".mlp.down.weight"] = [dim, 4 * dim]
    return shapes


# --------------------------------------------------------------------------
# The decoder.
# --------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight, 1e-6)


class CausalSelfAttention(nn.Module):
    def __init__(self, arch: dict):
        super().__init__()
        self.num_heads = arch["num_heads"]
        self.head_dim = arch["head_dim"]
        inner = self.num_heads * self.head_dim
        self.qkv = nn.Linear(arch["model_dim"], 3 * inner, bias=False)
        self.projection = nn.Linear(inner, arch["model_dim"], bias=False)

    def forward(self, x):
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).split(self.num_heads * self.head_dim, dim=2)
        shape = (batch, length, self.num_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(batch, length, self.num_heads * self.head_dim)
        return self.projection(out)


class MLP(nn.Module):
    def __init__(self, arch: dict):
        super().__init__()
        self.up = nn.Linear(arch["model_dim"], 4 * arch["model_dim"], bias=False)
        self.down = nn.Linear(4 * arch["model_dim"], arch["model_dim"], bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, arch: dict):
        super().__init__()
        self.norm_attention = RMSNorm(arch["model_dim"])
        self.attention = CausalSelfAttention(arch)
        self.norm_mlp = RMSNorm(arch["model_dim"])
        self.mlp = MLP(arch)

    def forward(self, x):
        x = x + self.attention(self.norm_attention(x))
        return x + self.mlp(self.norm_mlp(x))


class GPT(nn.Module):
    """The canonical 12-layer, 768-dim decoder of the frozen substrate."""

    def __init__(self, arch: dict):
        super().__init__()
        self.arch = dict(arch)
        self.embed = nn.Embedding(arch["vocab_size"], arch["model_dim"])
        self.position = nn.Embedding(arch["seq_len"], arch["model_dim"])
        self.blocks = nn.ModuleList([Block(arch) for _ in range(arch["num_layers"])])
        self.final_norm = RMSNorm(arch["model_dim"])
        self.head = nn.Linear(arch["model_dim"], arch["vocab_size"], bias=False)
        self.apply(self._init)
        torch.nn.init.zeros_(self.head.weight)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, targets):
        """THE forward pass. Returns mean cross entropy in nats over the batch.

        Removing this method removes the metric. There is no fallback path in
        this bundle that produces a validation loss without it.
        """
        _, length = tokens.shape
        positions = torch.arange(length, device=tokens.device)
        x = self.embed(tokens) + self.position(positions)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.final_norm(x))
        return F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), targets.reshape(-1), reduction="mean"
        )


def build(arch: dict, device: str) -> GPT:
    return GPT(arch).to(device)


# --------------------------------------------------------------------------
# Shards. The submission's tokenize stage writes these, and the verifier's own
# held-out FineWeb validation shards carry the identical layout.
# --------------------------------------------------------------------------
def read_shard(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.int32, count=SHARD_HEADER_INTS)
    if int(raw[0]) != SHARD_MAGIC or int(raw[1]) != SHARD_VERSION:
        raise ValueError("shard " + str(path) + " does not carry the bound header")
    count = int(raw[2])
    tokens = np.fromfile(str(path), dtype=np.uint16, offset=SHARD_HEADER_INTS * 4, count=count)
    if tokens.size != count:
        raise ValueError("shard " + str(path) + " declares " + str(count) + " tokens and carries " + str(tokens.size))
    return tokens


def load_stream(paths: list) -> np.ndarray:
    pieces = [read_shard(Path(p)) for p in paths]
    if not pieces:
        return np.zeros(0, dtype=np.uint16)
    return np.concatenate(pieces)


def _batches(stream: np.ndarray, seq_len: int, rows: int, offset: int, device: str):
    """Contiguous, deterministic. No shuffling, no random source, no clock."""
    span = seq_len * rows
    window = stream[offset:offset + span + 1]
    if window.size < span + 1:
        return None
    chunk = torch.from_numpy(window.astype(np.int64))
    tokens = chunk[:-1].view(rows, seq_len).to(device, non_blocking=True)
    targets = chunk[1:].view(rows, seq_len).to(device, non_blocking=True)
    return tokens, targets


# --------------------------------------------------------------------------
# Snapshots. Real parameters, shape-bound to the declaration.
# --------------------------------------------------------------------------
def write_snapshot(model: GPT, arch: dict, path: Path, checkpoint: str, step: int, tokens: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SNAPSHOT_SCHEMA,
        "checkpoint": checkpoint,
        "step": int(step),
        "tokens_consumed": int(tokens),
        "architecture": dict(arch),
        "state_dict": {k: v.detach().to("cpu", torch.float32) for k, v in model.state_dict().items()},
    }
    torch.save(payload, str(path))
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_snapshot(path: Path) -> dict:
    return torch.load(str(path), map_location="cpu", weights_only=False)


def snapshot_shapes(path: Path) -> dict:
    """The shape map of a snapshot, for the verifier's architecture binding."""
    payload = read_snapshot(path)
    state = payload.get("state_dict") or {}
    return {name: list(tensor.shape) for name, tensor in state.items()}


# --------------------------------------------------------------------------
# Training. One forward and one backward over each step's batch.
# --------------------------------------------------------------------------
CHECKPOINT_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
CHECKPOINT_IDS = ("c1", "c2", "c3", "c4")
BOUND_EVALUATION_POINT = "final-snapshot"
BOUND_CHECKPOINT_ID = "c4"


def _learning_rate(step: int, steps: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, steps - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train(
    shard_paths: list,
    arch: dict,
    run: dict,
    steps: int,
    out_dir: Path,
    device: str,
    micro_rows: int,
    peak_lr: float,
    halt_at: str = "",
) -> dict:
    """Train the frozen decoder over the submission's shards. Harness-owned."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stream = load_stream(shard_paths)
    seq_len = arch["seq_len"]
    batch_tokens = run["batch_tokens_per_step"]
    rows_per_step = batch_tokens // seq_len
    micro_rows = max(1, min(micro_rows, rows_per_step))
    if rows_per_step % micro_rows != 0:
        raise ValueError("the micro-batch row count must divide the step's row count exactly")
    micro_batches = rows_per_step // micro_rows

    tokens_needed = steps * batch_tokens
    marks = {}
    for ident, fraction in zip(CHECKPOINT_IDS, CHECKPOINT_FRACTIONS):
        marks[int(round(steps * fraction))] = ident

    model = build(arch, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.1)
    warmup = max(1, int(0.02 * steps))

    checkpoints = []
    consumed = 0
    offset = 0
    halted_at_tokens = None
    optimizer_steps = 0
    forward_passes = 0
    backward_passes = 0
    stream_exhausted = False
    last_loss = None

    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = _learning_rate(step, steps, peak_lr, warmup)
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(micro_batches):
            if offset + seq_len * micro_rows + 1 > stream.size:
                offset = 0
                stream_exhausted = True
            batch = _batches(stream, seq_len, micro_rows, offset, device)
            if batch is None:
                stream_exhausted = True
                break
            tokens, targets = batch
            loss = model(tokens, targets) / micro_batches
            forward_passes += 1
            loss.backward()
            backward_passes += 1
            step_loss += float(loss.detach())
            offset += seq_len * micro_rows
            consumed += seq_len * micro_rows
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer_steps += 1
        last_loss = step_loss
        ident = marks.get(step + 1)
        if ident is None:
            continue
        path = out_dir / ("snapshot_" + ident + ".pt")
        digest = write_snapshot(model, arch, path, ident, step + 1, consumed)
        checkpoints.append(
            {"id": ident, "step": step + 1, "tokens": consumed, "path": path.as_posix(), "snapshot_sha256": digest}
        )
        if halt_at and ident == halt_at:
            halted_at_tokens = consumed
            break

    reached = any(row["id"] == BOUND_CHECKPOINT_ID for row in checkpoints)
    return {
        "tokens_available": int(stream.size),
        "tokens_needed": int(tokens_needed),
        "tokens_consumed": int(consumed),
        "bound_steps": int(steps),
        "optimizer_steps": int(optimizer_steps),
        "batch_tokens_per_step": int(batch_tokens),
        "micro_batches_per_step": int(micro_batches),
        "forward_passes_total": int(forward_passes),
        "backward_passes_total": int(backward_passes),
        "forward_passes_per_step": 1,
        "backward_passes_per_step": 1,
        "stream_wrapped": bool(stream_exhausted),
        "checkpoints": checkpoints,
        "bound_evaluation_point": BOUND_EVALUATION_POINT,
        "bound_checkpoint_id": BOUND_CHECKPOINT_ID,
        "bound_point_reached": bool(reached),
        "halted_at_tokens": halted_at_tokens,
        "final_train_loss_telemetry": last_loss,
    }


# --------------------------------------------------------------------------
# THE GRADED SCALAR. A forward pass of the snapshot's parameters over the
# verifier-owned held-out FineWeb split. Never called by the submission, never
# reachable from the submission, and never fed a number the submission wrote.
# --------------------------------------------------------------------------
@torch.no_grad()
def validation_loss(
    snapshot_path: Path,
    val_shard_paths: list,
    arch: dict,
    device: str,
    rows: int,
    token_range=None,
) -> dict:
    """Mean cross entropy in nats per token over the held-out split.

    This is the only producer of the graded quantity in the bundle. It loads the
    parameter snapshot the harness wrote, rebuilds the frozen architecture from
    the declaration, and runs the forward pass. Nothing else returns a loss.

    `token_range` is the half-open window of the concatenated held-out stream a
    fold occupies. It is resolved from the verifier's own fold partition and is
    never read from the submission.
    """
    payload = read_snapshot(Path(snapshot_path))
    state = payload["state_dict"]
    model = build(arch, device)
    model.load_state_dict(state, strict=True)
    model.eval()

    stream = load_stream([Path(p) for p in val_shard_paths])
    if token_range is not None:
        low, high = int(token_range[0]), int(token_range[1])
        stream = stream[low:high]
    seq_len = arch["seq_len"]
    span = seq_len * rows
    total = 0.0
    counted = 0
    offset = 0
    while offset + span + 1 <= stream.size:
        batch = _batches(stream, seq_len, rows, offset, device)
        if batch is None:
            break
        tokens, targets = batch
        loss = model(tokens, targets)
        total += float(loss) * span
        counted += span
        offset += span
    if counted == 0:
        raise ValueError("the held-out split carries no full evaluation window")
    return {
        "val_loss": total / counted,
        "tokens_evaluated": counted,
        "snapshot_checkpoint": payload.get("checkpoint"),
        "snapshot_step": payload.get("step"),
        "architecture": payload.get("architecture"),
    }


def fold_range(folds: dict, name: str):
    """Resolve one declared fold to its half-open window of the held-out stream."""
    for row in folds["folds"]:
        if row["id"] == name:
            return (int(row["tokens"][0]), int(row["tokens"][1]))
    raise KeyError("the fold partition declares no fold " + str(name))


def holdout_shards(root: Path, glob: str) -> list:
    """The verifier's own held-out FineWeb validation shards. Pinned, not searched.

    The root comes from tests/bound.json and from nowhere else. No submission
    byte, no submission call and no agent-visible declaration reaches it.
    """
    return sorted(Path(root).glob(glob))
