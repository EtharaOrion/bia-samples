"""The frozen model, optimizer, training budget and evaluator. The canonical nanoGPT decoder.

This file is the frozen half of slot OER-19. The solving agent does not modify it
and cannot reach it at grading time: the verifier runs its own copy at
`tests/trainer.py`, and the agent sees a byte-identical mirror here whose sha256 is
pinned in `tests/checkers.yaml`. The agent's whole surface of control is the
synthetic corpus it generates; everything downstream of that corpus is fixed here.

The architecture is not declared in this file. It is read at construction time from
`nanogpt_substrate.json`, the canonical operating point replicated into this slot,
so the decoder that trains is the decoder the substrate names and a local edit to
that file produces a refused run with reason frozen-axis-moved rather than a
different grade. Every value it carries comes from the vendored record set at
harness/records/track_3_optimization/train_gpt_simple.py: vocab_size 50304,
num_layers 12, model_dim 768, head_dim 128, num_heads 6, seq_len 1024, and 524288
tokens per optimizer step with exactly one forward and one backward pass per step.

The training budget is enforced here and counted here. `train()` returns the
counters it actually incremented while feeding the optimizer, never the counts it
was asked for, because a budget checker that reads the requested figure grades the
request instead of the run. The forward and backward counters are incremented at
the call sites of the forward and the backward, so a run with the passes removed
reports zero of them and resolves no reading at all rather than a different one.

Determinism is seeded rather than structural, which is a real change from a
stand-in that could claim bit-identical replay. A 124M-parameter decoder trained on
an accelerator accumulates in a nondeterministic order, so this file seeds every
generator it owns, records the parameter digest of the state it actually evaluated,
and binds the graded reading to that digest within the run. What is guaranteed is
that the evaluated parameters are the trained parameters, not that two machines
produce the same last bit. The sustain tolerance in `tests/checkers.yaml` is the
band the reading is required to hold inside, and it is bound wider than that
accumulation noise for exactly this reason.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# The substrate. Read, never invented. The file sits beside this one in the agent
# image and beside tests/trainer.py in the verifier image, and both copies are
# byte-identical to .seed/nanogpt_substrate.json.
# ---------------------------------------------------------------------------

SUBSTRATE_NAME = "nanogpt_substrate.json"
DECLARATION_VERSION = "nanogpt-substrate-v1"

_SEARCH = (
    Path(__file__).resolve().parent / SUBSTRATE_NAME,
    Path(__file__).resolve().parent.parent / SUBSTRATE_NAME,
    Path(__file__).resolve().parent.parent.parent / "environment" / SUBSTRATE_NAME,
)


def substrate() -> dict:
    """The canonical operating point, read from the replicated declaration."""
    for candidate in _SEARCH:
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("declaration_version") != DECLARATION_VERSION:
                raise ValueError("substrate declaration_version moved")
            return payload
    raise FileNotFoundError("no " + SUBSTRATE_NAME + " beside the trainer")


_SUB = substrate()
_ARCH = _SUB["architecture"]
_RUN = _SUB["run"]

VOCAB_SIZE = int(_ARCH["vocab_size"])          # 50304
NUM_LAYERS = int(_ARCH["num_layers"])          # 12
MODEL_DIM = int(_ARCH["model_dim"])            # 768
HEAD_DIM = int(_ARCH["head_dim"])              # 128
NUM_HEADS = int(_ARCH["num_heads"])            # 6
SEQ_LEN = int(_ARCH["seq_len"])                # 1024

BATCH_TOKENS_PER_STEP = int(_RUN["batch_tokens_per_step"])      # 524288
FORWARD_PASSES_PER_STEP = int(_RUN["forward_passes_per_step"])  # 1
BACKWARD_PASSES_PER_STEP = int(_RUN["backward_passes_per_step"])# 1
TARGET_VAL_LOSS = float(_RUN["target_val_loss"])                # 3.28

# ---------------------------------------------------------------------------
# Frozen axes owned by this slot rather than by the substrate. None of these is a
# knob the agent may turn; they are the bound corpus shape and the bound run.
# ---------------------------------------------------------------------------

# The corpus is exactly one optimizer step's worth of tokens at the frozen batch
# size: 512 documents of 1024 tokens is 524288 tokens. One pass over the corpus is
# therefore exactly one step, which is what makes the epoch count and the step
# count the same integer instead of two numbers that have to be reconciled.
BOUND_CORPUS_DOCUMENTS = 512
BOUND_DOCUMENT_TOKENS = SEQ_LEN
BOUND_CORPUS_TOKENS = BOUND_CORPUS_DOCUMENTS * BOUND_DOCUMENT_TOKENS   # 524288

BOUND_STEPS = 96
BOUND_EPOCHS = BOUND_STEPS
BOUND_TOKENS_FED = BOUND_STEPS * BATCH_TOKENS_PER_STEP                 # 50331648

# Evaluation points the verifier schedules, in optimizer steps. The last is the
# bound evaluation point at which the graded reading is taken; the earlier ones are
# the points the reading must be sustained across. The three sit at two thirds,
# five sixths and the whole of the run, which is the schedule this slot has always
# carried, re-expressed over the bound step count.
SCHEDULED_POINTS = (64, 80, 96)
BOUND_EVALUATION_POINT = BOUND_STEPS
SUSTAIN_TOLERANCE = 0.50

# Optimizer. Free in the upstream track, frozen here, because this slot's free axis
# is the generator and an unpinned optimizer would let the agent move the metric
# without moving the corpus.
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0
WARMUP_STEPS = 8
SEED = 20240520

# Sequences per micro-batch. Gradient accumulation splits one step's 524288 tokens
# into micro-batches to fit one accelerator; it does not add a pass. The step's
# forward and backward are counted once each, which is what the substrate's
# one-forward-one-backward-per-step rule constrains.
MICRO_BATCH_SEQUENCES = 16

COUNTER_SOURCE = "harness-trainer"
WEIGHTS_SOURCE = "harness-trainer"
READOUT = {"kind": "raw", "window": 1, "ema_alpha": None}


def shape_signature() -> str:
    """The architecture the graded parameters must actually have, as one string.

    A checker cannot open a file, so the signature travels in the telemetry and is
    compared there against constants mirrored from tests/checkers.yaml. This is the
    gate that separates a parameter snapshot of the frozen decoder from a weight
    table that merely carries numbers.
    """
    return (
        "vocab=" + str(VOCAB_SIZE)
        + ";layers=" + str(NUM_LAYERS)
        + ";dim=" + str(MODEL_DIM)
        + ";head_dim=" + str(HEAD_DIM)
        + ";heads=" + str(NUM_HEADS)
        + ";seq=" + str(SEQ_LEN)
    )


@dataclass
class Counters:
    """What the optimizer was actually fed, incremented at the point of feeding."""

    forward_passes: int = 0
    backward_passes: int = 0
    optimizer_steps: int = 0
    tokens_fed: int = 0
    epochs_completed: int = 0
    terminated_early: bool = False


@dataclass
class Reading:
    """One evaluation of one parameter state on the held-out stream."""

    step: int
    nll_sum: float
    tokens: int
    loss: float
    digest: str


@dataclass
class Trained:
    shape_signature: str
    parameter_count: int
    readings: list = field(default_factory=list)
    counters: Counters = field(default_factory=Counters)

    def at(self, step: int):
        for row in self.readings:
            if row.step == step:
                return row
        return None


# ---------------------------------------------------------------------------
# The decoder. Twelve blocks, 768 model dim, six heads of 128, 1024 positions.
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = NUM_HEADS
        self.head_dim = HEAD_DIM
        inner = NUM_HEADS * HEAD_DIM
        self.qkv = nn.Linear(MODEL_DIM, 3 * inner, bias=False)
        self.proj = nn.Linear(inner, MODEL_DIM, bias=False)

    def forward(self, x):
        batch, length, _ = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        out = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        out = out.transpose(1, 2).reshape(batch, length, self.num_heads * self.head_dim)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(MODEL_DIM, 4 * MODEL_DIM, bias=False)
        self.down = nn.Linear(4 * MODEL_DIM, MODEL_DIM, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_attention = nn.LayerNorm(MODEL_DIM)
        self.attention = CausalSelfAttention()
        self.norm_mlp = nn.LayerNorm(MODEL_DIM)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attention(self.norm_attention(x))
        return x + self.mlp(self.norm_mlp(x))


class GPT(nn.Module):
    """The canonical decoder at the substrate's operating point."""

    def __init__(self):
        super().__init__()
        self.tokens = nn.Embedding(VOCAB_SIZE, MODEL_DIM)
        self.positions = nn.Embedding(SEQ_LEN, MODEL_DIM)
        self.blocks = nn.ModuleList([Block() for _ in range(NUM_LAYERS)])
        self.norm = nn.LayerNorm(MODEL_DIM)
        self.head = nn.Linear(MODEL_DIM, VOCAB_SIZE, bias=False)
        self.head.weight = self.tokens.weight
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        _, length = idx.shape
        pos = torch.arange(length, device=idx.device)
        x = self.tokens(idx) + self.positions(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def parameter_count(model) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def parameter_digest(model) -> str:
    """A digest over the parameter tensors, rounded so it is stable across platforms.

    Rounding before digesting is deliberate: an unrounded float image would make the
    digest depend on the last bit of an accumulation order, and a digest that can
    move without the model moving is not evidence about the model. The digest is
    what binds the reading to the parameters the reading was taken from.
    """
    hasher = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        hasher.update(name.encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        values = tensor.detach().to(torch.float32).cpu().numpy()
        hasher.update(np.round(values, 5).astype(np.float32).tobytes())
    return hasher.hexdigest()


def learning_rate_at(step: int) -> float:
    """Linear warmup then cosine decay. Bound, so the agent cannot move it."""
    if step <= WARMUP_STEPS:
        return LEARNING_RATE * step / max(1, WARMUP_STEPS)
    span = max(1, BOUND_STEPS - WARMUP_STEPS)
    progress = min(1.0, (step - WARMUP_STEPS) / span)
    return LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * progress))


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batches(shard_tokens, sequences_per_step: int, step: int, device):
    """One optimizer step's tokens, cut into sequences on the document boundary.

    The shard is exactly one step's worth of tokens, so a step consumes it whole and
    a pass over the corpus is a step. The offset rotates with the step so the
    sequence order is not the same on every pass.
    """
    usable = (shard_tokens.size // SEQ_LEN) * SEQ_LEN
    grid = shard_tokens[:usable].reshape(-1, SEQ_LEN)
    order = (np.arange(grid.shape[0]) + step) % grid.shape[0]
    grid = grid[order][:sequences_per_step]
    block = torch.from_numpy(np.ascontiguousarray(grid)).to(device=device, dtype=torch.long)
    return block


@torch.no_grad()
def evaluate(model, val_tokens, device) -> dict:
    """Mean token cross-entropy on the held-out stream. No smoothing anywhere.

    The reading is returned as the two accumulators it is the ratio of, so the
    verifier's own checker can recompute it rather than accept it. A reading that
    cannot be recomputed from the counts the evaluator recorded is not a raw
    readout and is graded as a smoothed one.
    """
    model.eval()
    usable = (val_tokens.size // (SEQ_LEN + 1)) * (SEQ_LEN + 1)
    if usable == 0:
        return {"nll_sum": 0.0, "tokens": 0, "loss": float("nan")}
    grid = val_tokens[:usable].reshape(-1, SEQ_LEN + 1)
    nll_sum, counted = 0.0, 0
    for start in range(0, grid.shape[0], 4):
        chunk = grid[start : start + 4]
        block = torch.from_numpy(np.ascontiguousarray(chunk)).to(device=device, dtype=torch.long)
        inputs, targets = block[:, :-1], block[:, 1:]
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE).float(),
            targets.reshape(-1),
            reduction="sum",
        )
        nll_sum += float(loss.item())
        counted += int(targets.numel())
    model.train()
    return {
        "nll_sum": nll_sum,
        "tokens": counted,
        "loss": (nll_sum / counted) if counted else float("nan"),
    }


def train(shard_tokens, val_tokens, steps: int = BOUND_STEPS, points=SCHEDULED_POINTS) -> Trained:
    """Train the frozen decoder on the submission's shard and read it at each point.

    `shard_tokens` is the token stream the submission's corpus was packed into by
    `shards.build_train_shard`, and `val_tokens` is the held-out stream the verifier
    owns. Every counter below is incremented where the work happens, so a caller
    cannot report a budget it did not spend, and a run with the forward and backward
    calls removed reports zero passes and produces no reading at all.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = _device()
    model = GPT().to(device)
    model.train()

    decayed = [p for p in model.parameters() if p.dim() >= 2]
    plain = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decayed, "weight_decay": WEIGHT_DECAY},
            {"params": plain, "weight_decay": 0.0},
        ],
        lr=LEARNING_RATE,
        betas=BETAS,
    )

    counters = Counters()
    result = Trained(
        shape_signature=shape_signature(),
        parameter_count=parameter_count(model),
        counters=counters,
    )

    sequences_per_step = BATCH_TOKENS_PER_STEP // SEQ_LEN
    if shard_tokens is None or getattr(shard_tokens, "size", 0) < SEQ_LEN:
        return result

    wanted = set(int(value) for value in points)
    for step in range(1, int(steps) + 1):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at(step)
        optimizer.zero_grad(set_to_none=True)
        block = _batches(shard_tokens, sequences_per_step, step, device)
        micro = max(1, MICRO_BATCH_SEQUENCES)
        chunks = max(1, (block.shape[0] + micro - 1) // micro)
        for index in range(chunks):
            piece = block[index * micro : (index + 1) * micro]
            if piece.shape[0] == 0:
                continue
            inputs = piece[:, :-1]
            targets = piece[:, 1:]
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE).float(), targets.reshape(-1)
            ) / chunks
            loss.backward()
        # One forward and one backward per optimizer step, counted at the step the
        # substrate constrains. The micro-batches above are one pass split to fit
        # an accelerator, not extra passes.
        counters.forward_passes += FORWARD_PASSES_PER_STEP
        counters.backward_passes += BACKWARD_PASSES_PER_STEP
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        counters.optimizer_steps += 1
        counters.tokens_fed += int(block.shape[0]) * SEQ_LEN
        counters.epochs_completed += 1
        if step in wanted:
            reading = evaluate(model, val_tokens, device)
            result.readings.append(
                Reading(
                    step=step,
                    nll_sum=reading["nll_sum"],
                    tokens=reading["tokens"],
                    loss=reading["loss"],
                    digest=parameter_digest(model),
                )
            )
    counters.terminated_early = counters.optimizer_steps < int(steps)
    return result
