"""The frozen training stack for OER-18: the nanoGPT decoder, its optimizer, and the feeder.

FROZEN. This file is part of the task substrate, not part of the submission. Editing your
copy changes nothing that is graded: the verifier runs its own pinned copy of these bytes,
whose sha256 is bound in tests/checkers.yaml. What is free is the corpus your generator
emits, and nothing else.

What this stack is, and what it is not. It is the canonical nanoGPT decoder declared in
nanogpt_substrate.json: vocab_size 50304, num_layers 12, model_dim 768, head_dim 128,
num_heads 6, seq_len 1024, one forward pass and one backward pass per optimizer step, at
524288 tokens per step. It is not a bag-of-tokens scorer, not a perceptron and not a cost
model. An earlier revision of this slot carried a 512-bucket hashed linear scorer here; that
stand-in has been removed, because a metric that resolves against it does not resolve
against a nanoGPT run.

How your corpus reaches the model. Your emitted samples are encoded with the GPT-2 BPE the
upstream loader uses, one end-of-text delimiter is appended per sample, and the resulting
uint16 stream is written into shards in the upstream shard format. Training reads those
shards in emission order. The token that the budget counts and the token that the model
trains on are therefore the same token, counted once.

Determinism. Initialization is seeded from a bound constant, the sample order is the
emission order, and the pass ledger below is checked before any score is returned. Two calls
with the same corpus and the same mark produce the same parameter digest.

The pass ledger is load-bearing. Each optimizer step records exactly one forward pass and
one backward pass. A snapshot whose ledger does not equal its step count is refused by
`evaluate`, which returns None rather than a number. A run with the forward and backward
pass removed therefore has no score at all, rather than a different score.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

# cuBLAS reads this when it creates its handle, which must happen after the variable is
# set, so it is set here before the first CUDA call rather than left to the image alone.
# Without it a deterministic reduction is not available and two identical runs disagree.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# The canonical substrate. Read from the replicated declaration, never inlined.
# ---------------------------------------------------------------------------

SUBSTRATE_NAME = "nanogpt_substrate.json"


def _load_substrate() -> dict:
    """Read this tree's own copy of the canonical operating point.

    Replication rather than reference is the rule: a slot never reads a sibling slot's
    bytes, so each tree carries its own copy and each side reads the copy beside it. An
    absent or unreadable declaration is a refused run, never a defaulted architecture.
    """
    for candidate in (HERE / SUBSTRATE_NAME, HERE.parent / SUBSTRATE_NAME):
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise RuntimeError("frozen-axis-declaration-absent: " + SUBSTRATE_NAME)


SUBSTRATE = _load_substrate()
_ARCH = SUBSTRATE["architecture"]
_RUN = SUBSTRATE["run"]

VOCAB_SIZE = int(_ARCH["vocab_size"])
NUM_LAYERS = int(_ARCH["num_layers"])
MODEL_DIM = int(_ARCH["model_dim"])
HEAD_DIM = int(_ARCH["head_dim"])
NUM_HEADS = int(_ARCH["num_heads"])
SEQ_LEN = int(_ARCH["seq_len"])

BATCH_TOKENS_PER_STEP = int(_RUN["batch_tokens_per_step"])
FORWARD_PASSES_PER_STEP = int(_RUN["forward_passes_per_step"])
BACKWARD_PASSES_PER_STEP = int(_RUN["backward_passes_per_step"])

# The declaration must be internally consistent before a single parameter is allocated.
if MODEL_DIM // HEAD_DIM != NUM_HEADS:
    raise RuntimeError("frozen-axis-moved: model_dim // head_dim does not equal num_heads")
if BATCH_TOKENS_PER_STEP % SEQ_LEN != 0:
    raise RuntimeError("frozen-axis-moved: batch_tokens_per_step is not a multiple of seq_len")

SEQUENCES_PER_STEP = BATCH_TOKENS_PER_STEP // SEQ_LEN

# ---------------------------------------------------------------------------
# Frozen optimizer and schedule. None of these is a free variable in this slot.
# The free axis here is the corpus the generator emits, and nothing else.
# ---------------------------------------------------------------------------

INIT_SEED = 20260820
LEARNING_RATE = 6e-4
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
WARMUP_STEPS = 2

# 524288 tokens do not fit one H100 in a single resident batch, so the step's batch is
# accumulated over micro-batches of this many sequences. This is a memory split of one
# pass over the step's batch, not an extra pass: the gradient the optimizer consumes is
# the gradient of the whole 524288-token batch, and the ledger counts that pass once.
MICRO_SEQUENCES = 8
EVAL_SEQUENCES = 8

# The GPT-2 BPE end-of-text id, which delimits one emitted sample from the next in the
# shard stream. It is the upstream delimiter, not a value this slot chose.
EOT_TOKEN = 50256

# The upstream shard header: 256 int32 words, magic 20240520, version 1, token count.
SHARD_MAGIC = 20240520
SHARD_VERSION = 1
SHARD_HEADER_WORDS = 256

_ENCODER = None


def encoder():
    """The GPT-2 BPE the upstream loader uses. Resolved offline from the image cache."""
    global _ENCODER
    if _ENCODER is None:
        import tiktoken

        _ENCODER = tiktoken.get_encoding("gpt2")
    return _ENCODER


def tokenize(text: str) -> list:
    """The frozen tokenization: GPT-2 BPE, then one end-of-text delimiter.

    This is the only tokenization in the slot. The budget counts these ids and the model
    trains on these ids, so a sample cannot cost one thing and teach another.
    """
    return encoder().encode_ordinary(str(text)) + [EOT_TOKEN]


def token_count(sample) -> int:
    """The tokens one sample costs against the frozen budget."""
    return len(tokenize(sample.get("text", "")))


def feed(samples, budget_tokens: int) -> tuple:
    """Feed samples in emission order and report what was actually consumed.

    Returns (prefix, tokens_fed_by_the_prefix, tokens_offered_by_the_whole_corpus). The
    feeder stops at the budget; it never silently trims a corpus that overshoots, because
    the overshoot is exactly what the budget checker reads.
    """
    prefix, fed, offered = [], 0, 0
    for row in samples:
        cost = token_count(row)
        offered += cost
        if fed + cost <= budget_tokens:
            prefix.append(row)
            fed += cost
    return prefix, fed, offered


def build_shards(samples, out_dir) -> dict:
    """Write the emitted samples into upstream-format uint16 shards, in emission order.

    This is the whole of the path from a submitted generator's output to the tokens the
    decoder trains on. No sample is reordered, deduplicated or rewritten on the way.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stream = []
    for row in samples:
        stream.extend(tokenize(row.get("text", "")))
    shard = out / "generated_train_000000.bin"
    header = [0] * SHARD_HEADER_WORDS
    header[0], header[1], header[2] = SHARD_MAGIC, SHARD_VERSION, len(stream)
    with open(shard, "wb") as handle:
        handle.write(struct.pack("<" + "i" * SHARD_HEADER_WORDS, *header))
        handle.write(np.asarray(stream, dtype="<u2").tobytes())
    return {"shard": str(shard), "tokens": len(stream)}


# ---------------------------------------------------------------------------
# The canonical decoder. Twelve layers, 768 model dim, six heads of head dim 128.
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads = NUM_HEADS
        self.head_dim = HEAD_DIM
        self.qkv = nn.Linear(MODEL_DIM, 3 * NUM_HEADS * HEAD_DIM, bias=False)
        self.proj = nn.Linear(NUM_HEADS * HEAD_DIM, MODEL_DIM, bias=False)

    def forward(self, x):
        batch, length, _ = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.num_heads, self.head_dim)
        query, key, value = (qkv[:, :, index].transpose(1, 2) for index in range(3))
        with sdpa_kernel(SDPBackend.MATH):
            out = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        out = out.transpose(1, 2).reshape(batch, length, self.num_heads * self.head_dim)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(MODEL_DIM, 4 * MODEL_DIM, bias=False)
        self.proj = nn.Linear(4 * MODEL_DIM, MODEL_DIM, bias=False)

    def forward(self, x):
        return self.proj(F.relu(self.fc(x)).square())


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(MODEL_DIM, bias=False)
        self.attn = CausalSelfAttention()
        self.mlp_norm = nn.LayerNorm(MODEL_DIM, bias=False)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    """The frozen architecture. Every dimension is read from the substrate declaration."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, MODEL_DIM)
        self.pos = nn.Embedding(SEQ_LEN, MODEL_DIM)
        self.blocks = nn.ModuleList([Block() for _ in range(NUM_LAYERS)])
        self.final_norm = nn.LayerNorm(MODEL_DIM, bias=False)
        self.head = nn.Linear(MODEL_DIM, VOCAB_SIZE, bias=False)

    def forward(self, idx):
        _batch, length = idx.shape
        positions = torch.arange(length, device=idx.device)
        x = self.embed(idx) + self.pos(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.head(self.final_norm(x))


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_determinism() -> None:
    """Pin every reduction the run touches to a deterministic kernel.

    The harness grades a submission partly by retraining the same prefix and comparing
    parameter digests, so a nondeterministic kernel would fail an honest run. Flash
    attention's backward and the default cuBLAS reduction are both nondeterministic, so
    attention is pinned to the math backend and TF32 is refused.
    """
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _build_model() -> GPT:
    configure_determinism()
    torch.manual_seed(INIT_SEED)
    torch.cuda.manual_seed_all(INIT_SEED)
    return GPT().to(_device())


# ---------------------------------------------------------------------------
# The parameter snapshot. This is the graded artifact.
# ---------------------------------------------------------------------------


@dataclass
class Snapshot:
    """A real parameter snapshot of the frozen architecture, plus its pass ledger.

    `owner` is the string the harness stamps on state it produced itself. `steps`,
    `forward_passes` and `backward_passes` are the ledger: the run is well formed only when
    the ledger equals one forward and one backward pass per step, and `evaluate` refuses a
    snapshot whose ledger does not.
    """

    state: dict = field(default_factory=dict)
    steps: int = 0
    forward_passes: int = 0
    backward_passes: int = 0
    tokens_seen: int = 0
    owner: str = "harness"
    architecture: dict = field(default_factory=dict)

    def ledger_holds(self) -> bool:
        if self.steps < 0:
            return False
        return (
            self.forward_passes == self.steps * FORWARD_PASSES_PER_STEP
            and self.backward_passes == self.steps * BACKWARD_PASSES_PER_STEP
        )

    def shape_holds(self) -> bool:
        """The snapshot is shape-bound to the declared architecture, or it is not this model."""
        declared = {
            "vocab_size": VOCAB_SIZE,
            "num_layers": NUM_LAYERS,
            "model_dim": MODEL_DIM,
            "head_dim": HEAD_DIM,
            "num_heads": NUM_HEADS,
            "seq_len": SEQ_LEN,
        }
        if self.architecture != declared:
            return False
        if "embed.weight" not in self.state:
            return False
        if tuple(self.state["embed.weight"].shape) != (VOCAB_SIZE, MODEL_DIM):
            return False
        return len([key for key in self.state if key.startswith("blocks.")]) > 0


def state_digest(snapshot) -> str:
    """A stable sha256 over the parameter tensors and the pass ledger."""
    if not isinstance(snapshot, Snapshot):
        return ""
    digest = hashlib.sha256()
    for key in sorted(snapshot.state):
        tensor = snapshot.state[key].detach().to(torch.float32).cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    digest.update(
        json.dumps(
            {
                "steps": snapshot.steps,
                "forward_passes": snapshot.forward_passes,
                "backward_passes": snapshot.backward_passes,
                "architecture": snapshot.architecture,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Training. One forward pass and one backward pass per optimizer step.
# ---------------------------------------------------------------------------


def _stream(samples) -> list:
    stream = []
    for row in samples:
        stream.extend(tokenize(row.get("text", "")))
    return stream


def _batches(stream, steps):
    """Cut the token stream into the frozen per-step batch, in emission order."""
    needed = SEQ_LEN + 1
    cursor = 0
    for _ in range(steps):
        rows = []
        for _ in range(SEQUENCES_PER_STEP):
            if cursor + needed > len(stream):
                cursor = 0
            rows.append(stream[cursor : cursor + needed])
            cursor += SEQ_LEN
        yield rows


def _lr_at(step: int, total: int) -> float:
    if total <= 0:
        return 0.0
    if step < WARMUP_STEPS:
        return LEARNING_RATE * float(step + 1) / float(max(1, WARMUP_STEPS))
    span = max(1, total - WARMUP_STEPS)
    progress = float(step - WARMUP_STEPS) / float(span)
    return LEARNING_RATE * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))


def _train_steps(stream, steps: int) -> Snapshot:
    """Run the decoder for the given number of optimizer steps over the token stream."""
    device = _device()
    model = _build_model()
    architecture = {
        "vocab_size": VOCAB_SIZE,
        "num_layers": NUM_LAYERS,
        "model_dim": MODEL_DIM,
        "head_dim": HEAD_DIM,
        "num_heads": NUM_HEADS,
        "seq_len": SEQ_LEN,
    }
    if steps <= 0 or len(stream) < SEQ_LEN + 1:
        model.eval()
        return Snapshot(
            state={key: value.detach().clone() for key, value in model.state_dict().items()},
            steps=0,
            forward_passes=0,
            backward_passes=0,
            tokens_seen=0,
            owner="harness",
            architecture=architecture,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, betas=BETAS, weight_decay=WEIGHT_DECAY
    )
    forward_passes, backward_passes, tokens_seen = 0, 0, 0
    model.train()
    for step, rows in enumerate(_batches(stream, steps)):
        for group in optimizer.param_groups:
            group["lr"] = _lr_at(step, steps)
        optimizer.zero_grad(set_to_none=True)
        batch = torch.tensor(rows, dtype=torch.long, device=device)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        chunks = max(1, SEQUENCES_PER_STEP // MICRO_SEQUENCES)
        for chunk_inputs, chunk_targets in zip(
            inputs.chunk(chunks, dim=0), targets.chunk(chunks, dim=0)
        ):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(chunk_inputs)
                loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), chunk_targets.reshape(-1))
            (loss / chunks).backward()
        forward_passes += 1
        backward_passes += 1
        tokens_seen += SEQUENCES_PER_STEP * SEQ_LEN
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

    model.eval()
    return Snapshot(
        state={key: value.detach().clone() for key, value in model.state_dict().items()},
        steps=steps,
        forward_passes=forward_passes,
        backward_passes=backward_passes,
        tokens_seen=tokens_seen,
        owner="harness",
        architecture=architecture,
    )


def train(samples) -> Snapshot:
    """Train the frozen decoder on the samples in emission order, for the full budget."""
    stream = _stream(samples)
    return _train_steps(stream, len(stream) // BATCH_TOKENS_PER_STEP)


def train_to_mark(samples, mark_tokens: int) -> tuple:
    """Train on the corpus prefix whose cumulative token count reaches the mark.

    Returns (snapshot, tokens_consumed, reached), where `reached` is False when the corpus
    ran out before the mark or when the pass ledger does not hold. A mark that is not
    reached is not an evaluation point that happened.
    """
    prefix, consumed = [], 0
    for row in samples:
        if consumed >= mark_tokens:
            break
        prefix.append(row)
        consumed += token_count(row)
    steps = int(mark_tokens) // BATCH_TOKENS_PER_STEP
    stream = _stream(prefix)
    snapshot = _train_steps(stream, steps if consumed >= mark_tokens else 0)
    reached = bool(consumed >= mark_tokens and steps > 0 and snapshot.ledger_holds())
    return snapshot, consumed, reached


# ---------------------------------------------------------------------------
# Evaluation. The graded scalar, computed from the snapshot on held-out windows.
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(snapshot, items):
    """Held-out next-token top-1 accuracy in [0, 1], higher is better.

    Returns None, not a number, when the snapshot is not a real parameter snapshot of the
    frozen architecture or when its pass ledger does not hold. Removing the forward and
    backward pass therefore leaves the metric undefined rather than merely different, which
    is the property that separates a trained model from a simulator.
    """
    rows = [row for row in (items or []) if row.get("tokens")]
    if not rows:
        return None
    if not isinstance(snapshot, Snapshot):
        return None
    if not snapshot.shape_holds() or not snapshot.ledger_holds():
        return None

    device = _device()
    model = GPT().to(device)
    model.load_state_dict(snapshot.state)
    model.eval()

    windows = [list(row["tokens"])[: SEQ_LEN + 1] for row in rows]
    windows = [window for window in windows if len(window) == SEQ_LEN + 1]
    if not windows:
        return None

    hits, total = 0, 0
    for start in range(0, len(windows), EVAL_SEQUENCES):
        batch = torch.tensor(
            windows[start : start + EVAL_SEQUENCES], dtype=torch.long, device=device
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(batch[:, :-1])
        guess = logits.float().argmax(dim=-1)
        hits += int((guess == batch[:, 1:]).sum().item())
        total += int(batch[:, 1:].numel())
    if total == 0:
        return None
    return hits / total


def detokenize(tokens) -> str:
    """The inverse of the frozen tokenization, for the near-duplicate comparison only."""
    return encoder().decode([int(value) for value in tokens if int(value) != EOT_TOKEN])


def held_out_windows(payload: dict, count: int, window: int, stride: int) -> list:
    """Cut the verifier's held-out token slice into evaluation windows.

    The payload this reads lives only under tests/. It is never assembled onto the agent
    surface, and no call here resolves an evaluation split from anywhere a solver can reach.
    """
    raw = base64.b64decode(payload["held_out_split"]["payload_base64"])
    ids = np.frombuffer(raw, dtype="<u2").astype(np.int64).tolist()
    out = []
    for index in range(count):
        start = index * stride
        chunk = ids[start : start + window + 1]
        if len(chunk) < window + 1:
            break
        out.append({"index": index, "tokens": chunk})
    return out


__all__ = [
    "GPT",
    "Snapshot",
    "build_shards",
    "detokenize",
    "encoder",
    "evaluate",
    "feed",
    "held_out_windows",
    "state_digest",
    "token_count",
    "tokenize",
    "train",
    "train_to_mark",
]
