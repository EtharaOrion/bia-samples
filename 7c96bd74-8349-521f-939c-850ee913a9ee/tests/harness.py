"""The frozen nanoGPT decoder, the frozen optimizer, the frozen budget, and the evaluation.

This module is the verifier's own trainer. It is the only thing that ever touches the
held-out FineWeb validation split, it is the only thing that produces the graded number,
and every counter a checker reads is incremented here rather than parsed out of anything
the submission emitted.

The model is the canonical modded-nanogpt decoder declared in tests/nanogpt_substrate.json:
vocab_size 50304, num_layers 12, model_dim 768, head_dim 128, num_heads 6, seq_len 1024.
The optimizer is the record-set pair, AdamW on the embedding, the output projection and
the one-dimensional parameters, Muon on the hidden block matrices, both read from
harness/records/track_3_optimization/train_gpt_simple.py. The budget is 524288 tokens per
step with exactly one forward and one backward pass per step.

The training stream is a mix. Each step draws `synthetic_tokens_per_step` tokens from a
shard this module builds by tokenizing the submitted generator's documents, and
`fineweb_tokens_per_step` tokens from the FineWeb10B train shards, and the two sums to the
frozen batch. That is the sense in which the generator is graded: it owns a bounded share
of a real pretraining corpus and nothing else.

The graded quantity is the held-out benchmark score, which is exp(-val_loss) where
val_loss is the mean per-token cross entropy the snapshot assigns to the held-out FineWeb
validation documents. It is the geometric-mean per-token probability of held-out text, it
lies on (0.0, 1.0], higher is better, and it is undefined without a forward pass.

environment/trainer.py is a mirror of this training core for the agent to iterate against
locally. The graded run is this file, executed by the verifier, on a validation split that
is absent from every agent-visible container.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

SCHEMA = "forge.oer17.run/v1"

HERE = Path(__file__).resolve().parent
SUBSTRATE = HERE / "nanogpt_substrate.json"

# The modded-nanogpt shard container: 256 int32 of header, then uint16 tokens.
SHARD_MAGIC = 20240520
SHARD_VERSION = 1
SHARD_HEADER_INTS = 256

# Where the FineWeb10B train shards are staged inside the verifier image. The validation
# shards are NOT staged here; the held-out split this module scores against is
# tests/heldout/benchmark.jsonl, which lives in the grading tree alone.
FINEWEB_TRAIN_ROOT = Path(os.environ.get("OER17_FINEWEB_ROOT", "/verifier/data/fineweb10B"))
FINEWEB_TRAIN_GLOB = "fineweb_train_*.bin"


def load_substrate(path: Path = SUBSTRATE) -> dict:
    """The verifier's own copy of the canonical operating point.

    A slot replicates the same bytes into environment/ so the solver can read the
    declaration, and this function never reads that copy: the numbers the graded run is
    bound to come from the verifier's tree, so editing the agent-visible copy changes the
    declaration the solver sees and nothing about how the run is scored.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list:
    """The held-out items. The leading provenance banner row is metadata, not an item."""
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict) and "text" in row:
                rows.append(row)
    return rows


def document_text(row: dict) -> str:
    return str(row.get("text", ""))


def token_count(sample: dict) -> int:
    """Whitespace token count, used only for the corpus-size counters, never for training."""
    return len(document_text(sample).split())


# --- tokenizer ----------------------------------------------------------------


def encoder():
    """The GPT-2 BPE the FineWeb10B shards were built with, vendored into the image.

    tests/Dockerfile installs tiktoken and pre-caches the gpt2 encoding at build time,
    because the verifier surface binds egress denied and a run-time fetch would turn a
    graded run into a network dependency.
    """
    import tiktoken

    return tiktoken.get_encoding("gpt2")


def encode_documents(documents: list, enc, vocab_size: int) -> list:
    """Tokenize documents into one flat stream, delimited by end-of-text, clipped to vocab."""
    eot = enc.eot_token
    stream = []
    usable = 0
    for row in documents:
        text = document_text(row)
        if not text:
            continue
        ids = enc.encode_ordinary(text)
        if not ids:
            continue
        stream.append(eot)
        stream.extend(token for token in ids if 0 <= token < vocab_size)
        usable += 1
    return stream, usable


def write_shard(tokens: list, path: Path) -> int:
    """Write a uint16 shard in the modded-nanogpt container. Returns the token count."""
    import numpy

    array = numpy.asarray(tokens, dtype=numpy.uint16)
    header = numpy.zeros(SHARD_HEADER_INTS, dtype=numpy.int32)
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = int(array.size)
    with Path(path).open("wb") as handle:
        handle.write(header.tobytes())
        handle.write(array.tobytes())
    return int(array.size)


def read_shard(path: Path):
    """Read a uint16 shard written in the modded-nanogpt container."""
    import numpy

    with Path(path).open("rb") as handle:
        header = numpy.frombuffer(handle.read(SHARD_HEADER_INTS * 4), dtype=numpy.int32)
        if int(header[0]) != SHARD_MAGIC or int(header[1]) != SHARD_VERSION:
            raise ValueError("shard container mismatch at " + str(path))
        count = int(header[2])
        return numpy.frombuffer(handle.read(2 * count), dtype=numpy.uint16)


# --- the frozen architecture --------------------------------------------------
#
# Read from harness/records/track_3_optimization/train_gpt_simple.py. Nothing below is
# authored here: the RMSNorm, the biased Linear, the half-truncate rotary, the causal
# attention at scale 0.12, the squared-ReLU MLP, the logit soft-cap at 15 and the
# summed cross entropy are the record set's own definitions.


def build_model(architecture: dict):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    head_dim = int(architecture["head_dim"])

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
            self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)]))

        def forward(self, x_BTHD: Tensor):
            pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
            theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
            cos, sin = theta.cos(), theta.sin()
            x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
            y1 = x1 * cos + x2 * sin
            y2 = x1 * (-sin) + x2 * cos
            return torch.cat((y1, y2), 3).type_as(x_BTHD)

    class CausalSelfAttention(nn.Module):
        def __init__(self, dim: int):
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
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=0.12, is_causal=True
            ).transpose(1, 2)
            y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
            return self.proj(y)

    class MLP(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.fc = Linear(dim, 4 * dim)
            self.proj = Linear(4 * dim, dim)

        def forward(self, x: Tensor):
            return self.proj(self.fc(x).relu().square())

    class Block(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.attn = CausalSelfAttention(dim)
            self.mlp = MLP(dim)
            self.norm1 = RMSNorm(dim)
            self.norm2 = RMSNorm(dim)

        def forward(self, x: Tensor):
            x = x + self.attn(self.norm1(x))
            return x + self.mlp(self.norm2(x))

    class GPT(nn.Module):
        def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
            self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
            self.proj = Linear(model_dim, vocab_size)
            self.norm1 = RMSNorm(model_dim)
            self.norm2 = RMSNorm(model_dim)

        def forward(self, inputs: Tensor, targets: Tensor):
            x = self.norm1(self.embed(inputs))
            for block in self.blocks:
                x = block(x)
            logits = self.proj(self.norm2(x)).float()
            logits = 15 * logits * (logits.square() + 15 ** 2).rsqrt()
            return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")

    model = GPT(
        vocab_size=int(architecture["vocab_size"]),
        num_layers=int(architecture["num_layers"]),
        model_dim=int(architecture["model_dim"]),
    )
    for name, parameter in model.named_parameters():
        w = parameter.data
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
            raise ValueError("uninitialized parameter: " + name)
    return model


# --- the frozen optimizer -----------------------------------------------------


def zeropower_via_newtonschulz5(G):
    """Newton-Schulz quintic iteration, as vendored in the record set."""
    import torch

    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(5):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def build_optimizers(model, bound: dict):
    """AdamW on the embedding, the head and the 1D parameters; Muon on the block matrices."""
    import torch
    from torch.optim import AdamW

    class Muon(torch.optim.Optimizer):
        def __init__(self, params, lr, weight_decay, momentum=0.95):
            super().__init__(list(params), dict(lr=lr, weight_decay=weight_decay, momentum=momentum))

        @torch.no_grad()
        def step(self):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    buf.lerp_(p.grad, 1 - group["momentum"])
                    update = p.grad.lerp_(buf, group["momentum"])
                    update = zeropower_via_newtonschulz5(update)
                    update = update * max(1, p.size(-2) / p.size(-1)) ** 0.5
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])

    adamw = AdamW(
        [
            dict(params=[model.embed.weight], lr=float(bound["adamw_embed_lr"])),
            dict(params=[model.proj.weight], lr=float(bound["adamw_head_lr"])),
            dict(params=[p for p in model.parameters() if p.ndim < 2], lr=float(bound["adamw_scalar_lr"])),
        ],
        betas=(0.8, 0.95),
        eps=1e-10,
        weight_decay=float(bound["adamw_weight_decay"]),
    )
    muon = Muon(
        [p for p in model.blocks.parameters() if p.ndim >= 2],
        lr=float(bound["learning_rate"]),
        weight_decay=float(bound["muon_weight_decay"]),
    )
    optimizers = [adamw, muon]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return optimizers


def set_hparams(optimizers, step: int, budget: int, cooldown_frac: float = 0.7) -> None:
    progress = step / budget
    eta = 1.0 if progress < 1 - cooldown_frac else (1 - progress) / cooldown_frac
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * eta


# --- snapshots ----------------------------------------------------------------


def snapshot_descriptor(model, architecture: dict, path: Path) -> dict:
    """A real parameter snapshot, written to a verifier-private path and shape-bound."""
    import torch

    state = {name: parameter.detach().to("cpu", torch.float32) for name, parameter in model.named_parameters()}
    torch.save(state, path)
    tensors = []
    for name in sorted(state):
        tensor = state[name]
        payload = tensor.contiguous().numpy().tobytes()
        tensors.append([name, [int(size) for size in tensor.shape], hashlib.sha256(payload).hexdigest()])
    return {
        "architecture": {
            "vocab_size": int(architecture["vocab_size"]),
            "num_layers": int(architecture["num_layers"]),
            "model_dim": int(architecture["model_dim"]),
            "head_dim": int(architecture["head_dim"]),
            "num_heads": int(architecture["num_heads"]),
            "seq_len": int(architecture["seq_len"]),
        },
        "tensors": tensors,
        "parameter_count": sum(int(tensor.numel()) for tensor in state.values()),
        "snapshot_path": str(path),
    }


def weights_digest(descriptor: dict) -> str:
    """The digest binding a parameter snapshot to a run, recomputable by any reader.

    Only the architecture and the per-tensor shape and content digests enter the preimage.
    The host path does not, so two runs of the same bytes on different hosts agree.
    """
    architecture = descriptor.get("architecture") or {}
    rows = [[str(key), int(architecture[key])] for key in sorted(architecture)]
    tensors = [[str(name), [int(size) for size in shape], str(digest)] for name, shape, digest in descriptor.get("tensors") or []]
    payload = json.dumps([rows, sorted(tensors)], sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --- evaluation ---------------------------------------------------------------


def held_out_loss(model, benchmark: list, enc, architecture: dict) -> float:
    """Mean per-token cross entropy of the held-out documents under these parameters."""
    import torch

    seq_len = int(architecture["seq_len"])
    vocab_size = int(architecture["vocab_size"])
    stream, _ = encode_documents(benchmark, enc, vocab_size)
    usable = (len(stream) - 1) // seq_len * seq_len
    if usable <= 0:
        raise ValueError("the held-out split tokenized to fewer than one sequence")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = torch.tensor(stream[:usable], dtype=torch.int32, device=device).view(-1, seq_len)
    targets = torch.tensor(stream[1:usable + 1], dtype=torch.int64, device=device).view(-1, seq_len)
    total = 0.0
    model.eval()
    with torch.no_grad():
        for index in range(inputs.size(0)):
            total += float(model(inputs[index:index + 1], targets[index:index + 1]))
    model.train()
    return total / float(usable)


def benchmark_score(loss: float) -> float:
    """The graded scalar: exp(-val_loss), on (0.0, 1.0], higher better.

    This is the geometric-mean per-token probability the parameters assign to held-out
    FineWeb text. It is a strictly decreasing function of the validation loss the record
    set targets, so the direction bound in task.toml is honoured without inventing a
    scale, and it is undefined when no forward pass ran because no loss exists to map.
    """
    return math.exp(-float(loss))


def evaluate(descriptor: dict, benchmark: list, relative_tolerance: float) -> float:
    """The verifier's own recomputation, from the snapshot the harness wrote.

    The parameters are reloaded from the snapshot file, the held-out documents are
    tokenized in this process, and the score is recomputed from scratch. The in-loop
    reading recorded at the same step must agree within `relative_tolerance`, so a
    snapshot that does not reproduce its own reading is a refused run rather than a grade.
    """
    import torch

    if not isinstance(descriptor, dict) or not descriptor.get("snapshot_path"):
        raise ValueError("no parameter snapshot was handed to the verifier's evaluation")
    architecture = descriptor["architecture"]
    model = build_model(architecture)
    state = torch.load(descriptor["snapshot_path"], map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    if torch.cuda.is_available():
        model = model.cuda()
    recomputed = benchmark_score(held_out_loss(model, benchmark, encoder(), architecture))
    in_loop = descriptor.get("in_loop_score")
    if in_loop is not None:
        deviation = abs(recomputed - float(in_loop))
        if deviation > float(relative_tolerance) * max(1e-12, abs(recomputed)):
            raise ValueError(
                "the snapshot does not reproduce its in-loop reading: recomputed "
                + format(recomputed, ".8f") + " against " + format(float(in_loop), ".8f")
            )
    return recomputed


def eval_schedule(budget_steps: int, fractions: list) -> list:
    """The evaluation points the verifier schedules. The submission chooses none."""
    points = []
    for fraction in fractions:
        step = int(round(float(fraction) * int(budget_steps)))
        step = max(1, min(int(budget_steps), step))
        if step not in points:
            points.append(step)
    return sorted(points)


# --- the graded run -----------------------------------------------------------


def batch_stream(synthetic, fineweb_files, bound: dict, architecture: dict):
    """One frozen batch per step: the generator's token quota plus the FineWeb remainder."""
    import numpy

    synthetic_per_step = int(bound["synthetic_tokens_per_step"])
    fineweb_per_step = int(bound["fineweb_tokens_per_step"])
    seq_len = int(architecture["seq_len"])
    synthetic_pos = 0
    file_index = 0
    fineweb = read_shard(fineweb_files[0])
    fineweb_pos = 0
    while True:
        if synthetic_pos + synthetic_per_step + 1 >= len(synthetic):
            synthetic_pos = 0
        if fineweb_pos + fineweb_per_step + 1 >= len(fineweb):
            file_index = (file_index + 1) % len(fineweb_files)
            fineweb = read_shard(fineweb_files[file_index])
            fineweb_pos = 0
        left = synthetic[synthetic_pos:synthetic_pos + synthetic_per_step + 1]
        right = fineweb[fineweb_pos:fineweb_pos + fineweb_per_step + 1]
        synthetic_pos += synthetic_per_step
        fineweb_pos += fineweb_per_step
        buffer = numpy.concatenate([left[:-1], right[:-1]]).astype(numpy.int64)
        following = numpy.concatenate([left[1:], right[1:]]).astype(numpy.int64)
        yield buffer.reshape(-1, seq_len), following.reshape(-1, seq_len)


def train(samples: list, benchmark: list, bound: dict) -> dict:
    """One frozen training run, its counters, and the verifier's own evaluations."""
    import torch

    substrate = load_substrate()
    architecture = substrate["architecture"]
    budget = int(bound["frozen_train_steps"])
    batch_tokens = int(bound["batch_tokens_per_step"])
    schedule = eval_schedule(budget, bound["eval_point_fractions"])
    tolerance = float(bound["evaluation_relative_tolerance"])

    enc = encoder()
    stream, usable = encode_documents(samples, enc, int(architecture["vocab_size"]))

    workspace = Path(tempfile.mkdtemp(prefix="oer17-run-"))
    synthetic_tokens = 0
    if stream:
        synthetic_tokens = write_shard(stream, workspace / "synthetic_train_000000.bin")

    fineweb_files = sorted(FINEWEB_TRAIN_ROOT.glob(FINEWEB_TRAIN_GLOB))
    if not fineweb_files:
        raise FileNotFoundError("no FineWeb10B train shard under " + str(FINEWEB_TRAIN_ROOT))

    steps_fed = 0
    tokens_fed = 0
    eval_points = []
    weights_by_step = {}
    halted_at_step = 0
    final_descriptor = {}

    if usable <= 0:
        # No usable document means no shard, so no mixed batch and no forward pass. The
        # run establishes no score, and grade.py attributes that zero by reason.
        return {
            "optimizer": str(bound["optimizer_id"]),
            "learning_rate": float(bound["learning_rate"]),
            "steps_fed": 0,
            "tokens_fed": 0,
            "corpus_samples": len(samples),
            "usable_samples": 0,
            "synthetic_tokens": 0,
            "halted_at_step": 0,
            "completed": False,
            "checkpoint_selected_by": "harness",
            "weights_by_step": {},
            "eval_points": [],
            "final_weights": {},
            "schedule": schedule,
        }

    model = build_model(architecture)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizers = build_optimizers(model, bound)
    batches = batch_stream(read_shard(workspace / "synthetic_train_000000.bin"), fineweb_files, bound, architecture)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for step in range(1, budget + 1):
        set_hparams(optimizers, step - 1, budget)
        inputs, targets = next(batches)
        inputs = torch.from_numpy(inputs).to(device=device, dtype=torch.int32)
        targets = torch.from_numpy(targets).to(device=device, dtype=torch.int64)
        # Exactly one forward and exactly one backward pass per step, as the frozen axis
        # requires. Delete either and there is no loss, so there is no score.
        loss = model(inputs, targets)
        loss.backward()
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
        steps_fed += 1
        tokens_fed += batch_tokens
        halted_at_step = step
        if step in schedule:
            descriptor = snapshot_descriptor(model, architecture, workspace / ("snapshot_step" + str(step) + ".pt"))
            score = benchmark_score(held_out_loss(model, benchmark, enc, architecture))
            descriptor["in_loop_score"] = score
            digest = weights_digest(descriptor)
            weights_by_step[str(step)] = digest
            eval_points.append(
                {
                    "step": step,
                    "raw_score": score,
                    "smoothed_score": None,
                    "weights_sha256": digest,
                }
            )
            final_descriptor = descriptor

    return {
        "optimizer": str(bound["optimizer_id"]),
        "learning_rate": float(bound["learning_rate"]),
        "steps_fed": steps_fed,
        "tokens_fed": tokens_fed,
        "corpus_samples": len(samples),
        "usable_samples": usable,
        "synthetic_tokens": synthetic_tokens,
        "halted_at_step": halted_at_step,
        "completed": halted_at_step == budget,
        "checkpoint_selected_by": "harness",
        "weights_by_step": weights_by_step,
        "eval_points": eval_points,
        "final_weights": final_descriptor,
        "schedule": schedule,
        "_tolerance": tolerance,
    }


def digest_tree(root: Path, skip: tuple = ("__pycache__",)) -> tuple:
    """Path-independent digest over a directory, plus the file count behind it."""
    rows = []
    for entry in sorted(Path(root).rglob("*")):
        relative = entry.relative_to(root)
        if set(relative.parts) & set(skip):
            continue
        if entry.is_file():
            rows.append([relative.as_posix(), hashlib.sha256(entry.read_bytes()).hexdigest()])
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), len(rows)
