#!/usr/bin/env python3
"""The frozen trainer. The HARNESS runs this; a submission never does.

This is the real nanoGPT decoder, not a description of one. It instantiates the
canonical 12-layer, 768-dimension architecture declared in
environment/nanogpt_substrate.json, trains it on the feed the pipeline produced with
exactly one forward and one backward pass per step, and writes a real parameter
snapshot of that architecture at the bound evaluation point and at every sustain point.
The architecture body mirrors the vendored record script at
harness/records/track_3_optimization/train_gpt_simple.py: RMSNorm, half-truncate rotary
embeddings, relu-squared feedforward, a softcapped output projection, and a token-summed
cross entropy.

Every axis this script reads comes from environment/frozen_recipe.json and
environment/nanogpt_substrate.json, and none of them is a submission input. The one
thing that varies between runs is the feed the pipeline produced, which is exactly the
free surface of this slot. If the two files disagree on any architecture, batch or
corpus value, this script refuses to run rather than training something nobody froze.

This file is agent-visible so the agent can read what is frozen. It is not a file the
agent edits: the harness digests the frozen axes at open and at close, and a run whose
frozen axes moved is graded as having mutated a frozen axis rather than as an
improvement.

THIS SCRIPT NEVER COMPUTES THE GRADED NUMBER. It never opens a held-out shard, and it
refuses to start if one is reachable from the container it runs in. What it emits on
stdout is telemetry: a training loss the run can watch and nothing that scores anything.
The verifier loads the parameter snapshots this script wrote and evaluates them itself,
unsmoothed, on its own copy of the held-out FineWeb10B validation shards, at the bound
evaluation point and at every sustain point the harness schedules. Delete the forward
and backward pass below and no snapshot is written, so the graded loss becomes undefined
rather than merely different.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
RECIPE = HERE / "frozen_recipe.json"
SUBSTRATE = HERE / "nanogpt_substrate.json"

SNAPSHOT_DIR = pathlib.Path(os.environ.get("OER12_SNAPSHOT_DIR", "/logs/harness/snapshots"))
MICROBATCH_SEQUENCES = 64

REFUSAL_AXIS_DISAGREES = "frozen-axis-moved"
REFUSAL_HELD_OUT_VISIBLE = "held-out-split-visible-in-container"
REFUSAL_FEED_ABSENT = "feed-absent"
REFUSAL_FEED_EXHAUSTED = "feed-exhausted-before-bound-schedule"
REFUSAL_NO_SNAPSHOT = "no-snapshot-at-bound-eval-step"


def load_recipe() -> dict:
    return json.loads(RECIPE.read_text(encoding="utf-8"))


def load_substrate() -> dict:
    return json.loads(SUBSTRATE.read_text(encoding="utf-8"))


def refuse(reason: str, detail: str) -> int:
    print(json.dumps({"trainer": "frozen", "refused": reason, "detail": detail},
                     sort_keys=True), file=sys.stderr)
    return 3


def check_axes_against_substrate(axes: dict, substrate: dict) -> str:
    """Every frozen architecture and batch value must equal the substrate's own."""
    model = axes["model"]
    architecture = substrate["architecture"]
    for key in ("vocab_size", "num_layers", "model_dim", "head_dim", "num_heads", "seq_len"):
        if model.get(key) != architecture.get(key):
            return ("model." + key + " is " + repr(model.get(key))
                    + " while the substrate declares " + repr(architecture.get(key)))
    budget = axes["token_budget"]
    run = substrate["run"]
    for key in ("batch_tokens_per_step", "forward_passes_per_step", "backward_passes_per_step"):
        if budget.get(key) != run.get(key):
            return ("token_budget." + key + " is " + repr(budget.get(key))
                    + " while the substrate declares " + repr(run.get(key)))
    derived = int(axes["max_steps"]) * int(budget["batch_tokens_per_step"])
    if int(budget["tokens"]) != derived:
        return ("token_budget.tokens is " + repr(budget["tokens"])
                + " while max_steps times batch_tokens_per_step is " + repr(derived))
    corpus = axes["corpus"]
    for key, name in (("train_glob", "train_glob"), ("held_out_glob", "val_glob")):
        if corpus.get(key) != substrate["corpus"].get(name):
            return ("corpus." + key + " is " + repr(corpus.get(key))
                    + " while the substrate declares " + repr(substrate["corpus"].get(name)))
    return ""


def held_out_shards_visible(axes: dict) -> list:
    """The graded split must be unreachable from here. Its presence is a refusal."""
    pattern = axes["corpus"]["held_out_glob"]
    roots = [pathlib.Path.cwd(), pathlib.Path("/app"), pathlib.Path("/workspace"), HERE]
    found = []
    for root in roots:
        found.extend(sorted(glob.glob(str(root / pattern))))
    return sorted(set(found))


def build_model(model_axes: dict, seed: int):
    """The canonical decoder. Shape-bound to the declared architecture, nothing less."""
    import torch
    from torch import nn
    import torch.nn.functional as functional

    torch.manual_seed(int(seed))

    class RMSNorm(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.gains = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            return functional.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))

    class Linear(nn.Linear):
        def __init__(self, in_features: int, out_features: int):
            super().__init__(in_features, out_features, bias=True)

        def forward(self, x):
            return functional.linear(x, self.weight.type_as(x), self.bias.type_as(x))

    class Rotary(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
            self.register_buffer("angular_freq",
                                 torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)]))

        def forward(self, x_BTHD):
            pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
            theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
            cos, sin = theta.cos(), theta.sin()
            x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
            return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), 3).type_as(x_BTHD)

    class CausalSelfAttention(nn.Module):
        def __init__(self, dim: int, head_dim: int):
            super().__init__()
            self.num_heads = dim // head_dim
            self.head_dim = head_dim
            hdim = self.num_heads * self.head_dim
            self.q = Linear(dim, hdim)
            self.k = Linear(dim, hdim)
            self.v = Linear(dim, hdim)
            self.proj = Linear(hdim, dim)
            self.rotary = Rotary(head_dim)

        def forward(self, x):
            batch, time = x.size(0), x.size(1)
            q = self.q(x).view(batch, time, self.num_heads, self.head_dim)
            k = self.k(x).view(batch, time, self.num_heads, self.head_dim)
            v = self.v(x).view(batch, time, self.num_heads, self.head_dim)
            q = functional.rms_norm(q, (q.size(-1),))
            k = functional.rms_norm(k, (k.size(-1),))
            q, k = self.rotary(q), self.rotary(k)
            y = functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                scale=0.12, is_causal=True).transpose(1, 2)
            y = y.contiguous().view(batch, time, self.num_heads * self.head_dim)
            return self.proj(y)

    class MLP(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.fc = Linear(dim, 4 * dim)
            self.proj = Linear(4 * dim, dim)

        def forward(self, x):
            return self.proj(self.fc(x).relu().square())

    class Block(nn.Module):
        def __init__(self, dim: int, head_dim: int):
            super().__init__()
            self.attn = CausalSelfAttention(dim, head_dim)
            self.mlp = MLP(dim)
            self.norm1 = RMSNorm(dim)
            self.norm2 = RMSNorm(dim)

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            return x + self.mlp(self.norm2(x))

    class GPT(nn.Module):
        def __init__(self, vocab_size: int, num_layers: int, model_dim: int, head_dim: int):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
            self.blocks = nn.ModuleList([Block(model_dim, head_dim) for _ in range(num_layers)])
            self.proj = Linear(model_dim, vocab_size)
            self.norm1 = RMSNorm(model_dim)
            self.norm2 = RMSNorm(model_dim)

        def forward(self, inputs, targets):
            x = self.norm1(self.embed(inputs))
            for block in self.blocks:
                x = block(x)
            logits = self.proj(self.norm2(x)).float()
            logits = 15 * logits * (logits.square() + 15 ** 2).rsqrt()
            return functional.cross_entropy(
                logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")

    return GPT(int(model_axes["vocab_size"]), int(model_axes["num_layers"]),
               int(model_axes["model_dim"]), int(model_axes["head_dim"]))


def feed_batches(feed_path: pathlib.Path, batch_tokens: int, seq_len: int):
    """Encode the pipeline's admitted feed with the FROZEN encoder and yield step batches.

    The vocabulary is frozen with the model, so the encoder is the substrate's GPT-2 BPE
    and the pipeline does not author one. What the pipeline decides is which documents
    reach this stream, in what order, and in what form, which is exactly the parse
    surface this slot grades. Tokens are consumed once, in order: nothing is repeated to
    pad a short feed, so a feed that runs out before the bound schedule ends halts the
    run instead of quietly recycling itself.
    """
    import tiktoken

    encoder = tiktoken.get_encoding("gpt2")
    buffer: list = []
    need = batch_tokens + 1
    with feed_path.open("r", encoding="utf-8", errors="strict") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if chunk:
                buffer.extend(encoder.encode_ordinary(chunk))
            while len(buffer) >= need:
                window = buffer[:need]
                del buffer[:batch_tokens]
                yield window
            if not chunk:
                return


def snapshot(model, step: int, target: pathlib.Path) -> dict:
    """Write a real parameter snapshot and return its shape-bound digest row."""
    import torch

    target.mkdir(parents=True, exist_ok=True)
    path = target / ("step-%06d.pt" % step)
    state = {name: tensor.detach().to("cpu") for name, tensor in model.state_dict().items()}
    torch.save(state, str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "step": step,
        "path": path.name,
        "digest": digest,
        "parameter_count": sum(tensor.numel() for tensor in state.values()),
    }


def main(argv: list) -> int:
    recipe = load_recipe()
    substrate = load_substrate()
    axes = recipe["frozen_axes"]

    disagreement = check_axes_against_substrate(axes, substrate)
    if disagreement:
        return refuse(REFUSAL_AXIS_DISAGREES, disagreement)

    visible = held_out_shards_visible(axes)
    if visible:
        return refuse(REFUSAL_HELD_OUT_VISIBLE,
                      "the graded split must be absent from this container, found: "
                      + ", ".join(visible))

    if len(argv) < 2:
        print("usage: train_frozen.py <feed.txt>", file=sys.stderr)
        print("frozen axes: " + json.dumps(axes, sort_keys=True), file=sys.stderr)
        return 2
    feed = pathlib.Path(argv[1])
    if not feed.is_file():
        return refuse(REFUSAL_FEED_ABSENT, str(feed))

    import torch

    model_axes = axes["model"]
    budget = axes["token_budget"]
    evaluation = axes["evaluation"]
    optimizer_axes = axes["optimizer"]
    max_steps = int(axes["max_steps"])
    seq_len = int(model_axes["seq_len"])
    batch_tokens = int(budget["batch_tokens_per_step"])
    snapshot_steps = sorted({int(evaluation["bound_eval_step"])}
                            | {int(step) for step in evaluation["sustain_points"]})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(model_axes, axes["seed"]).to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_axes["lr"]),
        betas=tuple(float(value) for value in optimizer_axes["betas"]),
        weight_decay=float(optimizer_axes["weight_decay"]),
    )
    warmup = int(optimizer_axes["warmup_steps"])

    def learning_rate(step: int) -> float:
        base = float(optimizer_axes["lr"])
        if step < warmup:
            return base * (step + 1) / warmup
        progress = (step - warmup) / max(1, max_steps - warmup)
        return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    batches = feed_batches(feed, batch_tokens, seq_len)
    manifest: list = []
    forward_passes = 0
    backward_passes = 0
    fed_tokens = 0
    halt_kind = "bound-schedule-complete"
    halted_at_step = 0

    for step in range(max_steps + 1):
        # A snapshot exists only where the passes that produced it exist. The counters
        # are incremented by the forward and backward block below and nowhere else, so
        # a run with that block deleted reaches the bound evaluation point having
        # written nothing, and the graded loss is undefined rather than merely worse.
        if step in snapshot_steps and forward_passes == step and backward_passes == step:
            manifest.append(snapshot(model, step, SNAPSHOT_DIR))
        halted_at_step = step
        if step == max_steps:
            break

        window = next(batches, None)
        if window is None:
            halt_kind = REFUSAL_FEED_EXHAUSTED
            break

        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)

        # THE FORWARD AND BACKWARD PASS. One of each per step, over the whole step
        # batch, accumulated across microbatches exactly as the vendored record script
        # does. Delete this block and no parameter ever moves, no snapshot after the
        # first differs from initialization, and the verifier has nothing to grade.
        inputs = torch.tensor(window[:-1], dtype=torch.int32, device=device).view(-1, seq_len)
        targets = torch.tensor(window[1:], dtype=torch.int64, device=device).view(-1, seq_len)
        step_loss = 0.0
        for start in range(0, inputs.size(0), MICROBATCH_SEQUENCES):
            stop = start + MICROBATCH_SEQUENCES
            loss = model(inputs[start:stop], targets[start:stop])
            loss.backward()
            step_loss += float(loss.detach())
        forward_passes += 1
        backward_passes += 1
        fed_tokens += batch_tokens

        optimizer.step()
        model.zero_grad(set_to_none=True)

        print(json.dumps({"trainer": "frozen", "step": step + 1,
                          "train_loss_telemetry": step_loss / batch_tokens,
                          "fed_tokens": fed_tokens,
                          "note": "telemetry only, never the graded number"},
                         sort_keys=True), flush=True)

    if not any(row["step"] == int(evaluation["bound_eval_step"]) for row in manifest):
        halt_kind = REFUSAL_NO_SNAPSHOT

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    (SNAPSHOT_DIR / "manifest.json").write_text(
        json.dumps({
            "trainer": "frozen",
            "architecture": {key: model_axes[key] for key in
                             ("vocab_size", "num_layers", "model_dim", "head_dim",
                              "num_heads", "seq_len")},
            "snapshots": manifest,
            "halt_kind": halt_kind,
            "halted_at_step": halted_at_step,
            "forward_passes": forward_passes,
            "backward_passes": backward_passes,
            "fed_tokens": fed_tokens,
            "budget_tokens": int(budget["tokens"]),
            "graded_by": "the verifier, on its own held-out split, from these snapshots",
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"trainer": "frozen", "snapshots": len(manifest),
                      "halt_kind": halt_kind, "halted_at_step": halted_at_step,
                      "forward_passes": forward_passes,
                      "backward_passes": backward_passes,
                      "note": "no loss here is the grade; the verifier evaluates the snapshots"},
                     sort_keys=True))
    return 0 if halt_kind == "bound-schedule-complete" else 4


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
