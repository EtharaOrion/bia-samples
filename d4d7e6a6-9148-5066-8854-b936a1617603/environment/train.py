#!/usr/bin/env python3
"""Frozen training entry point for OER-09.

This script trains the canonical nanoGPT decoder declared in
environment/nanogpt_substrate.json over the FineWeb10B train shards, under the frozen
budget in environment/frozen_config.yaml, on the corpus the curated pool admits. The
architecture, the batch size, the corpus and the one-forward-backward-per-step rule are
frozen. The only thing a submission changes is the pool handed to `--pool`.

How curation reaches the corpus. The trainer walks the train shards in document order,
splitting on the GPT-2 end-of-text id, and admits corpus document i when register row
(i mod 4096) survived the filter chain. Shedding rows sheds corpus documents in the same
proportion, so a pool that did not move feeds the same tokens the uncurated pool feeds.
A row carrying `holdout` true is reserved by the verifier: admitting its stratum puts
reserved material in the fed stream and is graded `eval-split-leaked`.

What this script does NOT do, on purpose:

  * it does not decide the graded loss. The verifier recomputes the graded validation
    loss itself on the verifier-owned held-out FineWeb split, from harness-owned weights,
    at the bound evaluation point, unsmoothed. That split is absent from this container.
    Any loss this script prints is for the operator's eyes.
  * it does not choose which checkpoint is evaluated. It writes one parameter snapshot at
    each point the schedule names, under names the schedule fixes, and the harness digests
    and ledgers them. A snapshot written anywhere else is never read as graded state.
  * it does not decide when to stop. The evaluation schedule is the verifier's; halting
    early is graded as not having established the loss.

As it feeds, the harness records its own observation of the pool, the feed ledger and the
held-out leak audit under /logs/harness/. Those records are the graded state. This script
writes none of them.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "frozen_config.yaml"
SUBSTRATE = HERE / "nanogpt_substrate.json"
REGISTER = HERE / "pool" / "source_register.jsonl"
SNAPSHOTS = Path("/logs/harness/weights")

HEADER_BYTES = 1024
EOT = 50256


# ---------------------------------------------------------------------------------------
# The frozen declarations, and the refusal that binds them together.
# ---------------------------------------------------------------------------------------
def load_config() -> dict:
    import yaml

    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def load_substrate() -> dict:
    return json.loads(SUBSTRATE.read_text(encoding="utf-8"))


def bind_to_substrate(config: dict, substrate: dict) -> None:
    """Refuse to train when the declared architecture is not the canonical one."""
    arch = substrate["architecture"]
    model = config["model"]
    pairs = [
        ("vocab", "vocab_size"),
        ("layers", "num_layers"),
        ("d_model", "model_dim"),
        ("head_dim", "head_dim"),
        ("heads", "num_heads"),
        ("context", "seq_len"),
    ]
    for local, canonical in pairs:
        if int(model[local]) != int(arch[canonical]):
            raise SystemExit(
                "frozen-axis-moved: frozen_config model."
                + local
                + " is "
                + repr(model[local])
                + " and the substrate declares "
                + canonical
                + " "
                + repr(arch[canonical])
            )
    if int(config["budget"]["tokens_per_step"]) != int(substrate["run"]["batch_tokens_per_step"]):
        raise SystemExit("frozen-axis-moved: tokens_per_step is not the canonical batch")
    if int(substrate["run"]["forward_passes_per_step"]) != 1:
        raise SystemExit("frozen-axis-moved: the substrate does not declare one forward pass per step")


# ---------------------------------------------------------------------------------------
# The canonical decoder, shape for shape with the vendored record set's train_gpt_simple.py.
# ---------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))


class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)]))

    def forward(self, x_BTHD: Tensor) -> Tensor:
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int = 128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor) -> Tensor:
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=0.12, is_causal=True
        ).transpose(1, 2)
        return self.proj(y.contiguous().view(B, T, self.num_heads * self.head_dim))


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = Linear(dim, 4 * dim)
        self.proj = Linear(4 * dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.fc(x).relu().square())


class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
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

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")


def initialise(model: nn.Module, seed: int) -> None:
    torch.manual_seed(seed)
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            if "proj" in name:
                w.zero_()
            elif "embed" in name:
                w.normal_()
            else:
                w.normal_(std=0.33**0.5 / w.size(-1) ** 0.5)
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)
        else:
            raise SystemExit("uninitialised parameter: " + name)


# ---------------------------------------------------------------------------------------
# The corpus, admitted by the curated pool.
# ---------------------------------------------------------------------------------------
def load_pool(path: Path) -> list:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def admitted_rows(curated: list, register: list) -> set:
    """The register row indices the curated pool retained, by doc_id."""
    order = {str(row.get("doc_id")): index for index, row in enumerate(register)}
    return {order[str(row.get("doc_id"))] for row in curated if str(row.get("doc_id")) in order}


def shard_paths(config: dict) -> list:
    root = Path(config["corpus"]["container_train_path"])
    paths = sorted(root.glob("fineweb_train_*.bin"))
    if not paths:
        raise SystemExit(
            "no FineWeb10B train shard is staged at "
            + str(root)
            + "; the loader is "
            + str(config["corpus"]["loader"])
        )
    return paths


def stream(config: dict, admitted: set, rows: int, seq_len: int, budget: int):
    """Yield (seq_len + 1) token windows drawn once from the admitted corpus documents."""
    carry = np.empty(0, dtype=np.uint16)
    fed = 0
    document = 0
    for path in shard_paths(config):
        tokens = np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_BYTES)
        breaks = np.flatnonzero(np.asarray(tokens[:] == EOT))
        start = 0
        for stop in breaks:
            piece = tokens[start : stop + 1]
            start = stop + 1
            if (document % rows) in admitted:
                carry = np.concatenate([carry, np.asarray(piece, dtype=np.uint16)])
                while carry.size >= seq_len + 1 and fed < budget:
                    yield torch.from_numpy(carry[: seq_len + 1].astype(np.int64))
                    carry = carry[seq_len:]
                    fed += seq_len
            document += 1
            if fed >= budget:
                return


# ---------------------------------------------------------------------------------------
# The run.
# ---------------------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="train the frozen decoder on a curated pool")
    parser.add_argument("--pool", required=True)
    parser.add_argument("--micro-batch", type=int, default=64)
    args = parser.parse_args(argv)

    config = load_config()
    substrate = load_substrate()
    bind_to_substrate(config, substrate)

    budget = int(config["budget"]["token_budget_tokens"])
    per_step = int(config["budget"]["tokens_per_step"])
    steps = int(config["budget"]["steps"])
    seq_len = int(config["model"]["context"])
    schedule = [int(config["evaluation"]["bound_point"])] + [
        int(point) for point in config["evaluation"]["sustain_points"]
    ]

    curated = load_pool(Path(args.pool))
    if not curated:
        print("refusing to train on an empty pool", file=sys.stderr)
        return 2
    register = load_pool(REGISTER)
    admitted = admitted_rows(curated, register)
    if not admitted:
        print("refusing to train: no curated document resolves to a register row", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print("refusing to train: no accelerator is present", file=sys.stderr)
        return 2
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    model = GPT(
        vocab_size=int(config["model"]["vocab"]),
        num_layers=int(config["model"]["layers"]),
        model_dim=int(config["model"]["d_model"]),
        head_dim=int(config["model"]["head_dim"]),
    ).to(device)
    initialise(model, int(config["model"]["init_seed"]))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["lr"]),
        betas=tuple(float(b) for b in config["optimizer"]["betas"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
        fused=True,
    )
    warmup = int(config["optimizer"]["warmup_steps"])
    clip = float(config["optimizer"]["grad_clip"])

    print("pool documents: " + str(len(curated)))
    print("admitted register rows: " + str(len(admitted)) + " of " + str(len(register)))
    print("frozen token budget: " + str(budget) + " tokens, " + str(steps) + " steps")
    print(
        "model: "
        + str(config["model"]["arch"])
        + ", layers="
        + str(config["model"]["layers"])
        + ", d_model="
        + str(config["model"]["d_model"])
        + ", vocab="
        + str(config["model"]["vocab"])
    )
    print("optimizer: " + str(config["optimizer"]["name"]))
    print("evaluation split: " + str(config["evaluation"]["split_id"]) + ", owned by the verifier")
    print("verifier evaluation points: bound=" + str(schedule[0]) + " sustain=" + str(schedule[1:]))
    print("")

    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    micro = max(1, int(args.micro_batch))
    per_micro = micro * seq_len
    if per_step % per_micro:
        print("refusing to train: the batch does not divide into whole micro-batches", file=sys.stderr)
        return 2
    accumulation = per_step // per_micro

    windows = stream(config, admitted, len(register), seq_len, budget)
    fed = 0
    for step in range(1, steps + 1):
        # One forward and one backward pass over the step's batch, accumulated over
        # micro-batches exactly as the vendored record set does at batch_size 8 * 64 * 1024.
        optimizer.zero_grad(set_to_none=True)
        taken = 0
        for _ in range(accumulation):
            batch = []
            for _ in range(micro):
                try:
                    batch.append(next(windows))
                except StopIteration:
                    break
            if not batch:
                break
            block = torch.stack(batch).to(device, non_blocking=True)
            inputs = block[:, :-1].contiguous()
            targets = block[:, 1:].contiguous()
            loss = model(inputs, targets)
            (loss / per_step).backward()
            taken += len(batch) * seq_len
        if taken == 0:
            print("the admitted corpus was exhausted before the schedule completed", file=sys.stderr)
            return 3
        fed += taken

        progress = min(1.0, step / max(1, warmup)) if step <= warmup else 1.0
        cosine = 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
        scale = progress if step <= warmup else cosine
        for group in optimizer.param_groups:
            group["lr"] = float(config["optimizer"]["lr"]) * scale
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if step in schedule:
            # A parameter snapshot of the frozen architecture, at a point the schedule
            # named. The submission chooses neither the point nor the name.
            torch.save(
                {"step": step, "state_dict": model.state_dict()},
                SNAPSHOTS / ("snapshot-{:06d}.pt".format(step)),
            )

    print("tokens fed: " + str(fed))
    print("snapshots written at: " + str(schedule))
    print("")
    print("the graded validation loss is recomputed by the verifier and is not printed here")
    return 0


if __name__ == "__main__":
    sys.exit(main())
