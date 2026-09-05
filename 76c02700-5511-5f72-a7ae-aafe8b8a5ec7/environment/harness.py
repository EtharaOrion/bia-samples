#!/usr/bin/env python3
"""Agent-side simulator for the OER-08 data-curation substrate.

This is the copy you iterate against. It reads the same pool spec the graded
harness reads, it builds the same frozen nanoGPT decoder the graded harness
builds, and it trains it on the tokens your plan names, so you can measure a
mixture locally before you submit it. It then evaluates on the agent-visible
`dev_probe`, which is a slice of a FineWeb10B TRAIN shard that carries no pool
slice.

It does NOT hold the graded evaluation split and it does not produce the graded
number. The verifier owns its own trainer, its own parameter snapshot and its
own held-out validation slice, recomputes the loss from the snapshot that
trainer produced, and grades that.

The model here is the real one: vocab_size 50304, num_layers 12, model_dim 768,
head_dim 128, num_heads 6, seq_len 1024, read from
environment/nanogpt_substrate.json and cross-checked against
environment/corpus_spec.json. Every graded step is one forward pass and one
backward pass over 524288 tokens. Delete the forward and backward pass and there
is no loss to report, because the loss is a function of the parameters those
passes produced and of nothing else.

Nothing here reads a clock, a network or an unseeded random source. Every
quantity is a deterministic function of the shard bytes, the pool spec, the plan
and the frozen seed.

Usage:
    python3 harness.py --plan plan.json                 # train and score on dev_probe
    python3 harness.py --from-mixture mixture.yaml      # turn the default template into a plan
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent

SEED = 1337
MICRO_SEQUENCES = 32


# --------------------------------------------------------------------------
# substrate


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def architecture(corpus: dict, substrate: dict) -> dict:
    """The frozen architecture, taken from the substrate and cross-checked.

    The substrate declaration is canonical. corpus_spec.json replicates it so a
    drifted pool spec cannot quietly train a different model, and the two are
    required to agree.
    """
    declared = substrate["architecture"]
    mirrored = corpus["substrate"]["architecture"]
    for key in ("vocab_size", "num_layers", "model_dim", "head_dim", "num_heads", "seq_len"):
        if int(declared[key]) != int(mirrored[key]):
            raise ValueError(
                "frozen architecture disagrees between nanogpt_substrate.json and "
                "corpus_spec.json at " + key + ": "
                + str(declared[key]) + " against " + str(mirrored[key])
            )
    return {key: int(declared[key]) for key in declared if not key.startswith("_")}


def budget_tokens(corpus: dict) -> int:
    """The frozen token budget a graded run feeds, exactly.

    `budget_tokens` is canonical and `total_tokens` is the same quantity under its
    earlier name. Both are read and required to agree, so the alias cannot drift
    into a second budget that this simulator and the graded harness disagree on.
    """
    budget = corpus["budget"]
    value = budget["budget_tokens"]
    legacy = budget.get("total_tokens", value)
    if int(legacy) != int(value):
        raise ValueError(
            "corpus_spec.json carries two disagreeing budgets: budget_tokens="
            + str(value) + " total_tokens=" + str(legacy)
        )
    return int(value)


def data_root(corpus: dict) -> Path:
    return Path(os.environ.get("OER08_DATA", corpus["corpus"]["container_train_path"]))


def read_tokens(root: Path, shard: str, offset: int, count: int) -> np.ndarray:
    """Read a contiguous token range from a FineWeb10B shard, header skipped."""
    header = int(64 * 4)
    path = root / shard
    array = np.memmap(path, dtype=np.uint16, mode="r", offset=header)
    window = np.asarray(array[offset:offset + count], dtype=np.int64)
    if window.shape[0] != count:
        raise ValueError(
            "shard " + shard + " holds " + str(window.shape[0])
            + " tokens at offset " + str(offset) + ", not the " + str(count) + " the pool declares"
        )
    return window


def slice_digest(tokens: np.ndarray) -> str:
    return hashlib.sha256(tokens.astype("<u2").tobytes()).hexdigest()


def rare_fraction(tokens: np.ndarray) -> float:
    return float((tokens >= 20000).sum()) / float(tokens.shape[0])


def pool_index(corpus: dict, root: Path) -> dict:
    """Every pool slice by id, carrying its shard, offset, band and content digest.

    Band membership is recomputed here from the shard bytes by the rule
    corpus_spec.json states. No band membership is transcribed into any file, so
    the pool a plan names is the pool the bytes describe.
    """
    pool = corpus["pool"]
    width = int(pool["slice_tokens"])
    per_shard = int(pool["slices_per_shard"])
    rows = {}
    for shard in corpus["corpus"]["shards"]:
        stem = shard.replace(".bin", "")
        for index in range(per_shard):
            tokens = read_tokens(root, shard, index * width, width)
            rows[stem + ":" + str(index)] = {
                "id": stem + ":" + str(index),
                "shard": shard,
                "offset": index * width,
                "tokens": width,
                "rare_fraction": rare_fraction(tokens),
                "digest": slice_digest(tokens),
            }
    order = sorted(rows, key=lambda ident: (rows[ident]["rare_fraction"], rows[ident]["shard"], rows[ident]["offset"]))
    group = len(order) // int(pool["bands"]["count"])
    for position, ident in enumerate(order):
        rows[ident]["band"] = pool["bands"]["ids"][min(position // group, int(pool["bands"]["count"]) - 1)]
    return rows


def band_members(index: dict, band: str) -> list:
    return sorted(
        (ident for ident in index if index[ident]["band"] == band),
        key=lambda ident: (index[ident]["shard"], index[ident]["offset"]),
    )


# --------------------------------------------------------------------------
# the frozen decoder


class SelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int):
        super().__init__()
        self.heads = dim // head_dim
        self.head_dim = head_dim
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        batch, time, dim = x.shape
        q, k, v = self.qkv(x).split(dim, dim=2)
        shape = (batch, time, self.heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).contiguous().view(batch, time, dim))


class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int):
        super().__init__()
        self.norm_attention = nn.LayerNorm(dim, bias=False)
        self.attention = SelfAttention(dim, head_dim)
        self.norm_mlp = nn.LayerNorm(dim, bias=False)
        self.up = nn.Linear(dim, 4 * dim, bias=False)
        self.down = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        x = x + self.attention(self.norm_attention(x))
        return x + self.down(F.gelu(self.up(self.norm_mlp(x))))


class Decoder(nn.Module):
    """The frozen nanoGPT decoder. Its shape is the graded architecture."""

    def __init__(self, arch: dict):
        super().__init__()
        self.arch = arch
        dim = arch["model_dim"]
        self.embed = nn.Embedding(arch["vocab_size"], dim)
        self.position = nn.Embedding(arch["seq_len"], dim)
        self.blocks = nn.ModuleList([Block(dim, arch["head_dim"]) for _ in range(arch["num_layers"])])
        self.norm = nn.LayerNorm(dim, bias=False)
        self.head = nn.Linear(dim, arch["vocab_size"], bias=False)

    def forward(self, tokens):
        position = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.embed(tokens) + self.position(position)[None]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def build(arch: dict, device: str) -> Decoder:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    return Decoder(arch).to(device)


def parameter_shapes(model: Decoder) -> dict:
    return {name: list(tensor.shape) for name, tensor in sorted(model.state_dict().items())}


def parameter_digest(model: Decoder) -> str:
    running = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        running.update(name.encode("ascii"))
        running.update(tensor.detach().to(torch.float32).cpu().numpy().tobytes())
    return running.hexdigest()


# --------------------------------------------------------------------------
# feed and train


def expand(corpus: dict, index: dict, plan: dict) -> list:
    """Turn a plan into the ordered list of slice ids the trainer will feed."""
    budget = budget_tokens(corpus)
    width = int(corpus["pool"]["slice_tokens"])
    mode = plan.get("mode")
    order = []
    if mode == "constant":
        weights = plan.get("weights") or {}
        for band in sorted(weights):
            weight = float(weights[band])
            if weight <= 0.0:
                continue
            members = band_members(index, band)
            if not members:
                raise KeyError("plan names a band outside the pool: " + str(band))
            instances = int(round(weight * budget / width))
            for position in range(instances):
                order.append(members[position % len(members)])
    elif mode == "schedule":
        for draw in plan.get("draws") or []:
            ident = str(draw.get("slice"))
            if ident not in index:
                raise KeyError("plan names a slice outside the pool: " + ident)
            repeat = int(draw.get("repeat", 1))
            if repeat < 1:
                raise ValueError("repeat below one is not a draw: " + str(draw))
            order.extend([ident] * repeat)
    else:
        raise ValueError("plan mode outside the grammar: " + repr(mode))
    return order


def batches(root: Path, index: dict, order: list, seq_len: int):
    """Yield one (inputs, targets) tensor pair per fed slice, in plan order."""
    for ident in order:
        row = index[ident]
        tokens = read_tokens(root, row["shard"], row["offset"], row["tokens"])
        block = torch.from_numpy(tokens).view(-1, seq_len)
        yield ident, block[:, :-1].contiguous(), block[:, 1:].contiguous()


def train(corpus: dict, arch: dict, index: dict, order: list, root: Path, device: str) -> dict:
    """Train the frozen decoder on the fed slices. One forward and one backward per step.

    A step is `tokens_per_step` fed tokens. The gradient over those tokens is
    accumulated across micro-sequences and applied once, so the run performs
    exactly one forward pass and one backward pass over each step's batch, which
    is the substrate's frozen rule.
    """
    model = build(arch, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
    width = int(corpus["pool"]["slice_tokens"])
    per_step = int(corpus["budget"]["tokens_per_step"])
    slices_per_step = per_step // width
    steps_planned = max(1, len(order) // slices_per_step)
    ledger = []
    fed = 0
    steps = 0
    pending = []
    model.train()
    for ident, inputs, targets in batches(root, index, order, arch["seq_len"]):
        ledger.append({"index": len(ledger), "slice": ident, "band": index[ident]["band"], "tokens": width})
        fed += width
        pending.append((inputs, targets))
        if len(pending) < slices_per_step:
            continue
        rate = 6e-4 * 0.5 * (1.0 + math.cos(math.pi * steps / steps_planned))
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        rows = sum(chunk[0].shape[0] for chunk in pending)
        for inputs_chunk, targets_chunk in pending:
            for start in range(0, inputs_chunk.shape[0], MICRO_SEQUENCES):
                x = inputs_chunk[start:start + MICRO_SEQUENCES].to(device)
                y = targets_chunk[start:start + MICRO_SEQUENCES].to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits = model(x)
                    loss = F.cross_entropy(logits.float().view(-1, arch["vocab_size"]), y.reshape(-1))
                (loss * (x.shape[0] / rows)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        steps += 1
        pending = []
    return {"model": model, "ledger": ledger, "tokens_fed": fed, "steps": steps}


@torch.no_grad()
def evaluate(model: Decoder, tokens: np.ndarray, seq_len: int, device: str) -> dict:
    """One forward pass of the trained snapshot. Returns the raw accumulators.

    The scalar is `nats_sum / token_count` and nothing else. Both accumulators
    travel with it so a reader recomputes the scalar rather than trusting it.
    """
    model.eval()
    block = torch.from_numpy(tokens[: (tokens.shape[0] // seq_len) * seq_len]).view(-1, seq_len)
    inputs, targets = block[:, :-1], block[:, 1:]
    nats = 0.0
    counted = 0
    for start in range(0, inputs.shape[0], MICRO_SEQUENCES):
        x = inputs[start:start + MICRO_SEQUENCES].contiguous().to(device)
        y = targets[start:start + MICRO_SEQUENCES].contiguous().to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(x)
        total = F.cross_entropy(
            logits.float().view(-1, model.arch["vocab_size"]), y.reshape(-1), reduction="sum"
        )
        nats += float(total)
        counted += int(y.numel())
    return {"nats_sum": nats, "token_count": counted, "loss": nats / counted}


# --------------------------------------------------------------------------
# entry


def from_mixture(text: str) -> dict:
    """Turn the default weight template into a constant-mode plan, verbatim."""
    weights = {}
    inside = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.strip() == "weights:":
            inside = True
            continue
        if inside and line.startswith(" "):
            key, _, value = line.strip().partition(":")
            weights[key.strip()] = float(value.strip())
        elif inside:
            inside = False
    return {"schema": "oer08.plan/v2", "mode": "constant", "weights": weights}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(HERE / "corpus_spec.json"))
    parser.add_argument("--substrate", default=str(HERE / "nanogpt_substrate.json"))
    parser.add_argument("--plan")
    parser.add_argument("--from-mixture")
    parser.add_argument("--out", default="plan.json")
    args = parser.parse_args()

    if args.from_mixture:
        plan = from_mixture(Path(args.from_mixture).read_text(encoding="utf-8"))
        Path(args.out).write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("wrote " + args.out)
        return 0

    if not args.plan:
        parser.error("give --plan or --from-mixture")

    corpus = load_json(Path(args.corpus))
    substrate = load_json(Path(args.substrate))
    arch = architecture(corpus, substrate)
    readout = load_json(HERE / "graded_readout.json")
    root = data_root(corpus)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    index = pool_index(corpus, root)
    order = expand(corpus, index, json.loads(Path(args.plan).read_text(encoding="utf-8")))
    run = train(corpus, arch, index, order, root, device)

    probe = corpus["dev_probe"]
    measured = evaluate(
        run["model"],
        read_tokens(root, probe["shard"], int(probe["token_offset"]), int(probe["token_count"])),
        arch["seq_len"],
        device,
    )
    heldout = readout["heldout"]
    print(
        json.dumps(
            {
                "tokens_fed": run["tokens_fed"],
                "budget_tokens": budget_tokens(corpus),
                "optimizer_steps": run["steps"],
                "parameter_digest": parameter_digest(run["model"]),
                "dev_probe_loss": measured["loss"],
                "dev_probe_nats_sum": measured["nats_sum"],
                "dev_probe_token_count": measured["token_count"],
                "heldout_tokens": heldout["tokens"],
                "extrapolation_factor": heldout["tokens"] / probe["token_count"],
                "note": "dev_probe only, and drawn from a TRAIN shard. The graded loss is recomputed by the verifier on the frozen FineWeb10B validation slice, from the parameter snapshot the verifier's own trainer produced. This probe is smaller by extrapolation_factor, and it is one sample rather than two folds, so it cannot show you a fold artifact.",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
