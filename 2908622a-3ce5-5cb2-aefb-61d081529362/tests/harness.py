#!/usr/bin/env python3
"""The frozen training and evaluation harness for OER-10.

This file is BYTE-IDENTICAL on the agent surface and inside the verifier, and
tests/Dockerfile asserts that with `cmp` at build time. A mixture measured locally
with `python3 train_local.py` is measured at grading time by the same code, the same
frozen initialisation, the same frozen optimizer and the same budget, so the only
thing that can move the number is the mixture and its ordering.

WHAT IS FREE HERE IS WHICH TOKENS ARE SPENT, AND IN WHAT ORDER

The budget is 3072 blocks of 8192 tokens. `plan` turns a submitted mixture into a
list of exactly 3072 (shard, block) pairs, and `order` arranges them. Nothing else
about the run changes. In particular:

  * the NUMBER of blocks is 3072 whatever the mixture says. A weight vector is
    normalised and then converted to integer block counts by largest-remainder, so
    the counts sum to 3072 exactly. Overspending is not a rule a submission is
    trusted to respect; it is a state the representation cannot express.

  * a block reads 8193 tokens from INSIDE one shard -- 8192 inputs and the one extra
    token the last target needs. The targets of a block therefore never depend on
    which block was placed after it, so two orderings of the same multiset of blocks
    train on exactly the same (input, target) pairs in a different sequence.

  * when a shard is asked for more blocks than it holds, it is read again from the
    start. That is not an error and it is not prevented: a mixture that concentrates
    on one shard is choosing to see the same tokens three times, and finding out what
    that costs is part of the task.

THREE LOAD-BEARING PROPERTIES

1. ONE model instance, ONE compiled graph, reused for every run in the process.
2. The compute is identical across mixtures: 3072 forward passes, 3072 backward
   passes, 3072 optimizer steps, 16 x 512 each, always.
3. The whole shard pool is resident on the accelerator as one int64 tensor, so a
   block is a slice and assembling a stream costs no copy.
"""

from __future__ import annotations

import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE / "frozen" / "task_spec.json"
MODEL_PATH = HERE / "model" / "nanogpt.py"

SHARD_MAGIC = 20240520
SHARD_VERSION = 1

GROUPS = ("embed", "hidden", "head", "scalar")


class HarnessError(RuntimeError):
    """The substrate on disk is not the frozen one. Refused, never coerced."""


def load_spec(path: Path = SPEC_PATH) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_model_module(path: Path = MODEL_PATH):
    spec = importlib.util.spec_from_file_location("nanogpt_frozen", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_backends() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_shard(path, ntokens=None) -> np.ndarray:
    """Read a FineWeb10B shard. Header-checked, never assumed."""
    path = Path(path)
    header = np.fromfile(path, dtype=np.int32, count=256)
    if header.size < 3 or int(header[0]) != SHARD_MAGIC or int(header[1]) != SHARD_VERSION:
        raise HarnessError(f"{path} is not a FineWeb10B shard: header {header[:3].tolist()}")
    available = int(header[2])
    want = available if ntokens is None else int(ntokens)
    if want > available:
        raise HarnessError(f"{path} holds {available} tokens, {want} were asked for")
    with path.open("rb") as fh:
        fh.seek(1024)
        buf = fh.read(2 * want)
    return np.frombuffer(buf, dtype=np.uint16)


class ShardPool:
    """Every shard, concatenated once, resident on the accelerator.

    `base[i]` is where shard i starts in the pool, so block b of shard i is the
    slice [base[i] + b * stride, base[i] + b * stride + 8193). Because every shard
    is the same declared length, a block index is bounds-checked by construction.
    """

    def __init__(self, directory, spec: dict, device="cuda"):
        corpus = spec["corpus"]
        self.names = list(corpus["shards"])
        self.shard_tokens = int(corpus["shard_tokens"])
        self.blocks_per_shard = int(corpus["blocks_per_shard"])
        self.block_tokens = int(spec["budget"]["block_tokens"])
        self.stride = self.block_tokens
        self.read = int(spec["budget"]["block_stride"])
        if self.blocks_per_shard * self.stride + 1 > self.shard_tokens:
            raise HarnessError(
                f"a shard of {self.shard_tokens} tokens cannot yield "
                f"{self.blocks_per_shard} blocks of stride {self.stride}")
        directory = Path(directory)
        chunks = []
        for name in self.names:
            arr = load_shard(directory / (name + ".bin"))
            if arr.size != self.shard_tokens:
                raise HarnessError(
                    f"{name} holds {arr.size} tokens, the substrate declares "
                    f"{self.shard_tokens}")
            chunks.append(arr.astype(np.int32))
        self.base = [i * self.shard_tokens for i in range(len(self.names))]
        self.tokens = torch.from_numpy(np.concatenate(chunks)).to(device).long()
        self.index = {name: i for i, name in enumerate(self.names)}

    def block(self, shard: int, block: int) -> torch.Tensor:
        lo = self.base[shard] + (block % self.blocks_per_shard) * self.stride
        return self.tokens[lo: lo + self.read]


def to_device_tokens(arr: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.int32)).to(device).long()


# ---------------------------------------------------------------------------
# The mixture -- the one thing a submission supplies
# ---------------------------------------------------------------------------

def block_counts(mixture: dict, names, total: int) -> dict:
    """Turn weights into integer block counts that sum to `total`, exactly.

    Largest-remainder apportionment, with ties broken by the shard's declared order
    so the result depends on the weights alone and never on dictionary iteration
    order. A shard whose weight rounds to zero blocks gets zero blocks; that is the
    submission's choice and not a floor this function applies on its behalf.
    """
    weights = [max(0.0, float(mixture[name])) for name in names]
    mass = sum(weights)
    if mass <= 0.0:
        raise HarnessError("the mixture weights sum to zero, so no block can be drawn")
    exact = [w / mass * total for w in weights]
    counts = [int(math.floor(value)) for value in exact]
    short = total - sum(counts)
    order = sorted(range(len(names)), key=lambda i: (-(exact[i] - counts[i]), i))
    for k in range(short):
        counts[order[k % len(order)]] += 1
    return dict(zip(names, counts))


def plan(document: dict, names, total: int) -> list:
    """The exact sequence of (shard index, block index) the run will consume.

    Deterministic in the submission and in nothing else. The three policies are
    genuinely different arrangements of the SAME multiset of blocks:

      interleave  round-robin, spacing each shard's blocks as evenly as the counts
                  allow, so the mixture is stationary over the run
      sequential  every block of one shard, then the next, in declared shard order;
                  the run sees one distribution at a time
      blocked     the interleaved sequence cut into chunks of `block_group` and the
                  chunks permuted by `seed`, so the mixture is stationary in the
                  large and clumped in the small
    """
    counts = block_counts(document["mixture"], names, total)
    per_shard = {name: list(range(counts[name])) for name in names}
    policy = document["order"]["policy"]

    if policy == "sequential":
        sequence = [(i, b) for i, name in enumerate(names) for b in per_shard[name]]
    else:
        # Even spacing: give every block a position in [0, 1) at the centre of its
        # share of that shard's allocation, then sort by position.
        marked = []
        for i, name in enumerate(names):
            n = counts[name]
            for b in range(n):
                marked.append(((b + 0.5) / n, i, b))
        marked.sort(key=lambda row: (row[0], row[1], row[2]))
        sequence = [(i, b) for _p, i, b in marked]

    if policy == "blocked":
        group = int(document["order"]["block_group"])
        chunks = [sequence[i: i + group] for i in range(0, len(sequence), group)]
        generator = torch.Generator()
        generator.manual_seed(int(document["order"]["seed"]))
        order = torch.randperm(len(chunks), generator=generator).tolist()
        sequence = [pair for k in order for pair in chunks[k]]

    if len(sequence) != total:
        raise HarnessError(f"the plan holds {len(sequence)} blocks, the budget is {total}")
    return sequence


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class Runner:
    def __init__(self, pool: ShardPool, device="cuda", spec: dict | None = None,
                 compile_model: bool = True):
        self.spec = spec if spec is not None else load_spec()
        arch = self.spec["architecture"]
        self.device = device
        self.pool = pool
        self.micro_batch = int(self.spec["budget"]["micro_batch"])
        self.seq_len = int(arch["seq_len"])
        self.micro_steps = int(self.spec["budget"]["micro_steps"])
        self.grad_accum = int(self.spec["budget"]["grad_accum"])
        self.optimizer_cfg = self.spec["optimizer"]
        expected = self.micro_batch * self.seq_len * self.micro_steps
        if expected != int(self.spec["budget"]["total_train_tokens"]):
            raise HarnessError("budget is inconsistent: micro_batch * seq_len * micro_steps "
                               f"is {expected}, total_train_tokens is "
                               f"{self.spec['budget']['total_train_tokens']}")
        if self.micro_batch * self.seq_len != int(self.spec["budget"]["block_tokens"]):
            raise HarnessError("budget is inconsistent: a block is not one micro-batch")
        if arch["model_dim"] // arch["head_dim"] != arch["num_heads"]:
            raise HarnessError("architecture is inconsistent")
        if self.optimizer_cfg["algorithm"] != "adamw":
            raise HarnessError("the frozen optimizer is not adamw")

        nanogpt = load_model_module()
        torch.manual_seed(int(self.spec["init"]["seed"]))
        model = nanogpt.GPT(
            vocab_size=int(arch["vocab_size"]),
            num_layers=int(arch["num_layers"]),
            model_dim=int(arch["model_dim"]),
            head_dim=int(arch["head_dim"]),
        ).to(device)
        self.model = model
        self.init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.fn = torch.compile(model) if compile_model else model
        self.groups = self._param_groups()

    def _param_groups(self) -> dict:
        out = {g: [] for g in GROUPS}
        for name, param in self.model.named_parameters():
            if param.ndim < 2:
                out["scalar"].append(param)
            elif name.startswith("embed"):
                out["embed"].append(param)
            elif name.startswith("proj"):
                out["head"].append(param)
            else:
                out["hidden"].append(param)
        empty = [g for g, ps in out.items() if not ps]
        if empty:
            raise HarnessError("parameter role(s) with no tensors: " + ", ".join(empty))
        return out

    def reset(self) -> None:
        self.model.load_state_dict(self.init_state)

    def steps_for(self, document: dict) -> int:
        return self.micro_steps // self.grad_accum

    def lr_multiplier(self, step: int, steps: int) -> float:
        cfg = self.optimizer_cfg
        warm = int(round(float(cfg["schedule_warmup_frac"]) * steps))
        final = float(cfg["schedule_final_frac"])
        if step < warm:
            return (step + 1) / max(1, warm)
        progress = (step - warm) / max(1, steps - warm)
        if cfg["schedule_shape"] != "wsd":
            raise HarnessError("unknown frozen schedule shape")
        stable = float(cfg["schedule_stable_frac"])
        if progress < stable:
            return 1.0
        tail = (progress - stable) / max(1e-9, 1.0 - stable)
        return 1.0 + (final - 1.0) * tail

    def train(self, document: dict, progress=None) -> dict:
        """Train from the frozen initialisation on the SUBMITTED mixture."""
        mb, seq = self.micro_batch, self.seq_len
        accum = self.grad_accum
        steps = self.micro_steps // accum
        sequence = plan(document, self.pool.names, self.micro_steps)

        self.reset()
        cfg = self.optimizer_cfg
        param_groups = [
            {"params": self.groups[g], "lr": float(cfg["lr_" + g]),
             "base_lr": float(cfg["lr_" + g]), "role": g}
            for g in GROUPS
        ]
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(float(cfg["beta1"]), float(cfg["beta2"])),
            eps=float(cfg["eps"]),
            weight_decay=float(cfg["weight_decay"]),
            foreach=True,
        )
        clip = float(cfg["grad_clip"])
        params = list(self.model.parameters())

        started = time.time()
        micro = 0
        for step in range(steps):
            mult = self.lr_multiplier(step, steps)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * mult
            for _ in range(accum):
                shard, block = sequence[micro]
                window = self.pool.block(shard, block)
                inputs = window[: mb * seq].view(mb, seq)
                targets = window[1: mb * seq + 1].view(mb, seq)
                micro += 1
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = self.fn(inputs, targets)
                (loss / accum).backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(params, clip, foreach=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if progress is not None and (step % progress == 0 or step == steps - 1):
                torch.cuda.synchronize()
                value = float(loss.detach())
                print(f"    step {step + 1}/{steps}  train_loss {value:.4f}"
                      f"  lr_mult {mult:.4f}  {time.time() - started:.1f}s", flush=True)
                if not math.isfinite(value):
                    break
        torch.cuda.synchronize()
        final = float(loss.detach())
        counts = block_counts(document["mixture"], self.pool.names, self.micro_steps)
        return {
            "optimizer_steps": steps,
            "micro_steps": self.micro_steps,
            "grad_accum": accum,
            "blocks_per_shard": counts,
            "tokens_per_shard": {k: v * self.pool.block_tokens for k, v in counts.items()},
            "tokens_fed": sum(counts.values()) * self.pool.block_tokens,
            "order_policy": document["order"]["policy"],
            "final_micro_loss": final,
            "diverged": not math.isfinite(final),
            "train_seconds": round(time.time() - started, 2),
        }

    @torch.no_grad()
    def evaluate(self, tokens: torch.Tensor) -> float:
        mb, seq = self.micro_batch, self.seq_len
        batches = int(self.spec["evaluation"]["tokens"]) // (mb * seq)
        need = batches * mb * seq + 1
        if tokens.numel() < need:
            raise HarnessError(f"eval stream holds {tokens.numel()} tokens, {need} are needed")
        self.model.eval()
        total = torch.zeros((), device=self.device, dtype=torch.float64)
        for i in range(batches):
            lo = i * mb * seq
            inputs = tokens[lo: lo + mb * seq].view(mb, seq)
            targets = tokens[lo + 1: lo + mb * seq + 1].view(mb, seq)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                total += self.fn(inputs, targets).double()
        self.model.train()
        return float(total.item()) / batches

    def run(self, document: dict, eval_tokens, progress=None) -> dict:
        report = self.train(document, progress=progress)
        if report["diverged"]:
            report["val_loss"] = float("nan")
            return report
        started = time.time()
        report["val_loss"] = self.evaluate(eval_tokens)
        report["eval_seconds"] = round(time.time() - started, 2)
        report["diverged"] = not math.isfinite(report["val_loss"])
        return report
