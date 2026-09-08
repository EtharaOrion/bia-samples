#!/usr/bin/env python3
"""The frozen training and evaluation harness for OER-09.

BYTE-IDENTICAL on the agent surface and inside the verifier; tests/Dockerfile
asserts that with `cmp` at build time. That identity is the contract of this
task: a filter chain measured locally with `python3 train_local.py` is measured at
grading time by the same code, the same frozen initialisation, the same frozen
optimizer and the same block pool, so the only thing that can move the number is
which blocks the chain admitted.

Three properties are load-bearing.

1. ONE model instance, ONE compiled graph, reused for every run in the process.
   `Runner` builds the network once under the frozen seed, keeps a pristine copy
   of the initial parameters and restores them before each run. The control, the
   reference and the submission are trained by literally the same kernels from
   literally the same starting tensors. This is also what keeps three training
   runs inside the grading budget: torch.compile is paid once, not three times.

2. The compute budget is fixed in BLOCKS. Exactly `budget.blocks` admitted blocks
   of `budget.block_tokens` tokens are consumed, one per optimizer step. A chain
   that admits more has the surplus ignored; a chain that admits fewer is refused
   upstream. No filter can buy itself more compute or spend less.

3. The optimizer and the schedule are read out of the frozen spec and are the
   same object for every run. This slot has no optimizer axis.
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
    """Read a FineWeb10B-format shard. Header-checked, never assumed."""
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


def to_device_tokens(arr: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.int32)).to(device).long()


class BlockPool:
    """The pool of candidate blocks, resident on the accelerator.

    Held as [pool_blocks, block_tokens]. A run is a selection of row indices; the
    stream it trains on is those rows concatenated in the order given.
    """

    def __init__(self, blocks_file, spec: dict, device="cuda"):
        self.block_tokens = int(spec["pool"]["block_tokens"])
        self.expect_blocks = int(spec["pool"]["blocks"])
        flat = load_shard(blocks_file)
        if flat.size != self.expect_blocks * self.block_tokens:
            raise HarnessError(
                f"{blocks_file} holds {flat.size} tokens, the frozen pool is "
                f"{self.expect_blocks} x {self.block_tokens} = "
                f"{self.expect_blocks * self.block_tokens}")
        self.blocks = to_device_tokens(flat, device).view(self.expect_blocks, self.block_tokens)
        self.device = device

    def stream(self, block_ids) -> torch.Tensor:
        """Concatenate the given blocks into one token stream, plus the single
        shifted-target token the last micro-batch needs.

        That trailing token is the first token of the first admitted block. It is
        one token out of 25165824 and it is chosen by a rule rather than by the
        submission, so it can carry no information and cannot be gamed.
        """
        idx = torch.as_tensor(list(block_ids), dtype=torch.long, device=self.device)
        if idx.numel() == 0:
            raise HarnessError("no blocks were selected")
        if int(idx.max()) >= self.expect_blocks or int(idx.min()) < 0:
            raise HarnessError("a selected block id is outside the pool")
        flat = self.blocks[idx].reshape(-1)
        return torch.cat([flat, self.blocks[idx[0], :1]])


# ---------------------------------------------------------------------------
# Schedule -- frozen, read from the spec
# ---------------------------------------------------------------------------

def lr_multiplier(schedule: dict, step: int, steps: int) -> float:
    warm = int(round(float(schedule["warmup_frac"]) * steps))
    final = float(schedule["final_frac"])
    if step < warm:
        return (step + 1) / max(1, warm)
    progress = (step - warm) / max(1, steps - warm)
    shape = schedule["shape"]
    if shape == "constant":
        return 1.0
    if shape == "linear":
        return 1.0 + (final - 1.0) * progress
    if shape == "cosine":
        return final + (1.0 - final) * 0.5 * (1.0 + math.cos(math.pi * progress))
    if shape == "wsd":
        stable = float(schedule["stable_frac"])
        if progress < stable:
            return 1.0
        tail = (progress - stable) / max(1e-9, 1.0 - stable)
        return 1.0 + (final - 1.0) * tail
    raise HarnessError("unknown schedule shape " + repr(shape))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class Runner:
    def __init__(self, pool: BlockPool, device="cuda", spec: dict | None = None,
                 compile_model: bool = True):
        self.spec = spec if spec is not None else load_spec()
        arch = self.spec["architecture"]
        budget = self.spec["budget"]
        self.device = device
        self.pool = pool
        self.micro_batch = int(budget["micro_batch"])
        self.seq_len = int(arch["seq_len"])
        self.block_tokens = int(budget["block_tokens"])
        self.blocks = int(budget["blocks"])
        if self.micro_batch * self.seq_len != self.block_tokens:
            raise HarnessError("budget is inconsistent: micro_batch * seq_len is "
                               f"{self.micro_batch * self.seq_len}, block_tokens is "
                               f"{self.block_tokens}")
        if self.blocks * self.block_tokens != int(budget["total_train_tokens"]):
            raise HarnessError("budget is inconsistent: blocks * block_tokens is "
                               f"{self.blocks * self.block_tokens}, total_train_tokens is "
                               f"{budget['total_train_tokens']}")
        if arch["model_dim"] // arch["head_dim"] != arch["num_heads"]:
            raise HarnessError("architecture is inconsistent")

        nanogpt = load_model_module()
        torch.manual_seed(int(self.spec["init"]["seed"]))
        self.model = nanogpt.GPT(
            vocab_size=int(arch["vocab_size"]),
            num_layers=int(arch["num_layers"]),
            model_dim=int(arch["model_dim"]),
            head_dim=int(arch["head_dim"]),
        ).to(device)
        self.init_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.fn = torch.compile(self.model) if compile_model else self.model
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

    def steps_for(self, _doc=None) -> int:
        return self.blocks

    def train(self, block_ids, progress=None) -> dict:
        """Train from the frozen initialisation on the given blocks."""
        used = list(block_ids)[: self.blocks]
        if len(used) != self.blocks:
            raise HarnessError(f"{len(used)} blocks were offered, the budget consumes {self.blocks}")
        stream = self.pool.stream(used)

        self.reset()
        opt_cfg = self.spec["optimizer"]
        param_groups = [
            {"params": self.groups[g], "lr": float(opt_cfg["lr_" + g]),
             "base_lr": float(opt_cfg["lr_" + g]), "role": g}
            for g in GROUPS
        ]
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(float(opt_cfg["beta1"]), float(opt_cfg["beta2"])),
            eps=float(opt_cfg["eps"]),
            weight_decay=float(opt_cfg["weight_decay"]),
            foreach=True,
        )
        clip = float(opt_cfg["grad_clip"])
        params = list(self.model.parameters())
        mb, seq = self.micro_batch, self.seq_len
        steps = self.blocks

        started = time.time()
        for step in range(steps):
            mult = lr_multiplier(self.spec["schedule"], step, steps)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * mult
            lo = step * self.block_tokens
            inputs = stream[lo: lo + mb * seq].view(mb, seq)
            targets = stream[lo + 1: lo + mb * seq + 1].view(mb, seq)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = self.fn(inputs, targets)
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(params, clip, foreach=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if progress is not None and (step % progress == 0 or step == steps - 1):
                torch.cuda.synchronize()
                print(f"    step {step + 1}/{steps}  train_loss {float(loss.detach()):.4f}"
                      f"  lr_mult {mult:.4f}  {time.time() - started:.1f}s", flush=True)
        torch.cuda.synchronize()
        final = float(loss.detach())
        del stream
        torch.cuda.empty_cache()
        return {
            "optimizer_steps": steps,
            "blocks_consumed": steps,
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

    def run(self, block_ids, eval_tokens, progress=None) -> dict:
        report = self.train(block_ids, progress=progress)
        if report["diverged"]:
            report["val_loss"] = float("nan")
            return report
        started = time.time()
        report["val_loss"] = self.evaluate(eval_tokens)
        report["eval_seconds"] = round(time.time() - started, 2)
        report["diverged"] = not math.isfinite(report["val_loss"])
        return report
