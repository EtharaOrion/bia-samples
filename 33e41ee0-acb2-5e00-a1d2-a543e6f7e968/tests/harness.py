#!/usr/bin/env python3
"""The frozen training and evaluation harness for OER-04.

BYTE-IDENTICAL on the agent surface and inside the verifier; tests/Dockerfile
asserts that with `cmp` at build time. That identity is the contract of this task:
a shape measured locally with `python3 train_local.py` is measured at grading time
by the same code, the same frozen optimizer, the same frozen schedule and the same
token stream, so the only thing that can move the number is the shape.

WHY THIS HARNESS DOES NOT REUSE ONE COMPILED GRAPH
    The sibling slots on this substrate keep a single network and restore its
    pristine initial tensors before each run, which lets one compiled graph serve
    every arm. That trick is unavailable here for the honest reason that two
    different shapes are two different networks with different tensors. Each arm
    therefore builds and compiles its own model. What is held identical across
    arms instead is everything that is not the shape: the token stream, the
    optimizer, the schedule, the number of forward and backward passes, the
    initialisation seed and the evaluation window.

WHAT IS FIXED REGARDLESS OF SHAPE
    `budget.micro_steps` forward and backward passes over `budget.micro_batch` x
    `seq_len` tokens, one optimizer step each. Micro-batch i is always tokens
    [i*8192, (i+1)*8192] of the stream. No shape can buy itself more tokens, more
    passes or more optimizer steps.
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
    # Every arm of this slot compiles a DIFFERENT shape from the SAME source
    # function, and torch.compile counts its recompiles per code object rather
    # than per module instance. The stock limit is 8, and a grading pass needs
    # one entry per shape per grad mode; hitting the limit makes dynamo fall
    # back to eager silently, which would change both the wall clock and the
    # arithmetic mid-run. Raise it so that never happens by accident.
    try:
        import torch._dynamo as _dynamo
        for attr in ("recompile_limit", "cache_size_limit", "accumulated_recompile_limit",
                     "accumulated_cache_size_limit"):
            if hasattr(_dynamo.config, attr):
                setattr(_dynamo.config, attr, 256)
    except Exception:                       # pragma: no cover - never fatal
        pass


def load_shard(path, ntokens=None) -> np.ndarray:
    """Read a FineWeb10B shard slice. Header-checked, never assumed."""
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


def build_model(shape: dict, spec: dict, device="cuda"):
    """Instantiate a shape under the frozen seed. Returns (model, parameter_count)."""
    comp = spec["computation"]
    nanogpt = load_model_module()
    torch.manual_seed(int(spec["init"]["seed"]))
    model = nanogpt.GPT(
        vocab_size=int(comp["vocab_size"]),
        num_layers=int(shape["num_layers"]),
        model_dim=int(shape["model_dim"]),
        head_dim=int(shape["head_dim"]),
        mlp_ratio=int(shape["mlp_ratio"]),
    ).to(device)
    return model, sum(p.numel() for p in model.parameters())


class ShapeRunner:
    """One shape: build it, compile it, train it on the frozen budget, evaluate it."""

    def __init__(self, shape: dict, spec: dict, device="cuda", compile_model: bool = True):
        self.spec = spec
        self.shape = dict(shape)
        self.device = device
        budget = spec["budget"]
        self.micro_batch = int(budget["micro_batch"])
        self.seq_len = int(spec["computation"]["seq_len"])
        self.micro_steps = int(budget["micro_steps"])
        if self.micro_batch * self.seq_len * self.micro_steps != int(budget["total_train_tokens"]):
            raise HarnessError("budget is inconsistent with total_train_tokens")
        self.model, self.parameters = build_model(shape, spec, device)
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

    def train(self, tokens: torch.Tensor, progress=None) -> dict:
        mb, seq = self.micro_batch, self.seq_len
        steps = self.micro_steps
        need = steps * mb * seq + 1
        if tokens.numel() < need:
            raise HarnessError(f"train stream holds {tokens.numel()} tokens, the budget needs {need}")

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

        started = time.time()
        for step in range(steps):
            mult = lr_multiplier(self.spec["schedule"], step, steps)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * mult
            lo = step * mb * seq
            inputs = tokens[lo: lo + mb * seq].view(mb, seq)
            targets = tokens[lo + 1: lo + mb * seq + 1].view(mb, seq)
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
        return {
            "optimizer_steps": steps,
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

    def run(self, train_tokens, eval_tokens, progress=None) -> dict:
        report = self.train(train_tokens, progress=progress)
        report["parameters"] = self.parameters
        report["shape"] = dict(self.shape)
        if report["diverged"]:
            report["val_loss"] = float("nan")
            return report
        started = time.time()
        report["val_loss"] = self.evaluate(eval_tokens)
        report["eval_seconds"] = round(time.time() - started, 2)
        report["diverged"] = not math.isfinite(report["val_loss"])
        return report

    def release(self) -> None:
        """Drop the model and its graph so the next shape starts from a clean device."""
        del self.fn
        del self.model
        self.groups = {}
        torch.cuda.empty_cache()
