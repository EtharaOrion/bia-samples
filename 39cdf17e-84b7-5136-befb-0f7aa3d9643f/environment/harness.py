#!/usr/bin/env python3
"""The frozen training and evaluation harness for OER-07.

This file is BYTE-IDENTICAL on the agent surface and inside the verifier, and
tests/Dockerfile asserts that with `cmp` at build time. That identity is the whole
contract of this task: a schedule measured locally with `python3 train_local.py` is
measured at grading time by the same code, the same frozen initialisation, the same
frozen optimizer and the same token stream, so the only thing that can move the
number is the schedule.

WHAT IS FREE HERE IS THE SHAPE AND ONLY THE SHAPE

`multiplier` below returns a number in [0, 1] for each of the 3072 optimizer steps,
and that number multiplies the four FROZEN per-role peak learning rates declared in
frozen/task_spec.json. A submission cannot raise a peak, cannot lower a peak, cannot
change the moments, epsilon, the weight decay, the gradient clip or the number of
steps. It decides one thing: how the run's step size travels from its first step to
its last.

The space is not a single family with a knob on it. It carries a warmup segment with
its own ramp shape, seven decay families, and a hold-then-decay split point, and the
families genuinely disagree about where a 3072-step budget should spend its steps.
A cosine and a hold-then-decay with the same warmup and the same floor put a
materially different number of steps near the peak, and at this budget that
difference is worth more than a tenth of a nat.

THREE LOAD-BEARING PROPERTIES

1. ONE model instance, ONE compiled graph, reused for every run in the process. The
   anchors and the submission are trained by literally the same kernels from
   literally the same starting tensors, and torch.compile is paid once, not three
   times.

2. The micro-batch SHAPE is frozen at 16 x 512 and grad_accum is frozen at 1, so no
   schedule can make grading slower than any other.

3. The token stream is a deterministic function of position alone. Micro-batch i is
   always tokens [i*8192, (i+1)*8192].
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

# The four parameter roles the FROZEN optimizer prices separately. A submission does
# not touch these; they are here because the harness has to sort the tensors.
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


# ---------------------------------------------------------------------------
# The schedule -- the one thing a submission supplies
# ---------------------------------------------------------------------------

def multiplier(schedule: dict, step: int, steps: int) -> float:
    """Peak-relative learning-rate multiplier at `step` of `steps`.

    Warmup runs first, over `round(warmup_frac * steps)` steps, from one step's worth
    of the peak up to the peak. `warmup_shape` picks the ramp: `linear` is the
    straight line, `poly` raises the linear fraction to `warmup_power`, so a power
    below one front-loads the ramp and a power above one holds the run low for
    longer before releasing it.

    After warmup the named decay family carries the multiplier from 1.0 down to
    `final_frac` as `progress` runs from 0 to 1. Every family lands on `final_frac`
    at the last step; they differ only in the path they take to get there, which is
    exactly the axis this slot grades.
    """
    warm = int(round(float(schedule["warmup_frac"]) * steps))
    final = float(schedule["final_frac"])
    if step < warm:
        fraction = (step + 1) / max(1, warm)
        if schedule["warmup_shape"] == "poly":
            return fraction ** float(schedule["warmup_power"])
        return fraction

    progress = (step - warm) / max(1, steps - warm)
    shape = schedule["shape"]
    power = float(schedule["decay_power"])

    if shape == "constant":
        return 1.0
    if shape == "linear":
        return 1.0 + (final - 1.0) * progress
    if shape == "cosine":
        return final + (1.0 - final) * 0.5 * (1.0 + math.cos(math.pi * progress))
    if shape == "poly":
        return final + (1.0 - final) * ((1.0 - progress) ** power)
    if shape == "exp":
        # A true exponential never reaches its floor, so the tail is rescaled to land
        # on `final_frac` exactly at the last step. The family keeps its shape and the
        # endpoint stays comparable with every other family's.
        decayed = math.exp(-power * progress)
        floor = math.exp(-power)
        return final + (1.0 - final) * (decayed - floor) / max(1e-12, 1.0 - floor)
    if shape == "inv_sqrt":
        decayed = 1.0 / math.sqrt(1.0 + power * progress)
        floor = 1.0 / math.sqrt(1.0 + power)
        return final + (1.0 - final) * (decayed - floor) / max(1e-12, 1.0 - floor)
    if shape == "wsd":
        stable = float(schedule["stable_frac"])
        if progress < stable:
            return 1.0
        tail = (progress - stable) / max(1e-9, 1.0 - stable)
        return 1.0 + (final - 1.0) * tail
    raise HarnessError("unknown schedule shape " + repr(shape))


def envelope_trace(schedule: dict, steps: int, points: int = 32) -> list:
    """A coarse trace of the envelope, for a human reading the score document."""
    if steps <= 0:
        return []
    idx = sorted({int(round(i * (steps - 1) / max(1, points - 1))) for i in range(points)})
    return [[i, round(multiplier(schedule, i, steps), 6)] for i in idx]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class Runner:
    def __init__(self, device="cuda", spec: dict | None = None, compile_model: bool = True):
        self.spec = spec if spec is not None else load_spec()
        arch = self.spec["architecture"]
        self.device = device
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
        if self.micro_steps // self.grad_accum != int(self.spec["budget"]["optimizer_steps"]):
            raise HarnessError("budget is inconsistent: micro_steps // grad_accum is "
                               f"{self.micro_steps // self.grad_accum}, optimizer_steps is "
                               f"{self.spec['budget']['optimizer_steps']}")
        if arch["model_dim"] // arch["head_dim"] != arch["num_heads"]:
            raise HarnessError("architecture is inconsistent: model_dim // head_dim "
                               f"is {arch['model_dim'] // arch['head_dim']}, num_heads is "
                               f"{arch['num_heads']}")
        if self.optimizer_cfg["algorithm"] != "adamw":
            raise HarnessError("the frozen optimizer is not adamw: "
                               + repr(self.optimizer_cfg["algorithm"]))

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
        """Frozen. Present so a caller can print it, not so a submission can move it."""
        return self.micro_steps // self.grad_accum

    def train(self, document: dict, tokens: torch.Tensor, progress=None) -> dict:
        """Train from the frozen initialisation under the submitted schedule."""
        mb, seq = self.micro_batch, self.seq_len
        accum = self.grad_accum
        steps = self.micro_steps // accum
        need = self.micro_steps * mb * seq + 1
        if tokens.numel() < need:
            raise HarnessError(f"train stream holds {tokens.numel()} tokens, the budget needs {need}")

        schedule = document["schedule"]
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
            mult = multiplier(schedule, step, steps)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * mult
            for _ in range(accum):
                lo = micro * mb * seq
                inputs = tokens[lo: lo + mb * seq].view(mb, seq)
                targets = tokens[lo + 1: lo + mb * seq + 1].view(mb, seq)
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
        return {
            "optimizer_steps": steps,
            "micro_steps": self.micro_steps,
            "grad_accum": accum,
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

    def run(self, document: dict, train_tokens, eval_tokens, progress=None) -> dict:
        report = self.train(document, train_tokens, progress=progress)
        if report["diverged"]:
            report["val_loss"] = float("nan")
            return report
        started = time.time()
        report["val_loss"] = self.evaluate(eval_tokens)
        report["eval_seconds"] = round(time.time() - started, 2)
        report["diverged"] = not math.isfinite(report["val_loss"])
        return report
