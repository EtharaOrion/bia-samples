#!/usr/bin/env python3
"""The frozen training and evaluation harness.

This file is BYTE-IDENTICAL on the agent surface and inside the verifier, and
tests/Dockerfile asserts that with `cmp` at build time. That identity is the whole
contract of this task: a shape measured locally with `python3 train_local.py` is
measured at grading time by the same code, the same frozen recipe and the same token
stream, so the only thing that can move the number is the shape.

Four properties are load-bearing and are worth stating explicitly.

1. ONE model per shape, and the model is built HERE from the frozen decoder module.
   Two different shapes cannot share a parameter tensor, so what is held constant
   across runs is the seed, the initialisation law and the construction order rather
   than the values. `torch.manual_seed` is re-applied immediately before each build,
   so building the same shape twice in one process gives the same parameters.

2. The model is NOT torch.compile'd, and that is a consequence of the task rather
   than an oversight. A compiled graph is specialised to a shape, so compiling would
   pay a fresh compilation for each of the three shapes a grading run builds -- a cost
   that depends on what was submitted. Eager execution costs every shape the same.

3. The TRAINING RECIPE is frozen and identical for every run: the same four per-role
   learning rates, the same AdamW constants, the same clip, the same warmup-hold-decay
   envelope over the same number of optimizer steps. A shape that only wins once it is
   given its own learning rate has not answered the question this slot asks.

4. The TOKEN STREAM is a deterministic function of position alone. Micro-batch i is
   always tokens [i*8192, (i+1)*8192]. Every shape sees the same tokens in the same
   order, and every shape is read on the same evaluation window at the same geometry.
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
# The frozen schedule
# ---------------------------------------------------------------------------

def lr_multiplier(recipe: dict, step: int, steps: int) -> float:
    warm = int(round(float(recipe["warmup_frac"]) * steps))
    final = float(recipe["final_frac"])
    if step < warm:
        return (step + 1) / max(1, warm)
    progress = (step - warm) / max(1, steps - warm)
    shape = recipe["schedule_shape"]
    if shape == "constant":
        return 1.0
    if shape == "linear":
        return 1.0 + (final - 1.0) * progress
    if shape == "cosine":
        return final + (1.0 - final) * 0.5 * (1.0 + math.cos(math.pi * progress))
    if shape == "wsd":
        stable = float(recipe["stable_frac"])
        if progress < stable:
            return 1.0
        tail = (progress - stable) / max(1e-9, 1.0 - stable)
        return 1.0 + (final - 1.0) * tail
    raise HarnessError("unknown schedule shape " + repr(shape))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class Runner:
    """Builds and trains ONE decoder shape under the frozen recipe and budget."""

    def __init__(self, shape: dict, device="cuda", spec: dict | None = None):
        self.spec = spec if spec is not None else load_spec()
        frozen = self.spec["architecture_frozen"]
        self.device = device
        self.shape = {k: int(shape[k]) for k in
                      ("num_layers", "model_dim", "head_dim", "mlp_ratio")}
        self.micro_batch = int(self.spec["budget"]["micro_batch"])
        self.seq_len = int(frozen["seq_len"])
        self.micro_steps = int(self.spec["budget"]["micro_steps"])
        expected = self.micro_batch * self.seq_len * self.micro_steps
        if expected != int(self.spec["budget"]["total_train_tokens"]):
            raise HarnessError("budget is inconsistent: micro_batch * seq_len * micro_steps "
                               f"is {expected}, total_train_tokens is "
                               f"{self.spec['budget']['total_train_tokens']}")
        if self.shape["model_dim"] % self.shape["head_dim"]:
            raise HarnessError("model_dim is not a multiple of head_dim")

        nanogpt = load_model_module()
        if nanogpt.LOGIT_SOFTCAP != frozen["logit_softcap"] or \
           nanogpt.ATTENTION_SCALE != frozen["attention_scale"]:
            raise HarnessError("the decoder module does not carry the frozen constants")
        # The seed is re-applied immediately before the build, so the same shape built
        # twice in one process draws the same tensors.
        torch.manual_seed(int(self.spec["init"]["seed"]))
        self.model = nanogpt.GPT(
            vocab_size=int(frozen["vocab_size"]),
            num_layers=self.shape["num_layers"],
            model_dim=self.shape["model_dim"],
            head_dim=self.shape["head_dim"],
            mlp_ratio=self.shape["mlp_ratio"],
        ).to(device)
        self.groups = self._param_groups()

    # -- accounting ---------------------------------------------------------
    def parameter_counts(self) -> dict:
        embed = sum(p.numel() for n, p in self.model.named_parameters()
                    if n.startswith("embed") or n.startswith("proj"))
        total = sum(p.numel() for p in self.model.parameters())
        return {"total": total, "embedding_and_head": embed, "non_embedding": total - embed}

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

    # -- the run ------------------------------------------------------------
    def train(self, tokens: torch.Tensor, progress=None) -> dict:
        mb, seq = self.micro_batch, self.seq_len
        recipe = self.spec["recipe"]
        accum = int(recipe["grad_accum"])
        steps = self.micro_steps // accum
        need = self.micro_steps * mb * seq + 1
        if tokens.numel() < need:
            raise HarnessError(f"train stream holds {tokens.numel()} tokens, "
                               f"the budget needs {need}")

        param_groups = [
            {"params": self.groups[g], "lr": float(recipe["lr_" + g]),
             "base_lr": float(recipe["lr_" + g]), "role": g}
            for g in GROUPS
        ]
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(float(recipe["beta1"]), float(recipe["beta2"])),
            eps=float(recipe["eps"]),
            weight_decay=float(recipe["weight_decay"]),
            foreach=True,
        )
        clip = float(recipe["grad_clip"])
        params = list(self.model.parameters())

        started = time.time()
        micro = 0
        for step in range(steps):
            mult = lr_multiplier(recipe, step, steps)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * mult
            for _ in range(accum):
                lo = micro * mb * seq
                inputs = tokens[lo: lo + mb * seq].view(mb, seq)
                targets = tokens[lo + 1: lo + mb * seq + 1].view(mb, seq)
                micro += 1
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = self.model(inputs, targets)
                (loss / accum).backward()
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
            "shape": dict(self.shape),
            "parameters": self.parameter_counts(),
            "optimizer_steps": steps,
            "micro_steps": self.micro_steps,
            "tokens_consumed": self.micro_steps * mb * seq,
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
                total += self.model(inputs, targets).double()
        self.model.train()
        return float(total.item()) / batches

    def release(self) -> None:
        """Free the shape's tensors before the next shape is built."""
        self.model = None
        self.groups = None
        torch.cuda.empty_cache()


def run_shape(shape: dict, train_tokens, eval_tokens, spec: dict,
              device="cuda", progress=None) -> dict:
    """Build one shape, train it under the frozen recipe, read it on the eval window."""
    runner = Runner(shape, device, spec)
    report = runner.train(train_tokens, progress=progress)
    if report["diverged"]:
        report["val_loss"] = float("nan")
    else:
        started = time.time()
        report["val_loss"] = runner.evaluate(eval_tokens)
        report["eval_seconds"] = round(time.time() - started, 2)
        report["diverged"] = not math.isfinite(report["val_loss"])
    runner.release()
    return report
