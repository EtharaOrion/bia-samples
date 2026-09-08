#!/usr/bin/env python3
"""The frozen training, snapshotting, averaging and evaluation harness.

This file is BYTE-IDENTICAL on the agent surface and inside the verifier, and
tests/Dockerfile asserts that with `cmp` at build time. That identity is the whole
contract of this task: a selection measured locally with `python3 train_local.py` is
combined and evaluated at grading time by the same code, from the same frozen run, so
the only thing that can move the graded number is the selection.

Three properties are load-bearing.

1. THE TRAINING RUN IS FROZEN AND IS RUN ONCE PER PROCESS.
   `Runner.train_snapshots()` executes the recipe in frozen/task_spec.json from the
   frozen seed and returns the 16 parameter states it passed through. No submission
   touches it, so every selection in a grading pass is scored against literally the
   same snapshots, and grading pays for ONE training run rather than one per anchor.

2. THE LEARNING RATE IS CONSTANT.
   After a short warmup the schedule never decays. The final iterate is therefore a
   sample from a stationary distribution around a basin rather than the bottom of one,
   which is precisely the regime where averaging iterates buys something. This is the
   reason the task exists, and it is stated rather than hidden.

3. AVERAGING IS DONE IN FLOAT64 AND APPLIED TO EVERY TENSOR.
   `average_states` combines the FULL state dict, including the RMSNorm gains and every
   bias, in double precision, so a combination is not quietly a different model class
   from a single snapshot.
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
# Combination
# ---------------------------------------------------------------------------

def average_states(states: list, weights: list) -> dict:
    """Weighted average of full state dicts, accumulated in float64.

    `states` and `weights` are parallel. The weights are used as given: the schema has
    already established that they are non-negative and sum to one, so nothing is
    renormalised here and a caller cannot smuggle a rescaling past the schema.
    """
    if not states:
        raise HarnessError("no snapshots were selected")
    if len(states) != len(weights):
        raise HarnessError("selection and weights differ in length")
    out = {}
    for key in states[0]:
        acc = torch.zeros_like(states[0][key], dtype=torch.float64)
        for state, weight in zip(states, weights):
            acc += state[key].to(torch.float64) * float(weight)
        out[key] = acc.to(states[0][key].dtype)
    return out


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
        expected = self.micro_batch * self.seq_len * self.micro_steps
        if expected != int(self.spec["budget"]["total_train_tokens"]):
            raise HarnessError("budget is inconsistent: micro_batch * seq_len * micro_steps "
                               f"is {expected}, total_train_tokens is "
                               f"{self.spec['budget']['total_train_tokens']}")
        if arch["model_dim"] // arch["head_dim"] != arch["num_heads"]:
            raise HarnessError("architecture is inconsistent")

        snaps = self.spec["snapshots"]
        self.snapshot_every = int(snaps["every_steps"])
        self.snapshot_count = int(snaps["count"])
        if self.snapshot_every * self.snapshot_count != self.micro_steps:
            raise HarnessError("snapshot schedule does not tile the budget: "
                               f"{self.snapshot_every} * {self.snapshot_count} != {self.micro_steps}")

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
        # ONE compiled graph. It is reused by the training run and by every evaluation
        # of every combination, which is what keeps a grading pass to one compile.
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

    # -- the frozen run -----------------------------------------------------

    def train_snapshots(self, tokens: torch.Tensor, progress=None) -> dict:
        """Run the FROZEN recipe once and return every snapshot it passed through.

        Returns {"snapshots": {id: state_dict on cpu}, ...}. Nothing about this run is
        parameterised by a submission; the only argument is the token stream.
        """
        mb, seq = self.micro_batch, self.seq_len
        recipe = self.spec["training_recipe"]
        accum = int(recipe["grad_accum"])
        steps = self.micro_steps // accum
        need = self.micro_steps * mb * seq + 1
        if tokens.numel() < need:
            raise HarnessError(f"train stream holds {tokens.numel()} tokens, the budget needs {need}")

        self.reset()
        opt_cfg = recipe["optimizer"]
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
        clip = float(recipe["grad_clip"])
        params = list(self.model.parameters())
        warm = int(round(float(recipe["schedule"]["warmup_frac"]) * steps))

        started = time.time()
        snapshots = {}
        micro = 0
        for step in range(steps):
            # Constant after warmup. There is no decay, on purpose.
            mult = (step + 1) / max(1, warm) if step < warm else 1.0
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
            if (step + 1) % self.snapshot_every == 0:
                ident = (step + 1) // self.snapshot_every
                snapshots[ident] = {k: v.detach().to("cpu", copy=True)
                                    for k, v in self.model.state_dict().items()}
                if progress is not None:
                    torch.cuda.synchronize()
                    print(f"    snapshot {ident}/{self.snapshot_count} at step {step + 1}"
                          f"  train_loss {float(loss.detach()):.4f}"
                          f"  {time.time() - started:.1f}s", flush=True)
        torch.cuda.synchronize()
        if len(snapshots) != self.snapshot_count:
            raise HarnessError(f"the frozen run produced {len(snapshots)} snapshots, "
                               f"{self.snapshot_count} were declared")
        return {
            "snapshots": snapshots,
            "optimizer_steps": steps,
            "micro_steps": self.micro_steps,
            "train_seconds": round(time.time() - started, 2),
        }

    # -- evaluation ---------------------------------------------------------

    @torch.no_grad()
    def evaluate_state(self, state: dict, tokens: torch.Tensor) -> float:
        """Mean token-level cross-entropy of `state` over the frozen evaluation window."""
        self.model.load_state_dict({k: v.to(self.device) for k, v in state.items()})
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
        value = float(total.item()) / batches
        return value

    def evaluate_selection(self, selection: dict, snapshots: dict, tokens: torch.Tensor) -> dict:
        """Combine the kept snapshots under the selection and evaluate the result."""
        ids = [int(i) for i in selection["keep"]]
        weights = [float(w) for w in selection["weights"]]
        missing = [i for i in ids if i not in snapshots]
        if missing:
            raise HarnessError("selection names snapshots the frozen run did not produce: "
                               + ", ".join(str(i) for i in missing))
        started = time.time()
        combined = average_states([snapshots[i] for i in ids], weights)
        val_loss = self.evaluate_state(combined, tokens)
        return {
            "kept": ids,
            "weights": weights,
            "kept_count": len(ids),
            "val_loss": val_loss,
            "diverged": not math.isfinite(val_loss),
            "eval_seconds": round(time.time() - started, 2),
        }
