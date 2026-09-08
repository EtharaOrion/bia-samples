#!/usr/bin/env python3
"""The frozen training and evaluation harness.

This file is BYTE-IDENTICAL on the agent surface and inside the verifier, and
tests/Dockerfile asserts that with `cmp` at build time. That identity is the whole
contract of this task: a plan measured locally with `python3 train_local.py` is
measured at grading time by the same code, the same frozen initialisation and the
same token stream, so the only thing that can move the number is the plan.

Five properties are load-bearing and are worth stating explicitly.

1. ONE model instance, reused for every run in the process. `Runner` builds the
   network once under the frozen seed, keeps a pristine copy of the initial
   parameters, and restores them before each run. The anchors and the submission are
   therefore trained by literally the same kernels from literally the same starting
   tensors, and building a 49M-parameter network is paid once rather than three times.

2. The model is NOT torch.compile'd, and that is a consequence of the task rather
   than an oversight. A plan may change the micro-batch shape between phases, and a
   compiled graph is specialised to a shape: compiling would pay a fresh compilation
   for every distinct geometry any of the three runs names, which is a cost that
   depends on what was submitted. Eager execution costs every plan the same.

3. The TOKEN budget is frozen, the optimizer-step count is not. Every plan consumes
   the same tokens from the same stream in the same order. What a plan chooses is the
   shape they arrive in, and therefore how many optimizer steps the same tokens buy
   and how much left context each predicted token has.

4. The learning-rate envelope is clocked in TOKENS CONSUMED, never in step index, so
   two plans see the same peak-relative multiplier at the same point of the same
   budget. On top of it the peak is scaled by the square root of the optimizer-step
   token count against a frozen reference, which is the standard adaptive-optimizer
   batch rule. Both are frozen; neither is a number a plan may set.

5. Evaluation geometry is FIXED at the declared rows x seq_len for every run,
   whatever geometry the run trained under. A plan cannot move the measurement.
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

# The four parameter roles the frozen optimizer prices separately.
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
    """Resident int64 token stream, so no host-to-device copy sits in the step."""
    return torch.from_numpy(arr.astype(np.int32)).to(device).long()


# ---------------------------------------------------------------------------
# The frozen schedule
# ---------------------------------------------------------------------------

def lr_multiplier(schedule: dict, progress: float) -> float:
    """Peak-relative multiplier at `progress`, the fraction of the TOKEN budget spent.

    Warmup is linear from zero up to peak over the first `warmup_frac` of the budget.
    After warmup the named shape carries the multiplier from 1.0 down to `final_frac`.
    """
    progress = min(max(float(progress), 0.0), 1.0)
    warm = float(schedule["warmup_frac"])
    final = float(schedule["final_frac"])
    if warm > 0.0 and progress < warm:
        return progress / warm
    rest = (progress - warm) / max(1e-9, 1.0 - warm)
    shape = schedule["shape"]
    if shape == "constant":
        return 1.0
    if shape == "linear":
        return 1.0 + (final - 1.0) * rest
    if shape == "cosine":
        return final + (1.0 - final) * 0.5 * (1.0 + math.cos(math.pi * rest))
    if shape == "wsd":
        stable = float(schedule["stable_frac"])
        if rest < stable:
            return 1.0
        tail = (rest - stable) / max(1e-9, 1.0 - stable)
        return 1.0 + (final - 1.0) * tail
    raise HarnessError("unknown schedule shape " + repr(shape))


def batch_scale(schedule: dict, step_tokens: int) -> float:
    """The frozen batch rule: sqrt of the optimizer-step token count, normalised."""
    rule = schedule.get("batch_scaling", "sqrt")
    reference = float(schedule["batch_scaling_reference_tokens"])
    ratio = float(step_tokens) / reference
    if rule == "sqrt":
        return math.sqrt(ratio)
    if rule == "linear":
        return ratio
    if rule == "none":
        return 1.0
    raise HarnessError("unknown batch scaling rule " + repr(rule))


# ---------------------------------------------------------------------------
# The plan, expanded into the phases the harness will actually run
# ---------------------------------------------------------------------------

def expand(plan: dict, spec: dict) -> list:
    """Resolve a validated plan into concrete phases at the frozen token budget.

    The budget is partitioned in order. A phase runs whole optimizer steps only, and
    whatever its share cannot fill is CARRIED FORWARD into the next phase rather than
    dropped, so the whole plan consumes the budget to within one optimizer step of its
    final phase. The residue is reported, never hidden.
    """
    total = int(spec["budget"]["total_train_tokens"])
    phases = []
    carried = 0
    remaining_frac = 1.0
    for index, row in enumerate(plan["phases"]):
        rows = int(row["rows"])
        seq_len = int(row["seq_len"])
        accum = int(row["grad_accum"])
        step_tokens = rows * seq_len * accum
        last = index == len(plan["phases"]) - 1
        if last:
            share = int(round(remaining_frac * total)) + carried
        else:
            share = int(float(row["budget_frac"]) * total) + carried
            remaining_frac -= float(row["budget_frac"])
        steps = share // step_tokens
        carried = share - steps * step_tokens
        phases.append({
            "index": index,
            "rows": rows,
            "seq_len": seq_len,
            "grad_accum": accum,
            "microbatch_tokens": rows * seq_len,
            "optimizer_step_tokens": step_tokens,
            "optimizer_steps": steps,
            "tokens": steps * step_tokens,
        })
    consumed = sum(p["tokens"] for p in phases)
    if consumed <= 0:
        raise HarnessError("the plan buys no optimizer step at the frozen budget")
    return phases


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class Runner:
    def __init__(self, device="cuda", spec: dict | None = None):
        self.spec = spec if spec is not None else load_spec()
        arch = self.spec["architecture"]
        self.device = device
        self.total_tokens = int(self.spec["budget"]["total_train_tokens"])
        self.schedule = self.spec["schedule"]
        self.opt_cfg = self.spec["optimizer"]
        if arch["model_dim"] // arch["head_dim"] != arch["num_heads"]:
            raise HarnessError("architecture is inconsistent: model_dim // head_dim "
                               f"is {arch['model_dim'] // arch['head_dim']}, "
                               f"num_heads is {arch['num_heads']}")

        nanogpt = load_model_module()
        torch.manual_seed(int(self.spec["init"]["seed"]))
        model = nanogpt.GPT(
            vocab_size=int(arch["vocab_size"]),
            num_layers=int(arch["num_layers"]),
            model_dim=int(arch["model_dim"]),
            head_dim=int(arch["head_dim"]),
        ).to(device)
        self.model = model
        # The frozen initialisation, held on the device and restored before each run.
        self.init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.groups = self._param_groups()

    def _param_groups(self) -> dict:
        out = {g: [] for g in GROUPS}
        for name, param in self.model.named_parameters():
            if param.ndim < 2:
                out["scalar"].append(param)        # RMSNorm gains and every bias
            elif name.startswith("embed"):
                out["embed"].append(param)         # token embedding
            elif name.startswith("proj"):
                out["head"].append(param)          # untied output projection
            else:
                out["hidden"].append(param)        # every block matrix
        empty = [g for g, ps in out.items() if not ps]
        if empty:
            raise HarnessError("parameter role(s) with no tensors: " + ", ".join(empty))
        return out

    def reset(self) -> None:
        self.model.load_state_dict(self.init_state)

    def _optimizer(self) -> torch.optim.Optimizer:
        cfg = self.opt_cfg
        param_groups = [
            {"params": self.groups[g], "lr": float(cfg["lr_" + g]),
             "base_lr": float(cfg["lr_" + g]), "role": g}
            for g in GROUPS
        ]
        return torch.optim.AdamW(
            param_groups,
            betas=(float(cfg["beta1"]), float(cfg["beta2"])),
            eps=float(cfg["eps"]),
            weight_decay=float(cfg["weight_decay"]),
            foreach=True,
        )

    def train(self, plan: dict, tokens: torch.Tensor, progress=None) -> dict:
        """Train from the frozen initialisation under `plan`. Returns a report."""
        phases = expand(plan, self.spec)
        need = sum(p["tokens"] for p in phases) + 1
        if tokens.numel() < need:
            raise HarnessError(f"train stream holds {tokens.numel()} tokens, the plan needs {need}")

        self.reset()
        optimizer = self._optimizer()
        clip = float(self.opt_cfg["grad_clip"])
        params = list(self.model.parameters())

        started = time.time()
        position = 0
        consumed = 0
        total_steps = 0
        last_loss = float("nan")
        for phase in phases:
            rows, seq_len = phase["rows"], phase["seq_len"]
            accum = phase["grad_accum"]
            span = rows * seq_len
            scale = batch_scale(self.schedule, phase["optimizer_step_tokens"])
            for _ in range(phase["optimizer_steps"]):
                mult = lr_multiplier(self.schedule, consumed / self.total_tokens) * scale
                for group in optimizer.param_groups:
                    group["lr"] = group["base_lr"] * mult
                for _ in range(accum):
                    inputs = tokens[position: position + span].view(rows, seq_len)
                    targets = tokens[position + 1: position + span + 1].view(rows, seq_len)
                    position += span
                    consumed += span
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = self.model(inputs, targets)
                    (loss / accum).backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, clip, foreach=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                total_steps += 1
                if progress is not None and total_steps % progress == 0:
                    torch.cuda.synchronize()
                    last_loss = float(loss.detach())
                    print(f"    phase {phase['index']} {rows}x{seq_len}x{accum}  "
                          f"step {total_steps}  train_loss {last_loss:.4f}  "
                          f"lr_mult {mult:.4f}  {time.time() - started:.1f}s", flush=True)
        torch.cuda.synchronize()
        final = float(loss.detach())
        return {
            "phases": phases,
            "optimizer_steps": total_steps,
            "tokens_consumed": consumed,
            "tokens_budget": self.total_tokens,
            "budget_residue": self.total_tokens - consumed,
            "final_micro_loss": final,
            "diverged": not math.isfinite(final),
            "train_seconds": round(time.time() - started, 2),
        }

    @torch.no_grad()
    def evaluate(self, tokens: torch.Tensor) -> float:
        """Mean token-level cross entropy over the frozen evaluation window.

        The geometry here is the FROZEN evaluation geometry and never the geometry the
        run trained under, so every run is read at the same context length.
        """
        rows = int(self.spec["evaluation"]["rows"])
        seq_len = int(self.spec["evaluation"]["seq_len"])
        batches = int(self.spec["evaluation"]["tokens"]) // (rows * seq_len)
        need = batches * rows * seq_len + 1
        if tokens.numel() < need:
            raise HarnessError(f"eval stream holds {tokens.numel()} tokens, {need} are needed")
        self.model.eval()
        total = torch.zeros((), device=self.device, dtype=torch.float64)
        for i in range(batches):
            lo = i * rows * seq_len
            inputs = tokens[lo: lo + rows * seq_len].view(rows, seq_len)
            targets = tokens[lo + 1: lo + rows * seq_len + 1].view(rows, seq_len)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                total += self.model(inputs, targets).double()
        self.model.train()
        return float(total.item()) / batches

    def run(self, plan: dict, train_tokens, eval_tokens, progress=None) -> dict:
        report = self.train(plan, train_tokens, progress=progress)
        if report["diverged"]:
            report["val_loss"] = float("nan")
            return report
        started = time.time()
        report["val_loss"] = self.evaluate(eval_tokens)
        report["eval_seconds"] = round(time.time() - started, 2)
        report["diverged"] = not math.isfinite(report["val_loss"])
        return report
