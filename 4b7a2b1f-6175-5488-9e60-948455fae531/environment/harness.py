#!/usr/bin/env python3
"""The frozen training and evaluation harness, and the plan interpreter.

BYTE-IDENTICAL on the agent surface and inside the verifier; tests/Dockerfile asserts
that with `cmp` at build time. That identity is the whole contract of this task: a
plan measured locally with `python3 probe_local.py` is measured at grading time by
the same code, the same frozen initialisation and the same assembled token stream, so
the only thing that can move the number is the plan.

Three properties are load-bearing.

1. ONE model instance, ONE compiled graph, reused for every run in the process.
   `Runner` builds the network once under the frozen seed, keeps a pristine copy of
   the initial parameters, and restores them before each run. The anchors and the
   submission are trained by literally the same kernels from literally the same
   starting tensors. This is also what keeps three training runs inside the grading
   budget: torch.compile is paid once, not three times.

2. THE BUDGET IS FIXED IN TOKENS AND IN PASSES. `assemble` refuses any plan whose
   draws do not total exactly the frozen budget. Nothing a plan says changes the
   number of forward or backward passes, so no plan can make grading slower than any
   other, and no plan can win by training longer.

3. ORDER IS PART OF THE PLAN AND IS NOT NORMALISED AWAY. `assemble` concatenates the
   draws in the order they were written. Two plans with identical per-source totals
   in different orders are two different training streams, and the harness treats
   them as such.
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
RECIPE_PATH = HERE / "frozen" / "train_recipe.json"
MODEL_PATH = HERE / "model" / "nanogpt.py"

SHARD_MAGIC = 20240520
SHARD_VERSION = 1

GROUPS = ("embed", "hidden", "head", "scalar")


class HarnessError(RuntimeError):
    """The substrate on disk is not the frozen one. Refused, never coerced."""


def load_spec(path: Path = SPEC_PATH) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_train_recipe(path: Path = RECIPE_PATH) -> dict:
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


def load_pool(spec: dict, root) -> dict:
    """Every declared source, read from `root`. A missing source is refused."""
    root = Path(root)
    pool = {}
    for name in sorted(spec["pool"]["sources"]):
        path = root / (name + ".bin")
        tokens = load_shard(path)
        if tokens.size < int(spec["pool"]["source_tokens"]):
            raise HarnessError(f"source {name} holds {tokens.size} tokens, "
                               f"{spec['pool']['source_tokens']} were declared")
        pool[name] = tokens
    return pool


def to_device_tokens(arr: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.int32)).to(device).long()


# ---------------------------------------------------------------------------
# The plan interpreter
# ---------------------------------------------------------------------------

def assemble(plan: dict, spec: dict, pool: dict) -> np.ndarray:
    """Build the training stream a plan describes. Order-preserving, exact.

    Each draw takes `tokens` tokens from `source`, starting at `offset` within that
    source and CYCLING if the draw runs past the end. Repetition is therefore an
    allocation a plan may choose, not an error -- what it costs is that the repeated
    tokens are spent out of the same fixed budget as any others.

    The assembled stream is one token longer than the budget, because the last target
    is the token after the last input.
    """
    budget = int(spec["budget"]["total_train_tokens"])
    pieces = []
    total = 0
    for i, draw in enumerate(plan["draws"]):
        name = draw["source"]
        if name not in pool:
            raise HarnessError(f"draws[{i}].source {name!r} is not in the pool")
        want = int(draw["tokens"])
        src = pool[name]
        start = int(draw.get("offset", 0)) % src.size
        reps = (start + want) // src.size + 1
        run = np.tile(src, reps)[start:start + want] if reps > 1 else src[start:start + want]
        pieces.append(run)
        total += want
    if total != budget:
        raise HarnessError(f"the plan draws {total} tokens, the frozen budget is {budget}")
    stream = np.concatenate(pieces)
    # one more token for the shifted target, taken from the first draw's source
    tail = pool[plan["draws"][0]["source"]][:1]
    return np.concatenate([stream, tail])


def lr_multiplier(schedule: dict, step: int, steps: int) -> float:
    """Peak-relative learning-rate multiplier at `step` of `steps`."""
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

    def train(self, recipe: dict, tokens: torch.Tensor, progress=None) -> dict:
        """Train from the frozen initialisation over the assembled stream."""
        mb, seq = self.micro_batch, self.seq_len
        steps = self.micro_steps
        need = steps * mb * seq + 1
        if tokens.numel() < need:
            raise HarnessError(f"assembled stream holds {tokens.numel()} tokens, "
                               f"the budget needs {need}")

        self.reset()
        opt_cfg = recipe["optimizer"]
        optimizer = torch.optim.AdamW(
            [{"params": self.groups[g], "lr": float(opt_cfg["lr_" + g]),
              "base_lr": float(opt_cfg["lr_" + g]), "role": g} for g in GROUPS],
            betas=(float(opt_cfg["beta1"]), float(opt_cfg["beta2"])),
            eps=float(opt_cfg["eps"]),
            weight_decay=float(opt_cfg["weight_decay"]),
            foreach=True,
        )
        clip = float(recipe["grad_clip"])
        params = list(self.model.parameters())

        started = time.time()
        for step in range(steps):
            mult = lr_multiplier(recipe["schedule"], step, steps)
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
                      f"  {time.time() - started:.1f}s", flush=True)
        torch.cuda.synchronize()
        final = float(loss.detach())
        return {
            "micro_steps": steps,
            "train_tokens": steps * mb * seq,
            "diverged": not math.isfinite(final),
            "train_seconds": round(time.time() - started, 2),
        }

    @torch.no_grad()
    def evaluate(self, tokens: torch.Tensor) -> float:
        """Mean token-level cross-entropy over the frozen evaluation window."""
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

    def run(self, plan, recipe, spec, pool, eval_tokens, progress=None) -> dict:
        stream = to_device_tokens(assemble(plan, spec, pool), self.device)
        report = self.train(recipe, stream, progress=progress)
        del stream
        if report["diverged"]:
            report["val_loss"] = float("nan")
            return report
        started = time.time()
        report["val_loss"] = self.evaluate(eval_tokens)
        report["eval_seconds"] = round(time.time() - started, 2)
        report["diverged"] = not math.isfinite(report["val_loss"])
        return report
