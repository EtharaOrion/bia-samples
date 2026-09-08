#!/usr/bin/env python3
"""The frozen training and evaluation harness for OER-05.

This file is BYTE-IDENTICAL on the agent surface and inside the verifier, and
tests/Dockerfile asserts that with `cmp` at build time. That identity is the whole
contract of this task: an update rule measured locally with `python3 train_local.py`
is measured at grading time by the same code, the same frozen initialisation, the
same frozen step-size envelope and the same token stream, so the only thing that can
move the number is the rule.

WHAT IS FREE HERE, AND WHY IT IS THE UPDATE RULE AND NOT A LEARNING RATE

`apply_update` below is a single parameterised update law, and the seven numbers a
submission supplies per parameter role select a point in it. The law is

    m   <- decay1 * m + (1 - decay1) * g                    first moment
    v   <- decay2 * v + (1 - decay2) * g * g                second moment
    m^  <- m / (1 - decay1**t)      when decay1 > 0         bias correction
    v^  <- v / (1 - decay2**t)      when decay2 > 0
    d   <- m^ / (v^ ** precond_power + eps)                 preconditioning
    d   <- (1 - sign_mix) * d + sign_mix * sign(m^)         sign blending
    w   <- w - step * (d + weight_decay * w)                decoupled decay

and the named optimizers are interior points of it rather than special cases bolted
on beside it:

    precond_power 0.5, decay1 0.9, decay2 0.999, sign_mix 0     ->  AdamW
    precond_power 0.0, decay1 0.9, decay2 0,     sign_mix 0     ->  heavy-ball SGD
    precond_power 0.5, decay1 0.0, decay2 0.99,  sign_mix 0     ->  RMSProp
    precond_power 0.0, decay1 0.95, decay2 0,    sign_mix 1     ->  sign momentum
    precond_power 0.25 ...                                      ->  a rule with no name

Because the four parameter roles are priced independently, a submission may run a
DIFFERENT law on the embedding than on the block matrices. That is the axis this
slot is about, and it is not reachable by tuning a learning rate.

THREE LOAD-BEARING PROPERTIES

1. ONE model instance, ONE compiled graph, reused for every run in the process.
   `Runner` builds the network once under the frozen seed, keeps a pristine copy of
   the initial parameters, and restores them before each run. The anchors and the
   submission are therefore trained by literally the same kernels from literally the
   same starting tensors. This is also what keeps three training runs inside the
   grading budget: torch.compile is paid once, not three times.

2. The micro-batch SHAPE is frozen at 16 x 512 and grad_accum is frozen at 1.
   Nothing a submission can say changes the number of forward passes, of backward
   passes or of optimizer steps, so no rule can make grading slower than any other.
   The compute envelope is a property of the harness rather than a rule a submission
   is trusted to respect.

3. The step-size ENVELOPE is frozen in frozen/task_spec.json and is applied
   identically to every run. A submission sets the magnitude of its per-role step
   sizes; the shape of those step sizes over the 3072 steps is not its to choose.
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

# The four parameter roles a submission may give different update laws.
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
    """Resident int64 token stream. 25M tokens is 200 MB; keeping it on the
    accelerator removes a host-to-device copy from every micro-step."""
    return torch.from_numpy(arr.astype(np.int32)).to(device).long()


# ---------------------------------------------------------------------------
# The FROZEN step-size envelope
# ---------------------------------------------------------------------------

def envelope(schedule: dict, step: int, steps: int) -> float:
    """Peak-relative step-size multiplier at `step` of `steps`.

    Read from frozen/task_spec.json and identical for every run. It is here rather
    than in the submission because this slot's free axis is the update law, not the
    envelope: freeing both would make a rule that merely rediscovers a good schedule
    indistinguishable from a rule that is genuinely better.
    """
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
    raise HarnessError("unknown frozen envelope shape " + repr(shape))


# ---------------------------------------------------------------------------
# The update law
# ---------------------------------------------------------------------------

class RoleState:
    """The optimizer state of one parameter role, and the law that advances it."""

    def __init__(self, params, block: dict):
        self.params = list(params)
        self.step_size = float(block["step_size"])
        self.decay1 = float(block["decay1"])
        self.decay2 = float(block["decay2"])
        self.power = float(block["precond_power"])
        self.sign_mix = float(block["sign_mix"])
        self.weight_decay = float(block["weight_decay"])
        self.eps = float(block["eps"])
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    @torch.no_grad()
    def apply_update(self, scale: float) -> None:
        """One application of the law to every tensor in this role.

        `scale` is the frozen envelope multiplier for this step. Written with the
        foreach kernels so that ninety-odd tensors cost a handful of launches rather
        than a Python loop; the arithmetic is exactly the law in the module
        docstring and nothing is approximated for speed.
        """
        grads = [p.grad for p in self.params]
        if any(g is None for g in grads):
            raise HarnessError("a parameter in this role received no gradient")
        self.t += 1

        # --- first moment -------------------------------------------------
        if self.decay1 > 0.0:
            torch._foreach_mul_(self.m, self.decay1)
            torch._foreach_add_(self.m, grads, alpha=1.0 - self.decay1)
            mhat = torch._foreach_div(self.m, 1.0 - self.decay1 ** self.t)
        else:
            mhat = list(grads)

        # --- second moment ------------------------------------------------
        if self.power != 0.0:
            if self.decay2 > 0.0:
                torch._foreach_mul_(self.v, self.decay2)
                torch._foreach_addcmul_(self.v, grads, grads, value=1.0 - self.decay2)
                vhat = torch._foreach_div(self.v, 1.0 - self.decay2 ** self.t)
            else:
                vhat = torch._foreach_mul(grads, grads)
            if self.power == 0.5:
                denom = torch._foreach_sqrt(vhat)
            else:
                denom = torch._foreach_pow(vhat, self.power)
            torch._foreach_add_(denom, self.eps)
            direction = torch._foreach_div(mhat, denom)
        else:
            # v^0 is 1 for every entry, so the law reduces to m^ / (1 + eps).
            direction = torch._foreach_div(mhat, 1.0 + self.eps)

        # --- sign blending --------------------------------------------------
        if self.sign_mix > 0.0:
            signs = torch._foreach_sign(mhat)
            torch._foreach_mul_(direction, 1.0 - self.sign_mix)
            torch._foreach_add_(direction, signs, alpha=self.sign_mix)

        # --- decoupled weight decay and the step ----------------------------
        if self.weight_decay > 0.0:
            torch._foreach_add_(direction, self.params, alpha=self.weight_decay)
        torch._foreach_add_(self.params, direction, alpha=-self.step_size * scale)


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
        self.schedule = self.spec["schedule"]
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
        self.fn = torch.compile(model) if compile_model else model
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

    def steps_for(self, rule: dict) -> int:
        """Frozen. Present so a caller can print it, not so a rule can change it."""
        return self.micro_steps // self.grad_accum

    def train(self, rule: dict, tokens: torch.Tensor, progress=None) -> dict:
        """Train from the frozen initialisation under `rule`. Returns a report."""
        mb, seq = self.micro_batch, self.seq_len
        accum = self.grad_accum
        steps = self.micro_steps // accum
        need = self.micro_steps * mb * seq + 1
        if tokens.numel() < need:
            raise HarnessError(f"train stream holds {tokens.numel()} tokens, the budget needs {need}")

        self.reset()
        states = [RoleState(self.groups[g], rule["rules"][g]) for g in GROUPS]
        clip = float(rule["grad_clip"])
        params = list(self.model.parameters())
        for p in params:
            p.grad = None

        started = time.time()
        micro = 0
        last_loss = float("nan")
        for step in range(steps):
            scale = envelope(self.schedule, step, steps)
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
            for state in states:
                state.apply_update(scale)
            for p in params:
                p.grad = None
            if progress is not None and (step % progress == 0 or step == steps - 1):
                torch.cuda.synchronize()
                last_loss = float(loss.detach())
                print(f"    step {step + 1}/{steps}  train_loss {last_loss:.4f}"
                      f"  envelope {scale:.4f}  {time.time() - started:.1f}s", flush=True)
                if not math.isfinite(last_loss):
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

    def run(self, rule: dict, train_tokens, eval_tokens, progress=None) -> dict:
        report = self.train(rule, train_tokens, progress=progress)
        if report["diverged"]:
            report["val_loss"] = float("nan")
            return report
        started = time.time()
        report["val_loss"] = self.evaluate(eval_tokens)
        report["eval_seconds"] = round(time.time() - started, 2)
        report["diverged"] = not math.isfinite(report["val_loss"])
        return report
