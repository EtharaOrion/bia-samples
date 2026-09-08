#!/usr/bin/env python3
"""The frozen training and evaluation harness.

This file is BYTE-IDENTICAL on the agent surface and inside the verifier, and
tests/Dockerfile asserts that with `cmp` at build time. That identity is the whole
contract of this task: a policy measured locally with `python3 train_local.py` is
measured at grading time by the same code, the same frozen initialisation and the
same token stream, so the only thing that can move the number is the policy.

Three properties are load-bearing and are worth stating explicitly.

1. ONE model instance, ONE compiled graph, reused for every run in the process.
   `Runner` builds the network once under the frozen seed, keeps a pristine copy
   of the initial parameters, and restores them before each run. The anchors and
   the submission are therefore trained by literally the same kernels from
   literally the same starting tensors. This is also what keeps three training
   runs inside the grading budget: torch.compile is paid once, not three times.

2. EVERY POLICY CONTROL IS A 0-DIM TENSOR, never a Python float, and that is what
   makes point 1 possible. A Python float is burned into the compiled graph as a
   constant, so three policies with three different softcaps would pay three
   compilations and grading would cost a function of what was submitted. Passed as
   tensors, the same graph serves every policy, and a policy cannot make grading
   slower than any other.

3. The micro-batch SHAPE, the micro-step count and the accumulation are ALL frozen.
   Nothing a submission can say changes the number of forward or backward passes,
   so every policy costs the same FLOPs.

4. The token stream is a deterministic function of position alone. Micro-batch i
   is always tokens [i*8192, (i+1)*8192].

5. The TRAINING objective and the GRADED reading are different functions. Training
   uses the policy's smoothing and log-partition penalty; the reading is plain
   cross entropy through the same logit softcap the run trained under. `evaluate`
   calls the decoder's `cross_entropy`, never its training `forward`.
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

# The four parameter roles a policy decays separately.
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
# Schedule
# ---------------------------------------------------------------------------

def lr_multiplier(schedule: dict, step: int, steps: int) -> float:
    """Peak-relative learning-rate multiplier at `step` of `steps`. FROZEN.

    `schedule` is the frozen optimizer block of the substrate declaration, never a
    submission. Warmup is linear from one step's worth of peak up to peak. After
    warmup the named shape carries the multiplier from 1.0 down to `final_frac`.
    """
    warm = int(round(float(schedule["warmup_frac"]) * steps))
    final = float(schedule["final_frac"])
    if step < warm:
        return (step + 1) / max(1, warm)
    progress = (step - warm) / max(1, steps - warm)
    shape = schedule["schedule_shape"]
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
        self.grad_accum = int(self.spec["budget"]["grad_accum"])
        self.opt_cfg = self.spec["optimizer"]
        expected = self.micro_batch * self.seq_len * self.micro_steps
        if expected != int(self.spec["budget"]["total_train_tokens"]):
            raise HarnessError("budget is inconsistent: micro_batch * seq_len * micro_steps "
                               f"is {expected}, total_train_tokens is "
                               f"{self.spec['budget']['total_train_tokens']}")
        if arch["model_dim"] // arch["head_dim"] != arch["num_heads"]:
            raise HarnessError("architecture is inconsistent: model_dim // head_dim "
                               f"is {arch['model_dim'] // arch['head_dim']}, num_heads is {arch['num_heads']}")

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

    def steps_for(self, policy: dict | None = None) -> int:
        """Frozen. The optimizer-step count is not a function of the submission."""
        del policy
        return self.micro_steps // self.grad_accum

    def _control(self, value: float) -> torch.Tensor:
        """A policy control, as a 0-dim device tensor. Never a Python float."""
        return torch.tensor(float(value), device=self.device, dtype=torch.float32)

    def train(self, policy: dict, tokens: torch.Tensor, progress=None) -> dict:
        """Train from the frozen initialisation under `policy`. Returns a report."""
        mb, seq = self.micro_batch, self.seq_len
        accum = self.grad_accum
        steps = self.micro_steps // accum
        need = self.micro_steps * mb * seq + 1
        if tokens.numel() < need:
            raise HarnessError(f"train stream holds {tokens.numel()} tokens, the budget needs {need}")

        self.reset()
        cfg = self.opt_cfg
        decay = policy["decay"]
        objective = policy["objective"]
        # DECOUPLED, PER ROLE. AdamW's own weight_decay is left at zero and each role
        # carries its own, so the four decays are the policy's and nothing else's.
        param_groups = [
            {"params": self.groups[g], "lr": float(cfg["lr_" + g]),
             "base_lr": float(cfg["lr_" + g]), "role": g,
             "weight_decay": float(decay["wd_" + g])}
            for g in GROUPS
        ]
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(float(cfg["beta1"]), float(cfg["beta2"])),
            eps=float(cfg["eps"]),
            weight_decay=0.0,
            foreach=True,
        )
        clip = float(cfg["grad_clip"])
        params = list(self.model.parameters())

        softcap = self._control(objective["logit_softcap"])
        smoothing = self._control(objective["label_smoothing"])
        z_weight = self._control(objective["z_loss"])

        started = time.time()
        micro = 0
        last_loss = float("nan")
        for step in range(steps):
            mult = lr_multiplier(cfg, step, steps)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * mult
            for _ in range(accum):
                lo = micro * mb * seq
                inputs = tokens[lo: lo + mb * seq].view(mb, seq)
                targets = tokens[lo + 1: lo + mb * seq + 1].view(mb, seq)
                micro += 1
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = self.fn(inputs, targets, softcap, smoothing, z_weight)
                (loss / accum).backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(params, clip, foreach=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if progress is not None and (step % progress == 0 or step == steps - 1):
                torch.cuda.synchronize()
                last_loss = float(loss.detach())
                print(f"    step {step + 1}/{steps}  train_objective {last_loss:.4f}"
                      f"  lr_mult {mult:.4f}  {time.time() - started:.1f}s", flush=True)
        torch.cuda.synchronize()
        final = float(loss.detach())
        return {
            "optimizer_steps": steps,
            "micro_steps": self.micro_steps,
            "grad_accum": accum,
            "final_micro_objective": final,
            "diverged": not math.isfinite(final),
            "train_seconds": round(time.time() - started, 2),
        }

    @torch.no_grad()
    def evaluate(self, tokens: torch.Tensor, softcap: torch.Tensor) -> float:
        """Mean token-level cross entropy over the frozen evaluation window.

        PLAIN cross entropy: the policy's smoothing and log-partition penalty are
        training-time terms and do not appear here. The softcap does, because it is
        part of the model the run produced rather than a choice about scoring.
        """
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
                total += self.model.cross_entropy(inputs, targets, softcap).double()
        self.model.train()
        return float(total.item()) / batches

    def run(self, policy: dict, train_tokens, eval_tokens, progress=None) -> dict:
        report = self.train(policy, train_tokens, progress=progress)
        softcap = self._control(policy["objective"]["logit_softcap"])
        if report["diverged"]:
            report["val_loss"] = float("nan")
            return report
        started = time.time()
        report["val_loss"] = self.evaluate(eval_tokens, softcap)
        report["eval_seconds"] = round(time.time() - started, 2)
        report["diverged"] = not math.isfinite(report["val_loss"])
        return report
