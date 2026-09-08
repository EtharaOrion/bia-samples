#!/usr/bin/env python3
"""The frozen training and evaluation harness for OER-11.

This file is BYTE-IDENTICAL on the agent surface and inside the verifier, and
tests/Dockerfile asserts that with `cmp` at build time. An initialisation measured
locally with `python3 train_local.py` is measured at grading time by the same code,
the same frozen optimizer, the same frozen schedule and the same token stream, so
the only thing that can move the number is the initialisation.

WHAT IS FREE HERE IS THE SCALE OF THE DRAW, NOT THE DRAW

`unit_draw` builds, once per process, one standard-normal tensor for every parameter
in the network, from a generator seeded by the frozen seed and by that parameter's
own name. `apply_init` then writes each parameter as

    parameter = unit_draw[name] * scale(role of name)

so two initialisations in this task differ by their scales and by nothing else. They
see the same random numbers in the same places. This is what makes a measured
difference between two submissions a fact about the scaling rather than a resampling
artifact, and it is why this harness does not simply call `torch.nn.init.normal_`
with a different std each time -- that would redraw, and a redraw at 49M parameters
is worth a few thousandths of a nat all on its own.

THE SEVEN ROLES

    embed        the token embedding
    attn_qkv     the q, k and v projections in every block
    attn_proj    the attention output projection in every block
    mlp_fc       the MLP input projection in every block
    mlp_proj     the MLP output projection in every block
    head         the untied output projection
    bias         every Linear bias in the network

plus `norm_gain`, which is a CONSTANT written into every RMSNorm gain rather than a
scale on a draw, because a gain is not a random quantity.

`residual_depth_power` is the one non-scalar knob. The two projections that write
into the residual stream -- attn_proj and mlp_proj -- have their scale divided by
(2 * num_layers) ** residual_depth_power. At power 0 the knob is off; at power 0.5
it is the classic 1/sqrt(2L) residual taper that keeps the variance of the residual
stream from growing with depth. It is exposed as a continuous exponent rather than a
switch because the right amount of taper at six layers is not obviously either
endpoint.

THREE LOAD-BEARING PROPERTIES

1. ONE model instance, ONE compiled graph, reused for every run. Only the parameter
   VALUES are rewritten between runs, so the anchors and the submission are trained
   by literally the same kernels and torch.compile is paid once, not three times.
2. The micro-batch shape is frozen at 16 x 512 and grad_accum at 1.
3. The token stream is a deterministic function of position alone.
"""

from __future__ import annotations

import hashlib
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
# not touch these; they are here because the optimizer has to sort the tensors. They
# are NOT the initialisation roles, which are finer -- see INIT_ROLES.
GROUPS = ("embed", "hidden", "head", "scalar")

# The initialisation roles a submission scales independently.
INIT_ROLES = ("embed", "attn_qkv", "attn_proj", "mlp_fc", "mlp_proj", "head", "bias")


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
# Roles and the frozen schedule
# ---------------------------------------------------------------------------

def init_role(name: str) -> str:
    """Which initialisation role a parameter belongs to. Total over the network.

    The mapping is by parameter NAME and is deliberately exhaustive: a parameter this
    function could not classify would have to be initialised by some default the
    submission never chose, so an unrecognised name is an error rather than a guess.
    """
    if name.endswith("bias"):
        return "bias"
    if name.endswith("gains"):
        return "norm_gain"
    if name.startswith("embed"):
        return "embed"
    if name.startswith("proj"):
        return "head"
    if ".attn.q." in name or ".attn.k." in name or ".attn.v." in name:
        return "attn_qkv"
    if ".attn.proj." in name:
        return "attn_proj"
    if ".mlp.fc." in name:
        return "mlp_fc"
    if ".mlp.proj." in name:
        return "mlp_proj"
    raise HarnessError("no initialisation role for parameter " + repr(name))


def lr_multiplier(cfg: dict, step: int, steps: int) -> float:
    """The FROZEN learning-rate envelope, read from frozen/task_spec.json."""
    warm = int(round(float(cfg["schedule_warmup_frac"]) * steps))
    final = float(cfg["schedule_final_frac"])
    if step < warm:
        return (step + 1) / max(1, warm)
    progress = (step - warm) / max(1, steps - warm)
    shape = cfg["schedule_shape"]
    if shape == "constant":
        return 1.0
    if shape == "linear":
        return 1.0 + (final - 1.0) * progress
    if shape == "cosine":
        return final + (1.0 - final) * 0.5 * (1.0 + math.cos(math.pi * progress))
    if shape == "wsd":
        stable = float(cfg["schedule_stable_frac"])
        if progress < stable:
            return 1.0
        tail = (progress - stable) / max(1e-9, 1.0 - stable)
        return 1.0 + (final - 1.0) * tail
    raise HarnessError("unknown frozen schedule shape " + repr(shape))


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
        self.num_layers = int(arch["num_layers"])
        self.optimizer_cfg = self.spec["optimizer"]
        expected = self.micro_batch * self.seq_len * self.micro_steps
        if expected != int(self.spec["budget"]["total_train_tokens"]):
            raise HarnessError("budget is inconsistent: micro_batch * seq_len * micro_steps "
                               f"is {expected}, total_train_tokens is "
                               f"{self.spec['budget']['total_train_tokens']}")
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
        self.fn = torch.compile(model) if compile_model else model
        self.groups = self._param_groups()
        self.unit = self._unit_draw()
        self.roles = {name: init_role(name) for name, _ in model.named_parameters()}

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

    def _unit_draw(self) -> dict:
        """One frozen standard-normal tensor per parameter, keyed by parameter name.

        The generator is seeded by the frozen seed mixed with a digest of the
        parameter's name, so the draw for one tensor does not depend on how many
        tensors were drawn before it, on the order they are visited in, or on
        anything a submission can say. Held on the device: 49M floats is 200 MB and
        it removes a host copy from every run.
        """
        base = int(self.spec["init"]["seed"])
        out = {}
        for name, param in self.model.named_parameters():
            digest = hashlib.sha256(name.encode("utf-8")).digest()
            seed = (base + int.from_bytes(digest[:6], "big")) % (2 ** 63 - 1)
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            out[name] = torch.randn(param.shape, generator=generator,
                                    device=self.device, dtype=param.dtype)
        return out

    def scales(self, document: dict) -> dict:
        """The realised per-role scale of every parameter, after depth tapering.

        Returned rather than only applied, so the score document can record exactly
        what the submission's numbers came out to at this depth without the reader
        having to redo the arithmetic.
        """
        block = document["init"]
        taper = (2.0 * self.num_layers) ** float(block["residual_depth_power"])
        out = {}
        for role in INIT_ROLES:
            value = float(block[role + "_std"])
            if role in ("attn_proj", "mlp_proj"):
                value = value / taper
            out[role] = value
        out["norm_gain"] = float(block["norm_gain"])
        out["residual_taper_divisor"] = taper
        return out

    @torch.no_grad()
    def apply_init(self, document: dict) -> dict:
        """Write every parameter as its frozen unit draw times the submitted scale."""
        realised = self.scales(document)
        for name, param in self.model.named_parameters():
            role = self.roles[name]
            if role == "norm_gain":
                param.fill_(realised["norm_gain"])
            else:
                param.copy_(self.unit[name] * realised[role])
        return realised

    def steps_for(self, document: dict) -> int:
        """Frozen. Present so a caller can print it, not so a submission can move it."""
        return self.micro_steps // self.grad_accum

    def train(self, document: dict, tokens: torch.Tensor, progress=None) -> dict:
        """Train from the SUBMITTED initialisation under the frozen optimizer."""
        mb, seq = self.micro_batch, self.seq_len
        accum = self.grad_accum
        steps = self.micro_steps // accum
        need = self.micro_steps * mb * seq + 1
        if tokens.numel() < need:
            raise HarnessError(f"train stream holds {tokens.numel()} tokens, the budget needs {need}")

        realised = self.apply_init(document)
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
            mult = lr_multiplier(cfg, step, steps)
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
            "realised_scales": realised,
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
