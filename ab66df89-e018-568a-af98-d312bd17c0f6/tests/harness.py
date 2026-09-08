#!/usr/bin/env python3
"""The frozen training and evaluation harness, and the FLOP accountant.

BYTE-IDENTICAL on the agent surface and inside the verifier; tests/Dockerfile asserts
that with `cmp` at build time. That identity is the whole contract of this task: an
allocation measured locally with `python3 train_local.py` is measured at grading time
by the same code, the same frozen initialisation and the same token stream, so the
only thing that can move the number is the allocation.

Three properties are load-bearing.

1. THE FLOP BUDGET IS THE SCARCE THING, AND IT IS COUNTED HERE. `micro_batch_flops`
   is an explicit arithmetic model of one forward-plus-backward pass over one
   16 x 512 micro-batch of a given architecture. It is DECLARED rather than profiled:
   the same formula bounds a submission on the agent surface and charges it at
   grading time, so what an allocation costs is never a matter of which machine ran
   it. Every admissible allocation spends at most the same budget, which is what
   makes the three runs of a grading pass comparable in cost as well as in outcome.

2. THE LEARNING RATE FOLLOWS THE WIDTH. A single fixed learning rate would make this
   a test of which menu entry happened to suit one constant, not of how to spend
   compute. frozen/train_recipe.json therefore declares a width scaling -- the hidden
   matrices are priced at base_lr * (reference_dim / model_dim) -- and `scaled_recipe`
   applies it. The rule is public, fixed, and applied identically to every allocation.

3. THE TOKEN STREAM IS A DETERMINISTIC FUNCTION OF POSITION ALONE. Micro-batch i is
   always tokens [i * 8192, (i + 1) * 8192]. Two allocations that run a different
   number of micro-batches still consume the same tokens in the same order for as
   long as they both run.
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


def to_device_tokens(arr: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.int32)).to(device).long()


# ---------------------------------------------------------------------------
# The FLOP accountant
# ---------------------------------------------------------------------------

def model_shape(spec: dict, model_id: str) -> dict:
    """The declared architecture of one menu entry. Refused if it is not on the menu."""
    menu = spec["model_menu"]
    if model_id not in menu:
        raise HarnessError(f"{model_id!r} is not on the model menu {sorted(menu)}")
    shape = dict(spec["architecture_common"])
    shape.update(menu[model_id])
    if shape["model_dim"] % shape["head_dim"] != 0:
        raise HarnessError(f"{model_id}: model_dim {shape['model_dim']} is not a multiple "
                           f"of head_dim {shape['head_dim']}")
    shape["num_heads"] = shape["model_dim"] // shape["head_dim"]
    return shape


def micro_batch_flops(spec: dict, model_id: str) -> int:
    """FLOPs for one forward-and-backward pass over one micro-batch. DECLARED.

    Multiply-accumulates per token, counted over the matmuls that dominate:

        per block   4 * d * d        the q, k, v and output projections
                  + 8 * d * d        the MLP, d -> 4d -> d
                  + 2 * seq * d      the attention score and value products
        output      d * vocab        the untied head

    two FLOPs per multiply-accumulate, and the backward pass is charged at twice the
    forward, which is the standard convention and is stated in frozen/task_spec.json
    rather than left implicit. Nothing here is profiled: this arithmetic is the
    budget, on both surfaces.
    """
    shape = model_shape(spec, model_id)
    d = int(shape["model_dim"])
    seq = int(shape["seq_len"])
    layers = int(shape["num_layers"])
    vocab = int(shape["vocab_size"])
    macs_per_token = layers * (4 * d * d + 8 * d * d + 2 * seq * d) + d * vocab
    fwd_bwd = int(spec["accounting"]["forward_backward_multiplier"])
    tokens = int(spec["budget"]["micro_batch"]) * seq
    return 2 * macs_per_token * fwd_bwd * tokens


def max_micro_steps(spec: dict, model_id: str) -> int:
    """The most micro-batches this menu entry can buy inside the frozen FLOP budget."""
    return int(spec["budget"]["flop_budget"]) // micro_batch_flops(spec, model_id)


def scaled_recipe(recipe: dict, shape: dict) -> dict:
    """Apply the declared width scaling. Public, fixed, identical for every allocation."""
    out = json.loads(json.dumps(recipe))
    ref_dim = float(recipe["width_scaling"]["reference_model_dim"])
    factor = ref_dim / float(shape["model_dim"])
    for role in recipe["width_scaling"]["scaled_roles"]:
        out["optimizer"]["lr_" + role] = float(recipe["optimizer"]["lr_" + role]) * factor
    out["width_scaling_factor"] = factor
    return out


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
    """One architecture, built once and reused for every allocation that names it.

    Two allocations on the same menu entry share one compiled graph and one pristine
    set of initial tensors, so they differ by how the budget was spent and by nothing
    else. Allocations on different entries necessarily build different models; that
    is the axis being graded.
    """

    def __init__(self, spec: dict, model_id: str, device="cuda", compile_model: bool = True):
        self.spec = spec
        self.model_id = model_id
        self.shape = model_shape(spec, model_id)
        self.device = device
        self.micro_batch = int(spec["budget"]["micro_batch"])
        self.seq_len = int(self.shape["seq_len"])

        nanogpt = load_model_module()
        torch.manual_seed(int(spec["init"]["seed"]))
        model = nanogpt.GPT(
            vocab_size=int(self.shape["vocab_size"]),
            num_layers=int(self.shape["num_layers"]),
            model_dim=int(self.shape["model_dim"]),
            head_dim=int(self.shape["head_dim"]),
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

    def train(self, allocation: dict, recipe: dict, tokens: torch.Tensor, progress=None) -> dict:
        """Train from the frozen initialisation under `allocation`. Returns a report."""
        mb, seq = self.micro_batch, self.seq_len
        micro_steps = int(allocation["micro_steps"])
        accum = int(allocation["grad_accum"])
        steps = micro_steps // accum
        need = micro_steps * mb * seq + 1
        # THE CORPUS IS FINITE AND THE STREAM CYCLES. An allocation at the small end
        # of the menu can buy more tokens than data/train_slice.bin holds; when it
        # does, the run sees the corpus again from the start rather than being
        # refused. Repetition is an allocation, not an error -- what it costs is that
        # the repeated tokens teach less, which is the data-constrained half of the
        # trade this task is about.
        corpus = tokens.numel()
        epochs = need / corpus

        self.reset()
        cfg = scaled_recipe(recipe, self.shape)
        opt_cfg = cfg["optimizer"]
        optimizer = torch.optim.AdamW(
            [{"params": self.groups[g], "lr": float(opt_cfg["lr_" + g]),
              "base_lr": float(opt_cfg["lr_" + g]), "role": g} for g in GROUPS],
            betas=(float(opt_cfg["beta1"]), float(opt_cfg["beta2"])),
            eps=float(opt_cfg["eps"]),
            weight_decay=float(opt_cfg["weight_decay"]),
            foreach=True,
        )
        clip = float(cfg["grad_clip"])
        params = list(self.model.parameters())

        started = time.time()
        micro = 0
        for step in range(steps):
            mult = lr_multiplier(cfg["schedule"], step, steps)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * mult
            for _ in range(accum):
                lo = (micro * mb * seq) % (corpus - mb * seq - 1)
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
                print(f"    step {step + 1}/{steps}  train_loss {float(loss.detach()):.4f}"
                      f"  {time.time() - started:.1f}s", flush=True)
        torch.cuda.synchronize()
        final = float(loss.detach())
        return {
            "model": self.model_id,
            "model_dim": int(self.shape["model_dim"]),
            "num_layers": int(self.shape["num_layers"]),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "optimizer_steps": steps,
            "micro_steps": micro_steps,
            "grad_accum": accum,
            "tokens": micro_steps * mb * seq,
            "corpus_tokens": corpus,
            "epochs": round(epochs, 3),
            "width_scaling_factor": cfg["width_scaling_factor"],
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

    def run(self, allocation: dict, recipe: dict, train_tokens, eval_tokens, progress=None) -> dict:
        report = self.train(allocation, recipe, train_tokens, progress=progress)
        if report["diverged"]:
            report["val_loss"] = float("nan")
            return report
        started = time.time()
        report["val_loss"] = self.evaluate(eval_tokens)
        report["eval_seconds"] = round(time.time() - started, 2)
        report["diverged"] = not math.isfinite(report["val_loss"])
        return report
