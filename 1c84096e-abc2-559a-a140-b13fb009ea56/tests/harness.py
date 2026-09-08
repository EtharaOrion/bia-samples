#!/usr/bin/env python3
"""The frozen training-and-measurement harness for OER-12.

BYTE-IDENTICAL on the agent surface and inside the verifier; tests/Dockerfile
asserts that with `cmp` at build time. A policy measured locally with
`python3 probe_local.py` is replayed at grading time by this same code over the
same frozen initialisation and the same token stream, so the only thing that can
move the graded number is the policy.

Three properties are load-bearing.

1. THE TRAINING RUN IS FROZEN AND HAPPENS ONCE. The optimizer, the schedule, the
   token stream and the number of passes are fixed by frozen/train_recipe.json and
   frozen/task_spec.json. A policy cannot change what is trained; it chooses only
   where to LOOK. So one training pass serves the shipped default policy, the
   verifier's private reference policy and the submission alike, and no policy can
   make grading cost more than any other.

2. THE MEASUREMENTS ARE TAKEN ONCE AND SHARED. At every candidate step the harness
   records the per-batch probe losses and the held-out loss. A policy that asks for
   k batches at step s is answered with the mean of the FIRST k of the batch losses
   recorded at s -- the identical numbers every other policy asking for k at s
   receives. Two policies therefore differ by their choices and by nothing else.

3. THE EVALUATION BUDGET IS ENFORCED ON THE POLICY, NOT ON THE HARNESS. Spending
   more probe tokens buys a less noisy estimate of the probe loss; the budget caps
   the total. Nothing about the budget touches the held-out loss, which is measured
   at every candidate step regardless and is never shown to a policy.
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
    raise HarnessError("unknown schedule shape " + repr(shape))


# ---------------------------------------------------------------------------
# The degradation ramp
# ---------------------------------------------------------------------------

def degrade(tokens: np.ndarray, ramp_start: int, ramp_end: int, block: int,
            tokens_per_step: int, seed: int) -> np.ndarray:
    """Apply the declared pipeline fault to a clean token stream.

    From micro-step `ramp_start` onward an increasing fraction of the stream's
    `block`-token windows are permuted inside the window, rising linearly to all of
    them at `ramp_end`. Unigram statistics are untouched; local structure decays.

    THE DRAW IS THE POINT. `ramp_start` and `seed` are chosen by the verifier on
    every grading run out of the declared choice set in frozen/task_spec.json, so
    where the held-out curve turns is a property of THIS run and not something a
    submission can precompute. A policy is a procedure that has to work without
    knowing the draw; this function is what makes that true. It is on the agent
    surface, and calling it with each declared choice is how a policy is designed.
    """
    out = np.asarray(tokens).copy()
    rng = np.random.default_rng(int(seed))
    steps = out.size // tokens_per_step
    blocks_per_step = tokens_per_step // block
    corrupted = 0
    for step in range(steps):
        p = (step - ramp_start) / max(1, ramp_end - ramp_start)
        p = min(max(p, 0.0), 1.0)
        if p <= 0.0:
            continue
        base = step * tokens_per_step
        pick = rng.random(blocks_per_step) < p
        idx = np.nonzero(pick)[0]
        if idx.size == 0:
            continue
        starts = base + idx * block
        window = np.stack([out[s:s + block] for s in starts])
        order = np.argsort(rng.random(window.shape), axis=1)
        window = np.take_along_axis(window, order, axis=1)
        for row, s in enumerate(starts):
            out[s:s + block] = window[row]
        corrupted += idx.size
    return out, corrupted


def draw_degradation(spec: dict, rng) -> dict:
    """Draw this run's fault. Uniform over the DECLARED choices, fresh seed."""
    corpus = spec["corpus"]
    choices = list(corpus["ramp_start_choices"])
    start = int(choices[rng.randrange(len(choices))])
    return {
        "ramp_start": start,
        "ramp_end": start + int(corpus["ramp_span_micro_steps"]),
        "block": int(corpus["degradation_block_tokens"]),
        "seed": rng.randrange(2 ** 62),
    }


def candidate_steps(spec: dict) -> list:
    """The grid of steps a policy may probe. A property of the substrate."""
    grid = spec["evaluation"]["candidate_grid"]
    lo, hi, stride = int(grid["first"]), int(grid["last"]), int(grid["stride"])
    return list(range(lo, hi + 1, stride))


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

    @torch.no_grad()
    def _batch_losses(self, tokens: torch.Tensor, batches: int) -> list:
        """Per-batch cross-entropy over the first `batches` windows of `tokens`."""
        mb, seq = self.micro_batch, self.seq_len
        need = batches * mb * seq + 1
        if tokens.numel() < need:
            raise HarnessError(f"eval stream holds {tokens.numel()} tokens, {need} are needed")
        self.model.eval()
        out = []
        for i in range(batches):
            lo = i * mb * seq
            inputs = tokens[lo: lo + mb * seq].view(mb, seq)
            targets = tokens[lo + 1: lo + mb * seq + 1].view(mb, seq)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(float(self.fn(inputs, targets).double()))
        self.model.train()
        return out

    def sweep(self, recipe, train_tokens, streams, steps_wanted, progress=None) -> dict:
        """ONE frozen training run, instrumented at every candidate step.

        `streams` maps a name to (token tensor, batch count). At every step in
        `steps_wanted` the harness records the per-batch cross-entropy of each named
        stream. Returns {step: {name: [per-batch losses]}} plus a report.

        Nothing here reads a policy. The same measurements are produced on every
        grading run whatever was submitted, which is what lets the shipped default,
        the private reference and the submission be compared without any of them
        costing the verifier a training run of its own.
        """
        mb, seq = self.micro_batch, self.seq_len
        steps = self.micro_steps
        need = steps * mb * seq + 1
        if train_tokens.numel() < need:
            raise HarnessError(f"train stream holds {train_tokens.numel()} tokens, "
                               f"the budget needs {need}")
        wanted = sorted(set(int(s) for s in steps_wanted))
        if wanted and (wanted[0] < 1 or wanted[-1] > steps):
            raise HarnessError(f"candidate steps {wanted[0]}..{wanted[-1]} fall outside 1..{steps}")

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
        curves = {}
        pending = set(wanted)
        for step in range(steps):
            mult = lr_multiplier(recipe["schedule"], step, steps)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * mult
            lo = step * mb * seq
            inputs = train_tokens[lo: lo + mb * seq].view(mb, seq)
            targets = train_tokens[lo + 1: lo + mb * seq + 1].view(mb, seq)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = self.fn(inputs, targets)
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(params, clip, foreach=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            done = step + 1
            if done in pending:
                pending.discard(done)
                curves[done] = {name: self._batch_losses(tokens, batches)
                                for name, (tokens, batches) in streams.items()}
                if progress:
                    shown = "  ".join(f"{n} {float(np.mean(v)):.4f}"
                                      for n, v in sorted(curves[done].items()))
                    print(f"    step {done}/{steps}  {shown}  "
                          f"{time.time() - started:.1f}s", flush=True)
        torch.cuda.synchronize()
        if pending:
            raise HarnessError("candidate steps were never reached: " + repr(sorted(pending)))
        report = {
            "optimizer_steps": steps,
            "measured_steps": sorted(curves),
            "streams": {name: batches for name, (_tokens, batches) in streams.items()},
            "wall_seconds": round(time.time() - started, 2),
        }
        return {"curves": curves, "report": report}


# ---------------------------------------------------------------------------
# Applying a policy to measurements the harness already took
# ---------------------------------------------------------------------------

def probe_estimate(curves: dict, step: int, tokens: int, tokens_per_batch: int,
                   stream: str = "probe") -> float:
    """A policy's own reading of the probe stream at `step`, at its own resolution.

    The policy bought `tokens` worth of evaluation here, so it is answered with the
    mean of the first `tokens // tokens_per_batch` batch losses recorded at this
    step -- fewer tokens is a noisier reading of the same underlying curve, and
    that is the entire cost side of the budget this task hands out.
    """
    batches = int(tokens) // int(tokens_per_batch)
    losses = curves[int(step)][stream][:batches]
    if not losses:
        raise HarnessError(f"probe at step {step} bought no batches")
    return float(np.mean(losses))


def apply_policy(policy: dict, curves: dict, tokens_per_batch: int,
                 stream: str = "probe") -> dict:
    """Select a checkpoint step using ONLY what this policy paid to see.

    The held-out loss is never consulted here. This function sees the probe
    readings the policy bought and nothing else, which is what makes the graded
    number a test of the policy rather than of hindsight.
    """
    readings = []
    for probe in policy["probes"]:
        step = int(probe["step"])
        readings.append({
            "step": step,
            "tokens": int(probe["tokens"]),
            "probe_loss": probe_estimate(curves, step, probe["tokens"], tokens_per_batch, stream),
        })
    readings.sort(key=lambda r: r["step"])

    rule = policy["selection"]["rule"]
    if rule == "argmin":
        scores = [r["probe_loss"] for r in readings]
    elif rule == "argmin_smoothed":
        width = int(policy["selection"]["window"])
        half = width // 2
        scores = []
        for i in range(len(readings)):
            lo, hi = max(0, i - half), min(len(readings), i + half + 1)
            scores.append(float(np.mean([r["probe_loss"] for r in readings[lo:hi]])))
    else:
        raise HarnessError("unknown selection rule " + repr(rule))

    best = int(np.argmin(scores))
    return {
        "readings": readings,
        "scores": scores,
        "selected_step": readings[best]["step"],
        "tokens_spent": sum(r["tokens"] for r in readings),
    }
