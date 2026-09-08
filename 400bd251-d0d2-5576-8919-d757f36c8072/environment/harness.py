#!/usr/bin/env python3
"""The frozen harness for OER-28: build once, check equivalence, then time.

BYTE-IDENTICAL on the agent surface and inside the verifier; tests/Dockerfile
asserts that with `cmp` at build time. A plan measured locally with
`python3 bench_local.py` is measured at grading time by the same code, on the same
parameters, over the same batch shape.

THREE PROPERTIES ARE LOAD-BEARING.

1. ONE set of parameters, built once under the frozen seed and reused by every
   plan. Two plans differ by their execution and by nothing else, so the
   equivalence check is a comparison of implementations rather than of models.

2. EQUIVALENCE IS CHECKED BEFORE ANYTHING IS TIMED. A plan that does not agree
   with the reference plan on the loss and on the gradient norm is refused. Speed
   bought by computing a different function is not speed.

3. TIMING IS INTERLEAVED AND TAKES A MEDIAN. The plans are timed round-robin and
   each plan's reported latency is the MEDIAN over its rounds. A busy machine is
   busy for everybody within a round, and the reward is a ratio of DIFFERENCES of
   latencies, so a slowdown that multiplies all three equally cancels exactly.

   The median rather than the minimum is a measurement rather than a preference.
   The first round after warmup is systematically faster than the rest here, and
   its plans are timed at different points of that ramp, so a minimum picks the
   one anomalous round and picks it unevenly across plans. Rounds after the first
   agree with each other to within half a percent. `min_ms` is still reported in
   the score document because it is informative to a reader; nothing grades on it.
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


def load_shard(path, ntokens=None) -> np.ndarray:
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


def reference_plan(spec: dict) -> dict:
    """The canonical execution every other plan is checked against. Public."""
    ref = spec["reference_plan"]
    return {k: v for k, v in ref.items() if k != "note"}


class Bench:
    """The frozen parameters, plus the two things that can be done with them."""

    def __init__(self, spec: dict, device="cuda"):
        self.spec = spec
        self.device = device
        arch = spec["architecture"]
        self.micro_batch = int(spec["step"]["micro_batch"])
        self.seq_len = int(arch["seq_len"])
        if self.micro_batch * self.seq_len != int(spec["step"]["tokens_per_step"]):
            raise HarnessError("step is inconsistent: micro_batch * seq_len != tokens_per_step")
        nanogpt = load_model_module()
        torch.manual_seed(int(spec["init"]["seed"]))
        self.model = nanogpt.GPT(
            vocab_size=int(arch["vocab_size"]),
            num_layers=int(arch["num_layers"]),
            model_dim=int(arch["model_dim"]),
            head_dim=int(arch["head_dim"]),
        ).to(device)
        self.parameters = sum(p.numel() for p in self.model.parameters())
        self.params = list(self.model.parameters())

    def _batch(self, tokens: torch.Tensor, index: int):
        mb, seq = self.micro_batch, self.seq_len
        lo = index * mb * seq
        need = lo + mb * seq + 1
        if tokens.numel() < need:
            raise HarnessError(f"token stream holds {tokens.numel()}, batch {index} needs {need}")
        return (tokens[lo: lo + mb * seq].view(mb, seq),
                tokens[lo + 1: lo + mb * seq + 1].view(mb, seq))

    def _zero(self):
        for p in self.params:
            p.grad = None

    def forward_backward(self, plan: dict, tokens: torch.Tensor, index: int):
        """One step. Returns (loss, global gradient norm), both as floats."""
        inputs, targets = self._batch(tokens, index)
        self._zero()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = self.model(inputs, targets, plan)
        loss.backward()
        total = torch.zeros((), device=self.device, dtype=torch.float64)
        for p in self.params:
            if p.grad is not None:
                total += p.grad.double().pow(2).sum()
        norm = float(total.sqrt().item())
        value = float(loss.detach())
        self._zero()
        return value, norm

    def equivalence(self, plan: dict, tokens: torch.Tensor) -> dict:
        """Compare a plan against the reference plan on the same frozen weights."""
        batches = int(self.spec["equivalence"]["batches"])
        ref = reference_plan(self.spec)
        rows = []
        for i in range(batches):
            ref_loss, ref_norm = self.forward_backward(ref, tokens, i)
            got_loss, got_norm = self.forward_backward(plan, tokens, i)
            rows.append({
                "batch": i,
                "reference_loss": ref_loss,
                "plan_loss": got_loss,
                "loss_abs_delta": abs(got_loss - ref_loss),
                "reference_grad_norm": ref_norm,
                "plan_grad_norm": got_norm,
                "grad_relative_delta": (abs(got_norm - ref_norm) / ref_norm
                                        if ref_norm > 0 else float("inf")),
            })
        worst_loss = max(r["loss_abs_delta"] for r in rows)
        worst_grad = max(r["grad_relative_delta"] for r in rows)
        finite = all(math.isfinite(r["plan_loss"]) and math.isfinite(r["plan_grad_norm"])
                     for r in rows)
        tol_loss = float(self.spec["equivalence"]["loss_tolerance"])
        tol_grad = float(self.spec["equivalence"]["grad_relative_tolerance"])
        return {
            "batches": rows,
            "worst_loss_abs_delta": worst_loss,
            "worst_grad_relative_delta": worst_grad,
            "loss_tolerance": tol_loss,
            "grad_relative_tolerance": tol_grad,
            "finite": finite,
            "equivalent": bool(finite and worst_loss <= tol_loss and worst_grad <= tol_grad),
        }

    def time_round(self, plan: dict, tokens: torch.Tensor) -> float:
        """Mean CUDA-event-timed milliseconds per step over one round."""
        steps = int(self.spec["timing"]["steps_per_round"])
        inputs, targets = self._batch(tokens, 0)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(steps):
            self._zero()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = self.model(inputs, targets, plan)
            loss.backward()
        end.record()
        torch.cuda.synchronize()
        self._zero()
        return float(start.elapsed_time(end)) / steps

    def warmup(self, plan: dict, tokens: torch.Tensor) -> None:
        steps = int(self.spec["timing"]["warmup_steps"])
        inputs, targets = self._batch(tokens, 0)
        for _ in range(steps):
            self._zero()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = self.model(inputs, targets, plan)
            loss.backward()
        self._zero()
        torch.cuda.synchronize()


def interleaved_latencies(bench: Bench, plans: dict, tokens: torch.Tensor,
                          verbose: bool = True) -> dict:
    """Time every plan round-robin and report the minimum round for each.

    The round-robin is the point. Timing one plan to completion and then the next
    would let a machine that got busier halfway through show up as a difference
    between plans. Here every round pays whatever the machine is doing at that
    moment, for all of them.

    Every plan is warmed up before ANY plan is timed, so no plan pays another
    plan's first-call costs, and the caller grades on the median rather than the
    minimum of what comes back.
    """
    rounds = int(bench.spec["timing"]["rounds"])
    for label, plan in plans.items():
        bench.warmup(plan, tokens)
        if verbose:
            print(f"[warmup] {label} done", flush=True)
    samples = {label: [] for label in plans}
    for r in range(rounds):
        for label, plan in plans.items():
            samples[label].append(bench.time_round(plan, tokens))
        if verbose:
            row = "  ".join(f"{label}={samples[label][-1]:.2f}ms" for label in plans)
            print(f"[round {r + 1}/{rounds}] {row}", flush=True)
    out = {}
    for label, values in samples.items():
        out[label] = {
            "samples_ms": [round(v, 4) for v in values],
            "min_ms": min(values),
            "median_ms": sorted(values)[len(values) // 2],
            "mean_ms": sum(values) / len(values),
        }
    return out
