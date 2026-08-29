"""FROZEN. Reference emulation of the BIA-E4M3N-v1 numeric format for bia slot S07.

This module is harness owned. It is read only to the solving agent, it is hashed into the
bundle content hash, and tests/checkers/format_digest.json pins its sha256 alongside the
sha256 of environment/format.json. Nothing in submission/ may import a private name from
here and nothing may rebind Quantizer.quantize.

Every parameter of the format comes from environment/format.json at import time. There are no
format constants written into this file, so the fixture is the single source of truth and a
reader can confirm the two agree by inspection.

The emulation is pure elementwise software arithmetic over float32 tensors, so it produces
identical results on CPU and on an accelerator. That is what makes BIA_SMOKE=1 a real proof
of the overflow gate rather than a stub: the same Quantizer, the same reduce_in_format, and
the same overflow ledger run in both modes.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
from typing import Any, Dict, List, Sequence, Tuple

import torch

HERE = pathlib.Path(__file__).resolve().parent
FORMAT_PATH = HERE / "format.json"

with open(FORMAT_PATH, "r", encoding="utf-8") as _f:
    FORMAT: Dict[str, Any] = json.load(_f)

FORMAT_ID: str = FORMAT["format_id"]
MANTISSA_BITS: int = int(FORMAT["mantissa_bits"])
EXPONENT_BITS: int = int(FORMAT["exponent_bits"])
EXPONENT_BIAS: int = int(FORMAT["exponent_bias"])
MAX_NORMAL: float = float(FORMAT["max_normal"])
MIN_NORMAL: float = float(FORMAT["min_normal"])
MIN_SUBNORMAL: float = float(FORMAT["min_subnormal"])
SATURATES: bool = bool(FORMAT["saturates_on_overflow"])

# floor(log2(min_normal)), the exponent at which gradual underflow begins.
_E_MIN: float = float(1 - EXPONENT_BIAS)


class PlanError(ValueError):
    """Raised when a submitted reduction plan is not well formed."""


class OverflowLedger:
    """Harness owned counter of overflow events in the frozen format.

    No submitted policy can reach this object to reset or suppress it. The harness reads it,
    writes it into telemetry, and the ABSENCE checker grades the recorded event list.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.total_overflow_elements: int = 0
        self.quantize_calls: int = 0
        self.elements_quantized: int = 0
        self._site: str = "unattributed"
        self._step: int = -1

    def bind(self, step: int, site: str) -> None:
        self._step = int(step)
        self._site = str(site)

    def record(self, n_over: int, n_elem: int) -> None:
        self.quantize_calls += 1
        self.elements_quantized += int(n_elem)
        if n_over > 0:
            self.total_overflow_elements += int(n_over)
            # One record per (step, site) rather than one per element, so a run that
            # overflows on every step produces a bounded, readable event list.
            if self.events and self.events[-1]["step"] == self._step and self.events[-1]["site"] == self._site:
                self.events[-1]["elements"] += int(n_over)
                self.events[-1]["occurrences"] += 1
            else:
                self.events.append(
                    {
                        "step": self._step,
                        "site": self._site,
                        "elements": int(n_over),
                        "occurrences": 1,
                        "format_id": FORMAT_ID,
                        "max_normal": MAX_NORMAL,
                    }
                )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "format_id": FORMAT_ID,
            "overflow_event_count": len(self.events),
            "overflow_elements": self.total_overflow_elements,
            "quantize_calls": self.quantize_calls,
            "elements_quantized": self.elements_quantized,
            "events": list(self.events),
        }


LEDGER = OverflowLedger()


def quantize(x: torch.Tensor) -> torch.Tensor:
    """Round a float32 tensor to the nearest BIA-E4M3N-v1 value, round half to even.

    Gradual underflow through the subnormal range, no saturation, no infinity. A magnitude
    strictly greater than max_normal after rounding is an overflow event: it increments the
    ledger and propagates NaN, exactly as the fixture declares. A non finite input is itself
    counted as an overflow event, because the format encodes no non finite value.
    """
    if x.dtype != torch.float32:
        x = x.to(torch.float32)

    sign = torch.sign(x)
    a = x.abs()

    finite = torch.isfinite(a)
    a_safe = torch.where(finite, a, torch.zeros_like(a))

    # frexp is exact at binade boundaries, unlike floor(log2(.)).
    _mant, exp_i = torch.frexp(a_safe)
    e = (exp_i - 1).to(torch.float32)
    e = torch.clamp(e, min=_E_MIN)

    # Spacing of the 3 bit significand inside binade e is 2**(e - mantissa_bits).
    quantum = torch.exp2(e - float(MANTISSA_BITS))

    q = torch.round(a_safe / quantum) * quantum

    over = (q > MAX_NORMAL) | (~finite)
    n_over = int(over.sum().item())
    LEDGER.record(n_over, a.numel())

    if SATURATES:  # false for BIA-E4M3N-v1; kept so the fixture stays the source of truth
        q = torch.clamp(q, max=MAX_NORMAL)
    else:
        q = torch.where(over, torch.full_like(q, float("nan")), q)

    return sign * q


def is_representable(x: torch.Tensor) -> torch.Tensor:
    """True where x is already an exact BIA-E4M3N-v1 value. Used by the INVARIANT checker."""
    saved = (LEDGER.quantize_calls, LEDGER.elements_quantized)
    q = quantize(x)
    LEDGER.quantize_calls, LEDGER.elements_quantized = saved
    return torch.isnan(q) | (q == x.to(torch.float32))


# ---------------------------------------------------------------------------
# The frozen reduction. The submission supplies the plan; the arithmetic is here.
# ---------------------------------------------------------------------------

_VALID_SHAPES = ("chain", "pairwise")
_VALID_ALIGN = ("min", "max")


def validate_plan(plan: Dict[str, Any], n_chunks: int) -> Dict[str, Any]:
    """Reject a malformed plan loudly rather than silently substituting a default."""
    if not isinstance(plan, dict):
        raise PlanError("plan must be a dict")
    missing = {"order", "shape", "compensated", "leaf_exp", "align", "node_exp"} - set(plan)
    if missing:
        raise PlanError("plan is missing keys: " + ",".join(sorted(missing)))

    order = list(plan["order"])
    if sorted(order) != list(range(n_chunks)):
        raise PlanError("plan.order must be a permutation of range(%d)" % n_chunks)

    shape = plan["shape"]
    if shape not in _VALID_SHAPES:
        raise PlanError("plan.shape must be one of %s" % (_VALID_SHAPES,))

    align = plan["align"]
    if align not in _VALID_ALIGN:
        raise PlanError("plan.align must be one of %s" % (_VALID_ALIGN,))

    leaf_exp = list(plan["leaf_exp"])
    if len(leaf_exp) != n_chunks:
        raise PlanError("plan.leaf_exp must have one entry per chunk")
    for v in leaf_exp:
        if not isinstance(v, int) or isinstance(v, bool) or not (-64 <= v <= 64):
            raise PlanError("plan.leaf_exp entries must be ints in [-64, 64]")

    node_exp = plan["node_exp"]
    if not isinstance(node_exp, int) or isinstance(node_exp, bool) or not (-16 <= node_exp <= 16):
        raise PlanError("plan.node_exp must be an int in [-16, 16]")

    if not isinstance(plan["compensated"], bool):
        raise PlanError("plan.compensated must be a bool")

    return {
        "order": order,
        "shape": shape,
        "compensated": bool(plan["compensated"]),
        "leaf_exp": leaf_exp,
        "align": align,
        "node_exp": int(node_exp),
    }


def _rescale(word: torch.Tensor, shift: int) -> torch.Tensor:
    """Multiply by an exact power of two and re-quantize. The shift is exact; the landing is not."""
    if shift == 0:
        return word
    return quantize(word * (2.0 ** shift))


def _combine(
    a: Tuple[torch.Tensor, torch.Tensor, int],
    b: Tuple[torch.Tensor, torch.Tensor, int],
    align: str,
    node_exp: int,
    compensated: bool,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    ha, la, ea = a
    hb, lb, eb = b
    e = (min(ea, eb) if align == "min" else max(ea, eb)) + node_exp

    ha = _rescale(ha, e - ea)
    hb = _rescale(hb, e - eb)
    if compensated:
        la = _rescale(la, e - ea)
        lb = _rescale(lb, e - eb)

    s = quantize(ha + hb)
    if not compensated:
        return s, torch.zeros_like(s), e

    # TwoSum entirely inside the frozen format. Every intermediate is quantized, so the
    # compensation buys precision without buying range, which is the point.
    bb = quantize(s - ha)
    err = quantize(quantize(hb - bb) + quantize(ha - quantize(s - bb)))
    lo = quantize(quantize(err + la) + lb)
    s2 = quantize(s + lo)
    lo2 = quantize(lo - quantize(s2 - s))
    return s2, lo2, e


def reduce_in_format(
    chunks: Sequence[torch.Tensor],
    plan: Dict[str, Any],
    step: int = -1,
    site: str = "reduce",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Reduce chunk gradients under BIA-E4M3N-v1 following the submitted plan.

    The submission owns order, shape, per chunk exponent, alignment rule, node exponent and
    whether the accumulator is compensated. The submission never owns the arithmetic: every
    operand and every intermediate partial sum passes through quantize() here.

    Returns the unscaled float32 result and an audit record. The unscale is an exact power of
    two, so the only numerical effect of the plan is where values land inside the format.
    """
    n = len(chunks)
    if n == 0:
        raise PlanError("reduce_in_format needs at least one chunk")
    p = validate_plan(plan, n)

    LEDGER.bind(step, site)
    before_calls = LEDGER.quantize_calls
    before_over = LEDGER.total_overflow_elements

    nodes: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
    for idx in p["order"]:
        a = p["leaf_exp"][idx]
        hi = quantize(chunks[idx].to(torch.float32) * (2.0 ** a))
        nodes.append((hi, torch.zeros_like(hi), a))

    depth = 0
    if p["shape"] == "chain":
        acc = nodes[0]
        for nxt in nodes[1:]:
            acc = _combine(acc, nxt, p["align"], p["node_exp"], p["compensated"])
            depth += 1
        root = acc
    else:
        level = nodes
        while len(level) > 1:
            nxt: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
            for i in range(0, len(level) - 1, 2):
                nxt.append(_combine(level[i], level[i + 1], p["align"], p["node_exp"], p["compensated"]))
            if len(level) % 2 == 1:
                nxt.append(level[-1])
            level = nxt
            depth += 1
        root = level[0]

    hi, lo, e = root
    out = (hi.to(torch.float32) + lo.to(torch.float32)) * (2.0 ** (-e))

    audit = {
        "format_id": FORMAT_ID,
        "n_chunks": n,
        "shape": p["shape"],
        "align": p["align"],
        "compensated": p["compensated"],
        "node_exp": p["node_exp"],
        "leaf_exp_min": min(p["leaf_exp"]),
        "leaf_exp_max": max(p["leaf_exp"]),
        "reduction_depth": depth,
        "root_exponent": e,
        "quantize_calls": LEDGER.quantize_calls - before_calls,
        "overflow_elements": LEDGER.total_overflow_elements - before_over,
        "nonfinite_output": bool(not torch.isfinite(out).all().item()),
    }
    return out, audit


def reduce_float32(chunks: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """The full precision control path. No narrow format anywhere, no ledger traffic."""
    acc = chunks[0].to(torch.float32).clone()
    for c in chunks[1:]:
        acc = acc + c.to(torch.float32)
    return acc, {"format_id": "float32", "n_chunks": len(chunks), "quantize_calls": 0, "overflow_elements": 0}


def format_summary() -> Dict[str, Any]:
    return {
        "format_id": FORMAT_ID,
        "exponent_bits": EXPONENT_BITS,
        "mantissa_bits": MANTISSA_BITS,
        "max_normal": MAX_NORMAL,
        "min_normal": MIN_NORMAL,
        "min_subnormal": MIN_SUBNORMAL,
        "machine_epsilon": 2.0 ** (-MANTISSA_BITS - 1),
        "dynamic_range_decades": round(math.log10(MAX_NORMAL / MIN_SUBNORMAL), 4),
        "saturates_on_overflow": SATURATES,
    }
