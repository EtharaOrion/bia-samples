"""Independent scalar re-implementation of BIA-E4M3N-v1 and its plan driven reduction.

This file exists for the DIVERGENCE checker and nothing else. It deliberately shares no code
with environment/bia_numerics.py: it uses plain Python floats and the standard library rather
than torch tensors, it derives the quantum from math.frexp rather than from a vectorised
exponent, and it was written from the fixture in environment/format.json rather than from the
harness source. Two implementations that agree element for element on a recorded reduction is
evidence that the harness computed what the fixture describes. One implementation checking
itself would be evidence of nothing.
"""

from __future__ import annotations

import json
import math
import pathlib
from typing import Any, Dict, List, Sequence, Tuple

NAN = float("nan")


def load_format(path: pathlib.Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class ScalarFormat:
    def __init__(self, fmt: Dict[str, Any]) -> None:
        self.mant = int(fmt["mantissa_bits"])
        self.bias = int(fmt["exponent_bias"])
        self.max_normal = float(fmt["max_normal"])
        self.e_min = 1 - self.bias
        self.saturates = bool(fmt["saturates_on_overflow"])
        self.overflows = 0

    def q(self, x: float) -> float:
        if x != x or x in (float("inf"), float("-inf")):
            self.overflows += 1
            return NAN
        if x == 0.0:
            return 0.0
        sign = -1.0 if x < 0.0 else 1.0
        a = abs(x)
        _m, e = math.frexp(a)
        ex = max(e - 1, self.e_min)
        quantum = 2.0 ** (ex - self.mant)
        # Python's round() is round-half-to-even, matching the fixture's declared mode and
        # torch.round. The quantum is an exact power of two, so the division and the
        # multiplication below introduce no rounding of their own.
        v = round(a / quantum) * quantum
        if v > self.max_normal:
            self.overflows += 1
            return self.max_normal if self.saturates else NAN
        return sign * v


def _rescale(fmt: ScalarFormat, w: float, shift: int) -> float:
    return w if shift == 0 else fmt.q(w * (2.0 ** shift))


def _combine(
    fmt: ScalarFormat,
    a: Tuple[float, float, int],
    b: Tuple[float, float, int],
    align: str,
    node_exp: int,
    compensated: bool,
) -> Tuple[float, float, int]:
    ha, la, ea = a
    hb, lb, eb = b
    e = (min(ea, eb) if align == "min" else max(ea, eb)) + node_exp
    ha = _rescale(fmt, ha, e - ea)
    hb = _rescale(fmt, hb, e - eb)
    if compensated:
        la = _rescale(fmt, la, e - ea)
        lb = _rescale(fmt, lb, e - eb)
    s = fmt.q(ha + hb)
    if not compensated:
        return s, 0.0, e
    bb = fmt.q(s - ha)
    err = fmt.q(fmt.q(hb - bb) + fmt.q(ha - fmt.q(s - bb)))
    lo = fmt.q(fmt.q(err + la) + lb)
    s2 = fmt.q(s + lo)
    lo2 = fmt.q(lo - fmt.q(s2 - s))
    return s2, lo2, e


def reduce_scalar(fmt: ScalarFormat, values: Sequence[float], plan: Dict[str, Any]) -> float:
    """values[i] is chunk i's value for one element. Returns the unscaled result."""
    nodes: List[Tuple[float, float, int]] = []
    for idx in plan["order"]:
        a = int(plan["leaf_exp"][idx])
        nodes.append((fmt.q(values[idx] * (2.0 ** a)), 0.0, a))

    if plan["shape"] == "chain":
        acc = nodes[0]
        for nxt in nodes[1:]:
            acc = _combine(fmt, acc, nxt, plan["align"], int(plan["node_exp"]), bool(plan["compensated"]))
        root = acc
    else:
        level = nodes
        while len(level) > 1:
            nxt: List[Tuple[float, float, int]] = []
            for i in range(0, len(level) - 1, 2):
                nxt.append(_combine(fmt, level[i], level[i + 1], plan["align"], int(plan["node_exp"]), bool(plan["compensated"])))
            if len(level) % 2 == 1:
                nxt.append(level[-1])
            level = nxt
        root = level[0]

    hi, lo, e = root
    return (hi + lo) * (2.0 ** (-e))


def replay(fmt_path: pathlib.Path, sample: Dict[str, Any]) -> List[float]:
    fmt = ScalarFormat(load_format(fmt_path))
    chunks = sample["chunks"]
    plan = sample["plan"]
    n = int(sample["elements"])
    return [reduce_scalar(fmt, [chunks[c][i] for c in range(len(chunks))], plan) for i in range(n)]
