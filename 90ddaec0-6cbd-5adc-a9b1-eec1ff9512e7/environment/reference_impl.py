"""Frozen operator core for BIA-GSN-1.

This file defines the mathematical output of the task. It is frozen: the graded
comparison is bitwise against the tensors reference_forward returns. Two copies of
this file ship in the bundle and they are byte identical. The copy under
environment/ is the agent visible statement of the operator. The copy the verifier
imports is private and is the only one grading reads.

Every arithmetic primitive used here is add, subtract, multiply, divide, square
root, or maximum on IEEE-754 binary32. Each of those is correctly rounded, so the
value of every output element is fixed by the op sequence alone and does not depend
on the device, the backend, or the memory layout. Nothing here calls a transcendental
function, a matrix multiply, or a library reduction whose accumulation order is an
implementation detail. The two reductions are explicit adjacent pairwise binary
trees, so their summation order is part of the specification rather than a property
of whatever kernel happens to run.
"""

from __future__ import annotations

import json
import pathlib

import torch

OPERATOR_ID = "BIA-GSN-1"

# 2**-13. Exactly representable in binary32, so the constant is unambiguous.
EPS = 1.220703125e-04
# Exactly representable in binary32.
ALPHA = 0.5
BETA = 1.5

FIXTURE_KINDS = ("normal", "dense", "tiny", "huge", "sparse", "signed_zero")


def load_spec(path):
    """Read the frozen workload spec written beside this file."""
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def tree_sum_last(v):
    """Adjacent pairwise binary tree sum over the last dimension.

    The length of the last dimension must be a power of two. Level k adds element
    2*i to element 2*i+1 for every i, which fixes the summation order exactly.
    """
    n = int(v.shape[-1])
    if n & (n - 1):
        raise ValueError("tree_sum_last requires a power of two length")
    while n > 1:
        v = v[..., 0::2] + v[..., 1::2]
        n = int(v.shape[-1])
    return v[..., 0]


def tree_sum_first(v):
    """Adjacent pairwise binary tree sum over the first dimension.

    Identical pairing and identical order to tree_sum_last, applied to the leading
    axis instead of the trailing one, so a transposed layout reduces to the same
    value bit for bit.
    """
    n = int(v.shape[0])
    if n & (n - 1):
        raise ValueError("tree_sum_first requires a power of two length")
    while n > 1:
        v = v[0::2] + v[1::2]
        n = int(v.shape[0])
    return v[0]


def reference_forward(x, g, b, chunk_rows):
    """The frozen reference implementation of BIA-GSN-1.

    x is (N, C) float32 with C a power of two, g and b are (C,) float32. The return
    is (Y, R) where Y is (N, C) float32 and R is (N,) float32.

    Per row the operator is exactly:
        s   = tree_sum(x[i] * x[i])
        m   = s * (1 / C)
        d   = sqrt(m + EPS)
        n   = x[i] / d
        y   = n * g + b
        h   = max(y, y * ALPHA)
        p   = h * h
        q   = h + p * BETA
        e   = tree_sum(q * q)
        r   = sqrt(e * (1 / C))
    with Y[i] = q and R[i] = r.

    This implementation walks the rows in chunks, works on a transposed view of each
    chunk, and rebuilds its loop invariants on every iteration. Those are properties
    of this implementation and not of the operator. The operator is the per element
    op sequence above, and any implementation that reproduces that sequence element
    by element reproduces these bytes exactly.
    """
    if x.dtype != torch.float32 or g.dtype != torch.float32 or b.dtype != torch.float32:
        raise TypeError("BIA-GSN-1 is defined on float32 inputs only")
    if x.dim() != 2:
        raise ValueError("x must be two dimensional")
    n_rows, n_cols = int(x.shape[0]), int(x.shape[1])
    if n_cols & (n_cols - 1):
        raise ValueError("C must be a power of two")
    out = torch.empty_like(x)
    rms = torch.empty(n_rows, dtype=x.dtype, device=x.device)
    start = 0
    while start < n_rows:
        stop = min(start + int(chunk_rows), n_rows)
        inv_c = 1.0 / float(n_cols)
        gg = g.unsqueeze(1)
        bb = b.unsqueeze(1)
        w = x[start:stop].t()
        sq = w * w
        s = tree_sum_first(sq)
        m = s * inv_c
        d = torch.sqrt(m + EPS)
        nrm = w / d
        y = nrm * gg
        y = y + bb
        h = torch.maximum(y, y * ALPHA)
        p = h * h
        q = h + p * BETA
        e = tree_sum_first(q * q)
        r = torch.sqrt(e * inv_c)
        out[start:stop] = q.t()
        rms[start:stop] = r
        start = stop
    return out, rms


def make_inputs(kind, rows, cols, seed, device):
    """Deterministic fixture construction.

    The harness builds each fixture once and hands the identical tensors to the
    reference implementation and to the submitted implementation, so this function
    never has to reproduce a value across two devices. Every scale factor is a power
    of two, which keeps the mantissa of the drawn sample intact.
    """
    if kind not in FIXTURE_KINDS:
        raise ValueError("unknown fixture kind: %s" % kind)
    dev = torch.device(device)
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))
    base = torch.randn(rows, cols, generator=gen, device=dev, dtype=torch.float32)
    if kind == "normal":
        x = base
    elif kind == "dense":
        u = torch.rand(rows, cols, generator=gen, device=dev, dtype=torch.float32)
        pos = torch.ones((), device=dev, dtype=torch.float32)
        neg = torch.full((), -1.0, device=dev, dtype=torch.float32)
        sign = torch.where(base >= 0.0, pos, neg)
        x = (u + 1.0) * sign
    elif kind == "tiny":
        x = base * (2.0 ** -30)
    elif kind == "huge":
        x = base * (2.0 ** 30)
    elif kind == "sparse":
        x = base.clone()
        x[0::7] = 0.0
    else:
        x = base.clone()
        x[1::5] = -0.0
        x[2::9] = 0.0
    gen_p = torch.Generator(device=dev)
    gen_p.manual_seed(int(seed) + 977)
    g = torch.randn(cols, generator=gen_p, device=dev, dtype=torch.float32)
    b = torch.randn(cols, generator=gen_p, device=dev, dtype=torch.float32) * 0.25
    return x.contiguous(), g.contiguous(), b.contiguous()


def bitwise_equal(a, c):
    """Exact comparison of two float32 tensors on their raw 32 bit patterns.

    Bit patterns rather than values, so a NaN payload difference and a signed zero
    difference both count as a divergence. An implementation that is numerically
    close but not identical fails here.
    """
    if not isinstance(a, torch.Tensor) or not isinstance(c, torch.Tensor):
        return False, "not_a_tensor"
    if a.dtype != c.dtype:
        return False, "dtype_mismatch"
    if a.dtype != torch.float32:
        return False, "dtype_not_float32"
    if tuple(a.shape) != tuple(c.shape):
        return False, "shape_mismatch"
    ai = a.detach().contiguous().view(torch.int32)
    ci = c.detach().contiguous().view(torch.int32)
    same = bool(torch.equal(ai, ci))
    return same, "ok" if same else "bit_pattern_mismatch"


def first_divergence(a, c):
    """Flat index of the first differing 32 bit word, or -1 when none differs."""
    ai = a.detach().contiguous().view(torch.int32).reshape(-1)
    ci = c.detach().contiguous().view(torch.int32).reshape(-1)
    if ai.numel() != ci.numel():
        return -1
    ne = (ai != ci).nonzero()
    if ne.numel() == 0:
        return -1
    return int(ne[0][0].item())
