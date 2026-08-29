"""Private reference solution for the BIA-GSN-1 throughput task.

The submitted implementation must return the same bit patterns as the frozen
reference and must return them faster. This implementation earns its speedup by
removing work that belongs to the reference implementation rather than to the
operator, and by touching nothing that belongs to the operator.

What it removes, in order of expected effect on wall clock:

1. The row chunk loop. The reference walks the rows in chunks of chunk_rows and
   issues a full op sequence per chunk, so a workload of N rows costs N / chunk_rows
   times the launch overhead of a single pass. This implementation issues one op
   sequence for all rows. Elementwise operations are independent per element, so
   merging the chunks changes which elements share a kernel launch and changes no
   element's value.
2. The transposed working view. The reference transposes each chunk to (C, rows)
   and computes on that non contiguous view, so every access walks a stride of rows
   elements. This implementation keeps the natural (N, C) contiguous layout and
   reduces over the trailing axis. Memory layout is not an input to any of the
   arithmetic primitives in the operator, so the values are unchanged.
3. The loop invariants. The reference rebuilds 1 / C and the two unsqueezed parameter
   views on every chunk. This implementation builds each once.

What it deliberately does not do, because each of these would change the bit pattern
and would therefore fail the divergence gate:

  no float16, bfloat16, or TF32 anywhere,
  no fused multiply add, so n * g and + b stay two separately rounded operations,
  no rsqrt or reciprocal in place of the divide by d,
  no library reduction in place of the specified adjacent pairwise tree,
  no reassociation of the tree, no split reduction, no Kahan or compensated variant,
  no torch.compile, since its default fusion is free to contract a multiply and an
  add into a single fused multiply add and free to reassociate a reduction.

The value proposition of this solution is therefore not that it is clever. It is that
every transformation it applies is provably value preserving at the element level,
and that it applies no transformation that is merely close.
"""

from __future__ import annotations

import torch

EPS = 1.220703125e-04
ALPHA = 0.5
BETA = 1.5


def _tree_sum_last(v):
    n = int(v.shape[-1])
    while n > 1:
        v = v[..., 0::2] + v[..., 1::2]
        n = int(v.shape[-1])
    return v[..., 0]


def forward(x, g, b, chunk_rows):
    """Bitwise identical to reference_impl.reference_forward, in one pass over the rows.

    chunk_rows is accepted and ignored. The chunking is a property of the reference
    implementation and carries no arithmetic meaning, which is exactly why ignoring it
    is safe and why doing so is where most of the speedup comes from.
    """
    if x.dtype != torch.float32 or g.dtype != torch.float32 or b.dtype != torch.float32:
        raise TypeError("BIA-GSN-1 is defined on float32 inputs only")
    n_cols = int(x.shape[1])
    inv_c = 1.0 / float(n_cols)
    xc = x if x.is_contiguous() else x.contiguous()
    sq = xc * xc
    s = _tree_sum_last(sq)
    m = s * inv_c
    d = torch.sqrt(m + EPS)
    nrm = xc / d.unsqueeze(1)
    y = nrm * g
    y = y + b
    h = torch.maximum(y, y * ALPHA)
    p = h * h
    q = h + p * BETA
    e = _tree_sum_last(q * q)
    r = torch.sqrt(e * inv_c)
    return q, r
