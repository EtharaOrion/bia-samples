"""Starter submission for BIA-GSN-1. Copy this to the submission path and improve it.

As shipped this delegates straight to the frozen reference implementation, so it is
bitwise correct by construction and its measured speedup is approximately one, which
scores zero under the reward schema. The whole task is to replace the body of forward
with a faster implementation that still returns the identical bit patterns.

The contract is exactly:
    forward(x, g, b, chunk_rows) -> (Y, R)
with x float32 (N, C), g float32 (C,), b float32 (C,), C a power of two, Y float32
(N, C) and R float32 (N,). chunk_rows is the chunk size the frozen reference walks
with. Your implementation is free to ignore it, because the chunking is a property of
the reference implementation and not of the operator.
"""

from __future__ import annotations

import reference_impl


def forward(x, g, b, chunk_rows):
    return reference_impl.reference_forward(x, g, b, chunk_rows)
