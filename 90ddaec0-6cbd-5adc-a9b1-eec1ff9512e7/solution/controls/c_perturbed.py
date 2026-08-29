"""Known wrong control: numerically close and not identical.

This is the control that decides whether the correctness gate is a tolerance or a
divergence. It computes the operator correctly and then moves one output element by a
whole unit, which is a change no tolerance based comparison would care about at the
level of a whole tensor and which the bitwise comparison must reject.

Targets: divergence_bitwise. Every other checker is expected to pass, so the failure
is attributable to the divergence gate alone.
"""

from __future__ import annotations

import fast_impl


def forward(x, g, b, chunk_rows):
    q, r = fast_impl.forward(x, g, b, chunk_rows)
    q = q.clone()
    flat = q.view(-1)
    flat[0] = flat[0] + 1.0
    return q, r
