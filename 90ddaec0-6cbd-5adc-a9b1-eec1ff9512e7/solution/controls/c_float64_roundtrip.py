"""Known wrong control: a more accurate implementation is still a divergent one.

This computes the whole operator in float64 and rounds the result back to float32. It
is closer to the exact real valued answer than the frozen reference is. It still fails,
because the task freezes an output rather than an accuracy target, and that is the
property that makes approximation worthless here.

Targets: divergence_bitwise.
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
    xd = x.double()
    gd = g.double()
    bd = b.double()
    inv_c = 1.0 / float(x.shape[1])
    sq = xd * xd
    s = _tree_sum_last(sq)
    m = s * inv_c
    d = torch.sqrt(m + EPS)
    nrm = xd / d.unsqueeze(1)
    y = nrm * gd
    y = y + bd
    h = torch.maximum(y, y * ALPHA)
    p = h * h
    q = h + p * BETA
    e = _tree_sum_last(q * q)
    r = torch.sqrt(e * inv_c)
    return q.float(), r.float()
