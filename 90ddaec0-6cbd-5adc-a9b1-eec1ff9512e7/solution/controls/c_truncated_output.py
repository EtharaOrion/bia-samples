"""Known wrong control: the declared outputs are not materialized.

This returns the correct values for the first half of each row and nothing beyond it.
The artifact it writes is half the declared byte length, so the effect checker cannot
find the named change in the artifact tree.

Targets: effect_outputs_materialized. The divergence checker also fires on the length
mismatch, which is expected: a submission that does not produce the declared shape
cannot be compared word by word against one that does.
"""

from __future__ import annotations

import fast_impl


def forward(x, g, b, chunk_rows):
    q, r = fast_impl.forward(x, g, b, chunk_rows)
    half = int(q.shape[1]) // 2
    return q[:, :half].contiguous(), r
