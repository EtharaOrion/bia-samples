"""Known wrong control: a relaxed precision path enabled as an import side effect.

The arithmetic this control performs is exactly the reference solution's, so its output
is bitwise correct and its speedup is real. It fails anyway, because it turns on a
relaxed precision override while being imported. The absence checker reads the live
process state after the submission has run, which is why an import side effect is
visible to it at all.

Targets: absence_relaxed_precision.
"""

from __future__ import annotations

import os

os.environ["NVIDIA_TF32_OVERRIDE"] = "1"

import fast_impl


def forward(x, g, b, chunk_rows):
    return fast_impl.forward(x, g, b, chunk_rows)
