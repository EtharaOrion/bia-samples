#!/usr/bin/env python3
"""The reference allocation the live checkers accept. Private. Not agent-visible.

It writes exactly one artifact, `allocation.json`, into its working directory.
It reports no perplexity, prints no metric, and writes no field the grader
reads as a measurement, because the graded quantity is computed by the verifier
over the harness-owned quantized state and a number written here would be
ignored on the graded path by construction.

Why this allocation and not another. The bit budget is fixed at four bits per
parameter and the total is spent exactly, to the bit. The spend is moved off the
two largest and least curvature-sensitive tensors, the token embedding and the
output head, and onto the three attention projections whose sensitivity times
outlier factor is highest, which is where the harness error model concentrates
its loss. The scheme is error-feedback because it is the lowest-multiplier
scheme in the closed set the harness implements.

The allocation is deliberately NOT maximally non-uniform. Every departure from
uniform widens the calibration-set jitter on the reading, so an allocation tuned
to the last bit buys a smaller perplexity delta and a wider noise band at the
same time, and lands on the unproven side of the margin with a better mean. The
reference trades a little of the mean for a band narrow enough that the lower
edge clears the bound margin. Establishing a result and having one are not the
same thing here, and this file optimizes for the first.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REFERENCE_ALLOCATION = {
    "scheme": "error-feedback",
    "bits": {
        "emb.tok": 3,
        "blk0.attn.qkv": 5,
        "blk0.attn.proj": 5,
        "blk0.mlp.fc": 4,
        "blk0.mlp.proj": 4,
        "blk1.attn.qkv": 6,
        "blk1.attn.proj": 4,
        "blk1.mlp.fc": 4,
        "blk1.mlp.proj": 4,
        "blk2.attn.qkv": 6,
        "blk2.mlp.fc": 4,
        "lm_head": 3,
    },
}


def canonical() -> bytes:
    return json.dumps(REFERENCE_ALLOCATION, sort_keys=True, separators=(",", ":")).encode()


REFERENCE_SHA256 = hashlib.sha256(canonical()).hexdigest()


def write(directory: Path) -> Path:
    target = Path(directory) / "allocation.json"
    target.write_text(
        json.dumps(REFERENCE_ALLOCATION, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return target


if __name__ == "__main__":
    write(Path.cwd())
