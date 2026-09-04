#!/usr/bin/env python3
"""The reference allocation the live checkers accept. Private. Not agent-visible.

It writes exactly one artifact, `allocation.json`, into its working directory.
It reports no perplexity, prints no metric, and writes no field the grader reads
as a measurement, because the graded quantity is computed by the verifier from
real forward passes of the frozen checkpoint over a held-out slice this file has
never seen, and a number written here would be ignored on the graded path by
construction.

Why this allocation and not another. The bit budget is fixed at four bits per
parameter over the checkpoint's fifty weight matrices, and the total is spent
exactly, to the bit: 648806400 allocated against 648806400 budgeted. The spend is
moved off the token embedding and the output head, the two largest matrices and
the two whose rows are closest to uniform in scale, and onto the twelve fused
query-key-value projections, whose rows carry the widest dynamic range in the
checkpoint and lose the most to a coarse grid. Eleven of the twelve attention
output projections take a fifth bit and the twelfth stays at the control width,
because that is the residue that makes the budget divide exactly.

The scheme is error-feedback: per-row affine quantization with the rounding
residual carried along the row, which is the lowest-error scheme in the closed
set and also the slowest, since it is sequential in the input dimension.

The allocation is deliberately NOT maximally non-uniform. Pushing the embedding
to two bits buys a larger mean improvement and a wider band, because a coarse
embedding grid makes the reading depend on which tokens a held-out batch happens
to contain, and the graded quantity is the LOWER EDGE of the band rather than
its centre. Establishing a result and having one are not the same thing here,
and this file optimizes for the first.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REFERENCE_ALLOCATION = {
    "scheme": "error-feedback",
    "bits": {
        'wte.weight': 3,
        'blocks.0.attn.qkv.weight': 6,
        'blocks.0.attn.proj.weight': 5,
        'blocks.0.mlp.fc.weight': 5,
        'blocks.0.mlp.proj.weight': 4,
        'blocks.1.attn.qkv.weight': 6,
        'blocks.1.attn.proj.weight': 5,
        'blocks.1.mlp.fc.weight': 5,
        'blocks.1.mlp.proj.weight': 4,
        'blocks.2.attn.qkv.weight': 6,
        'blocks.2.attn.proj.weight': 5,
        'blocks.2.mlp.fc.weight': 5,
        'blocks.2.mlp.proj.weight': 4,
        'blocks.3.attn.qkv.weight': 6,
        'blocks.3.attn.proj.weight': 5,
        'blocks.3.mlp.fc.weight': 5,
        'blocks.3.mlp.proj.weight': 4,
        'blocks.4.attn.qkv.weight': 6,
        'blocks.4.attn.proj.weight': 5,
        'blocks.4.mlp.fc.weight': 5,
        'blocks.4.mlp.proj.weight': 4,
        'blocks.5.attn.qkv.weight': 6,
        'blocks.5.attn.proj.weight': 5,
        'blocks.5.mlp.fc.weight': 5,
        'blocks.5.mlp.proj.weight': 4,
        'blocks.6.attn.qkv.weight': 6,
        'blocks.6.attn.proj.weight': 5,
        'blocks.6.mlp.fc.weight': 5,
        'blocks.6.mlp.proj.weight': 4,
        'blocks.7.attn.qkv.weight': 6,
        'blocks.7.attn.proj.weight': 5,
        'blocks.7.mlp.fc.weight': 5,
        'blocks.7.mlp.proj.weight': 4,
        'blocks.8.attn.qkv.weight': 6,
        'blocks.8.attn.proj.weight': 5,
        'blocks.8.mlp.fc.weight': 5,
        'blocks.8.mlp.proj.weight': 4,
        'blocks.9.attn.qkv.weight': 6,
        'blocks.9.attn.proj.weight': 5,
        'blocks.9.mlp.fc.weight': 5,
        'blocks.9.mlp.proj.weight': 4,
        'blocks.10.attn.qkv.weight': 6,
        'blocks.10.attn.proj.weight': 5,
        'blocks.10.mlp.fc.weight': 5,
        'blocks.10.mlp.proj.weight': 4,
        'blocks.11.attn.qkv.weight': 6,
        'blocks.11.attn.proj.weight': 4,
        'blocks.11.mlp.fc.weight': 5,
        'blocks.11.mlp.proj.weight': 4,
        'lm_head.weight': 3,
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
