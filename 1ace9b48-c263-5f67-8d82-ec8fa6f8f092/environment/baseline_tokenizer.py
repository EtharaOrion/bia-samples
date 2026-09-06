#!/usr/bin/env python3
"""A complete, valid OER-16 submission. Copy it to /app/submission.py and it runs.

This is the byte-level vocabulary: 256 tokens, one per byte, no merges. It is
the thing to beat, and it is deliberately honest about what it does not know: it
writes no claim, so the divergence check records `claim_absent` and grades it on
the verifier's measurement alone.

It is a weak baseline on purpose. At 256 tokens the fixed budget of 3250
optimizer steps covers roughly one byte per token, so the run sees far less
FineWeb text than a merged vocabulary would let it see inside the same compute.

Usage, and the exact call the verifier makes:

    python3 -I -S submission.py <frozen_dir> <output_dir>

`<frozen_dir>` carries model_config.json, optimizer_config.json,
compute_budget.json, data_config.json and nanogpt_substrate.json. It carries NO
corpus text. The training shards are at the path data_config.json names, and the
graded evaluation slice is held out and is not present anywhere you can reach.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCHEMA = "oer16.tokenizer/v1"


def build() -> list[bytes]:
    """The byte-level vocabulary. No corpus is consulted at all."""
    return [bytes([i]) for i in range(256)]


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: submission.py <frozen_dir> <output_dir>", file=sys.stderr)
        return 2
    frozen = Path(argv[1])
    budget = json.loads((frozen / "compute_budget.json").read_text(encoding="utf-8"))
    tokens = build()
    if len(tokens) > int(budget["max_vocab_size"]):
        print("vocabulary exceeds the announced ceiling", file=sys.stderr)
        return 1
    out = Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    (out / "tokenizer.json").write_text(
        json.dumps(
            {"schema": SCHEMA, "tokens": [t.hex() for t in tokens]},
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    # No claim.json is written. A claim is never a source of reward, and this
    # script measured nothing, so there is nothing it could honestly assert. The
    # absence is recorded by the verifier and graded on the measurement.
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
