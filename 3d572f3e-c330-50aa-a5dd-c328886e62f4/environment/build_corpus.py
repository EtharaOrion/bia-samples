#!/usr/bin/env python3
"""Stage the FineWeb10B shards this slot trains on. Nothing here generates a corpus.

WHAT THIS FILE USED TO BE, AND WHY IT IS NOT THAT ANY MORE

Until the re-base, this file BUILT the corpus: it walked a pool of forty
authored sentences with fixed index arithmetic and wrote out a 21599-byte
training file and a 5739-byte evaluation file. That corpus was a stand-in. A
tokenizer measured on forty sentences about tokenizers is not measured on the
nanoGPT corpus, and the metric it produced resolved against nothing the
substrate names.

This file now STAGES the real thing. The corpus is FineWeb10B, the frozen
dataset of `environment/nanogpt_substrate.json`, in the upstream shard layout
`data/fineweb10B/fineweb_train_*.bin`. This mirrors the substrate's own
`data/cached_fineweb10B.py`: same shards, same names, same layout, same order.

The validation shards are deliberately NOT staged here. The number that scores a
submission is computed by the verifier on a held-out FineWeb slice that no
agent-visible container carries. Asking this file for one returns nothing.

The shards hold uint16 GPT-2 token ids behind a 1024-byte header. A tokenizer
slot needs bytes rather than someone else's tokens, so `environment/harness.py`
decodes them back to the original FineWeb bytes with the GPT-2 byte-pair
encoding, which is exactly invertible. The corpus identity therefore stays
pinned to the shards the substrate names while the free axis stays the
vocabulary that re-expresses them.

Usage:
    python3 environment/build_corpus.py                 # stage the default shard count
    python3 environment/build_corpus.py --shards 8      # stage more training shards
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
REPOSITORY = "kjj0/fineweb10B-gpt2"
TRAIN_PATTERN = "fineweb_train_%06d.bin"
HEADER_MAGIC = 20240520


def manifest() -> dict:
    return json.loads((BASE / "manifest.json").read_text(encoding="utf-8"))


def target_root() -> Path:
    return Path(manifest()["corpus"]["container_train_path"])


def fetch(name: str, root: Path) -> Path:
    """Pull one shard from the pinned upstream repository into the shard root."""
    from huggingface_hub import hf_hub_download

    root.mkdir(parents=True, exist_ok=True)
    local = root / name
    if local.is_file():
        return local
    hf_hub_download(
        repo_id=REPOSITORY,
        filename=name,
        repo_type="dataset",
        local_dir=str(root),
    )
    return local


def verify(path: Path) -> dict:
    """Read the upstream header back, so a truncated download is loud rather than quiet."""
    import numpy as np

    with path.open("rb") as handle:
        header = np.frombuffer(handle.read(256 * 4), dtype=np.int32)
    if int(header[0]) != HEADER_MAGIC:
        raise SystemExit(path.name + " does not carry the upstream magic " + str(HEADER_MAGIC))
    tokens = int(header[2])
    expected = 1024 + tokens * 2
    actual = path.stat().st_size
    if actual != expected:
        raise SystemExit(
            path.name + " is " + str(actual) + " bytes against a header declaring "
            + str(expected)
        )
    return {
        "shard": path.name,
        "tokens": tokens,
        "bytes": actual,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def stage(count: int) -> list:
    root = target_root()
    rows = []
    for index in range(1, count + 1):
        rows.append(verify(fetch(TRAIN_PATTERN % index, root)))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage the FineWeb10B training shards named by the frozen substrate."
    )
    parser.add_argument("--shards", type=int, default=2)
    args = parser.parse_args()
    if args.shards < 1:
        raise SystemExit("stage at least one training shard")
    rows = stage(args.shards)
    print(json.dumps({"root": str(target_root()), "staged": rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
