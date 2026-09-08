from __future__ import annotations

import os
import pathlib

import numpy as np

CORPUS_ROOT = pathlib.Path(os.environ.get("BIA_CORPUS_ROOT", "/workspace/data/fineweb10B"))

MAGIC = 20240520
VERSION = 1
HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * 4

VOCAB_SIZE = 50304
REAL_TOKEN_IDS = 50257

SPLIT_PATTERNS = {
    "train": "fineweb_train_*.bin",
    "dev": "fineweb_val_*.bin",
    "val": "fineweb_val_*.bin",
}

SPLIT_REGIONS = {
    "train": (0.0, 1.0),
    "dev": (0.0, 0.5),
    "val": (0.5, 1.0),
}

def shard_paths(split: str, root=None) -> list:
    if split not in SPLIT_PATTERNS:
        raise KeyError(f"unknown split {split!r}, expected one of {sorted(SPLIT_PATTERNS)}")
    base = pathlib.Path(root) if root is not None else CORPUS_ROOT
    files = sorted(base.glob(SPLIT_PATTERNS[split]))
    if not files:
        raise FileNotFoundError(
            f"no {split} shards matching {SPLIT_PATTERNS[split]!r} under {base}. "
            "The corpus is mounted by the task image at /workspace/data; set "
            "BIA_CORPUS_ROOT to point elsewhere.")
    return files

def shard_token_count(path) -> int:
    header = np.fromfile(str(path), dtype=np.int32, count=HEADER_INTS)
    if header.size < HEADER_INTS:
        raise ValueError(f"{path}: truncated header, read {header.size} of {HEADER_INTS} int32")
    if int(header[0]) != MAGIC:
        raise ValueError(f"{path}: magic number mismatch, got {int(header[0])} want {MAGIC}")
    if int(header[1]) != VERSION:
        raise ValueError(f"{path}: unsupported version {int(header[1])}, want {VERSION}")
    count = int(header[2])
    expected = HEADER_BYTES + 2 * count
    actual = os.path.getsize(path)
    if actual != expected:
        raise ValueError(
            f"{path}: header claims {count} tokens ({expected} bytes), file is {actual}")
    return count

def _shard_tokens(path, count: int):
    return np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_BYTES, shape=(count,))

def _read_range(files, counts, offset: int, length: int) -> np.ndarray:
    out = np.empty(length, dtype=np.int64)
    filled = 0
    cursor = 0
    for path, count in zip(files, counts):
        if filled >= length:
            break
        end = cursor + count
        if end <= offset:
            cursor = end
            continue
        local = max(0, offset - cursor)
        take = min(count - local, length - filled)
        tokens = _shard_tokens(path, count)
        out[filled:filled + take] = tokens[local:local + take]
        del tokens
        filled += take
        cursor = end
    if filled < length:
        raise ValueError(
            f"corpus exhausted: wanted {length} tokens from offset {offset}, got {filled}")
    return out

def corpus_stream(split: str, draw: int, length: int, root=None) -> np.ndarray:
    length = int(length)
    draw = int(draw)
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    files = shard_paths(split, root)
    counts = [shard_token_count(path) for path in files]
    total = sum(counts)
    low, high = SPLIT_REGIONS[split]
    region_start = int(total * low)
    region_tokens = int(total * high) - region_start
    if region_tokens < length:
        raise ValueError(
            f"split {split!r} holds {region_tokens} tokens in its region, {length} requested")
    windows = max(1, region_tokens // length)
    offset = region_start + (draw % windows) * length
    return _read_range(files, counts, offset, length)

def batch_order(seed: int, count: int, high: int) -> np.ndarray:
    return np.random.default_rng(int(seed) + 7919).integers(0, high, size=count)
