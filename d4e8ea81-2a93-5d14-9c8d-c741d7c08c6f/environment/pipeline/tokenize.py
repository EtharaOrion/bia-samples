#!/usr/bin/env python3
"""Stage 2 of 3: tokenize. FREE. You may rewrite this file completely.

Reads the parse artifact and writes the token shards the frozen nanoGPT training
stage consumes, a shard index, and a stage report.

The report this stage writes is LOCALLY HONEST and GLOBALLY INCOMPLETE. Its
bytes-per-token figure is a true statement about the compression achieved on the
stream this stage was fed. It says nothing about how well that stream matches the
held-out FineWeb validation split, and it cannot, because this stage never sees
that split. Coverage is only observable at the end of the chain.

What is frozen here and what is free. The token space is FROZEN: the graded
validation loss is a cross entropy over the substrate's vocab_size of 50304, and
the verifier's held-out shards are encoded with the GPT-2 byte pair encoder, so a
stream encoded in any other id space is not comparable to the split it is scored
against and is graded as a frozen-axis move. What is free is everything about
which text reaches those ids and in what order: which documents survive, how they
are delimited, whether near-duplicates are dropped, how the stream is packed into
shards, and how the shards are ordered. That is the parser construction problem
this slot poses, and it is where the whole trade lives.

Two facts make it a real trade rather than a free lunch. The frozen budget is
counted in TOKENS, at 65536 per step for a bound number of steps, so a stream
that spends fewer tokens on boilerplate lets the same budget cover more source
documents. And the graded loss is a mean cross entropy per token on held-out
FineWeb prose, so a stream that packs itself with text FineWeb does not contain
pays for it at evaluation.

Shard contract, which the frozen stage and the verifier's own held-out shards
both carry:
    a 256-entry int32 header, where header[0] is the magic 20240520, header[1] is
    the version 1 and header[2] is the token count, followed by that many uint16
    token ids. This is the layout data/cached_fineweb10B.py stages.

Usage:
    python3 tokenize.py --parsed <path.jsonl> --shards <dir>
                        --index <path.json> --report <path.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tiktoken

MAGIC = 20240520
VERSION = 1
HEADER_INTS = 256

# The frozen encoder. Not free: the held-out split is encoded with this one.
ENCODING = "gpt2"
EOT = 50256

# Delivered shard size in tokens. Free to change.
SHARD_TOKENS = 1 << 20


def load_parsed(path: Path) -> list:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def encode(rows: list) -> np.ndarray:
    """Delivered policy: every document, in the order parse emitted it.

    Each document is preceded by the end-of-text id, which is the delimiter the
    corpus this model is evaluated against uses to separate documents.
    """
    encoder = tiktoken.get_encoding(ENCODING)
    ids = []
    for row in rows:
        ids.append(EOT)
        ids.extend(encoder.encode_ordinary(row["text"]))
    return np.asarray(ids, dtype=np.uint16)


def write_shard(path: Path, tokens: np.ndarray) -> str:
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = int(tokens.size)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header.tobytes())
        handle.write(tokens.astype(np.uint16).tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(parsed: Path, shards: Path, index: Path, report: Path) -> dict:
    rows = load_parsed(parsed)
    text_bytes = sum(len(row["text"].encode("utf-8")) for row in rows)
    stream = encode(rows)

    written = []
    for position in range(0, max(1, stream.size), SHARD_TOKENS):
        piece = stream[position:position + SHARD_TOKENS]
        if piece.size == 0:
            break
        name = "shard_%03d.bin" % (len(written),)
        digest = write_shard(shards / name, piece)
        written.append({"name": name, "tokens": int(piece.size), "sha256": digest})

    index_payload = json.dumps(
        {
            "schema": "oer13.shards/v1",
            "encoding": ENCODING,
            "vocab_size": 50304,
            "shards": written,
            "tokens": int(stream.size),
        },
        indent=2,
        sort_keys=True,
    ) + "\n"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(index_payload, encoding="utf-8")

    row = {
        "stage": "tokenize",
        "tokens": int(stream.size),
        "shards": len(written),
        "encoding": ENCODING,
        "max_token_id": int(stream.max()) if stream.size else 0,
        "bytes_per_token": round(text_bytes / stream.size, 6) if stream.size else 0.0,
        "input_path": parsed.as_posix(),
        "input_sha256": hashlib.sha256(parsed.read_bytes()).hexdigest(),
        "index_path": index.as_posix(),
        "index_sha256": hashlib.sha256(index_payload.encode("utf-8")).hexdigest(),
        "stage_output_contract": "token-shards-v1",
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    row = run(Path(args.parsed), Path(args.shards), Path(args.index), Path(args.report))
    print("tokenize: tokens=%d shards=%d bpt=%.4f"
          % (row["tokens"], row["shards"], row["bytes_per_token"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
