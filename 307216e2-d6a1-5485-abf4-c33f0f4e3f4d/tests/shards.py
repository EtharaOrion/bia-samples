"""The frozen shard builder. Turns a submission corpus into nanoGPT training shards.

This file is half of the frozen path for slot OER-19. The agent does not modify it
and cannot reach it at grading time: the verifier runs its own copy at
`tests/shards.py` and the agent sees a byte-identical mirror here, whose sha256 is
pinned in `tests/checkers.yaml`.

What it does is the whole bridge between this slot's free axis and its substrate.
The agent's generator emits documents. This module encodes those documents with the
GPT-2 byte-pair tokenizer the nanoGPT corpus is built with, truncates each document
to the bound sequence length, and packs the result into a `.bin` shard in exactly
the layout `data/cached_fineweb10B.py` writes: a 256-int32 header whose first three
words are the magic number 20240520, the format version 1 and the token count,
followed by the tokens as uint16. The trainer then reads that shard with the same
reader it reads a FineWeb shard with, so the corpus the agent generated and the
corpus the substrate names are consumed by one code path and not by two.

Truncation rather than padding is deliberate. A padded document would let a
generator meet the token bound with filler that carries no gradient signal of its
own, and the packed shard would then describe the padding rather than the
generator. A document that does not reach the bound length is a short document and
is reported as one, never quietly extended.

The tokenizer vocabulary is warmed into the image at build time and read from
`TIKTOKEN_CACHE_DIR`. The verifier's egress is denied, so a tokenizer that reached
for the network at grading time would fail closed rather than grade; warming it at
build is what makes the denied-egress verifier able to run at all.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# The nanoGPT shard container, read from data/cached_fineweb10B.py in the vendored
# record set. Header is 256 int32 words; the tokens follow as uint16.
HEADER_WORDS = 256
HEADER_MAGIC = 20240520
HEADER_VERSION = 1

# The GPT-2 byte-pair encoding the FineWeb10B shards are built with. 50257 real
# tokens; the declared vocab_size 50304 is that count padded up for the matmul, so
# no token id above 50256 is ever emitted by the encoder.
ENCODING_NAME = "gpt2"
END_OF_TEXT = 50256

# Bound document length, in tokens. Equal to the substrate's seq_len, so one
# document is exactly one training sequence and the emission timeline the diversity
# statistic reads is the same timeline the optimizer consumes.
BOUND_DOCUMENT_TOKENS = 1024

_ENCODER = None


def encoder():
    """The GPT-2 encoder, loaded once. Reads the warmed cache, never the network."""
    global _ENCODER
    if _ENCODER is None:
        import tiktoken

        _ENCODER = tiktoken.get_encoding(ENCODING_NAME)
    return _ENCODER


def encode_document(text, bound_tokens: int = BOUND_DOCUMENT_TOKENS) -> dict:
    """Encode one document and truncate it to the bound length.

    Returns the token ids actually kept and the length the document reached before
    truncation, so a caller can tell a document that filled the bound from one that
    fell short of it. A short document is reported, never padded.
    """
    ids = encoder().encode_ordinary(str(text))
    reached = len(ids)
    kept = ids[:bound_tokens]
    return {"tokens": kept, "reached": reached, "short": reached < bound_tokens}


def encode_corpus(texts, bound_tokens: int = BOUND_DOCUMENT_TOKENS) -> dict:
    """Encode a corpus in emission order and report what each document reached."""
    kept, reached, short = [], [], 0
    for text in texts:
        row = encode_document(text, bound_tokens)
        kept.append(row["tokens"])
        reached.append(row["reached"])
        if row["short"]:
            short += 1
    return {
        "documents": kept,
        "reached": reached,
        "short_documents": short,
        "total_tokens": sum(len(row) for row in kept),
    }


def write_shard(path, token_lists) -> dict:
    """Write one nanoGPT shard in the exact layout the FineWeb loader reads.

    The documents are concatenated in emission order with no separator inserted
    between them, because each document is already exactly one sequence at the
    bound length and a separator would push the sequence boundary off the document
    boundary. A short document shifts every boundary after it, which is why a short
    document is a bound violation rather than a rounding detail.
    """
    flat = []
    for row in token_lists:
        flat.extend(int(value) for value in row)
    tokens = np.asarray(flat, dtype=np.uint32)
    if tokens.size and int(tokens.max()) > END_OF_TEXT:
        raise ValueError("token id above the gpt2 vocabulary reached the shard writer")
    header = np.zeros(HEADER_WORDS, dtype=np.int32)
    header[0] = HEADER_MAGIC
    header[1] = HEADER_VERSION
    header[2] = int(tokens.size)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        handle.write(header.tobytes())
        handle.write(tokens.astype(np.uint16).tobytes())
    return {"path": str(target), "tokens": int(tokens.size)}


def read_shard(path) -> np.ndarray:
    """Read a nanoGPT shard back, validating the header rather than trusting it."""
    with Path(path).open("rb") as handle:
        raw = handle.read(HEADER_WORDS * 4)
        if len(raw) != HEADER_WORDS * 4:
            raise ValueError("shard header is truncated")
        header = struct.unpack("<256i", raw)
        if header[0] != HEADER_MAGIC:
            raise ValueError("shard magic is " + str(header[0]) + ", expected " + str(HEADER_MAGIC))
        if header[1] != HEADER_VERSION:
            raise ValueError("shard version is " + str(header[1]))
        count = int(header[2])
        tokens = np.frombuffer(handle.read(count * 2), dtype=np.uint16)
    if tokens.size != count:
        raise ValueError("shard declares " + str(count) + " tokens and carries " + str(tokens.size))
    return tokens.astype(np.int64)


def build_train_shard(texts, directory, bound_tokens: int = BOUND_DOCUMENT_TOKENS) -> dict:
    """Encode a submission corpus and lay it down as fineweb_train_000001.bin.

    The filename follows the substrate's declared train glob
    data/fineweb10B/fineweb_train_*.bin, so the shard the agent's corpus produces is
    named and read exactly as a corpus shard and not as a special case.
    """
    encoded = encode_corpus(texts, bound_tokens)
    written = write_shard(Path(directory) / "fineweb_train_000001.bin", encoded["documents"])
    return {
        "shard": written["path"],
        "shard_tokens": written["tokens"],
        "documents": len(encoded["documents"]),
        "short_documents": encoded["short_documents"],
        "document_tokens_reached": encoded["reached"],
    }


def build_val_tokens(items, bound_tokens: int = BOUND_DOCUMENT_TOKENS) -> np.ndarray:
    """Encode the held-out split into one flat token stream for evaluation.

    Each held-out item is truncated to the same bound length as a training document
    and the stream is the concatenation in file order, so the evaluation windows sit
    on item boundaries. The end-of-text token separates items here, because unlike
    the training shard the evaluation stream is scored per token and a run-on
    between two unrelated documents would be scored as if it were coherent text.
    """
    flat = []
    for text in items:
        ids = encoder().encode_ordinary(str(text))[:bound_tokens]
        if not ids:
            continue
        flat.extend(int(value) for value in ids)
        flat.append(END_OF_TEXT)
    return np.asarray(flat, dtype=np.int64)
