"""The tokenizer contract for OER-16. Agent-visible, stdlib only.

This file is the FORMAT, not the grader. The verifier reads `tokenizer.json` as
data and never imports a line of your code, so anything you compute here is
yours to use and is never a score.

`tokenizer.json`
----------------

    {
      "schema": "oer16.tokenizer/v1",
      "tokens": ["61", "62", ..., "74686520"]
    }

`tokens` is a list of lowercase hex strings, one per vocabulary entry, each the
hex encoding of that token's bytes. Two rules the verifier enforces when it
loads the file:

  * every one of the 256 single bytes must appear, so encoding is total and no
    input can ever fall off the vocabulary;
  * the list carries no duplicates, no entry longer than `max_token_bytes`, and
    no more entries than `max_vocab_size`, both read from
    `frozen/compute_budget.json`.

Token ids are the positions in `tokens`, so the file fixes the id assignment.

Encoding
--------

Encoding is greedy longest match, left to right, capped at `max_token_bytes`.
`encode()` below is the exact rule the verifier applies; it is reproduced here
so you can measure locally without guessing. The verifier runs its own copy of
this rule inside its own process, over the vocabulary you shipped.
"""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA = "oer16.tokenizer/v1"


def byte_vocabulary() -> list[bytes]:
    """The 256 single bytes, in id order. Every valid vocabulary contains these."""
    return [bytes([i]) for i in range(256)]


def dump_tokens(tokens: list[bytes]) -> str:
    """The exact bytes of a conforming tokenizer.json."""
    return json.dumps(
        {"schema": SCHEMA, "tokens": [t.hex() for t in tokens]},
        sort_keys=True,
        separators=(",", ":"),
    )


def load_tokens(path: Path) -> list[bytes]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("tokenizer.json declares an unknown schema")
    return [bytes.fromhex(item) for item in payload["tokens"]]


def encode(data: bytes, tokens: list[bytes], max_token_bytes: int) -> list[int]:
    """Greedy longest match, left to right. Total, because the 256 bytes are present."""
    index: dict[bytes, int] = {}
    for ident, token in enumerate(tokens):
        index.setdefault(token, ident)
    longest = min(max_token_bytes, max((len(t) for t in tokens), default=1))
    out: list[int] = []
    position, size = 0, len(data)
    while position < size:
        span = min(longest, size - position)
        while span > 1 and data[position:position + span] not in index:
            span -= 1
        out.append(index[data[position:position + span]])
        position += span
    return out
