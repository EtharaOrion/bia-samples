#!/usr/bin/env python3
"""Stage 2 of 3: tokenize. FREE. You may rewrite this file completely.

Reads the parse artifact and writes a flat token stream, a vocabulary, and a
stage report.

The report this stage writes is LOCALLY HONEST and GLOBALLY INCOMPLETE. Its
bytes-per-token figure is a true statement about the compression achieved on the
stream this stage was fed. It says nothing about how well that vocabulary covers
the held-out split, and it cannot, because this stage never sees the held-out
split. Coverage is only observable at the end of the chain.

Two facts make this a real trade rather than a free lunch. The frozen token
budget is counted in TOKENS, so a vocabulary that compresses harder lets the same
budget cover more source bytes. And the graded loss is normalised per BYTE of the
held-out split, so a vocabulary that compresses the training stream by merging
units the held-out split never contains pays for it at evaluation, where every
unmatched byte falls back to the reserved unknown unit.

Vocabulary contract:
    index 0 is the reserved unknown unit "<unk>"; every other index is a literal
    string. Segmentation, at training and at evaluation alike, is greedy longest
    match over the unit set.

Usage:
    python3 tokenize.py --parsed <path.jsonl> --tokens <path.json>
                        --vocab <path.json> --report <path.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

UNK = "<unk>"

# Delivered vocabulary size for multi-character units. Free to change.
WORD_UNITS = 64

DOC_SEPARATOR = "\n\n"


def load_parsed(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_units(text: str) -> list[str]:
    """Delivered policy: every single character present, plus the commonest words."""
    singles = sorted({ch for ch in text})
    counts: dict[str, int] = {}
    for word in text.split():
        if len(word) >= 2:
            counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    words = [w for w, _ in ranked[:WORD_UNITS]]
    units = [UNK]
    for unit in singles + words:
        if unit not in units:
            units.append(unit)
    return units


def segment(text: str, units: list[str]) -> list[int]:
    """Greedy longest match. The bound segmentation rule, at training and at eval."""
    index = {unit: i for i, unit in enumerate(units)}
    longest = max((len(u) for u in units if u != UNK), default=1)
    out, i, n = [], 0, len(text)
    while i < n:
        hit = 0
        for size in range(min(longest, n - i), 0, -1):
            piece = text[i:i + size]
            if piece in index:
                out.append(index[piece])
                hit = size
                break
        if hit == 0:
            out.append(0)
            hit = 1
        i += hit
    return out


def run(parsed: Path, tokens: Path, vocab: Path, report: Path) -> dict:
    rows = load_parsed(parsed)
    text = DOC_SEPARATOR.join(row["text"] for row in rows)
    units = build_units(text)
    stream = segment(text, units)

    vocab_payload = json.dumps(
        {"schema": "oer13.vocab/v1", "units": units}, indent=2, sort_keys=True
    ) + "\n"
    tokens_payload = json.dumps(
        {"schema": "oer13.tokens/v1", "stream": stream}, sort_keys=True
    ) + "\n"
    for path, payload in ((vocab, vocab_payload), (tokens, tokens_payload)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")

    row = {
        "stage": "tokenize",
        "tokens": len(stream),
        "vocab_size": len(units),
        "bytes_per_token": round(len(text.encode("utf-8")) / len(stream), 6) if stream else 0.0,
        "input_path": parsed.as_posix(),
        "input_sha256": hashlib.sha256(parsed.read_bytes()).hexdigest(),
        "tokens_path": tokens.as_posix(),
        "tokens_sha256": hashlib.sha256(tokens_payload.encode("utf-8")).hexdigest(),
        "vocab_path": vocab.as_posix(),
        "vocab_sha256": hashlib.sha256(vocab_payload.encode("utf-8")).hexdigest(),
        "stage_output_contract": "tokens-stream-v1",
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    row = run(Path(args.parsed), Path(args.tokens), Path(args.vocab), Path(args.report))
    print("tokenize: tokens=%d vocab=%d bpt=%.4f"
          % (row["tokens"], row["vocab_size"], row["bytes_per_token"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
