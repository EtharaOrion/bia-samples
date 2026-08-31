#!/usr/bin/env python3
"""Reference tokenize stage. Composition half two of two."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

UNK = "<unk>"
SPACED_UNITS = 320
BARE_UNITS = 48
DOC_SEPARATOR = "\n\n"


def load_parsed(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_units(text):
    """Single-character coverage first, then leading-space word units.

    The leading space is carried inside the unit so a word and the space before it
    cost one token instead of two. Coverage comes first because an unmatched byte
    at evaluation time falls back to the unknown unit, and that fallback is far
    more expensive than a missing merge.
    """
    singles = sorted(set(text))
    counts = {}
    for word in text.split():
        counts[word] = counts.get(word, 0) + 1
    ranked = [w for w, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    units = [UNK]
    for unit in singles:
        if unit not in units:
            units.append(unit)
    for word in ranked[:SPACED_UNITS]:
        unit = " " + word
        if unit not in units:
            units.append(unit)
    for word in ranked[:BARE_UNITS]:
        if word not in units:
            units.append(word)
    return units


def segment(text, units):
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


def run(parsed, tokens, vocab, report):
    rows = load_parsed(parsed)
    text = DOC_SEPARATOR.join(row["text"] for row in rows)
    units = build_units(text)
    stream = segment(text, units)
    vocab_payload = json.dumps({"schema": "oer13.vocab/v1", "units": units}, indent=2, sort_keys=True) + "\n"
    tokens_payload = json.dumps({"schema": "oer13.tokens/v1", "stream": stream}, sort_keys=True) + "\n"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    row = run(Path(args.parsed), Path(args.tokens), Path(args.vocab), Path(args.report))
    print("tokenize: tokens=%d vocab=%d bpt=%.4f" % (row["tokens"], row["vocab_size"], row["bytes_per_token"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
