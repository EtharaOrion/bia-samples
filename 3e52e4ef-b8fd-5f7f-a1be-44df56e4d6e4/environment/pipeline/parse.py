#!/usr/bin/env python3
"""Stage 1 of 3: parse. FREE. You may rewrite this file completely.

Reads the raw corpus shards and writes one JSON record per document to the parse
artifact, plus a stage report.

The report this stage writes is LOCALLY HONEST and GLOBALLY INCOMPLETE. It tells
you truthfully what fraction of the source characters survived parsing. It cannot
tell you whether the characters that survived are the ones the model needed. That
question is only answerable at the end of the chain, and the end of the chain is
what is graded.

The delivered implementation keeps every source line verbatim, so it reports a
retention ratio of 1.0. That is the best number this stage can report and it is
not the best parse.

Record contract, as delivered:
    {"doc_id": "<4 digits>", "text": "<document text>"}

Usage:
    python3 parse.py --corpus <dir> --out <path.jsonl> --report <path.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DOC_MARKER = "=== DOC "


def read_shards(corpus: Path) -> list[str]:
    return [p.read_text(encoding="utf-8") for p in sorted(corpus.glob("shard_*.raw"))]


def split_documents(raw: str) -> list[tuple[str, list[str]]]:
    """Cut a shard into (doc_id, lines) pairs on the document marker."""
    docs: list[tuple[str, list[str]]] = []
    doc_id, lines = None, []
    for line in raw.splitlines():
        if line.startswith(DOC_MARKER):
            if doc_id is not None:
                docs.append((doc_id, lines))
            doc_id = line[len(DOC_MARKER):].split(" ")[0].strip()
            lines = []
            continue
        if doc_id is not None:
            lines.append(line)
    if doc_id is not None:
        docs.append((doc_id, lines))
    return docs


def keep(lines: list[str]) -> list[str]:
    """Delivered policy: keep every line exactly as it appeared.

    This maximises the retention ratio this stage reports. It also carries the
    navigation bar, the advertisement line and the copyright footer of every
    document into the training stream.
    """
    return list(lines)


def run(corpus: Path, out: Path, report: Path) -> dict:
    source_chars = 0
    records = []
    for raw in read_shards(corpus):
        source_chars += len(raw)
        for doc_id, lines in split_documents(raw):
            text = "\n".join(keep(lines))
            records.append({"doc_id": doc_id, "text": text})
    payload = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in records
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload, encoding="utf-8")
    output_chars = sum(len(row["text"]) for row in records)
    row = {
        "stage": "parse",
        "records": len(records),
        "source_chars": source_chars,
        "output_chars": output_chars,
        "chars_retained_ratio": round(output_chars / source_chars, 6) if source_chars else 0.0,
        "output_path": out.as_posix(),
        "output_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "stage_output_contract": "parsed-record-v1",
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    row = run(Path(args.corpus), Path(args.out), Path(args.report))
    print("parse: records=%d retention=%.4f" % (row["records"], row["chars_retained_ratio"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
