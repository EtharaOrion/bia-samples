#!/usr/bin/env python3
"""Reference parse stage. Composition half one of two."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

DOC_MARKER = "=== DOC "
DROP_PREFIXES = ("<nav>", "[ad]", "Copyright ")
ENTITIES = (("&amp;", "&"), ("&quot;", '"'), ("&#39;", "'"), ("&lt;", "<"), ("&gt;", ">"))
HEADING = re.compile(r"^<h1>(.*)</h1>$")
SPACES = re.compile(r"[ \t]+")


def read_shards(corpus):
    return [p.read_text(encoding="utf-8") for p in sorted(corpus.glob("shard_*.raw"))]


def split_documents(raw):
    docs, doc_id, lines = [], None, []
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


def unescape(text):
    for needle, replacement in ENTITIES:
        text = text.replace(needle, replacement)
    return text


def keep(lines, seen):
    """Drop boilerplate, unwrap headings, unescape entities, drop repeated lines."""
    out = []
    for line in lines:
        body = line.strip()
        if not body:
            continue
        if body.startswith(DROP_PREFIXES):
            continue
        match = HEADING.match(body)
        if match:
            body = match.group(1).strip()
            if not body.endswith("."):
                body = body + "."
        body = SPACES.sub(" ", unescape(body)).strip()
        if not body or body in seen:
            continue
        seen.add(body)
        out.append(body)
    return out


def run(corpus, out, report):
    source_chars, records, seen = 0, [], set()
    for raw in read_shards(corpus):
        source_chars += len(raw)
        for doc_id, lines in split_documents(raw):
            text = "\n".join(keep(lines, seen))
            if text:
                records.append({"doc_id": doc_id, "text": text})
    payload = "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in records)
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


def main():
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
