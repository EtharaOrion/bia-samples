#!/usr/bin/env python3
"""The reference composition the live checkers accept. Private to the verifier.

It is a composition, not two independent stage improvements. The parse stage
lowers its own retention readout on purpose, because the characters it drops are
the navigation bar, the advertisement line and the copyright footer of every
document, and none of those appear in the held-out split. The tokenize stage then
builds its vocabulary out of the cleaned stream, so its units are spent on text
the held-out split actually contains. Either half on its own moves the graded loss
much less than the pair does, which is the whole point of the slot.

`materialize(dest)` writes the three submission files into a tree shaped like
environment/, so the reference is graded through exactly the path a submission is.
"""

from __future__ import annotations

import shutil
from pathlib import Path

BANNER = "# Reference composition for OER-13. Private to the verifier.\n"

PARSE_SOURCE = '''#!/usr/bin/env python3
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
SPACES = re.compile(r"[ \\t]+")


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
            text = "\\n".join(keep(lines, seen))
            if text:
                records.append({"doc_id": doc_id, "text": text})
    payload = "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\\n" for row in records)
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
    report.write_text(json.dumps(row, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
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
'''

TOKENIZE_SOURCE = '''#!/usr/bin/env python3
"""Reference tokenize stage. Composition half two of two."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

UNK = "<unk>"
SPACED_UNITS = 320
BARE_UNITS = 48
DOC_SEPARATOR = "\\n\\n"


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
    vocab_payload = json.dumps({"schema": "oer13.vocab/v1", "units": units}, indent=2, sort_keys=True) + "\\n"
    tokens_payload = json.dumps({"schema": "oer13.tokens/v1", "stream": stream}, sort_keys=True) + "\\n"
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
    report.write_text(json.dumps(row, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
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
'''

COMPOSE_SOURCE = """{
  "schema": "oer13.compose/v1",
  "order": ["parse", "tokenize"],
  "stages": {
    "parse": {
      "script": "pipeline/parse.py",
      "args": ["--corpus", "corpus", "--out", "work/parsed.jsonl", "--report", "work/parse_report.json"],
      "produces": ["work/parsed.jsonl", "work/parse_report.json"],
      "consumes": []
    },
    "tokenize": {
      "script": "pipeline/tokenize.py",
      "args": ["--parsed", "work/parsed.jsonl", "--tokens", "work/tokens.json", "--vocab", "work/vocab.json", "--report", "work/tokenize_report.json"],
      "produces": ["work/tokens.json", "work/vocab.json", "work/tokenize_report.json"],
      "consumes": ["work/parsed.jsonl"]
    }
  },
  "handoff": {
    "tokens": "work/tokens.json",
    "vocab": "work/vocab.json"
  }
}
"""

SOURCES = {
    "pipeline/parse.py": PARSE_SOURCE,
    "pipeline/tokenize.py": TOKENIZE_SOURCE,
    "pipeline/compose.json": COMPOSE_SOURCE,
}


def materialize(environment: Path, dest: Path) -> Path:
    """Build a submission tree: the delivered environment with the reference stages."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(environment, dest, symlinks=False, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "work"))
    for rel, source in SOURCES.items():
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
    return dest
