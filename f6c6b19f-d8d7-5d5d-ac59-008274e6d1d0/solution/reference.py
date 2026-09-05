#!/usr/bin/env python3
"""The reference composition the live checkers accept. Private to the verifier.

It is a composition, not two independent stage improvements. The parse stage
lowers its own retention readout on purpose, because the characters it drops are
the navigation bar, the advertisement line and the copyright footer of every
document, and none of those is prose the held-out FineWeb split contains. The
tokenize stage then spends the frozen token budget on the cleaned stream: it
drops documents that survived parsing as fragments, deduplicates at the token
level, and packs what is left into shards with the end-of-text delimiter the
evaluated corpus uses. Either half on its own moves the graded loss much less
than the pair does, which is the whole point of the slot.

What the re-base changed about this file. The tokenize half used to design a
vocabulary, because the retired path graded a bigram over whatever vocabulary the
submission handed it. Under the nanoGPT substrate the token space is frozen at
the declared vocab_size of 50304 under the GPT-2 byte pair encoder, so the
vocabulary is no longer a free axis and the tokenize half now works on what
reaches those ids and in what order. The parse half is unchanged, and it is
unchanged because it was already an argument about which text belongs in a
training stream, which is exactly what the new graded path measures.

`materialize(environment, dest)` writes the three submission files into a tree
shaped like environment/, so the reference is graded through exactly the path a
submission is.
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
"""Reference tokenize stage. Composition half two of two.

The token space is frozen, so this half spends its effort on what reaches the
frozen ids and in what order: fragments that survived parsing without becoming
prose are dropped, documents that tokenize to the same stream are kept once, and
the survivors are packed with the end-of-text delimiter the evaluated corpus
uses. The parse half made the text clean; this half keeps the budget on it.
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
ENCODING = "gpt2"
EOT = 50256

MIN_DOCUMENT_TOKENS = 24
SHARD_TOKENS = 1 << 20


def load_parsed(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def encode(rows):
    encoder = tiktoken.get_encoding(ENCODING)
    ids, seen = [], set()
    for row in rows:
        piece = encoder.encode_ordinary(row["text"])
        if len(piece) < MIN_DOCUMENT_TOKENS:
            continue
        key = hashlib.blake2b(np.asarray(piece, dtype=np.uint16).tobytes(), digest_size=16).digest()
        if key in seen:
            continue
        seen.add(key)
        ids.append(EOT)
        ids.extend(piece)
    return np.asarray(ids, dtype=np.uint16)


def write_shard(path, tokens):
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = int(tokens.size)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header.tobytes())
        handle.write(tokens.astype(np.uint16).tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(parsed, shards, index, report):
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
        {"schema": "oer13.shards/v1", "encoding": ENCODING, "vocab_size": 50304,
         "shards": written, "tokens": int(stream.size)},
        indent=2, sort_keys=True) + "\\n"
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
    report.write_text(json.dumps(row, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    row = run(Path(args.parsed), Path(args.shards), Path(args.index), Path(args.report))
    print("tokenize: tokens=%d shards=%d bpt=%.4f" % (row["tokens"], row["shards"], row["bytes_per_token"]))
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
      "args": ["--parsed", "work/parsed.jsonl", "--shards", "work/shards", "--index", "work/shards/index.json", "--report", "work/tokenize_report.json"],
      "produces": ["work/shards/index.json", "work/tokenize_report.json"],
      "consumes": ["work/parsed.jsonl"]
    }
  },
  "handoff": {
    "shards": "work/shards/index.json"
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
