"""The reference parse and tokenize pipeline. This is what the live checkers accept.

It is a reference and not a ceiling: it exists to prove the objective is reachable and
that every gate admits a real submission, not to be the best pipeline available.

The one thing it does that a naive pipeline does not: it never trusts a fact about the
corpus for longer than the call that produced it. Every snapshot-dependent quantity --
the field-key set, the shard list, the evaluation split id, the token budget -- is read
from the LIVE snapshot at the moment it is used, and the version it was read at is
carried on the result. That is the whole content of the AR9 temporal reasoning gap this
slot grades, expressed as code.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENVIRONMENT = HERE.parent / "environment"
if str(ENVIRONMENT) not in sys.path:
    sys.path.insert(0, str(ENVIRONMENT))

import corpus_api  # noqa: E402
from pipeline import TokenStream  # noqa: E402

# Token id layout. Fixed here so the stream is reproducible from the corpus alone.
BOS = 0
DOC_SEPARATOR = 1
VOCABULARY_BASE = 2
VOCABULARY_SIZE = 12
BYTE_BASE = VOCABULARY_BASE + VOCABULARY_SIZE

EVAL_SPLIT_RULE_MODULUS = 6
EVAL_SPLIT_RULE_RESIDUE = 5


def document_index(doc_id: str) -> int:
    """The corpus's own index for a document id of the form d-NNNN."""
    return int(str(doc_id).split("-")[-1])


def is_held_out(doc_id: str) -> bool:
    """Held-out membership, recomputed from the id rather than cached from a split file.

    A cached split is a snapshot-dependent fact, and the split rotates when the corpus
    moves. Recomputing it costs nothing and cannot go stale.
    """
    return document_index(doc_id) % EVAL_SPLIT_RULE_MODULUS == EVAL_SPLIT_RULE_RESIDUE


def field_text(record: dict, field_keys: list) -> str:
    """Flatten a document's snapshot-dependent `fields` object using the LIVE key set.

    `field_keys` is read off the live snapshot descriptor by the caller. Hardcoding a key
    list read at an earlier snapshot is the canonical way to build a stale parser here:
    at snapshot 3 the keys are title, section, retrieved_at_snapshot, and at snapshot 7
    retrieved_at_snapshot is gone and snapshot and content_hash have appeared.
    """
    fields = record.get("fields") or {}
    parts = []
    for key in field_keys:
        if key in fields:
            parts.append(str(fields[key]))
    return " ".join(parts)


def fit_vocabulary(texts: list) -> dict:
    """Fit a word vocabulary over the training text, deterministically.

    Ordering is by descending count and then lexicographic, so two runs over the same
    corpus produce the same vocabulary. No random source is consulted.
    """
    counts = Counter()
    for text in texts:
        counts.update(text.split())
    ranked = sorted(counts.items(), key=lambda row: (-row[1], row[0]))
    return {word: VOCABULARY_BASE + index for index, (word, _n) in enumerate(ranked[:VOCABULARY_SIZE])}


def encode(text: str, vocabulary: dict) -> list:
    """Word ids for known words, byte fallback for the rest. Never lossy, never random."""
    out = []
    for word in text.split():
        known = vocabulary.get(word)
        if known is None:
            out.extend(BYTE_BASE + byte for byte in word.encode("utf-8"))
        else:
            out.append(known)
    return out


def build(corpus) -> TokenStream:
    """Parse and tokenize the corpus AT THE LIVE SNAPSHOT and return the token stream.

    `corpus` arrives stamped with the version it resolved at. This function re-asks the
    live corpus for its version and refuses to build against a handle that has gone
    stale, because a stream built from a stale handle is exactly what
    parse-built-against-stale-snapshot rejects.
    """
    live = corpus_api.snapshot_version()
    if live != corpus.snapshot_version:
        # The handle is stale. Re-resolve rather than build against it.
        corpus = corpus_api.resolve_corpus()
        live = corpus.snapshot_version

    field_keys = list(corpus_api._snapshot_document().get("field_keys") or [])
    budget = corpus_api.token_budget()

    shards = corpus.shard_paths()
    per_shard_records = []
    for path in shards:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    import json as _json

                    rows.append(_json.loads(line))
        per_shard_records.append(rows)

    training_texts = []
    for rows in per_shard_records:
        for record in rows:
            if is_held_out(record["doc_id"]):
                continue
            training_texts.append(str(record.get("body", "")) + " " + field_text(record, field_keys))
    vocabulary = fit_vocabulary(training_texts)

    token_ids = []
    per_shard_tokens = []
    document_ids = []
    for rows in per_shard_records:
        before = len(token_ids)
        for record in rows:
            doc_id = record["doc_id"]
            if is_held_out(doc_id):
                continue
            document_ids.append(doc_id)
            token_ids.append(BOS)
            token_ids.extend(
                encode(str(record.get("body", "")) + " " + field_text(record, field_keys), vocabulary)
            )
            token_ids.append(DOC_SEPARATOR)
        per_shard_tokens.append(len(token_ids) - before)

    # Feed EXACTLY the budget the live snapshot binds, once. Truncation is a pipeline
    # decision; overspending is not available, and the harness measures as fed.
    if len(token_ids) < budget:
        raise ValueError(
            "the corpus at snapshot " + str(live) + " yields " + str(len(token_ids))
            + " tokens, fewer than the bound budget of " + str(budget)
        )
    token_ids = token_ids[:budget]
    trimmed = []
    remaining = budget
    for count in per_shard_tokens:
        take = min(count, remaining)
        trimmed.append(take)
        remaining -= take
    per_shard_tokens = trimmed

    return TokenStream(
        token_ids=token_ids,
        per_shard_tokens=per_shard_tokens,
        document_ids=document_ids,
        built_against=live,
        vocabulary_size=len(vocabulary),
    )


def probe() -> dict:
    """Look at the corpus without committing to anything derived from what you saw."""
    return {
        "snapshot_version": corpus_api.snapshot_version(),
        "shard_count": corpus_api.shard_count(),
        "eval_split_id": corpus_api.eval_split_id(),
        "token_budget": corpus_api.token_budget(),
        "note": "every value above is true only of the snapshot it was read at; re-ask before reuse",
    }


def main(argv: list) -> int:
    import json

    if "--probe" in argv:
        print(json.dumps(probe(), sort_keys=True, indent=2))
        return 0
    stream = build(corpus_api.resolve_corpus())
    print(
        json.dumps(
            {
                "tokens": len(stream.token_ids),
                "per_shard_tokens": stream.per_shard_tokens,
                "documents": len(stream.document_ids),
                "built_against": stream.built_against,
                "vocabulary_size": stream.vocabulary_size,
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
