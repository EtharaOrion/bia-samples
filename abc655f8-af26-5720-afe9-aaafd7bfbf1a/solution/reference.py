"""The reference parse and tokenize pipeline. This is what the live checkers accept.

It is a reference and not a ceiling: it exists to prove the objective is reachable and
that every gate admits a real submission, not to be the best pipeline available.

The one thing it does that a naive pipeline does not: it never trusts a fact about the
corpus for longer than the call that produced it. Every snapshot-dependent quantity --
the field-key set, the shard list, the token budget, the upstream artifact the corpus is
cut from -- is read from the LIVE snapshot at the moment it is used, and the version it
was read at is carried on the result. That is the whole content of the AR9 temporal
reasoning gap this slot grades, expressed as code.

The evaluation split is the one fact that is NOT snapshot-dependent, and this file is
built around that asymmetry rather than against it. The split is pinned to the verifier's
own held-out FineWeb slice, named by `corpus_api.eval_split_id()`, and it is an ARTIFACT
and not a set of marked records. Hold-out is therefore decided here at the artifact level,
by asking whether the upstream artifact the live corpus is cut from is the pinned split,
and never by a per-record marking. An earlier revision of this file decided hold-out by a
rotating rule over a per-record `doc_id`, from the era when the split lived inline in the
corpus descriptor and rotated with it. The corpus records carry no `doc_id`, the split no
longer rotates, and that rule is gone rather than defaulted around.
"""

from __future__ import annotations

import fnmatch
import posixpath
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

# The marker a shard's own first line carries. It is the carrier's provenance header, not
# a document: it names the upstream collection the shard was cut from and carries no text.
PROVENANCE_KIND = "provenance"


def is_provenance(record: dict) -> bool:
    """True for a shard's own provenance header line, false for a corpus document.

    Each shard's first line is `{"_banner": ..., "external_source": ..., "kind":
    "provenance", "upstream": ...}`. Tokenizing it would feed the carrier's banner into
    the training stream and count it as a document. Every other line is a document and is
    read with strict subscripts below, so a record that is neither is a loud failure.
    """
    return "kind" in record and str(record["kind"]) == PROVENANCE_KIND


def corpus_source_artifacts() -> list:
    """Every upstream artifact the LIVE snapshot descriptor says this corpus is cut from.

    Re-read on every call, like every other snapshot-dependent fact here. A descriptor
    block qualifies when it carries a `provenance` object naming the upstream `shard` the
    carrier was cut from, so a re-base that moved the corpus onto a different upstream
    artifact is visible here rather than assumed away.
    """
    document = corpus_api._snapshot_document()
    artifacts = []
    for key in sorted(document):
        block = document[key]
        if not isinstance(block, dict) or "provenance" not in block:
            continue
        provenance = block["provenance"]
        if isinstance(provenance, dict) and "shard" in provenance:
            artifacts.append(str(provenance["shard"]))
    return artifacts


def is_held_out(source_artifact: str) -> bool:
    """Held-out membership, decided at the level the real substrate actually partitions at.

    The held-out split is the verifier's own FineWeb slice, an ARTIFACT named by a glob
    that `corpus_api.eval_split_id()` returns and that reads no snapshot state. Records
    are not individually marked and never were on this substrate: a record is held out
    exactly when the upstream artifact it was cut from is the pinned split. The pinned
    identity is re-asked on every call so that a name this file cached could not stand in
    for the name the verifier grades against.
    """
    pinned = str(corpus_api.eval_split_id())
    name = str(source_artifact)
    return fnmatch.fnmatch(name, pinned) or fnmatch.fnmatch(
        posixpath.basename(name), posixpath.basename(pinned)
    )


def field_text(record: dict, field_keys: list) -> str:
    """Flatten a document's snapshot-dependent `fields` object using the LIVE key set.

    `field_keys` is read off the live snapshot descriptor by the caller. Hardcoding a key
    list read at an earlier snapshot is the canonical way to build a stale parser here:
    at snapshot 3 the keys are title, section, retrieved_at_snapshot, and at snapshot 7
    retrieved_at_snapshot is gone and snapshot and content_hash have appeared. A record
    that carries no `fields` object contributes nothing, which is a snapshot fact about
    the record shape and not a missing key being defaulted away.
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


def document_text(record: dict, field_keys: list) -> str:
    """One document's text, at the LIVE field-key set. Strict on the record schema."""
    return str(record["text"]) + " " + field_text(record, field_keys)


def build(corpus) -> TokenStream:
    """Parse and tokenize the corpus AT THE LIVE SNAPSHOT and return the token stream.

    `corpus` arrives stamped with the version it resolved at. This function re-asks the
    live corpus for its version and refuses to build against a handle that has gone
    stale, because a stream built from a stale handle is exactly what
    parse-built-against-stale-snapshot rejects.

    Before a single record is read it establishes the hold-out at the artifact level: the
    live descriptor names the upstream artifacts this corpus is cut from, and if any of
    them is the pinned evaluation split the build refuses rather than training on it.
    """
    live = corpus_api.snapshot_version()
    if live != corpus.snapshot_version:
        # The handle is stale. Re-resolve rather than build against it.
        corpus = corpus_api.resolve_corpus()
        live = corpus.snapshot_version

    pinned_split = corpus_api.eval_split_id()
    source_artifacts = corpus_source_artifacts()
    if not source_artifacts:
        raise ValueError(
            "the corpus descriptor at snapshot " + str(live) + " names no upstream artifact, "
            "so this corpus cannot be shown disjoint from the held-out split " + str(pinned_split)
        )
    leaked = sorted(name for name in source_artifacts if is_held_out(name))
    if leaked:
        raise ValueError(
            "the corpus at snapshot " + str(live) + " is cut from " + ", ".join(leaked)
            + ", which the pinned held-out split " + str(pinned_split) + " matches; training on "
            "it would put the graded evaluation split into the training stream"
        )

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

                    record = _json.loads(line)
                    if is_provenance(record):
                        continue
                    # The record id is shard-local: every shard numbers its own documents
                    # from zero. Qualifying it by the shard file it was read from is what
                    # makes it a corpus-wide identity, and both halves are bytes on disk.
                    rows.append((path.stem + "/" + str(record["id"]), record))
        per_shard_records.append(rows)

    training_texts = []
    for rows in per_shard_records:
        for _document_id, record in rows:
            training_texts.append(document_text(record, field_keys))
    vocabulary = fit_vocabulary(training_texts)

    token_ids = []
    per_shard_tokens = []
    document_ids = []
    for rows in per_shard_records:
        before = len(token_ids)
        for document_id, record in rows:
            document_ids.append(document_id)
            token_ids.append(BOS)
            token_ids.extend(encode(document_text(record, field_keys), vocabulary))
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
        "corpus_source_artifacts": corpus_source_artifacts(),
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
                "corpus_source_artifacts": corpus_source_artifacts(),
                "held_out_split": corpus_api.eval_split_id(),
                "held_out_artifacts_in_corpus": sorted(
                    name for name in corpus_source_artifacts() if is_held_out(name)
                ),
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
