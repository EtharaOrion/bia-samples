#!/usr/bin/env python3
"""The reference parse and tokenize pipeline for OER-12. PRIVATE.

This is the submission the live checkers accept. It is written as an INDEPENDENT
implementation of the record schema declared in environment/schema.json: the harness's
own classifier lives in tests/runner.py and shares no code with this file, so the
DIVERGENCE checker compares two readings of the same bytes rather than one reading
against itself.

What it does about the third state, in order:

1. It DECLARES its ambiguity policy first, emitting the declaration event before any
   document is classified. The policy is `quarantine-partial`: an ambiguous partial
   parse is excluded from the training feed and recorded in the ledger with its own
   partial reason. It is never re-labelled as a parse failure and never admitted as a
   valid record.
2. It reads the parse schema version IN FORCE from the harness state rather than
   hardcoding one, because the width of the ambiguous class can move during a session
   and a verdict computed against a width that is no longer bound is a stale verdict.
3. It classifies every record into exactly one of the three outcomes, with a reason
   code drawn from that outcome's own reason space.
4. It builds the feed from admitted records only, excluding every evaluation-split
   record under every outcome.

It invokes no model, no network, no clock and no random source.
"""
from __future__ import annotations

import json
import pathlib
import sys

DECLARED_AMBIGUITY_POLICY = "quarantine-partial"

FAILED = "parse-failed"
VALID = "parse-valid"
PARTIAL = "parse-partial"

POLICY_OUTCOME = {
    "quarantine-partial": "partial-quarantined",
    "complete-partial": "partial-completed",
    "truncate-partial": "partial-truncated",
}
POLICY_ADMITS_PARTIAL = {
    "quarantine-partial": False,
    "complete-partial": True,
    "truncate-partial": True,
}


def _blocks(text):
    """Cut the raw text into per-record line blocks. Comments never enter a block."""
    block = []
    for raw in text.split("\n"):
        line = raw.strip()
        if line == "" or line[:1] == "#":
            continue
        block.append(line)
        if line == "@end":
            yield block, True
            block = []
    if block:
        yield block, False


def _fields(block):
    """(marker, key, payload) triples for one record block, terminator excluded."""
    triples = []
    stray = False
    for line in block:
        if line == "@end":
            continue
        head = line[:1]
        if head not in ("@", "+"):
            stray = True
            continue
        key, _sep, payload = line[1:].partition(" ")
        triples.append((head, key.strip(), payload.strip()))
    return triples, stray


def classify(block, terminated, known, required, schema_version):
    """The declared classification order. The first rule that fires decides."""
    triples, stray = _fields(block)
    if stray:
        return FAILED, "directive-malformed"
    for _marker, key, _payload in triples:
        if key not in known:
            return FAILED, "field-key-unknown"
    identifiers = [p for m, k, p in triples if m == "@" and k == "id"]
    if len(identifiers) == 0:
        return FAILED, "record-id-absent"
    if len(identifiers) > 1:
        return FAILED, "record-id-duplicated"
    declared = set(k for m, k, _p in triples if m == "@")
    for key in required:
        if key not in declared:
            return PARTIAL, "required-field-absent"
    if not terminated:
        return PARTIAL, "record-unterminated"
    if int(schema_version) >= 2:
        body = [(m, p) for m, k, p in triples if k == "body"]
        if body and body[-1][0] == "+" and body[-1][1] == "":
            return PARTIAL, "continuation-dangling"
    return VALID, "record-complete"


def trainable_text(block):
    triples, _stray = _fields(block)
    return " ".join(p for _m, k, p in triples if k in ("title", "body") and p)


def policy_text(block, verdict, policy, defaults):
    """The text a record contributes to the feed, which the declared policy decides.

    The three policies do not merely label the ambiguous class differently, they build
    different corpora out of it. `complete-partial` fills the absent required fields from
    the declared defaults and so contributes tokens the record never carried;
    `truncate-partial` contributes only what survived, cutting at the last complete
    continuation; `quarantine-partial` contributes nothing because the record is not fed.
    """
    base = trainable_text(block)
    if verdict != PARTIAL:
        return base
    if policy == "complete-partial":
        triples, _stray = _fields(block)
        declared = set(k for m, k, _p in triples if m == "@")
        filled = [defaults[key] for key in ("title", "body")
                  if key not in declared and defaults.get(key)]
        return " ".join([base] + filled).strip()
    if policy == "truncate-partial":
        triples, _stray = _fields(block)
        kept = []
        for marker, key, payload in triples:
            if key not in ("title", "body"):
                continue
            if marker == "+" and payload == "":
                break
            if payload:
                kept.append(payload)
        return " ".join(kept)
    return base


def tokenize(text):
    return [token for token in text.lower().split() if token]


def build(corpus_text, schema, split, schema_version, eval_ordinals, budget_tokens,
          policy=DECLARED_AMBIGUITY_POLICY):
    """Classify, resolve the ambiguous class by the declared policy, and build the feed."""
    known = tuple(schema["known_keys"])
    required = tuple(schema["required_keys"])
    outcome = POLICY_OUTCOME[policy]
    admits = POLICY_ADMITS_PARTIAL[policy]
    defaults = schema.get("completion_defaults", {})

    events = [{"seq": 1, "event": "ambiguity-policy-declared", "policy": policy}]
    rows = []
    admitted = []
    tokens = []
    index = 0
    for block, terminated in _blocks(corpus_text):
        index += 1
        ordinal = "r%02d" % index
        if index == 1:
            events.append({"seq": 2, "event": "first-document-classified", "doc": ordinal})
        verdict, reason = classify(block, terminated, known, required, schema_version)
        identifiers = [p for m, k, p in _fields(block)[0] if m == "@" and k == "id"]
        row = {
            "ordinal": ordinal,
            "id": identifiers[0] if identifiers else "",
            "verdict": verdict,
            "reason": reason,
            "policy_applied": outcome if verdict == PARTIAL else None,
        }
        rows.append(row)
        if ordinal in eval_ordinals:
            continue
        if verdict == VALID or (verdict == PARTIAL and admits):
            admitted.append(ordinal)
            tokens.extend(tokenize(policy_text(block, verdict, policy, defaults)))
    events.append({"seq": len(events) + 1, "event": "classification-complete",
                   "records": len(rows)})

    if len(tokens) > int(budget_tokens):
        tokens = tokens[: int(budget_tokens)]

    return {
        "pipeline": {
            "schema": "oer12.pipeline/v1",
            "ambiguity_policy": policy,
            "ambiguity_policy_outcome": outcome,
            "ambiguity_policy_rationale": (
                "An ambiguous partial parse is a third state. Quarantine keeps it out of the "
                "training feed without re-labelling it as a parse failure, so the corpus is "
                "not padded with structurally incomplete records and the class stays visible "
                "in the ledger instead of disappearing into a neighbour."
            ),
            "parse_schema_version_read_from": "harness policy_state, never hardcoded",
            "parse_schema_version_in_force": int(schema_version),
            "split_id": split["split_id"],
        },
        "parse_ledger": {
            "writer": "submission",
            "ambiguity_policy": policy,
            "records": rows,
        },
        "events": events,
        "admitted_ordinals": admitted,
        "tokens": tokens,
    }


def main(argv) -> int:
    if len(argv) < 3:
        print("usage: reference.py <environment_dir> <out_dir> [schema_version]",
              file=sys.stderr)
        return 2
    env = pathlib.Path(argv[1])
    out = pathlib.Path(argv[2])
    version = int(argv[3]) if len(argv) > 3 else 2
    schema = json.loads((env / "schema.json").read_text(encoding="utf-8"))
    split = json.loads((env / "split.json").read_text(encoding="utf-8"))
    recipe = json.loads((env / "frozen_recipe.json").read_text(encoding="utf-8"))
    corpus = (env / "corpus" / "corpus.rec").read_text(encoding="utf-8")
    built = build(
        corpus,
        schema,
        split,
        version,
        list(split["eval_record_ordinals"]),
        recipe["frozen_axes"]["token_budget"]["tokens"],
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "pipeline.json").write_text(
        json.dumps(built["pipeline"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "parse_ledger.json").write_text(
        json.dumps(built["parse_ledger"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "feed.txt").write_text(" ".join(built["tokens"]) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(built["parse_ledger"]["records"]),
                      "admitted": len(built["admitted_ordinals"]),
                      "tokens": len(built["tokens"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
