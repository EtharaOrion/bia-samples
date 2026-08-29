#!/usr/bin/env python3
"""Load the pinned behavioural corpus and compute its signatures with the probe.

The signatures are computed here, on every call, rather than stored. A stored
signature can drift away from the rule that produced it, and a corpus whose
recorded answer and whose rule disagree grades neither. Recomputation costs
microseconds and removes the drift entirely.

The corpus PIN is a digest over the entry declarations. It is what the
`corpus_pinned_before_novelty_verdict` checker compares a verdict against: a
verdict stamped with a corpus digest other than the pinned one was taken over a
corpus that has since moved, and is stale.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENTRIES = HERE / "entries.json"


def declaration() -> dict:
    return json.loads(ENTRIES.read_text(encoding="utf-8"))


def pinned_digest(payload=None) -> str:
    """A digest over the entry declarations, fixing the corpus a verdict is against."""
    doc = payload if payload is not None else declaration()
    rows = sorted(
        [row["id"], row["kind"], json.dumps(row["hyper"], sort_keys=True, separators=(",", ":"))]
        for row in doc["entries"]
    )
    blob = json.dumps(
        {"pin_id": doc["pin_id"], "entries": rows}, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(blob).hexdigest()


def signatures(build, probe, set_name: str) -> list:
    """One behavioural signature per corpus entry, over the named probe set.

    `build` and `probe` are handed in rather than imported, so this loader has
    no opinion about which implementation produced the comparison and a caller
    can never be surprised by an import this module performed on its behalf.
    """
    doc = declaration()
    out = []
    for row in sorted(doc["entries"], key=lambda item: item["id"]):
        rule = build(row["kind"], row["hyper"])
        found = probe.signature(rule, set_name)
        out.append({"id": row["id"], "transcript": found["transcript"], "digest": found["digest"]})
    return out
