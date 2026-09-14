#!/usr/bin/env python3
"""The reference solution for OER-09. Private. Imported by tests/grade.py for its identity only.

The reference does two things, and the second is the one the slot is about.

  1. It writes a filter chain against the field the REGISTER actually carries. The register
     schema drifted from curation/v3 to curation/v4 and `quality_bucket` became nested under
     `quality.bucket`. A predicate naming the documented field resolves nothing, matches
     nothing, and the runner's inapplicable-stage guard hands back the pool unchanged with
     exit status 0 and status "ok".

  2. It refuses to believe its own filter. After curating, it measures the pool itself and
     asserts the pool actually moved before a single training token is spent. Inside the
     live harness that measurement is the harness's own published observation; outside it,
     where nothing writes that document, the pool is measured here from the register bytes
     and the curated bytes. What it never reads is work/curation_report.json: that is the
     claim, and the claim is the thing under suspicion. That assertion, not the filter, is
     what separates a solve from a silent no-op.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REFERENCE_ID = "oer-09-reference-curation-verified-against-harness-observation"

BUNDLE = Path(__file__).resolve().parent.parent
REGISTER = BUNDLE / "environment" / "pool" / "source_register.jsonl"
HARNESS_OBSERVED = Path("/logs/harness/pool_observed.json")
CURATED_DEFAULT = Path("work") / "curated_pool.jsonl"

REFERENCE_FILTER_CHAIN = [
    {"name": "drop-junk-and-low", "field": "quality.bucket", "op": "drop_in", "values": ["junk", "low"]},
    {"name": "english-only", "field": "lang", "op": "keep_in", "values": ["en"]},
    {"name": "drop-holdout", "field": "holdout", "op": "drop_true"},
    {"name": "dedup-by-group", "field": "dup_group", "op": "first_per_group"},
]

REFERENCE_STAGE_COUNTS = [4096, 3072, 2939, 2613, 1959]
REFERENCE_CURATED_DOCUMENTS = 1959
REFERENCE_CURATED_TOKENS = 815884


def _resolve(document, field):
    node = document
    for part in str(field).split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def reference_pool(rows):
    """The curated pool the reference chain produces. Pure; the stages are applied in order."""
    kept = [r for r in rows if _resolve(r, "quality.bucket") not in ("junk", "low")]
    kept = [r for r in kept if _resolve(r, "lang") in ("en",)]
    kept = [r for r in kept if _resolve(r, "holdout") is not True]
    seen, deduped = set(), []
    for row in kept:
        key = _resolve(row, "dup_group")
        if key is None or key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _load_jsonl(path):
    """The rows of a jsonl pool, or None when the file is absent. Absent is absent, not empty."""
    path = Path(path)
    if not path.is_file():
        return None
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _pool_digest(rows):
    """The digest environment/curate.py publishes, recomputed here over the same bytes."""
    payload = json.dumps(
        [row.get("doc_id") for row in rows], sort_keys=False, separators=(",", ":")
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def observe_pool(curated=None):
    """An observation of the pool that does not come from the runner's own report.

    Inside the live harness /logs/harness/pool_observed.json is the observation and is used
    as it is published. Outside it -- producing this reference, or any run the harness did
    not start -- nothing in this bundle writes that document, and reading its absence as
    "the pool did not move" is what made the reference unproducible. Reading its absence as
    "the pool moved" would have been worse: it would reinstate the exact failure this slot
    is about. So when the harness has not spoken the pool is measured here, from the
    register bytes and the curated bytes, by the same digest the runner publishes.
    """
    if HARNESS_OBSERVED.is_file():
        observed = json.loads(HARNESS_OBSERVED.read_text(encoding="utf-8"))
        observed["observed_by"] = "harness"
        return observed

    source_rows = _load_jsonl(REGISTER)
    curated_rows = _load_jsonl(CURATED_DEFAULT if curated is None else curated)
    if source_rows is None or curated_rows is None:
        return {"observed_by": "reference", "source_digest": None, "curated_digest": None}
    return {
        "observed_by": "reference",
        "source_documents": len(source_rows),
        "curated_documents": len(curated_rows),
        "curated_tokens": sum(int(row.get("tokens") or 0) for row in curated_rows),
        "source_digest": _pool_digest(source_rows),
        "curated_digest": _pool_digest(curated_rows),
    }


def _moved(observed) -> bool:
    """A curation moved the pool. Emptying it did not: that is a collapse, not a curation.

    The distinction is load bearing on a register whose fields do not resolve. `keep_in`
    against a missing field keeps nothing rather than everything, so the chain clears the
    pool instead of passing it through, the digest duly changes, and a bare digest
    comparison would call that a move and hand the trainer an empty corpus.
    """
    if observed.get("curated_documents") == 0:
        return False
    source = observed.get("source_digest")
    curated = observed.get("curated_digest")
    return bool(source) and bool(curated) and source != curated


def pool_moved(curated=None) -> bool:
    """Did the pool actually move? Measured, never taken on the runner's word."""
    return _moved(observe_pool(curated))


def main(argv=None) -> int:
    work = Path("work")
    work.mkdir(parents=True, exist_ok=True)
    filters = work / "filters.json"
    filters.write_text(json.dumps({"stages": REFERENCE_FILTER_CHAIN}, indent=2) + "\n", encoding="utf-8")

    curate = BUNDLE / "environment" / "curate.py"
    train = BUNDLE / "environment" / "train.py"
    curated = work / "curated_pool.jsonl"

    subprocess.run(
        [sys.executable, str(curate), "--filters", str(filters), "--out", str(curated), "--explain"],
        check=True,
    )

    observed = observe_pool(curated)
    if not _moved(observed):
        if observed.get("curated_documents") == 0:
            print("the chain emptied the pool; refusing to train", file=sys.stderr)
            print("every document was dropped, which is a collapse and not a curation", file=sys.stderr)
        else:
            print("the pool did not move; refusing to train", file=sys.stderr)
            print("the filter exited zero and reported ok, and it changed nothing", file=sys.stderr)
        print(
            "observed_by={0} source_documents={1} curated_documents={2} curated_tokens={3}".format(
                observed.get("observed_by"),
                observed.get("source_documents"),
                observed.get("curated_documents"),
                observed.get("curated_tokens"),
            ),
            file=sys.stderr,
        )
        print(
            "the bound chain ends at {0} documents and {1} tokens over stage counts {2}".format(
                REFERENCE_CURATED_DOCUMENTS, REFERENCE_CURATED_TOKENS, REFERENCE_STAGE_COUNTS
            ),
            file=sys.stderr,
        )
        return 3

    subprocess.run([sys.executable, str(train), "--pool", str(curated)], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
