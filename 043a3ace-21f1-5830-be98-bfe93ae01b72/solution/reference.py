#!/usr/bin/env python3
"""The reference solution for OER-09. Private. Imported by tests/grade.py for its identity only.

The reference does two things, and the second is the one the slot is about.

  1. It writes a filter chain against the field the REGISTER actually carries. The register
     schema drifted from curation/v3 to curation/v4 and `quality_bucket` became nested under
     `quality.bucket`. A predicate naming the documented field resolves nothing, matches
     nothing, and the runner's inapplicable-stage guard hands back the pool unchanged with
     exit status 0 and status "ok".

  2. It refuses to believe its own filter. After curating, it reads the harness's independent
     observation of the pool and asserts the pool actually moved before a single training
     token is spent. That assertion, not the filter, is what separates a solve from a
     silent no-op.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REFERENCE_ID = "oer-09-reference-curation-verified-against-harness-observation"

BUNDLE = Path(__file__).resolve().parent.parent
REGISTER = BUNDLE / "environment" / "pool" / "source_register.jsonl"
HARNESS_OBSERVED = Path("/logs/harness/pool_observed.json")

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


def pool_moved() -> bool:
    """Read the HARNESS observation, not the runner's report. This is the whole point."""
    if not HARNESS_OBSERVED.is_file():
        return False
    observed = json.loads(HARNESS_OBSERVED.read_text(encoding="utf-8"))
    return bool(observed.get("source_digest")) and observed.get("source_digest") != observed.get("curated_digest")


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

    if not pool_moved():
        print("harness observation says the pool did not move; refusing to train", file=sys.stderr)
        print("the filter exited zero and reported ok, and it changed nothing", file=sys.stderr)
        return 3

    subprocess.run([sys.executable, str(train), "--pool", str(curated)], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
