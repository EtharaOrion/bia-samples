#!/usr/bin/env python3
"""Print what the HARNESS observed about the pool, stage by stage.

This is the affordance that makes the silent no-op detectable. The harness watches the pool
independently of the curation runner and publishes its observation at
/logs/harness/pool_observed.json. Your own work/curation_report.json is a claim. This is an
observation. When the two disagree, the observation is the one that grades.

Read it BEFORE you spend a training token:

    python3 environment/pool_probe.py

If `curated.digest` equals `source.digest`, your filter did not move the pool, whatever your
report says and whatever exit status the runner returned.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OBSERVED = Path("/logs/harness/pool_observed.json")
CLAIM = Path("work/curation_report.json")


def main() -> int:
    if not OBSERVED.is_file():
        print("harness observation absent at " + str(OBSERVED))
        print("the harness writes it once the pool has been read; run the curation step first")
        return 2
    observed = json.loads(OBSERVED.read_text(encoding="utf-8"))
    print("harness-observed pool")
    for stage in observed.get("stages", []):
        print(
            "  {stage:<10} documents={documents:<6} digest={digest}".format(
                stage=stage.get("stage", "?"),
                documents=stage.get("documents", "?"),
                digest=stage.get("digest", "?"),
            )
        )
    source = observed.get("source_digest")
    curated = observed.get("curated_digest")
    moved = source is not None and curated is not None and source != curated
    print("pool_moved: " + ("yes" if moved else "NO"))
    if not moved:
        print("  the curated pool the trainer would consume is byte-identical to the source pool")

    if CLAIM.is_file():
        claim = json.loads(CLAIM.read_text(encoding="utf-8"))
        print("your report claims")
        print("  pool_state=" + str(claim.get("pool_state")))
        print("  curated_documents=" + str(claim.get("curated_documents")))
        print("  curated_tokens=" + str(claim.get("curated_tokens")))
        print("  curated_digest=" + str(claim.get("curated_digest")))
        if claim.get("curated_digest") != curated:
            print("  MISMATCH: the digest you claim is not the digest the harness observed")
    else:
        print("no work/curation_report.json yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
