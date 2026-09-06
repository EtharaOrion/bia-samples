#!/usr/bin/env python3
"""The harness observation the AGENT reads while it works. Advisory, never graded.

`environment/pool_probe.py` prints `/logs/harness/pool_observed.json` and says, when the file
is missing, that "the harness writes it once the pool has been read; run the curation step
first". Nothing in the delivered bundle ever wrote it, so that sentence described a file that
did not exist and `environment/reference.py`'s `pool_moved()` returned False on every run.
This module is what writes it, and `environment/curate.py` calls it as it reads the pool.

STANDING OF THESE BYTES. Advisory. `/logs/harness/` sits inside the agent's container, so an
agent could overwrite anything here; that is exactly why nothing here is graded.
`tests/harness.py` recomputes every one of these records inside the verifier, after the
submission has exited, and OVERWRITES this directory before any checker reads it. The copy
written here exists so the agent can see, before it spends a training token, whether its
filter actually moved the pool. It buys the agent information and it buys no reward.

Nothing here reads a clock, a random source or the network.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HARNESS_LOGS = Path("/logs/harness")

# The same stamps tests/harness.py publishes, so the advisory copy and the graded copy
# describe the same ordering rather than two different ones.
SEQ_REGISTER = 3
SEQ_POOL = 4
SEQ_CURATION = 5
SEQ_FIRST_FEED = 6


def pool_digest(rows) -> str:
    """Byte-identical to environment/curate.py::pool_digest and tests/harness.py::pool_digest."""
    payload = json.dumps(
        [row.get("doc_id") for row in rows], sort_keys=False, separators=(",", ":")
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _write(name: str, payload: dict, logs: Path = HARNESS_LOGS) -> None:
    try:
        logs.mkdir(parents=True, exist_ok=True)
        (logs / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError:
        # The observation is an affordance. A host that will not let the harness publish it
        # costs the agent a diagnostic and never changes a graded number, so this never
        # raises into the curation runner.
        pass


def observe_pool(source_rows, curated_rows, claim: dict, register: dict, logs: Path = HARNESS_LOGS) -> None:
    """Publish what the harness saw of the pool, independently of what the runner reported."""
    source = pool_digest(source_rows)
    curated = pool_digest(curated_rows)
    _write(
        "register.json",
        {
            "schema": str(register.get("schema")),
            "quality_field": str(register.get("quality_field")),
            "observed_at_seq": SEQ_REGISTER,
        },
        logs,
    )
    _write(
        "pool_observed.json",
        {
            "source_documents": len(source_rows),
            "source_digest": source,
            "curated_documents": len(curated_rows),
            "curated_digest": curated,
            "observed_at_seq": SEQ_POOL,
            # The per-stage view environment/pool_probe.py prints for the agent.
            "stages": [
                {"stage": "source", "documents": len(source_rows), "digest": source},
                {"stage": "curated", "documents": len(curated_rows), "digest": curated},
            ],
        },
        logs,
    )
    _write("curation_claim.json", dict(claim), logs)
    _write(
        "curation_ledger.json",
        {
            "completed_at_seq": SEQ_CURATION,
            "verdict": "applied" if curated != source else "inapplicable-pass-through",
            "observed_documents_removed": len(source_rows) - len(curated_rows),
        },
        logs,
    )


def observe_feed(fed_rows, tokens_fed: int, budget: int, register: dict, evaluation: dict,
                 held_out_documents: int, overlap: int, logs: Path = HARNESS_LOGS) -> None:
    """Publish what the harness saw of the feed. Advisory, on the same terms."""
    _write(
        "feed_ledger.json",
        {
            "stream_digest": pool_digest(fed_rows) if fed_rows else None,
            "resolved_register_schema": str(register.get("schema")) if fed_rows else None,
            "resolved_quality_field": str(register.get("quality_field")) if fed_rows else None,
            "tokens_fed": int(tokens_fed),
            "budget_tokens": int(budget),
            "frozen_budget_tokens": int(budget),
            "first_feed_seq": SEQ_FIRST_FEED,
        },
        logs,
    )
    _write(
        "split_audit.json",
        {
            "audited_by": "harness",
            "audited_documents": int(held_out_documents),
            "audited_split_id": str(evaluation.get("split_id")),
            "audited_split_digest": str(evaluation.get("split_digest")),
            "eval_split_id": str(evaluation.get("split_id")),
            "eval_split_digest": str(evaluation.get("split_digest")),
            "train_eval_overlap": int(overlap),
        },
        logs,
    )
