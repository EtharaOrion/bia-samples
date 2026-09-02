"""The durable, agent-writable recovery area. Read-only code, writable data.

This module is on the agent-visible lower layer and cannot be edited. The
directory it writes to, `/workspace/ledger/`, is a separate writable path and is
not part of that layer. Making the ledger writable therefore takes no write bit
off `task.toml`, `instruction.md` or `environment/`.

The harness never truncates this file and never rewrites a line in it. It reads
it, and it captures a copy into the run telemetry at the end of every iteration
so the graded surface can compare what the summary asserts against what the
record shows.
"""

import json
import os
import pathlib

LEDGER_DIR = pathlib.Path(os.environ.get("OER_LEDGER_DIR", "/workspace/ledger"))
LEDGER_PATH = LEDGER_DIR / "ledger.jsonl"

# The fields a record carries. `reconstructed_from_ledger` and `recovers` are
# the two the graded run reads when it asks whether a compaction was recovered.
FIELDS = (
    "approach_id",
    "status",
    "established",
    "iteration",
    "reward",
    "reconstructed_from_ledger",
    "recovers",
)

STATUSES = ("tried", "refuted")


def append(record):
    """Append one record. Never rewrites, never truncates, never reorders."""
    if not isinstance(record, dict):
        raise TypeError("a ledger record is a mapping")
    if not str(record.get("approach_id", "")).strip():
        raise ValueError("a ledger record names the approach it is about")
    if record.get("status") not in STATUSES:
        raise ValueError("status is one of " + ", ".join(STATUSES))
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, separators=(",", ":"))
    with LEDGER_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return record


def read_all():
    """Every record, in the order it was appended. Malformed lines are skipped."""
    if not LEDGER_PATH.is_file():
        return []
    rows = []
    for raw in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def reconcile(summary):
    """What the ledger holds that the handed summary does not show.

    This is the whole detection primitive, and it is deliberately small: the
    summary is a view and the ledger is the record, so the difference between
    them is the loss. An agent that never calls this, or calls it and trusts the
    view anyway, is the failure this slot grades.
    """
    visible = {str(entry.get("approach_id")) for entry in (summary or {}).get("entries", [])}
    recorded, dropped, established = set(), [], {}
    for row in read_all():
        ident = str(row.get("approach_id", ""))
        if not ident:
            continue
        recorded.add(ident)
        established[ident] = list(row.get("established") or [])
    for ident in sorted(recorded - visible):
        dropped.append(ident)
    return {
        "visible": sorted(visible),
        "recorded": sorted(recorded),
        "dropped_from_summary": dropped,
        "established_by_dropped": {ident: established[ident] for ident in dropped},
    }


def reconstruction_record(iteration, dropped):
    """The record the graded run reads when it asks whether you recovered."""
    return {
        "approach_id": "recon-" + str(iteration),
        "status": "tried",
        "established": [],
        "iteration": int(iteration),
        "reconstructed_from_ledger": True,
        "recovers": sorted(str(item) for item in dropped),
    }
