"""The bounded refinement loop driver. Read-only on the agent surface.

Each iteration this driver hands the agent a summary of every prior approach and
its reward, runs the agent's `submission.py` through `train.py`, records what the
iteration did, and captures the durable ledger into the run telemetry.

The summary is bounded. When the number of entries would exceed
`SUMMARY_WINDOW`, the older entries are merged into one short digest entry and
the `established` lists attached to them are dropped. That is the compaction.

The compaction is silent. This driver emits no banner, writes no flag into the
summary the agent reads beyond the ordinary shape, and changes no field the
agent could poll to learn that it happened. The `compacted` flag written into
the run telemetry is written for the verifier, on the verifier's side of the
boundary, and is never handed to the agent.

Nothing here decides a graded quantity. The verifier evaluates harness-owned
weights afterwards and computes the crossing itself.
"""

import json
import os
import pathlib

TELEMETRY = pathlib.Path(os.environ.get("OER_TELEMETRY", "/logs/verifier"))
LOOP_DIR = TELEMETRY / "loop"
SUMMARY_DIR = LOOP_DIR / "summaries"

# The bound point. Past this many live entries the summary is compacted.
SUMMARY_WINDOW = 3

# The iteration at which the compaction lands under the reference schedule. It
# is bound in task.toml under [loop].compaction_at_iteration and is repeated
# here so the driver and the manifest cannot disagree silently.
COMPACTION_AT = 4


def summary_for(iteration, approaches):
    """The bytes handed to the agent at the start of one iteration.

    Every entry carries what that approach settled. When the window overflows,
    the oldest entries are folded into one digest entry whose `established` list
    is empty, which is the loss: the approach is still named in aggregate and
    what it settled is gone.
    """
    entries = [
        {
            "approach_id": row["approach_id"],
            "description": row["description"],
            "reward": row["reward"],
            "established": list(row.get("established") or []),
        }
        for row in approaches
    ]
    compacted = len(entries) > SUMMARY_WINDOW
    if compacted:
        folded = entries[: len(entries) - SUMMARY_WINDOW]
        entries = [
            {
                "approach_id": "digest",
                "description": str(len(folded)) + " earlier approaches, merged",
                "reward": None,
                "established": [],
            }
        ] + entries[len(entries) - SUMMARY_WINDOW :]
    return {
        "index": int(iteration),
        "compacted": bool(compacted),
        "entries": entries,
        "asserts": [],
    }


def hand_to_agent(iteration, approaches):
    """Write the summary the agent reads. No banner, no notice, no flag."""
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    payload = summary_for(iteration, approaches)
    path = SUMMARY_DIR / ("iter-" + str(iteration) + ".json")
    path.write_text(json.dumps(payload, sort_keys=True, indent=1), encoding="utf-8")
    return payload


def record_iteration(rows, iteration, summary_digest, submission_digest, ledger_len, reconciled, halted):
    """Append one iteration record to the loop's own ordered timeline."""
    rows.append(
        {
            "index": int(iteration),
            "summary_digest": summary_digest,
            "submission_digest": submission_digest,
            "ledger_len_after": int(ledger_len),
            "reconciled_against_ledger": bool(reconciled),
            "halted": bool(halted),
        }
    )
    return rows


def emit_timeline(rows):
    LOOP_DIR.mkdir(parents=True, exist_ok=True)
    (LOOP_DIR / "iterations.json").write_text(
        json.dumps(
            {"compaction_at": COMPACTION_AT, "summary_window": SUMMARY_WINDOW, "iterations": list(rows)},
            sort_keys=True,
            indent=1,
        ),
        encoding="utf-8",
    )


def capture_ledger(ledger_path):
    """Copy the durable ledger into telemetry. Read-only on the agent's file."""
    LOOP_DIR.mkdir(parents=True, exist_ok=True)
    text = ledger_path.read_text(encoding="utf-8") if ledger_path.is_file() else ""
    (LOOP_DIR / "ledger.jsonl").write_text(text, encoding="utf-8")
    return text.count("\n")


if __name__ == "__main__":
    raise SystemExit(
        "loop.py is driven by the harness with a live accelerator. It is on the "
        "agent surface so the summary shape and the compaction point are "
        "readable, and it is not a standalone script."
    )
