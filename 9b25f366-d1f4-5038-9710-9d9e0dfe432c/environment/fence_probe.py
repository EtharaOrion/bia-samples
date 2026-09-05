#!/usr/bin/env python3
"""Emit the harness telemetry for one fence state root. The harness owns the traversal.

The verifier does not trust a traversal the submission performed and it does not perform one
inside the grading interpreter either. `tests/runner.py` launches this module as an isolated
subprocess against the BUILT fence state, and this module walks the producer chain once and
writes what it observed as a record stream. Every checker then reads that stream.

Nothing here decides a verdict, computes an admitted set or counts anything the checkers are
supposed to establish for themselves. It reports observed built state: the registry as
declared, the segments as they sit on disk, which of them the chain links, and every crossing
the chain presents, in presentation order.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fencelib  # noqa: E402


def rows_for(fence):
    chain = fence.chain()
    on_disk = fence.segments_on_disk()
    linked = set(chain)

    yield {
        "row": "fence",
        "epoch": fence.epoch(),
        "salt": fence.salt(),
        "head": fence.head_segment_id(),
        "state_sha256": fence.state_digest(),
    }

    for entry in fence.entries():
        yield {
            "row": "registry_entry",
            "kind": entry.get("kind"),
            "status": entry.get("status"),
            "quota": entry.get("quota"),
            "seal": entry.get("seal"),
            "position": entry.get("position"),
        }

    for segment_id in sorted(on_disk):
        segment = on_disk[segment_id]
        yield {
            "row": "segment",
            "segment_id": segment_id,
            "next": segment["next"],
            "records": len(segment["records"]),
            "linked": segment_id in linked,
            "chain_position": chain.index(segment_id) if segment_id in linked else None,
        }

    index = 0
    for segment_id in chain:
        for record in on_disk[segment_id]["records"]:
            yield {
                "row": "crossing",
                "index": index,
                "segment_id": segment_id,
                "record": record,
            }
            index += 1


def main():
    parser = argparse.ArgumentParser(description="emit OER-26 harness fence telemetry")
    parser.add_argument("--state", default=None)
    parser.add_argument("--telemetry", required=True)
    args = parser.parse_args()

    fence = fencelib.open_fence(args.state)
    lines = [json.dumps(row, sort_keys=True) for row in rows_for(fence)]
    Path(args.telemetry).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
