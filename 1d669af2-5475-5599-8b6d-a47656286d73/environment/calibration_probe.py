#!/usr/bin/env python3
"""Ask the calibration set what version it is at. This is the handle, use it.

The calibration slice this task is allocated against is versioned and harness-owned.
An answer it gave earlier in the session is an answer about the version that was in
force THEN. This probe is the first-class way to ask what is in force NOW, so
reasoning about when a calibration statistic was true is a thing you can actually do
rather than something you have to guess at.

Temporal ordering is carried by `issued_at_tick`, a monotonically issued integer
handle the harness owns. NO CLOCK IS READ, here or anywhere on the graded path. Wall
time is not the ordering.

A version is a PIN into the frozen FineWeb10B train shards, not a table of numbers.
This probe returns the pin so you can go and measure the statistics yourself by
running the real checkpoint over the slice in force. It returns two digests and they
answer two different questions:

    pin_digest      WHICH slice is in force. A pure function of the pin record in
                    environment/calibration_stats.json, so it is reproducible from
                    bundle bytes alone.
    content_digest  WHAT that slice actually contains. sha256 over the raw staged
                    token bytes. It is established only in built environment state
                    and sits on no agent-visible byte, so a run that never read the
                    built state cannot produce it.

Carry both in your submission. Declaring a version you did not fit against is graded
as the disagreement it is.

Usage:
    python3 environment/calibration_probe.py [--workspace DIR] [--no-content]

Prints one JSON object on stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def resolve(workspace: str | None, name: str) -> Path:
    base = Path(workspace).resolve() if workspace else Path(__file__).resolve().parents[1]
    candidate = base / "environment" / name
    if candidate.is_file():
        return candidate
    return Path(__file__).resolve().parent / name


def canonical(payload) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def pin_digest(pin: dict) -> str:
    return hashlib.sha256(canonical(pin)).hexdigest()


def content_digest(root: Path, pin: dict) -> str:
    """sha256 over the raw staged token bytes of the pinned slice.

    Imported lazily from environment/model.py so that a container without numpy or
    torch still gets a usable version answer out of this probe, with the content
    digest reported as absent rather than as a wrong value.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import model  # noqa: PLC0415

    return model.slice_content_sha256(Path(root), pin)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--no-content", action="store_true")
    args = parser.parse_args(argv)

    state_path = resolve(args.workspace, "calibration_state.json")
    stats_path = resolve(args.workspace, "calibration_stats.json")
    if not state_path.is_file() or not stats_path.is_file():
        print(json.dumps({"error": "calibration-state-absent", "looked_at": str(state_path)}))
        return 2

    state = json.loads(state_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    in_force = int(state["calibration_version_in_force"])
    record = next(row for row in stats["versions"] if int(row["version"]) == in_force)

    content = None
    error = None
    if not args.no_content:
        try:
            content = content_digest(Path(stats["corpus_root"]), record["pin"])
        except Exception as problem:  # noqa: BLE001  reported, never swallowed into a wrong value
            error = type(problem).__name__ + ": " + str(problem)[:200]

    print(
        json.dumps(
            {
                "calibration_version_in_force": in_force,
                "issued_at_tick": int(state["issued_at_tick"]),
                "pin": record["pin"],
                "pin_digest": pin_digest(record["pin"]),
                "content_digest": content,
                "content_digest_error": error,
                "corpus_root": stats["corpus_root"],
                "ledger": state["ledger"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
