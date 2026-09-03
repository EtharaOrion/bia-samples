#!/usr/bin/env python3
"""Build the near-duplicate index over the frozen corpus.

This is the harness-owned builder. It is run once at image build time to populate the index
directory the solving agent reads, and it is run again, in an isolated subprocess, by the
verifier's own runner so that grading rests on an index the verifier produced rather than on
one a submission handed it.

The build is a pure function of two frozen inputs, the corpus and the index specification.
It reads no clock, opens no socket, consults no environment secret and draws from no random
source, so the same two inputs build the same index on any host.

  python3 build_index.py --corpus corpus.jsonl --spec index_spec.json --out /task/index
  python3 build_index.py --corpus corpus.jsonl --spec index_spec.json --telemetry rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nd_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="build the OER-27 near-duplicate index")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", default=None, help="directory the built index is written to")
    parser.add_argument("--telemetry", default=None, help="path a flat telemetry stream is written to")
    arguments = parser.parse_args()

    index = nd_index.build(Path(arguments.corpus), Path(arguments.spec))

    if arguments.out:
        out = Path(arguments.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (out / "calibration.json").write_text(
            json.dumps(index["calibration"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with (out / "collisions.jsonl").open("w", encoding="utf-8") as handle:
            for row in index["collisions"]:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    if arguments.telemetry:
        telemetry = Path(arguments.telemetry)
        telemetry.parent.mkdir(parents=True, exist_ok=True)
        with telemetry.open("w", encoding="utf-8") as handle:
            for row in nd_index.telemetry_rows(index):
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    if not arguments.out and not arguments.telemetry:
        json.dump(index["calibration"], sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
