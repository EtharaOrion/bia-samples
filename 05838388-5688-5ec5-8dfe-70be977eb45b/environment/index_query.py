#!/usr/bin/env python3
"""Read a built near-duplicate index back.

This is the read-back handle over the index the image built. Everything the index
established is reachable from here and nothing here computes an adjudication for you: the
handle reports what the index holds, and deciding what is a near-duplicate is the task.

  python3 index_query.py calibration
  python3 index_query.py collisions [--limit N] [--min-bands K]
  python3 index_query.py record --id r0000
  python3 index_query.py similarity --a r0000 --b r0001
  python3 index_query.py telemetry --out rows.jsonl

The index directory defaults to the OER27_INDEX environment variable and then to /task/index.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nd_index  # noqa: E402

DEFAULT_INDEX = os.environ.get("OER27_INDEX", "/task/index")


def load_index(directory: Path):
    path = Path(directory) / "index.json"
    if not path.is_file():
        raise SystemExit("no built index at " + path.as_posix())
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="query the OER-27 near-duplicate index")
    parser.add_argument("--index", default=DEFAULT_INDEX)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("calibration", help="what the index calibrated over its own candidate pairs")

    collisions = sub.add_parser("collisions", help="the pairs the band structure placed together")
    collisions.add_argument("--limit", type=int, default=0)
    collisions.add_argument("--min-bands", type=int, default=0)

    record = sub.add_parser("record", help="one record's shingle set")
    record.add_argument("--id", required=True)

    similarity = sub.add_parser("similarity", help="exact quantised similarity of two records")
    similarity.add_argument("--a", required=True)
    similarity.add_argument("--b", required=True)

    telemetry = sub.add_parser("telemetry", help="the whole index as a flat record stream")
    telemetry.add_argument("--out", required=True)

    arguments = parser.parse_args()
    index = load_index(Path(arguments.index))

    if arguments.command == "calibration":
        json.dump(index["calibration"], sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    if arguments.command == "collisions":
        rows = [
            row
            for row in index["collisions"]
            if int(row["bands"]) >= int(arguments.min_bands)
        ]
        if arguments.limit > 0:
            rows = rows[: arguments.limit]
        for row in rows:
            sys.stdout.write(json.dumps(row, sort_keys=True) + "\n")
        return 0

    shingle_sets = {row["id"]: set(row["shingles"]) for row in index["records"]}

    if arguments.command == "record":
        if arguments.id not in shingle_sets:
            raise SystemExit("no record " + arguments.id + " in the corpus")
        json.dump(
            {"id": arguments.id, "shingles": sorted(shingle_sets[arguments.id])},
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        sys.stdout.write("\n")
        return 0

    if arguments.command == "similarity":
        for identifier in (arguments.a, arguments.b):
            if identifier not in shingle_sets:
                raise SystemExit("no record " + identifier + " in the corpus")
        quantum = int(index["spec"]["similarity_quantum"])
        value = nd_index.quantised_similarity(
            shingle_sets[arguments.a], shingle_sets[arguments.b], quantum
        )
        json.dump(
            {"a": arguments.a, "b": arguments.b, "similarity": value, "quantum": quantum},
            sys.stdout,
            sort_keys=True,
        )
        sys.stdout.write("\n")
        return 0

    if arguments.command == "telemetry":
        out = Path(arguments.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            for row in nd_index.telemetry_rows(index):
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        sys.stdout.write("telemetry written to " + out.as_posix() + "\n")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
