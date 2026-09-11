#!/usr/bin/env python3
"""The RESUME driver. The verifier phase runs it, and you may run it yourself as a probe.

It opens a Tape over the segment that starts at the splice point and runs to the end of the
stream, seeds your kernel with a carry read out of a file, and feeds it the remaining blocks in
order. It charges every read on its own counter and writes one telemetry row per block plus a
closing summary row.

TWO PROPERTIES MATTER AND BOTH ARE DELIBERATE. First, the Tape refuses any read before the
splice point, so nothing about the agent phase reaches this segment except through the carry.
Second, the splice point must fall on a block boundary; a value that does not is refused with a
named reason rather than quietly rounded to one.

The verifier runs this same file, over the bundle's own frozen stream and parameters, in an
isolated subprocess. Running it yourself is how you see the charge your kernel will be graded
on before you hand anything over.

Usage:
    python3 phase_b.py --stream S --params P --kernel K --carry carry.json \
        --splice-offset N --telemetry resume.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The verifier launches this driver under `python3 -I`, which ignores PYTHONPATH by design, so
# the driver puts its own directory on the path rather than depending on the caller's.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import opstream  # noqa: E402


class ResumeError(Exception):
    """The resume could not be run. Never a score, always a reason."""


def load_kernel(path):
    source = Path(path).read_text(encoding="utf-8")
    namespace = {"__name__": "oer28_agent_kernel", "__file__": str(path)}
    exec(compile(source, str(path), "exec"), namespace)  # noqa: S102
    entry = namespace.get("run_block")
    if not callable(entry):
        raise ResumeError("the kernel at " + str(path) + " exposes no callable run_block")
    return entry


def resume(stream, params, run_block, carry, offset):
    samples, spans = opstream.flatten(stream)
    window = int(params["window"])
    starts = [start for start, _length in spans]
    if offset not in starts and offset != len(samples):
        raise ResumeError(
            "the splice offset " + str(offset) + " does not fall on a block boundary"
        )
    if offset >= len(samples):
        raise ResumeError(
            "the splice offset " + str(offset) + " leaves the resume segment empty"
        )
    first = starts.index(offset)

    carry = {
        "accumulator": int(carry["accumulator"]),
        "history": [int(value) for value in carry["history"]],
    }
    if len(carry["history"]) != window:
        raise ResumeError(
            "the handed-over carry history is "
            + str(len(carry["history"]))
            + " long against a window of "
            + str(window)
        )

    tape = opstream.Tape(samples, offset, len(samples))
    rows = []
    produced = 0
    for index in range(first, len(spans)):
        start, length = spans[index]
        before = tape.reads
        outputs, carry = run_block(tape, params, carry, start, length)
        outputs = [int(value) for value in outputs]
        if len(outputs) != length:
            raise ResumeError(
                "block "
                + str(index)
                + " returned "
                + str(len(outputs))
                + " outputs against a block length of "
                + str(length)
            )
        carry = {
            "accumulator": int(carry["accumulator"]),
            "history": [int(value) for value in carry["history"]],
        }
        if len(carry["history"]) != window:
            raise ResumeError(
                "block " + str(index) + " returned a carry history of the wrong length"
            )
        produced += length
        rows.append(
            {
                "kind": "block",
                "index": index,
                "start": start,
                "length": length,
                "reads": tape.reads - before,
                "out_sha256": opstream.outputs_digest(outputs),
            }
        )
    rows.append(
        {
            "kind": "summary",
            "blocks": len(rows),
            "outputs": produced,
            "reads": tape.reads,
            "splice_offset": offset,
            "final_accumulator": carry["accumulator"],
        }
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description="resume the operator from a handed-over carry")
    parser.add_argument("--stream", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--carry", required=True)
    parser.add_argument("--splice-offset", required=True, type=int)
    parser.add_argument("--telemetry", required=True)
    args = parser.parse_args()

    stream = opstream.load_json(args.stream)
    params = opstream.load_json(args.params)
    carry = opstream.load_json(args.carry)
    run_block = load_kernel(args.kernel)
    try:
        rows = resume(stream, params, run_block, carry, int(args.splice_offset))
    except ResumeError as failure:
        raise SystemExit("resume refused: " + str(failure))

    Path(args.telemetry).parent.mkdir(parents=True, exist_ok=True)
    Path(args.telemetry).write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps(rows[-1], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
