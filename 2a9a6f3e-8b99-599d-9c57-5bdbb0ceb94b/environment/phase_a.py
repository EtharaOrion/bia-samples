#!/usr/bin/env python3
"""The AGENT PHASE driver. It runs your kernel forward and it decides where to stop.

WHAT IT DOES. It opens a Tape over the whole stream, feeds your kernel one whole block at a
time in order, and charges every read on its own counter. After each block it appends one row
to the harness journal. It keeps a harness-owned work counter, being the number of outputs
produced so far multiplied by the frozen window, and the moment that counter reaches
`params.splice_charge_budget` it HALTS. The absolute sample offset it halts at is the splice
point. The carry your kernel held at that offset is the carried state, and the driver digests
it into the carried-state digest and records both in the journal halt row.

WHAT IT DOES NOT DO. It does not write your handoff. Reading the splice point and the
carried-state digest back off the journal and writing `/workspace/handoff.json` in the bound
shape is the agent's half of the phase contract.

WHERE THINGS GO. Three separate carriers, on purpose. The graded kernel is at /app/submission.py
and is overridable with OER28_SUBMISSION. The journal is harness-owned and lands under the
harness log root, which defaults to /logs/harness and is overridable with OER28_HARNESS_LOGS.
The workspace, which is where the handoff goes, defaults to /workspace and is overridable with
OER28_WORKSPACE.

Usage:
    python3 phase_a.py --workspace /workspace --kernel /app/submission.py \
        --stream /task/environment/stream.json --params /task/environment/params.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Put this driver's own directory on the path rather than depending on the caller's, so the
# driver runs identically whether it is launched by the image, by a shell or by an importer.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import opstream  # noqa: E402

# The harness log root. Bound here as a module constant as well as an environment default, so a
# reader of this file and a reader of the manifest resolve the same path.
HARNESS_LOGS = Path(os.environ.get("OER28_HARNESS_LOGS", "/logs/harness"))
WORKSPACE = Path(os.environ.get("OER28_WORKSPACE", "/workspace"))
SUBMISSION = Path(os.environ.get("OER28_SUBMISSION", "/app/submission.py"))

JOURNAL_NAME = "phase_a.jsonl"

# The first link of the journal hash chain. Fixed, public and carried in grounding.yaml.
GENESIS = "oer28.journal.genesis/v1"


class DriverError(Exception):
    """The agent phase could not be run. Never a score, always a reason."""


def load_kernel(path):
    """Load the kernel module by path without putting it on the import graph of this process."""
    source = Path(path).read_text(encoding="utf-8")
    namespace = {"__name__": "oer28_agent_kernel", "__file__": str(path)}
    exec(compile(source, str(path), "exec"), namespace)  # noqa: S102
    entry = namespace.get("run_block")
    if not callable(entry):
        raise DriverError("the kernel at " + str(path) + " exposes no callable run_block")
    return entry


def append(rows, row, previous):
    """Chain one row and append it. The chain link never enters its own preimage."""
    link = opstream.chain_step(previous, row)
    shaped = dict(row)
    shaped["chain"] = link
    rows.append(shaped)
    return link


def run(stream_path, params_path, kernel_path, journal_path):
    stream = opstream.load_json(stream_path)
    params = opstream.load_json(params_path)
    run_block = load_kernel(kernel_path)

    samples, spans = opstream.flatten(stream)
    window = int(params["window"])
    budget = int(params["splice_charge_budget"])

    tape = opstream.Tape(samples, 0, len(samples))
    carry = opstream.origin_carry(params)

    rows = []
    link = append(
        rows,
        {
            "kind": "open",
            "schema": opstream.JOURNAL_SCHEMA,
            "genesis": GENESIS,
            "stream_sha256": opstream.file_digest(stream_path),
            "params_sha256": opstream.file_digest(params_path),
            "kernel_sha256": opstream.file_digest(kernel_path),
            "window": window,
            "splice_charge_budget": budget,
            "block_count": len(spans),
        },
        GENESIS,
    )

    produced = 0
    charged = 0
    for index, (start, length) in enumerate(spans):
        before = tape.reads
        outputs, carry = run_block(tape, params, carry, start, length)
        outputs = [int(value) for value in outputs]
        if len(outputs) != length:
            raise DriverError(
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
            raise DriverError(
                "block "
                + str(index)
                + " returned a carry history of length "
                + str(len(carry["history"]))
                + " against a window of "
                + str(window)
            )
        produced += length
        charged += tape.reads - before
        link = append(
            rows,
            {
                "kind": "block",
                "index": index,
                "start": start,
                "length": length,
                "reads": tape.reads - before,
                "out_sha256": opstream.outputs_digest(outputs),
                "cumulative_outputs": produced,
                "cumulative_work": produced * window,
            },
            link,
        )
        if produced * window >= budget:
            break

    link = append(
        rows,
        {
            "kind": "halt",
            "splice_offset": produced,
            "splice_block_index": len(rows) - 1,
            "carried_state_sha256": opstream.carry_digest(carry),
            "phase_a_reads": charged,
            "cumulative_work": produced * window,
        },
        link,
    )

    Path(journal_path).parent.mkdir(parents=True, exist_ok=True)
    Path(journal_path).write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows, carry


def main():
    parser = argparse.ArgumentParser(description="run the agent phase up to the splice")
    parser.add_argument("--workspace", default=str(WORKSPACE))
    parser.add_argument("--kernel", default=None)
    parser.add_argument("--stream", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--journal", default=None)
    args = parser.parse_args()

    workspace = Path(args.workspace)
    kernel = Path(args.kernel) if args.kernel else SUBMISSION
    journal = Path(args.journal) if args.journal else HARNESS_LOGS / JOURNAL_NAME
    if not kernel.is_file():
        raise SystemExit("no kernel at " + kernel.as_posix())

    rows, carry = run(args.stream, args.params, kernel, journal)
    halt = rows[-1]
    print(
        json.dumps(
            {
                "journal": journal.as_posix(),
                "blocks_in_agent_phase": halt["splice_block_index"],
                "splice_offset": halt["splice_offset"],
                "carried_state_sha256": halt["carried_state_sha256"],
                "phase_a_reads": halt["phase_a_reads"],
                "journal_chain": halt["chain"],
                "carried_state": carry,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
