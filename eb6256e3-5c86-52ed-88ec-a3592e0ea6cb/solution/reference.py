#!/usr/bin/env python3
"""The oracle. It installs the fused kernel, runs the agent phase, and writes the handoff.

THE INSIGHT. The frozen operator is a length-K finite impulse response whose weights are the
geometric sequence r ** j mod P. A geometric weight vector is exactly the case in which the
convolution collapses into a first order recurrence, because multiplying the previous output by
r reindexes every term of the sum by one:

    r * y[i-1] = sum over j in 1..K of r ** j * x[i-j]

Adding x[i] restores the j = 0 term and subtracting r ** K * x[i-K] removes the term that fell
off the end, which leaves exactly y[i]. So

    y[i] = ( r * y[i-1] + x[i] - (r ** K mod P) * x[i-K] ) mod P

and the kernel reads x[i] and x[i-K] instead of K samples per output. The window is 24, so the
charge falls by very close to a factor of twelve. Every intermediate is an exact integer modulo
a prime, so this is not an approximation of the reference: it is the same function.

WHAT THE TWO PHASES REQUIRE OF IT. The recurrence carries state across the splice. The
accumulator is y at the last sample of the agent phase, and the history is the K samples before
the splice point, which the resume segment's Tape refuses to hand back. Both travel in the
carry, and the harness digests exactly that carry into the carried-state digest. The oracle
therefore does not have to invent a handoff: it runs the driver, reads the splice point and the
digest the driver established off the harness journal, and transcribes them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent

# The kernel bytes the oracle installs. Held as one string so the bytes the agent phase runs,
# the bytes the verifier phase grades and the bytes the fixtures record are the same bytes.
FUSED_KERNEL = '''"""A fused first order recurrence for the frozen geometric-weight operator.

y[i] = ( r * y[i-1] + x[i] - (r ** K mod P) * x[i-K] ) mod P

Two Tape reads per output once the segment is K deep, against K reads for the reference.
"""


def run_block(tape, params, carry, start, length):
    modulus = int(params["modulus"])
    window = int(params["window"])
    ratio = int(params["ratio"])
    tail_weight = pow(ratio, window, modulus)

    accumulator = int(carry["accumulator"])
    history = [int(value) for value in carry["history"]]
    outputs = []

    for index in range(start, start + length):
        head = tape.at(index)
        source = index - window
        if source >= start:
            tail = tape.at(source)
        else:
            tail = history[0]
        accumulator = (ratio * accumulator + head - tail_weight * tail) % modulus
        outputs.append(accumulator)
        history = history[1:] + [head]

    return outputs, {"accumulator": accumulator, "history": history}
'''

HANDOFF_SCHEMA = "oer28.handoff/v1"


def install_kernel(submission, source=FUSED_KERNEL):
    path = Path(submission)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def drive(bundle, workspace, journal, submission, kernel_source=FUSED_KERNEL):
    """Install the kernel, run the agent phase, probe the resume, and write the handoff."""
    bundle = Path(bundle)
    workspace = Path(workspace)
    journal = Path(journal)
    submission = Path(submission)
    sys.path.insert(0, str(bundle / "environment"))
    import opstream  # noqa: E402
    import phase_a  # noqa: E402
    import phase_b  # noqa: E402

    stream_path = bundle / "environment" / "stream.json"
    params_path = bundle / "environment" / "params.json"
    kernel_path = install_kernel(submission, kernel_source)

    rows, carry = phase_a.run(stream_path, params_path, kernel_path, journal)
    halt = rows[-1]

    # The resume probe. The verifier runs the same driver over the same frozen bytes in its own
    # process, so this is what the graded charge will be. It is reported so a substitution is
    # visible, and it is never the number that is graded.
    stream = opstream.load_json(stream_path)
    params = opstream.load_json(params_path)
    probe = phase_b.resume(
        stream,
        params,
        phase_b.load_kernel(kernel_path),
        carry,
        int(halt["splice_offset"]),
    )
    summary = probe[-1]

    handoff = {
        "schema": HANDOFF_SCHEMA,
        "splice_offset": int(halt["splice_offset"]),
        "splice_block_index": int(halt["splice_block_index"]),
        "carried_state": {
            "accumulator": int(carry["accumulator"]),
            "history": [int(value) for value in carry["history"]],
        },
        "carried_state_sha256": halt["carried_state_sha256"],
        "journal_chain": halt["chain"],
        "kernel_sha256": opstream.file_digest(kernel_path),
        "reported_resume_charge": int(summary["reads"]),
    }
    path = workspace / "handoff.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return handoff, rows, probe


def main():
    parser = argparse.ArgumentParser(description="run the OER-28 oracle agent phase")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--journal", required=True)
    parser.add_argument("--submission", required=True)
    args = parser.parse_args()
    handoff, _rows, _probe = drive(
        args.bundle, args.workspace, args.journal, args.submission
    )
    print(json.dumps(handoff, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
