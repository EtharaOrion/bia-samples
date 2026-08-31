#!/usr/bin/env python3
"""The frozen operator, the instrumented Tape, and the canonical encodings both phases share.

THE OPERATOR IS FROZEN. For the concatenated input stream x of length M:

    y[i] = ( sum over j in 0..K-1 of (r ** j mod P) * x[i - j] ) mod P,   x[t] = 0 for t < 0

with P = params["modulus"], r = params["ratio"] and K = params["window"]. Every quantity is an
exact integer. There is no tolerance anywhere on the graded path, because there is no float
anywhere on the graded path.

THE HARNESS OWNS THE COUNTER. A kernel never touches the sample list. It is handed a Tape, and
`Tape.at(t)` is the only route to a sample. Every call charges exactly one read. That charge is
the cost model: a kernel is fast when it reads less, and the graded speedup is a ratio of read
counts rather than of seconds. Counting reads instead of seconds is what makes the grading
reproducible on any host, and it is declared as a substrate limit rather than presented as a
wall clock measurement.

THE SEGMENT IS A WALL. A Tape is opened over one phase's segment and refuses a read outside it.
That is what makes the carry substantive: at a phase boundary the kernel cannot look backwards
through the Tape, so everything the next phase needs about the past has to travel in the carry.

THE KERNEL CONTRACT, which is the whole of what an agent writes:

    def run_block(tape, params, carry, start, length):
        '''Produce y[start .. start+length-1] and return (outputs, next_carry).'''

`carry` is a mapping with two keys. `accumulator` is y[start-1], or 0 at the stream origin.
`history` is the list of the K samples x[start-K .. start-1], oldest first, zero-filled where
the index is negative. `next_carry` must have the same shape at the far end of the block.
"""

from __future__ import annotations

import hashlib
import json

# The keys of the carry, in the order the canonical encoding fixes. A carry carrying anything
# else, or missing one of these, is not a carry.
CARRY_KEYS = ("accumulator", "history")

STREAM_SCHEMA = "oer28.stream/v1"
PARAMS_SCHEMA = "oer28.params/v1"
JOURNAL_SCHEMA = "oer28.journal/v1"
HANDOFF_SCHEMA = "oer28.handoff/v1"


class OutOfSegment(Exception):
    """A kernel read outside the segment its Tape was opened over."""


class Tape:
    """The only route to an input sample, and the counter that charges for it."""

    def __init__(self, samples, start, stop):
        self._samples = samples
        self.start = int(start)
        self.stop = int(stop)
        self.reads = 0

    def at(self, index):
        """Return x[index] and charge one read. Refuse anything outside the open segment."""
        index = int(index)
        if index < self.start or index >= self.stop:
            raise OutOfSegment(
                "read at absolute index "
                + str(index)
                + " outside the open segment ["
                + str(self.start)
                + ", "
                + str(self.stop)
                + ")"
            )
        self.reads += 1
        return self._samples[index]


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten(stream):
    """The frozen stream as one list of samples, and the per-block (start, length) spans."""
    samples = []
    spans = []
    for block in stream["blocks"]:
        spans.append((len(samples), len(block)))
        samples.extend(int(value) for value in block)
    return samples, spans


def origin_carry(params):
    """The carry at the stream origin: no history and a zero accumulator."""
    return {"accumulator": 0, "history": [0] * int(params["window"])}


def canonical_carry(carry):
    """The one encoding a carry digest is ever taken over. Sorted keys, no spaces."""
    shaped = {
        "accumulator": int(carry["accumulator"]),
        "history": [int(value) for value in carry["history"]],
    }
    return json.dumps(shaped, sort_keys=True, separators=(",", ":"))


def carry_digest(carry):
    """sha256 over the canonical carry encoding. This is the carried-state digest."""
    return hashlib.sha256(canonical_carry(carry).encode("utf-8")).hexdigest()


def outputs_digest(outputs):
    """sha256 over one block's outputs, in order, canonically encoded."""
    body = json.dumps([int(value) for value in outputs], separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def rollup(block_digests):
    """The whole-stream output rollup: sha256 over the per-block digests joined by newline."""
    return hashlib.sha256("\n".join(block_digests).encode("utf-8")).hexdigest()


def chain_step(previous, row):
    """One link of the journal hash chain: sha256 over the previous link and this row's bytes."""
    body = json.dumps(row, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((previous + "\n" + body).encode("utf-8")).hexdigest()


def file_digest(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def splice_offset(stream, params):
    """The splice point, derived from the frozen bytes alone.

    The agent phase halts after the first WHOLE block at which the cumulative harness work
    counter, being the number of outputs produced so far multiplied by the window, reaches the
    frozen splice charge budget. The value returned is the absolute sample offset of that halt,
    which is also the number of samples the agent phase covered.

    This is a pure function of environment/stream.json and environment/params.json, so the
    verifier re-derives it rather than believing a number the run declared. It is nowhere stated
    in the task statement: it falls out of the drawn block lengths and is established by running.
    """
    window = int(params["window"])
    budget = int(params["splice_charge_budget"])
    produced = 0
    for block in stream["blocks"]:
        produced += len(block)
        if produced * window >= budget:
            return produced
    return produced


def splice_block_index(stream, params):
    """How many whole blocks the agent phase covers. Derived, never declared."""
    window = int(params["window"])
    budget = int(params["splice_charge_budget"])
    produced = 0
    for index, block in enumerate(stream["blocks"]):
        produced += len(block)
        if produced * window >= budget:
            return index + 1
    return len(stream["blocks"])


def reference_run_block(tape, params, carry, start, length):
    """THE FROZEN REFERENCE KERNEL. It defines the operator and it defines the baseline cost.

    One Tape read per in-segment term, so an output at segment offset u charges min(u+1, K)
    reads. This is the direct transcription of the operator definition and nothing more.
    """
    modulus = int(params["modulus"])
    window = int(params["window"])
    ratio = int(params["ratio"])
    weights = [pow(ratio, power, modulus) for power in range(window)]
    history = [int(value) for value in carry["history"]]
    accumulator = int(carry["accumulator"])
    outputs = []
    for index in range(start, start + length):
        total = 0
        head = 0
        for power in range(window):
            source = index - power
            if source >= start:
                sample = tape.at(source)
            else:
                # `history` is kept sliding, so at this point it holds x[index-window ..
                # index-1] and the sample wanted is `power` places back from its far end.
                sample = history[window - power]
            if power == 0:
                head = sample
            total = (total + weights[power] * sample) % modulus
        outputs.append(total)
        accumulator = total
        history = history[1:] + [head]
    return outputs, {"accumulator": accumulator, "history": history}


def reference_charge(length, window, offset=0):
    """The reads the frozen reference kernel charges over one segment. Arithmetic, not a run.

    `offset` is how far into the segment the block starts, so the per-block figures compose.
    """
    total = 0
    for position in range(offset, offset + length):
        total += min(position + 1, window)
    return total
