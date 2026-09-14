#!/usr/bin/env python3
"""The verifier's own behavioural probe over an update rule.

This is the load-bearing half of the novelty gate, and it is deliberately NOT a
text-similarity check on the submission's source. Source is paraphraseable:
renaming, reordering, inlining and restyling all move a text digest and move no
behaviour. What is compared here is what the rule DOES.

Determinism is the property that makes the comparison recomputable by anyone
holding the frozen bytes, so every source of variation is nailed down:

- fixed probe inputs, carried in this file as literals and nowhere else
- fixed dtype: Python float throughout, quantized to a fixed number of decimals
  before anything is recorded, so a difference below the quantum cannot move a
  digest
- fixed reduction order: step-major, then coordinate order, never a set, never a
  dict iteration, never a sort keyed on a float
- a digest over the quantized transcript, computed here

No clock, no random source, no environment read, no network, no filesystem read.
Two runs of this probe over the same rule produce the same digest, and that is
asserted on the graded path rather than assumed.
"""

from __future__ import annotations

import hashlib
import json

SCHEMA = "oer05.probe/v1"

# The quantum. Six decimals, fixed. Recorded here so a reader can recompute.
QUANTUM_DECIMALS = 6

# The bound behavioural margin. A rule whose relative behavioural distance to
# its nearest corpus entry is at or below this is not behaviourally novel.
BEHAVIOURAL_MARGIN = 0.02

# Fixed probe inputs. Two named sets. The graded set is named by the admin
# plane's drift state, so rotating the probe set is an admin-plane operation and
# a verdict carried over from the retired set is stale.
PROBE_SETS = {
    "probe-set-a": {
        "width": 8,
        "steps": 6,
        "params0": [0.5, -0.25, 0.125, 0.0, -0.75, 0.375, -0.5, 0.25],
        "grads": [
            [0.4, -0.2, 0.1, 0.3, -0.5, 0.25, -0.15, 0.05],
            [0.2, -0.4, 0.3, 0.1, -0.25, 0.5, -0.05, 0.15],
            [-0.1, 0.35, -0.2, 0.45, 0.15, -0.3, 0.25, -0.4],
            [0.3, 0.1, -0.45, 0.2, -0.35, 0.05, 0.4, -0.25],
            [-0.25, 0.5, 0.15, -0.4, 0.3, -0.1, 0.2, 0.35],
            [0.15, -0.3, 0.4, -0.05, 0.45, -0.2, 0.1, -0.35],
        ],
    },
    "probe-set-b": {
        "width": 8,
        "steps": 6,
        "params0": [-0.375, 0.625, -0.125, 0.25, 0.5, -0.625, 0.0, -0.25],
        "grads": [
            [-0.35, 0.45, 0.2, -0.1, 0.4, -0.25, 0.05, 0.3],
            [0.25, -0.15, 0.5, 0.35, -0.2, 0.1, -0.4, 0.45],
            [0.1, 0.3, -0.25, -0.5, 0.15, 0.4, 0.35, -0.05],
            [-0.45, 0.2, 0.05, 0.4, -0.3, -0.15, 0.5, 0.25],
            [0.5, -0.4, 0.35, 0.15, 0.25, 0.3, -0.1, -0.2],
            [-0.2, 0.05, -0.3, 0.5, -0.45, 0.35, 0.15, 0.4],
        ],
    },
}


def quantize(value) -> float:
    """One fixed quantum, applied before anything is recorded or compared."""
    return round(float(value) + 0.0, QUANTUM_DECIMALS)


def probe_set(name: str) -> dict:
    if name not in PROBE_SETS:
        raise KeyError("probe set outside the fixed set: " + repr(name))
    return PROBE_SETS[name]


def transcript(rule, set_name: str) -> list:
    """Drive the rule over the fixed probe inputs and record quantized deltas.

    The recorded quantity is the per-step, per-coordinate parameter DELTA, not
    the parameter value, because the delta is what an update rule is. Two rules
    that reach the same place by different increments are behaviourally
    different and this records that difference.
    """
    spec = probe_set(set_name)
    params = [float(value) for value in spec["params0"]]
    state: dict = {}
    rows = []
    for index in range(spec["steps"]):
        grads = [float(value) for value in spec["grads"][index]]
        before = list(params)
        params, state = rule.step(params, grads, state)
        params = [float(value) for value in params]
        state = state if isinstance(state, dict) else {}
        rows.append([quantize(after - old) for old, after in zip(before, params)])
    return rows


def flatten(rows) -> list:
    """Step-major then coordinate order. Fixed, never a set and never a sort."""
    out = []
    for row in rows:
        for value in row:
            out.append(value)
    return out


def digest(rows) -> str:
    payload = json.dumps(
        {"schema": SCHEMA, "quantum": QUANTUM_DECIMALS, "transcript": rows},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def signature(rule, set_name: str) -> dict:
    rows = transcript(rule, set_name)
    return {"probe_set": set_name, "transcript": rows, "digest": digest(rows)}


def distance(left_rows, right_rows) -> float:
    """Relative behavioural distance between two transcripts.

    Scale-relative rather than absolute, so a rule cannot clear the gate by
    multiplying an existing rule's step by a constant: scaling moves the
    absolute difference and leaves the relative one where it was for a
    proportional rule, and for a genuinely different rule the relative measure
    is what separates it. Bounded below by the quantum so two transcripts that
    agree to the recorded precision score exactly zero.
    """
    left = flatten(left_rows)
    right = flatten(right_rows)
    if len(left) != len(right):
        return 1.0
    scale = 0.0
    spread = 0.0
    for a, b in zip(left, right):
        scale = max(scale, abs(a), abs(b))
        spread = max(spread, abs(a - b))
    if scale <= 0.0:
        return 0.0
    return quantize(spread / scale)


def nearest(rows, corpus_signatures) -> dict:
    """The closest corpus entry, by behavioural distance. Ties break by id.

    Iteration is over a list sorted by id, so the answer never depends on the
    order a mapping happened to be built in.
    """
    best_id = ""
    best = 1.0
    for entry in sorted(corpus_signatures, key=lambda row: row["id"]):
        found = distance(rows, entry["transcript"])
        if best_id == "" or found < best:
            best_id = entry["id"]
            best = found
    return {"nearest_corpus_id": best_id, "min_behavioural_distance": best}
