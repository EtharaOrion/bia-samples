"""Harness-owned measurement for slot OER-14. Every number on the graded path is born here.

This module is the verifier's own process. It imports the FROZEN substrate from
`environment/substrate.py`, runs it, and hands the resulting telemetry to `checkers.py`.

It never imports the submission. A submission is a JSON DOCUMENT, and this module reads that
document only to learn which allocation to measure and which attempt record to re-measure.
No number the document carries is copied onto the graded path: every bits-per-byte figure in
the context this builds is one this process computed, from the frozen evaluation corpus, at
the evaluation points the substrate schedules.

It reads no clock, no random source, no network and no environment secret. Two runs over the
same document produce byte-identical context.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]
TESTS = BUNDLE / "tests"
sys.path.insert(0, str(BUNDLE / "environment"))

import substrate  # noqa: E402

CONTROLS = TESTS / "controls.json"

#: The FIXED key list the grading path consumes out of the context. It is a constant of this
#: module, so no submission can add itself to it: there is no code path that appends. Every
#: entry names a harness-owned surface. A key naming a submission field would be a defect and
#: `no_submission_reported_number_on_graded_path` is the checker that reads this list.
CONSUMED_KEYS = (
    "harness.frozen",
    "harness.points",
    "harness.meter",
    "harness.anchors",
    "harness.attempts",
    "harness.graded.bpb",
)

#: Fields of the submission document that a grading path must never consume. Mirrors
#: `substrate.VocabSpec.reported_fields()`, which is the frozen statement of the same set.
FORBIDDEN_CONSUMED = (
    "denominator_override",
    "graded_readout_filter",
    "notes",
    "reported_bpb",
)

_MEASURED: dict = {}


def controls() -> dict:
    """The harness control table. Generated from solution/grounding.yaml by recompute.py."""
    with CONTROLS.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def corpora() -> tuple:
    return substrate.load_frozen("train.txt"), substrate.load_frozen("eval.txt")


def _key(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def telemetry_for(payload: dict) -> dict:
    """Run the frozen substrate over one submission payload and return its telemetry.

    Memoised on the canonical payload, because the control table and the attempt record ask
    for the same allocation more than once and the substrate is a pure function of it.
    """
    cached = _MEASURED.get(_key(payload))
    if cached is not None:
        return cached
    train, evaluation = corpora()
    spec = substrate.VocabSpec.from_payload(payload)
    meter = substrate.Meter(substrate.COMPUTE_BUDGET_UNITS)
    result = substrate.measure(spec, train, evaluation, meter)
    _MEASURED[_key(payload)] = result
    return result


def reading_for(allocation: dict) -> float:
    """The sustained bits per byte of one allocation, as the harness reads it."""
    telemetry = telemetry_for({"allocations": dict(allocation)})
    return substrate.graded_bpb(telemetry["points"], substrate.frozen_eval_byte_count())


def anchors() -> dict:
    """Resolve the reward formula's two anchors BY MEASUREMENT, never from a constant.

    `baseline_metric` is the best reading any single-direction sweep of the whole slot budget
    reaches, so a greedy sweep of one direction scores exactly 0.0 by construction.
    `target_metric` is the harness's own reallocated reference at the same frozen budget.
    Neither number is written anywhere in this bundle; both are produced by this process.
    """
    table = controls()
    sweep = {
        name: reading_for(allocation)
        for name, allocation in sorted(table["single_direction_sweep"].items())
    }
    return {
        "sweep": sweep,
        "baseline_metric": min(sweep.values()),
        "baseline_source": "measured: best single-direction sweep of the whole slot budget",
        "target_metric": reading_for(table["reference_allocation"]),
        "target_source": "measured: harness reference reallocation at the frozen compute budget",
        "separation_margin": float(table["separation_margin"]),
        "anchors_state": table["anchors_state"],
        "anchors_gap": table["anchors_gap"],
    }


def _attempt_rows(document: dict, flattened: str) -> list:
    """Re-measure every attempt the record carries. A fabricated frontier does not survive."""
    rows, previous = [], None
    for raw in document.get("attempts") or []:
        row = raw if isinstance(raw, dict) else {}
        allocation = {name: 0 for name in substrate.DIRECTIONS}
        for name, value in (row.get("allocation") or {}).items():
            if name in allocation:
                allocation[name] = int(value)
        reading = reading_for(allocation)
        carried = row.get("carried_frontier")
        rows.append(
            {
                "index": int(row.get("index", len(rows) + 1)),
                "direction": str(row.get("direction", "")),
                "allocation": allocation,
                "reallocated_from": row.get("reallocated_from"),
                "bpb": reading,
                "marginal_gain_bits": 0.0 if previous is None else previous - reading,
                "carried_frontier": dict(carried) if isinstance(carried, dict) else {},
                "first": previous is None,
            }
        )
        previous = reading
    del flattened
    return rows


def context_for(document) -> dict:
    """Assemble the whole harness-owned context one grading pass reads."""
    payload = document if isinstance(document, dict) else {}
    table = controls()
    telemetry = telemetry_for(payload)
    denominator = substrate.frozen_eval_byte_count()
    flattened = str(table["drift_state_final"].get("paying_direction", ""))
    return {
        "frozen": {
            "eval_bytes": denominator,
            "eval_sha256": substrate.frozen_eval_digest(),
            "budget_units": substrate.COMPUTE_BUDGET_UNITS,
            "band_tolerance_bpb": substrate.BAND_TOLERANCE_BPB,
            "scheduled_points": len(substrate.EVALUATION_POINT_FRACTIONS),
            "slot_budget": substrate.SLOT_BUDGET,
            "directions": list(substrate.DIRECTIONS),
            "meter_source": "harness-counter",
            "denominator_key": "frozen_eval_byte_count",
        },
        "points": [dict(row) for row in telemetry["points"]],
        "meter": dict(telemetry["meter"]),
        "spec_requests": dict(telemetry["spec_requests"]),
        "graded": {
            "bpb": substrate.graded_bpb(telemetry["points"], denominator),
            "allocations": {
                name: int((payload.get("allocations") or {}).get(name, 0) or 0)
                for name in substrate.DIRECTIONS
            },
            "slots_requested": telemetry["slots_requested"],
            "vocab_size": telemetry["vocab_size"],
            "vocab_digest": telemetry["vocab_digest"],
        },
        "anchors": anchors(),
        "attempts": _attempt_rows(payload, flattened),
        "flatten": {
            "epsilon_bits": float(table["flatten_epsilon_bits"]),
            "run_required": int(table["flatten_run_required"]),
            "direction": "merge_depth",
            "note": (
                "the merge family admits a byte pair only at or above the frozen frequency "
                "floor, so it saturates at a finite depth and nothing announces it"
            ),
        },
        "consumed_keys": list(CONSUMED_KEYS),
        "forbidden_consumed": list(FORBIDDEN_CONSUMED),
    }


def reward_for(context: dict) -> float:
    """The bound formula, over anchors this process measured, clipped to the closed interval."""
    baseline = context["anchors"]["baseline_metric"]
    target = context["anchors"]["target_metric"]
    agent = context["graded"]["bpb"]
    if agent is None or baseline is None or target is None:
        return 0.0
    span = baseline - target
    if span <= 0.0:
        return 0.0
    raw = (baseline - agent) / span
    return min(max(raw, 0.0), 1.0)
