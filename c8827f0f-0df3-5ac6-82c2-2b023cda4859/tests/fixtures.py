"""Fixture construction for the compiled checker suite of slot OER-14.

Two kinds of fixture live here and they answer different questions.

A CLEAN context is the harness-owned context produced by running the frozen substrate over
the reference session. It is the accepting half of every checker.

A PLANTED context is that same context with one defect applied. Some defects are reachable by
a submission and some are not: a submission never runs inside the grading process, so it can
never substitute the evaluated state, put a filter on the harness's own readout, or add a key
to the fixed list the grading path consumes. For those checkers the rejecting half is carried
on a planted telemetry fixture, and the substitution is recorded in `tests/checkers.yaml`
under `rejecting_half` rather than papered over.

The defect plan is `solution/fixtures/plan.json`, generated from `solution/grounding.yaml` by
`solution/recompute.py`. Nothing here is hand-chosen.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "tests"))

import harness  # noqa: E402

PLAN = BUNDLE / "solution" / "fixtures" / "plan.json"


def plan() -> dict:
    with PLAN.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def reference_document() -> dict:
    """The reference submission, rebuilt from the generated plan through the harness.

    Same construction `solution/reference.py` performs, over the same generated trajectory,
    so `adequacy.py` can bind the reference bytes by sha256 and still compare like for like.
    It reads back through `probe_reading_for`, which is the reading the SESSION takes on the
    agent surface, because that is the only reading an agent phase can take: the document
    this rebuilds has to be one a solver could have written.
    """
    trajectory = plan()["golden_trajectory"]
    directions = harness.substrate.DIRECTIONS
    attempts, frontier, previous = [], {}, None
    for row in trajectory["attempts"]:
        allocation = {name: 0 for name in directions}
        allocation.update(row["allocation"])
        reading = harness.probe_reading_for(allocation)
        direction = row["direction"]
        best = frontier.get(direction)
        frontier[direction] = reading if best is None else min(best, reading)
        attempts.append(
            {
                "index": int(row["index"]),
                "direction": direction,
                "allocation": allocation,
                "reallocated_from": row["reallocated_from"],
                "marginal_gain_bits": 0.0 if previous is None else previous - reading,
                "carried_frontier": dict(sorted(frontier.items())),
            }
        )
        previous = reading
    graded_index = int(trajectory["graded_attempt"])
    graded = next(row for row in attempts if row["index"] == graded_index)
    return {
        "allocations": graded["allocation"],
        "extra_tokens": [],
        "train_units_requested": harness.substrate.COMPUTE_BUDGET_UNITS,
        "graded_readout_filter": "none",
        "denominator_override": 0,
        "stop_after_point": 0,
        "reported_bpb": None,
        "notes": "reallocated off merge_depth once its marginal gain reached zero",
        "graded_attempt_index": graded_index,
        "attempts": attempts,
    }


def document_for(spec: dict) -> dict:
    """A submission document from a negative-control spec: the reference, then the defect."""
    document = reference_document()
    row = dict(spec or {})
    if "allocations" in row:
        allocation = {name: 0 for name in harness.substrate.DIRECTIONS}
        allocation.update(row.pop("allocations") or {})
        document["allocations"] = allocation
    document.update(row)
    return document


def clean_context() -> dict:
    return harness.context_for(reference_document())


# ---------------------------------------------------------------------------
# Defect application. One kind per planted fixture, applied to a deep copy.
# ---------------------------------------------------------------------------


def _points(context: dict) -> list:
    return context["points"]


def apply_defect(context: dict, defect: dict) -> dict:
    """Apply exactly one planted defect and return the mutated copy."""
    row = copy.deepcopy(context)
    kind = str((defect or {}).get("kind", ""))

    if kind == "set_point_field":
        for point in _points(row):
            point[defect["field"]] = defect["value"]
        return row

    if kind == "set_meter_field":
        row["meter"][defect["field"]] = defect["value"]
        return row

    if kind == "unreach_point":
        for point in _points(row):
            if int(point.get("point", 0)) == int(defect["point"]):
                point["reached"] = False
                point["nll_bits_total"] = None
                point["state_digest"] = None
        return row

    if kind == "scale_point_bits":
        for point in _points(row):
            if int(point.get("point", 0)) == int(defect["point"]):
                point["nll_bits_total"] = float(point["nll_bits_total"]) * float(defect["factor"])
        row["graded"]["bpb"] = max(
            float(item["nll_bits_total"]) / float(item["denominator_bytes"])
            for item in _points(row)
            if item.get("reached") and item.get("nll_bits_total") is not None
        )
        return row

    if kind == "add_consumed_key":
        row["consumed_keys"] = list(row["consumed_keys"]) + [defect["value"]]
        return row

    if kind == "truncate_attempts":
        row["attempts"] = row["attempts"][: int(defect["keep"])]
        return row

    if kind == "strip_reallocations":
        for attempt in row["attempts"]:
            attempt["reallocated_from"] = None
        return row

    if kind == "drop_frontier_key":
        for attempt in row["attempts"]:
            if int(attempt.get("index", 0)) == int(defect["attempt"]):
                attempt["carried_frontier"].pop(defect["key"], None)
        return row

    if kind == "perturb_corpus_param":
        name = str(defect["param"])
        row["corpus"][name] = int(row["corpus"][name]) + int(defect["delta"])
        return row

    if kind == "set_graded_allocation":
        allocation = {name: 0 for name in harness.substrate.DIRECTIONS}
        allocation.update(defect["allocation"])
        row["graded"]["allocations"] = allocation
        row["graded"]["bpb"] = harness.reading_for(allocation)
        return row

    raise ValueError("planted defect kind outside the closed set: " + repr(kind))


def _defect_of(checker_id: str) -> dict:
    for row in plan()["checkers"]:
        if row["id"] == checker_id:
            return row["defect"]
    raise KeyError("no planted defect is declared for checker " + repr(checker_id))


def planted_context(checker_id: str) -> dict:
    """The rejecting half for one checker: the clean context plus its one planted defect."""
    return apply_defect(clean_context(), _defect_of(checker_id))


# ---------------------------------------------------------------------------
# Negative controls. Each is a whole submission, not a hand-built context.
# ---------------------------------------------------------------------------


def negative_context(control: dict) -> dict:
    """The context one negative control produces, through the real harness."""
    if control.get("spec") is not None:
        return harness.context_for(document_for(control["spec"]))
    context = clean_context()
    for key in ("trajectory_defect", "telemetry_defect"):
        if control.get(key):
            context = apply_defect(context, control[key])
    return context


def negative_controls() -> list:
    return list(plan()["negative_controls"])
