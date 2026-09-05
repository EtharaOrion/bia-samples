"""The ten graded checkers. Pure, deterministic, and none of them reads a clock.

Every number a checker compares comes from one of two places and never from a third:

  1. state the HARNESS produced. The verifier opens the frozen warehouse read-only through
     `tests/runner.py`, dumps the three frozen tables, and runs both the submitted plan and the
     frozen baseline plan on `environment/planrun.py`, which reports the engine's own
     virtual-machine step count. A checker reads all of that through the `Harness` handle below.
     A checker never calls `time`, `datetime`, `perf_counter`, `monotonic`, or any other clock,
     directly or transitively, and it never opens a database itself.

  2. a bound constant handed in as the `Bound` argument, read by `tests/grade.py` from
     `tests/bound.json`, which `solution/recompute.py` derived from `solution/grounding.yaml`.

Neither discovery value is carried in `Bound` as a plain value. The excluded row identifier and
the closure cardinality are RE-DERIVED here, by re-applying the exclusion rule to the frozen
lineage rows and by walking the closure relation over the frozen edges. `Bound` carries only a
digest over that closure, so the verifier's derivation and the author's derivation are checked
against each other without either of them being written down as an answer.

No number the submission printed or reported reaches the reward. Four submission-authored
fields are read, and each only so that a substitution is detectable rather than ignorable:
`closure_cardinality`, `admitted_row_count`, `reported_plan_cost` and `reported_baseline_cost`.

Imports are confined to the allowlist: json, hashlib, pathlib, dataclasses, typing. Nothing here
imports the submission, opens a socket, consults an environment secret, draws from a random
source, or reads a file the author planted with an answer in it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Machine-readable zero reasons. A downstream grader branches on these strings.
# ---------------------------------------------------------------------------
REASON_FROZEN_INPUTS = "frozen-inputs-modified"
REASON_PLAN_SHAPE = "plan-not-read-only-single-statement"
REASON_ROW_MISIDENTIFIED = "quarantine-row-misidentified"
REASON_CLOSURE_INCOMPLETE = "closure-incomplete"
REASON_TRAVERSAL_UNWITNESSED = "traversal-order-unwitnessed"
REASON_CARDINALITY_UNATTESTED = "closure-cardinality-unattested"
REASON_EXCLUDED_ADMITTED = "excluded-row-admitted"
REASON_RESULT_DIVERGENCE = "result-set-divergence"
REASON_COST_READOUT = "cost-readout-mismatch"
REASON_NOT_FASTER = "plan-not-faster-than-baseline"

REQUIRED_COLUMNS = ("reading_id", "node_id", "value")


@dataclass(frozen=True)
class Bound:
    """Every constant the checkers compare against. Handed in, never authored here."""

    warehouse_sha256: str
    spec_sha256: str
    lineage_count: int
    derives_count: int
    reading_count: int
    check_alpha: int
    check_beta: int
    check_gamma: int
    check_modulus: int
    closure_digest: str
    required_rows_digest: str
    baseline_plan_sha256: str
    baseline_plan_cost: int
    reference_plan_cost: int
    instance_baseline_speedup: float
    instance_target_speedup: float

    @staticmethod
    def from_mapping(payload: Dict[str, Any]) -> "Bound":
        return Bound(
            warehouse_sha256=str(payload["warehouse_sha256"]),
            spec_sha256=str(payload["spec_sha256"]),
            lineage_count=int(payload["lineage_count"]),
            derives_count=int(payload["derives_count"]),
            reading_count=int(payload["reading_count"]),
            check_alpha=int(payload["check_alpha"]),
            check_beta=int(payload["check_beta"]),
            check_gamma=int(payload["check_gamma"]),
            check_modulus=int(payload["check_modulus"]),
            closure_digest=str(payload["closure_digest"]),
            required_rows_digest=str(payload["required_rows_digest"]),
            baseline_plan_sha256=str(payload["baseline_plan_sha256"]),
            baseline_plan_cost=int(payload["baseline_plan_cost"]),
            reference_plan_cost=int(payload["reference_plan_cost"]),
            instance_baseline_speedup=float(payload["instance_baseline_speedup"]),
            instance_target_speedup=float(payload["instance_target_speedup"]),
        )


@dataclass
class Harness:
    """The real handle. Every read below is of state a run actually produced.

    `lineage`, `derives` and `reading` are the frozen tables as the verifier's own read-only
    dump returned them. `plan_run` and `baseline_run` are the executor's reports for the
    submitted plan and for the frozen baseline plan. `workspace` is the run directory the
    submission produced.
    """

    lineage: List[Dict[str, Any]] = field(default_factory=list)
    derives: List[Dict[str, Any]] = field(default_factory=list)
    reading: List[Dict[str, Any]] = field(default_factory=list)
    warehouse_sha256: str = ""
    spec_sha256: str = ""
    plan_text: str = ""
    plan_run: Dict[str, Any] = field(default_factory=dict)
    baseline_run: Dict[str, Any] = field(default_factory=dict)
    workspace: Optional[Path] = None
    _cache: Dict[str, Any] = field(default_factory=dict)

    # -- workspace readers ---------------------------------------------------
    def submission(self) -> Dict[str, Any]:
        if "submission" in self._cache:
            return self._cache["submission"]
        value: Any = None
        if self.workspace is not None:
            path = Path(self.workspace) / "submission.json"
            if path.is_file():
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                except ValueError:
                    value = None
        value = value if isinstance(value, dict) else {}
        self._cache["submission"] = value
        return value

    # -- derived warehouse facts --------------------------------------------
    def quarantined(self, bound: "Bound") -> List[str]:
        """Every lineage row whose declared checksum disagrees with its own payload."""
        key = "quarantined"
        if key in self._cache:
            return self._cache[key]
        out = []
        for row in self.lineage:
            expected = (
                int(row["p_alpha"]) * bound.check_alpha
                + int(row["p_beta"]) * bound.check_beta
                + int(row["p_gamma"]) * bound.check_gamma
            ) % bound.check_modulus
            if int(row["declared_check"]) != expected:
                out.append(str(row["node_id"]))
        out.sort()
        self._cache[key] = out
        return out

    def edges(self) -> Dict[str, List[str]]:
        """The closure relation: parent to child, unioned with derives source to target."""
        if "edges" in self._cache:
            return self._cache["edges"]
        table: Dict[str, set] = {}
        for row in self.lineage:
            parent = row.get("parent_id")
            if parent is not None:
                table.setdefault(str(parent), set()).add(str(row["node_id"]))
        for row in self.derives:
            table.setdefault(str(row["source_node"]), set()).add(str(row["target_node"]))
        resolved = {key: sorted(value) for key, value in table.items()}
        self._cache["edges"] = resolved
        return resolved

    def closure_layers(self, bound: "Bound") -> List[List[str]]:
        """Breadth-first layers from the excluded row. Layer zero is the excluded row alone."""
        key = "layers"
        if key in self._cache:
            return self._cache[key]
        origins = self.quarantined(bound)
        if len(origins) != 1:
            self._cache[key] = []
            return []
        layers = breadth_first_layers(self.edges(), origins[0])
        self._cache[key] = layers
        return layers

    def closure_nodes(self, bound: "Bound") -> List[str]:
        nodes: List[str] = []
        for layer in self.closure_layers(bound):
            nodes.extend(layer)
        return sorted(nodes)

    def required_rows(self, bound: "Bound") -> List[List[Any]]:
        """The required result set: every reading outside the closure, in reading_id order."""
        key = "required_rows"
        if key in self._cache:
            return self._cache[key]
        excluded = set(self.closure_nodes(bound))
        rows = [
            [int(row["reading_id"]), str(row["node_id"]), int(row["value"])]
            for row in self.reading
            if str(row["node_id"]) not in excluded
        ]
        rows.sort(key=lambda entry: entry[0])
        self._cache[key] = rows
        return rows


@dataclass
class Outcome:
    """One checker's verdict. `reason` is empty exactly when the checker passed."""

    ident: str
    passed: bool
    reason: str
    detail: str
    value: Any = None


def _pass(ident: str, detail: str, value: Any = None) -> Outcome:
    return Outcome(ident, True, "", detail, value)


def _fail(ident: str, reason: str, detail: str, value: Any = None) -> Outcome:
    return Outcome(ident, False, reason, detail, value)


# ---------------------------------------------------------------------------
# Quantities the VERIFIER computes. Never a number the submission reported.
# ---------------------------------------------------------------------------


def breadth_first_layers(edges: Dict[str, Sequence[str]], origin: str) -> List[List[str]]:
    """The traversal itself. One node appears in exactly one layer, the first that reaches it."""
    seen = {origin}
    layers = [[origin]]
    frontier = [origin]
    while frontier:
        nxt: List[str] = []
        for node in frontier:
            for child in edges.get(node, ()):  # already sorted by Harness.edges
                if child not in seen:
                    seen.add(child)
                    nxt.append(child)
        if not nxt:
            break
        nxt.sort()
        layers.append(nxt)
        frontier = nxt
    return layers


def closure_digest(nodes: Sequence[str]) -> str:
    """A digest over the closure, so a bound file can bind it without spelling it out."""
    payload = "\n".join(sorted(str(node) for node in nodes)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def rows_digest(rows: Sequence[Sequence[Any]]) -> str:
    payload = json.dumps([list(row) for row in rows], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def plan_cost(harness: Harness) -> Optional[int]:
    value = harness.plan_run.get("cost_steps")
    return int(value) if isinstance(value, int) else None


def baseline_cost(harness: Harness) -> Optional[int]:
    value = harness.baseline_run.get("cost_steps")
    return int(value) if isinstance(value, int) else None


def speedup(harness: Harness) -> float:
    """The graded metric: the frozen baseline's cost over the submitted plan's cost."""
    mine = plan_cost(harness)
    base = baseline_cost(harness)
    if not mine or not base or mine <= 0:
        return 0.0
    return round(base / mine, 6)


def _as_list_of_str(value: Any) -> Optional[List[str]]:
    if not isinstance(value, list):
        return None
    out = []
    for entry in value:
        if not isinstance(entry, str):
            return None
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# The ten graded checkers, in grading order.
# ---------------------------------------------------------------------------


def check_warehouse_unmodified(harness: Harness, bound: Bound) -> Outcome:
    """VALUE. The frozen warehouse is byte-identical and re-derives the bound closure."""
    ident = "warehouse_unmodified"
    if harness.warehouse_sha256 != bound.warehouse_sha256:
        return _fail(
            ident,
            REASON_FROZEN_INPUTS,
            "the warehouse on disk digests to "
            + (harness.warehouse_sha256 or "<absent>")
            + " and the bound digest is "
            + bound.warehouse_sha256,
        )
    counted = (len(harness.lineage), len(harness.derives), len(harness.reading))
    expected = (bound.lineage_count, bound.derives_count, bound.reading_count)
    if counted != expected:
        return _fail(
            ident,
            REASON_FROZEN_INPUTS,
            "the frozen tables carry " + repr(counted) + " rows and the bound counts are "
            + repr(expected),
        )
    quarantined = harness.quarantined(bound)
    if len(quarantined) != 1:
        return _fail(
            ident,
            REASON_FROZEN_INPUTS,
            "the exclusion rule selects " + str(len(quarantined))
            + " lineage rows and the frozen warehouse establishes exactly one",
        )
    derived = closure_digest(harness.closure_nodes(bound))
    if derived != bound.closure_digest:
        return _fail(
            ident,
            REASON_FROZEN_INPUTS,
            "the closure the verifier derived from the frozen tables digests to "
            + derived
            + " and the bound closure digest is "
            + bound.closure_digest,
        )
    if rows_digest(harness.required_rows(bound)) != bound.required_rows_digest:
        return _fail(
            ident,
            REASON_FROZEN_INPUTS,
            "the required result set the verifier derived does not match the bound digest",
        )
    return _pass(ident, "the frozen warehouse is intact and re-derives the bound closure")


def check_plan_is_single_read_only_select(harness: Harness, bound: Bound) -> Outcome:
    """INVARIANT. Exactly one read-only select reached the executor, and it ran."""
    ident = "plan_is_single_read_only_select"
    if not str(harness.plan_text).strip():
        return _fail(ident, REASON_PLAN_SHAPE, "no plan.sql was submitted, or it is empty")
    if not harness.plan_run.get("ok"):
        return _fail(
            ident,
            REASON_PLAN_SHAPE,
            "the executor refused the plan with reason "
            + repr(harness.plan_run.get("reason", "<absent>"))
            + ": "
            + str(harness.plan_run.get("detail", ""))[:300],
        )
    return _pass(ident, "one read-only statement was admitted and executed by the harness")


def check_quarantine_row_identified(harness: Harness, bound: Bound) -> Outcome:
    """VALUE. The declared excluded row is the row the exclusion rule actually selects.

    This is the first of the two discovery values. It is not read from a constant: the verifier
    re-applies the checksum rule to the frozen lineage rows and takes whatever row fails.
    """
    ident = "quarantine_row_identified"
    quarantined = harness.quarantined(bound)
    if len(quarantined) != 1:
        return _fail(
            ident,
            REASON_ROW_MISIDENTIFIED,
            "the exclusion rule does not select exactly one row over the frozen lineage",
        )
    declared = harness.submission().get("excluded_node_id")
    if not isinstance(declared, str):
        return _fail(
            ident,
            REASON_ROW_MISIDENTIFIED,
            "the submission carries no string excluded_node_id",
        )
    if declared != quarantined[0]:
        return _fail(
            ident,
            REASON_ROW_MISIDENTIFIED,
            "the submission names " + repr(declared)
            + " and the exclusion rule selects a different lineage row",
        )
    return _pass(ident, "the declared excluded row is the row the rule selects", declared)


def check_closure_set_complete(harness: Harness, bound: Bound) -> Outcome:
    """DIVERGENCE. The declared closure equals the transitive closure, hop for hop."""
    ident = "closure_set_complete"
    truth = harness.closure_nodes(bound)
    declared = _as_list_of_str(harness.submission().get("closure_nodes"))
    if declared is None:
        return _fail(
            ident,
            REASON_CLOSURE_INCOMPLETE,
            "the submission carries no closure_nodes list of strings",
        )
    if declared != sorted(set(declared)):
        return _fail(
            ident,
            REASON_CLOSURE_INCOMPLETE,
            "closure_nodes is not a sorted list of distinct node identifiers",
        )
    truth_set = set(truth)
    declared_set = set(declared)
    if declared_set == truth_set:
        return _pass(ident, "the declared closure equals the transitive closure", len(truth))
    missing = sorted(truth_set - declared_set)
    extra = sorted(declared_set - truth_set)
    layers = harness.closure_layers(bound)
    first_hop = set(layers[0]) | set(layers[1] if len(layers) > 1 else [])
    if declared_set == first_hop:
        note = (
            "the declared closure is exactly the excluded row and its immediate successors, so "
            "the traversal stopped at the first hop instead of running to fixpoint; "
        )
    else:
        note = ""
    return _fail(
        ident,
        REASON_CLOSURE_INCOMPLETE,
        note
        + "the declared closure omits "
        + str(len(missing))
        + " node(s) and adds "
        + str(len(extra))
        + " that the relation does not reach",
        {"missing": len(missing), "extra": len(extra)},
    )


def check_closure_layers_witnessed(harness: Harness, bound: Bound) -> Outcome:
    """ORDERING. The declared breadth-first layers are the layers the traversal produces.

    This is what makes the cardinality a traversal result rather than a number. A submission
    that guessed the size cannot also produce the layer profile that size has to come out of.
    """
    ident = "closure_layers_witnessed"
    truth = harness.closure_layers(bound)
    declared = harness.submission().get("closure_layers")
    if not isinstance(declared, list) or not declared:
        return _fail(
            ident,
            REASON_TRAVERSAL_UNWITNESSED,
            "the submission carries no non-empty closure_layers witness",
        )
    shaped: List[List[str]] = []
    for entry in declared:
        layer = _as_list_of_str(entry)
        if layer is None:
            return _fail(
                ident,
                REASON_TRAVERSAL_UNWITNESSED,
                "closure_layers is not a list of lists of node identifiers",
            )
        shaped.append(layer)
    if shaped != truth:
        if len(shaped) == 1:
            note = (
                "the witness carries a single layer, so it records no traversal at all; "
            )
        elif len(shaped) < len(truth):
            note = (
                "the witness stops at distance " + str(len(shaped) - 1)
                + " and the relation reaches distance " + str(len(truth) - 1) + "; "
            )
        else:
            note = ""
        return _fail(
            ident,
            REASON_TRAVERSAL_UNWITNESSED,
            note
            + "the declared layer profile "
            + repr([len(layer) for layer in shaped])
            + " is not the profile the traversal produces",
        )
    return _pass(
        ident,
        "the declared layers reproduce the traversal, layer for layer",
        [len(layer) for layer in truth],
    )


def check_closure_cardinality_attested(harness: Harness, bound: Bound) -> Outcome:
    """VALUE. The declared cardinality is the size the traversal produced, three ways over.

    The number has to agree with the verifier's own closure, with the submission's own node
    list, and with the submission's own layer witness. A number that was reported rather than
    derived fails at least one of those three, which is the point of checking all three.
    """
    ident = "closure_cardinality_attested"
    truth = len(harness.closure_nodes(bound))
    submission = harness.submission()
    declared = submission.get("closure_cardinality")
    if not isinstance(declared, int) or isinstance(declared, bool):
        return _fail(
            ident,
            REASON_CARDINALITY_UNATTESTED,
            "the submission carries no integer closure_cardinality",
        )
    nodes = _as_list_of_str(submission.get("closure_nodes")) or []
    layers = submission.get("closure_layers")
    layer_total = 0
    if isinstance(layers, list):
        for entry in layers:
            if isinstance(entry, list):
                layer_total += len(entry)
    if declared != truth:
        return _fail(
            ident,
            REASON_CARDINALITY_UNATTESTED,
            "the submission reports a closure cardinality of " + str(declared)
            + ", which is not the size of the closure the relation produces; the reported figure "
            + "differs from the traversal by " + str(declared - truth),
            declared,
        )
    if declared != len(nodes) or declared != layer_total:
        return _fail(
            ident,
            REASON_CARDINALITY_UNATTESTED,
            "the reported cardinality " + str(declared)
            + " does not agree with the submission's own traversal: its node list holds "
            + str(len(nodes)) + " and its layers hold " + str(layer_total),
            declared,
        )
    return _pass(ident, "the cardinality agrees with the traversal on all three readings", truth)


def check_excluded_rows_absent(harness: Harness, bound: Bound) -> Outcome:
    """ABSENCE. Not one reading attached to a closure node survives into the result."""
    ident = "excluded_rows_absent"
    rows = harness.plan_run.get("rows")
    if not isinstance(rows, list):
        return _fail(ident, REASON_EXCLUDED_ADMITTED, "the plan produced no result rows to read")
    excluded = set(harness.closure_nodes(bound))
    offenders = []
    for row in rows:
        if isinstance(row, list) and len(row) >= 2 and str(row[1]) in excluded:
            offenders.append(int(row[0]) if isinstance(row[0], int) else row[0])
    if offenders:
        return _fail(
            ident,
            REASON_EXCLUDED_ADMITTED,
            "the plan admits " + str(len(offenders))
            + " reading(s) attached to nodes inside the closure, the first being reading_id "
            + str(offenders[0]),
            len(offenders),
        )
    return _pass(ident, "no reading inside the closure survives into the result", 0)


def check_result_set_equality(harness: Harness, bound: Bound) -> Outcome:
    """VALUE. The plan's result set is the required result set, row for row and in order."""
    ident = "result_set_equality"
    columns = harness.plan_run.get("columns")
    if list(columns or ()) != list(REQUIRED_COLUMNS):
        return _fail(
            ident,
            REASON_RESULT_DIVERGENCE,
            "the plan projects " + repr(list(columns or ()))
            + " and the required projection is " + repr(list(REQUIRED_COLUMNS)),
        )
    rows = harness.plan_run.get("rows")
    if not isinstance(rows, list):
        return _fail(ident, REASON_RESULT_DIVERGENCE, "the plan produced no result rows to read")
    normalised = [list(row) for row in rows]
    required = harness.required_rows(bound)
    if normalised != required:
        if len(normalised) != len(required):
            note = (
                "the plan returns " + str(len(normalised)) + " rows and the required result set "
                "holds " + str(len(required)) + "; "
            )
        else:
            note = "the plan returns the right row count in the wrong content or order; "
        return _fail(
            ident,
            REASON_RESULT_DIVERGENCE,
            note + "the result digest is " + rows_digest(normalised)
            + " and the required digest is " + bound.required_rows_digest,
        )
    declared = harness.submission().get("admitted_row_count")
    if declared != len(required):
        return _fail(
            ident,
            REASON_RESULT_DIVERGENCE,
            "the submission reports " + repr(declared)
            + " admitted rows and the plan returned " + str(len(required)),
        )
    return _pass(ident, "the result set equals the required result set exactly", len(required))


def check_plan_cost_readout_exact(harness: Harness, bound: Bound) -> Outcome:
    """VALUE. Both reported costs equal the costs the harness itself counted."""
    ident = "plan_cost_readout_exact"
    mine = plan_cost(harness)
    base = baseline_cost(harness)
    if mine is None or base is None:
        return _fail(ident, REASON_COST_READOUT, "the harness produced no cost for one of the plans")
    submission = harness.submission()
    reported = submission.get("reported_plan_cost")
    reported_base = submission.get("reported_baseline_cost")
    if reported != mine:
        return _fail(
            ident,
            REASON_COST_READOUT,
            "the submission reports a plan cost of " + repr(reported)
            + " and the harness counted " + str(mine),
        )
    if reported_base != base:
        return _fail(
            ident,
            REASON_COST_READOUT,
            "the submission reports a baseline cost of " + repr(reported_base)
            + " and the harness counted " + str(base),
        )
    return _pass(ident, "both reported costs equal the harness-counted costs", mine)


def check_plan_faster_than_baseline(harness: Harness, bound: Bound) -> Outcome:
    """EFFECT. The submitted plan actually costs less than the frozen baseline plan."""
    ident = "plan_faster_than_baseline"
    mine = plan_cost(harness)
    base = baseline_cost(harness)
    if mine is None or base is None or mine <= 0:
        return _fail(ident, REASON_NOT_FASTER, "the harness produced no usable cost")
    if mine >= base:
        return _fail(
            ident,
            REASON_NOT_FASTER,
            "the submitted plan costs " + str(mine)
            + " harness steps and the frozen baseline costs " + str(base)
            + ", so the plan establishes no speedup",
            speedup(harness),
        )
    return _pass(ident, "the submitted plan costs less than the frozen baseline", speedup(harness))
