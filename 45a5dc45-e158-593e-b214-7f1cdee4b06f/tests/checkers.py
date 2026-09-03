"""The eight graded checkers. Pure, deterministic, and none of them reads a clock.

Every quantity a checker compares comes from one of two places and never from a third:

  1. a telemetry row the HARNESS produced. `tests/runner.py` launches
     `environment/fence_probe.py` against the BUILT fence state in an isolated subprocess, and
     that process walks the producer chain once and writes what it observed. A checker reads
     those rows through the `Harness` handle below and derives the admitted set, the crossing
     count and every reference verdict for itself.

  2. a bound constant handed in as the `Bound` argument, read by `tests/grade.py` from
     `tests/bound.json`, which `solution/recompute.py` derived from `solution/grounding.yaml`.

Neither discovery value is a bound constant. `tests/bound.json` carries a commitment over each
of them and never the value, so a checker that compared against a literal would have nothing to
compare against. The values are recomputed here, on every run, from harness-owned state.

Imports are confined to the allowlist: json, hashlib, pathlib, dataclasses, typing. Nothing
here imports the submission, opens a socket, consults an environment secret, draws from a
random source, or reads a file the author planted with an answer in it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Machine-readable zero reasons. A downstream grader branches on these strings.
# ---------------------------------------------------------------------------
REASON_STATE_MODIFIED = "fence-state-modified"
REASON_COUNT_UNACCOUNTED = "crossing-count-unaccounted"
REASON_ADMITTED_SET_WRONG = "admitted-set-wrong"
REASON_VERDICT_DISORDERED = "verdict-sequence-disordered"
REASON_VERDICT_DIVERGENCE = "verdict-record-divergence"
REASON_FENCE_BREACHED = "fence-breached"
REASON_QUOTA_EXCEEDED = "quota-ledger-exceeded"
REASON_BLANKET_REFUSAL = "blanket-refusal"
REASON_SUBMISSION_MALFORMED = "submission-malformed"

# The four refusal reasons the statement fixes, in the precedence the statement fixes.
REFUSAL_MALFORMED = "malformed-crossing"
REFUSAL_UNATTESTED = "unattested-origin"
REFUSAL_NOT_ADMITTED = "kind-not-admitted"
REFUSAL_QUOTA = "quota-exhausted"
REFUSAL_REASONS = (REFUSAL_MALFORMED, REFUSAL_UNATTESTED, REFUSAL_NOT_ADMITTED, REFUSAL_QUOTA)

# The two preimages the statement fixes. Restated here rather than imported, because a verifier
# that borrowed the producer's own implementation would be checking it against itself.
SEAL_DOMAIN = "oer26.registry-seal/v1"
RECORD_FIELDS = frozenset({"seq", "kind", "size", "origin", "epoch", "sig"})

COMMITMENT_DOMAIN = "oer26.discovery-commitment/v1"


@dataclass(frozen=True)
class Bound:
    """Every constant the checkers compare against. Handed in, never authored here."""

    state_sha256: str
    crossing_count_commitment: str
    admitted_set_commitment: str
    baseline_agreement: float
    commitment_salt: str

    @staticmethod
    def from_mapping(payload: Dict[str, Any]) -> "Bound":
        return Bound(
            state_sha256=str(payload["state_sha256"]),
            crossing_count_commitment=str(payload["crossing_count_commitment"]),
            admitted_set_commitment=str(payload["admitted_set_commitment"]),
            baseline_agreement=float(payload["baseline_agreement"]),
            commitment_salt=str(payload["commitment_salt"]),
        )


@dataclass
class Harness:
    """The real handle. Every read below is of state a build actually produced."""

    telemetry: List[Dict[str, Any]] = field(default_factory=list)
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
        shaped = value if isinstance(value, dict) else {}
        self._cache["submission"] = shaped
        return shaped

    # -- telemetry readers ---------------------------------------------------
    def rows(self, name: str) -> List[Dict[str, Any]]:
        return [row for row in self.telemetry if row.get("row") == name]

    def fence_row(self) -> Dict[str, Any]:
        found = self.rows("fence")
        return found[0] if found else {}

    def registry_entries(self) -> List[Dict[str, Any]]:
        return self.rows("registry_entry")

    def segments(self) -> List[Dict[str, Any]]:
        return self.rows("segment")

    def crossings(self) -> List[Dict[str, Any]]:
        ordered = sorted(self.rows("crossing"), key=lambda row: int(row.get("index", -1)))
        return [row.get("record") or {} for row in ordered]


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
# The two discovery values, and the reference enforcement, recomputed by the
# VERIFIER from harness-owned telemetry. Never a literal, never a submitted number.
# ---------------------------------------------------------------------------


def admitted_kind_set(harness: Harness) -> List[str]:
    """Discovery value: the kinds this fence admits. Derived, not looked up."""
    admitted = []
    for entry in harness.registry_entries():
        if str(entry.get("status")) != "admitted":
            continue
        preimage = "|".join(
            [
                SEAL_DOMAIN,
                str(entry.get("kind")),
                str(entry.get("status")),
                str(int(entry.get("quota", 0))),
            ]
        )
        if hashlib.sha256(preimage.encode("utf-8")).hexdigest() == str(entry.get("seal")):
            admitted.append(str(entry.get("kind")))
    return sorted(admitted)


def crossing_count(harness: Harness) -> int:
    """Discovery value: how many crossings the linked chain presented at the fence."""
    return len(harness.crossings())


def quota_table(harness: Harness) -> Dict[str, int]:
    return {str(entry.get("kind")): int(entry.get("quota", 0)) for entry in harness.registry_entries()}


def _well_formed(record: Dict[str, Any]) -> bool:
    if set(record) != set(RECORD_FIELDS):
        return False
    for key in ("seq", "size", "epoch"):
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            return False
    for key in ("kind", "origin", "sig"):
        if not isinstance(record.get(key), str):
            return False
    return int(record["size"]) > 0


def _attested(record: Dict[str, Any], salt: str) -> bool:
    payload = "|".join(
        [
            str(record.get("seq")),
            str(record.get("kind")),
            str(record.get("size")),
            str(record.get("origin")),
            str(record.get("epoch")),
            str(salt),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16] == str(record.get("sig"))


def reference_verdicts(harness: Harness) -> List[Dict[str, Any]]:
    """The fence enforced correctly, in chain order, by the verifier itself.

    Quota is spent in chain order, so this is a fold over the traversal rather than a filter
    over a set. Nothing here consults the submission.
    """
    admitted = set(admitted_kind_set(harness))
    quotas = quota_table(harness)
    salt = str(harness.fence_row().get("salt", ""))
    spent: Dict[str, int] = {}
    verdicts = []
    for position, record in enumerate(harness.crossings()):
        kind = record.get("kind")
        if not _well_formed(record):
            decision, reason = "refuse", REFUSAL_MALFORMED
        elif not _attested(record, salt):
            decision, reason = "refuse", REFUSAL_UNATTESTED
        elif str(kind) not in admitted:
            decision, reason = "refuse", REFUSAL_NOT_ADMITTED
        elif spent.get(str(kind), 0) >= quotas.get(str(kind), 0):
            decision, reason = "refuse", REFUSAL_QUOTA
        else:
            spent[str(kind)] = spent.get(str(kind), 0) + 1
            decision, reason = "admit", ""
        verdicts.append(
            {
                "position": position,
                "seq": record.get("seq"),
                "kind": kind if isinstance(kind, str) else None,
                "decision": decision,
                "reason": reason,
            }
        )
    return verdicts


def submitted_verdicts(harness: Harness) -> List[Dict[str, Any]]:
    rows = harness.submission().get("verdicts")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def agreement(harness: Harness) -> float:
    """The graded quantity: verdict-by-verdict agreement, decision and reason both."""
    truth = reference_verdicts(harness)
    if not truth:
        return 0.0
    claimed = submitted_verdicts(harness)
    matched = 0
    for position, expected in enumerate(truth):
        if position >= len(claimed):
            break
        row = claimed[position]
        if str(row.get("decision", "")) != expected["decision"]:
            continue
        if str(row.get("reason", "")) != expected["reason"]:
            continue
        matched += 1
    return round(matched / len(truth), 6)


def commitment(kind: str, value: str, salt: str) -> str:
    """The commitment tests/bound.json carries in place of a discovery value."""
    return hashlib.sha256(
        "|".join([COMMITMENT_DOMAIN, kind, value, salt]).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# The eight checkers. Exactly one reduction each.
# ---------------------------------------------------------------------------


def check_fence_state_unmodified(harness: Harness, bound: Bound) -> Outcome:
    """VALUE. The built fence state is the state the bundle was frozen over."""
    ident = "fence_state_unmodified"
    observed = str(harness.fence_row().get("state_sha256", ""))
    if not observed:
        return _fail(ident, REASON_STATE_MODIFIED, "the harness emitted no fence state digest")
    if observed != bound.state_sha256:
        return _fail(
            ident,
            REASON_STATE_MODIFIED,
            "the built fence state digests to " + observed + ", bound is " + bound.state_sha256,
        )
    if not harness.registry_entries():
        return _fail(ident, REASON_STATE_MODIFIED, "the fence registry carries no declaration")
    if not harness.crossings():
        return _fail(ident, REASON_STATE_MODIFIED, "the producer chain presents no crossing")
    return _pass(ident, "the built fence state matches its bound digest", observed)


def check_crossing_count_accounted(harness: Harness, bound: Bound) -> Outcome:
    """VALUE. Every crossing the traversal presented is accounted for, and only those.

    This is the checker the first discovery value decides. The count is recomputed from the
    harness traversal on every run and cross-checked against the commitment the bundle froze,
    so a submission that reports a count it did not derive from the chain is refused whether the
    number it invented is larger or smaller.
    """
    ident = "crossing_count_accounted"
    observed = crossing_count(harness)
    expected_commitment = commitment("crossings", str(observed), bound.commitment_salt)
    if expected_commitment != bound.crossing_count_commitment:
        return _fail(
            ident,
            REASON_COUNT_UNACCOUNTED,
            "the traversal presented a crossing count whose commitment does not match the bound "
            "commitment, so the built state and the frozen bundle disagree",
        )

    submission = harness.submission()
    claimed = submission.get("crossings_observed")
    if not isinstance(claimed, int) or isinstance(claimed, bool):
        return _fail(
            ident,
            REASON_COUNT_UNACCOUNTED,
            "the submission reports crossings_observed " + repr(claimed) + ", which is not an integer",
        )
    if claimed != observed:
        return _fail(
            ident,
            REASON_COUNT_UNACCOUNTED,
            "the submission reports " + str(claimed) + " crossings and the harness traversal "
            "presented " + str(observed),
        )

    verdicts = submitted_verdicts(harness)
    if len(verdicts) != observed:
        return _fail(
            ident,
            REASON_COUNT_UNACCOUNTED,
            "the submission carries " + str(len(verdicts)) + " verdicts against " + str(observed)
            + " crossings",
        )

    admitted = submission.get("admitted")
    refused = submission.get("refused")
    if not isinstance(admitted, int) or not isinstance(refused, int):
        return _fail(
            ident,
            REASON_COUNT_UNACCOUNTED,
            "the submission does not carry integer admitted and refused counts",
        )
    if admitted + refused != observed:
        return _fail(
            ident,
            REASON_COUNT_UNACCOUNTED,
            "admitted " + str(admitted) + " plus refused " + str(refused) + " is not the crossing "
            "count " + str(observed),
        )

    refusals = submission.get("refusals")
    if not isinstance(refusals, dict) or sorted(refusals) != sorted(REFUSAL_REASONS):
        return _fail(
            ident,
            REASON_COUNT_UNACCOUNTED,
            "the submission's refusals block does not carry exactly the four bound refusal reasons",
        )
    if sum(int(refusals[name]) for name in REFUSAL_REASONS) != refused:
        return _fail(
            ident,
            REASON_COUNT_UNACCOUNTED,
            "the four refusal counts do not sum to the reported refused total " + str(refused),
        )

    counted_admits = sum(1 for row in verdicts if str(row.get("decision")) == "admit")
    if counted_admits != admitted:
        return _fail(
            ident,
            REASON_COUNT_UNACCOUNTED,
            "the verdict list carries " + str(counted_admits) + " admissions against a reported "
            + str(admitted),
        )
    return _pass(ident, "every crossing the traversal presented is accounted for", observed)


def check_admitted_kind_set_exact(harness: Harness, bound: Bound) -> Outcome:
    """VALUE. The submitted admitted set is the set the built registry establishes.

    This is the checker the second discovery value decides. The set is derived here from the
    registry declarations by the stated rule, so a set assembled by reading the status column
    alone carries the retired declarations and is refused, and a set that differs from the
    derived one by a single kind in either direction is refused.
    """
    ident = "admitted_kind_set_exact"
    derived = admitted_kind_set(harness)
    expected_commitment = commitment("admitted", ",".join(derived), bound.commitment_salt)
    if expected_commitment != bound.admitted_set_commitment:
        return _fail(
            ident,
            REASON_ADMITTED_SET_WRONG,
            "the admitted set derived from the built registry does not match the bound "
            "commitment, so the built state and the frozen bundle disagree",
        )
    claimed_raw = harness.submission().get("admitted_kinds")
    if not isinstance(claimed_raw, list) or not all(isinstance(item, str) for item in claimed_raw):
        return _fail(
            ident,
            REASON_ADMITTED_SET_WRONG,
            "the submission does not carry admitted_kinds as a list of strings",
        )
    claimed = sorted(claimed_raw)
    if len(set(claimed)) != len(claimed):
        return _fail(
            ident, REASON_ADMITTED_SET_WRONG, "the submitted admitted set repeats a kind"
        )
    if claimed != derived:
        missing = [item for item in derived if item not in claimed]
        extra = [item for item in claimed if item not in derived]
        return _fail(
            ident,
            REASON_ADMITTED_SET_WRONG,
            "the submitted admitted set carries " + str(len(claimed)) + " kinds against "
            + str(len(derived)) + " the registry establishes; missing " + str(len(missing))
            + ", unsupported " + str(len(extra)),
        )
    return _pass(ident, "the submitted admitted set is the set the registry establishes", len(derived))


def check_verdict_sequence_ordered(harness: Harness, bound: Bound) -> Outcome:
    """ORDERING. One verdict per crossing, in the order the chain presented them."""
    ident = "verdict_sequence_ordered"
    truth = reference_verdicts(harness)
    claimed = submitted_verdicts(harness)
    if len(claimed) != len(truth):
        return _fail(
            ident,
            REASON_VERDICT_DISORDERED,
            "the submission carries " + str(len(claimed)) + " verdicts against " + str(len(truth))
            + " crossings",
        )
    for position, expected in enumerate(truth):
        row = claimed[position]
        if row.get("seq") != expected["seq"]:
            return _fail(
                ident,
                REASON_VERDICT_DISORDERED,
                "verdict at position " + str(position) + " names seq " + repr(row.get("seq"))
                + " and the chain presented seq " + repr(expected["seq"]),
            )
        decision = str(row.get("decision", ""))
        if decision not in ("admit", "refuse"):
            return _fail(
                ident,
                REASON_VERDICT_DISORDERED,
                "verdict at position " + str(position) + " carries decision " + repr(decision),
            )
        reason = str(row.get("reason", ""))
        if decision == "admit" and reason != "":
            return _fail(
                ident,
                REASON_VERDICT_DISORDERED,
                "the admission at position " + str(position) + " carries reason " + repr(reason),
            )
        if decision == "refuse" and reason not in REFUSAL_REASONS:
            return _fail(
                ident,
                REASON_VERDICT_DISORDERED,
                "the refusal at position " + str(position) + " carries reason " + repr(reason)
                + ", which is not one of the four bound refusal reasons",
            )
    return _pass(ident, "one verdict per crossing, in chain order", len(truth))


def check_verdict_records_match_traversal(harness: Harness, bound: Bound) -> Outcome:
    """DIVERGENCE. Each verdict is about the record the traversal actually reached.

    Every verdict echoes the kind its record carried. A verdict list assembled from the union of
    the segment files on disk, or from any stream other than the linked chain, diverges here
    even when it happens to carry the right number of rows.
    """
    ident = "verdict_records_match_traversal"
    truth = reference_verdicts(harness)
    claimed = submitted_verdicts(harness)
    if len(claimed) != len(truth):
        return _fail(
            ident,
            REASON_VERDICT_DIVERGENCE,
            "the submission carries " + str(len(claimed)) + " verdicts against " + str(len(truth))
            + " crossings",
        )
    divergent = 0
    first = None
    for position, expected in enumerate(truth):
        echoed = claimed[position].get("kind")
        if echoed != expected["kind"]:
            divergent += 1
            if first is None:
                first = position
    if divergent:
        return _fail(
            ident,
            REASON_VERDICT_DIVERGENCE,
            str(divergent) + " verdicts echo a kind the traversal did not present at that "
            "position, first at position " + str(first),
        )
    return _pass(ident, "every verdict echoes the record the traversal presented", len(truth))


def check_no_unadmitted_kind_admitted(harness: Harness, bound: Bound) -> Outcome:
    """ABSENCE. Nothing outside the admitted set crossed the fence."""
    ident = "no_unadmitted_kind_admitted"
    admitted = set(admitted_kind_set(harness))
    truth = reference_verdicts(harness)
    claimed = submitted_verdicts(harness)
    breaches = []
    for position, expected in enumerate(truth):
        if position >= len(claimed):
            break
        if str(claimed[position].get("decision", "")) != "admit":
            continue
        if expected["kind"] is None or str(expected["kind"]) not in admitted:
            breaches.append(position)
    if breaches:
        return _fail(
            ident,
            REASON_FENCE_BREACHED,
            str(len(breaches)) + " crossings of a kind this fence does not admit were admitted, "
            "first at position " + str(breaches[0]),
        )
    return _pass(ident, "no crossing of an unadmitted kind was admitted", 0)


def check_quota_ledger_respected(harness: Harness, bound: Bound) -> Outcome:
    """EFFECT. Admitting spends quota, and no kind is admitted past its published quota."""
    ident = "quota_ledger_respected"
    quotas = quota_table(harness)
    truth = reference_verdicts(harness)
    claimed = submitted_verdicts(harness)
    spent: Dict[str, int] = {}
    for position, expected in enumerate(truth):
        if position >= len(claimed):
            break
        if str(claimed[position].get("decision", "")) != "admit":
            continue
        kind = expected["kind"]
        if kind is None:
            continue
        spent[kind] = spent.get(kind, 0) + 1
        if spent[kind] > quotas.get(kind, 0):
            return _fail(
                ident,
                REASON_QUOTA_EXCEEDED,
                "kind " + kind + " was admitted " + str(spent[kind]) + " times against a "
                "published quota of " + str(quotas.get(kind, 0)) + ", first exceeded at position "
                + str(position),
            )
    return _pass(ident, "no kind was admitted past its published quota", sum(spent.values()))


def check_admission_not_blanket(harness: Harness, bound: Bound) -> Outcome:
    """INVARIANT. The fence admits, and what it admits is admissible.

    Refusing everything is not enforcing a fence, and neither is admitting a crossing the
    semantics refuse. The invariant is that the submitted admissions are a non-empty subset of
    the admissions the semantics produce, which leaves an under-admitting run gradeable while
    closing both blanket policies.
    """
    ident = "admission_not_blanket"
    truth = reference_verdicts(harness)
    claimed = submitted_verdicts(harness)
    admissible = {row["position"] for row in truth if row["decision"] == "admit"}
    submitted = {
        position
        for position, row in enumerate(claimed)
        if str(row.get("decision", "")) == "admit"
    }
    if not submitted:
        return _fail(
            ident,
            REASON_BLANKET_REFUSAL,
            "the submission admits no crossing at all while the semantics admit "
            + str(len(admissible)),
        )
    outside = sorted(submitted - admissible)
    if outside:
        return _fail(
            ident,
            REASON_BLANKET_REFUSAL,
            str(len(outside)) + " admitted crossings are refused by the fence semantics, first at "
            "position " + str(outside[0]),
        )
    return _pass(
        ident,
        "the submitted admissions are a non-empty subset of the admissible crossings",
        len(submitted),
    )
