"""Pure, deterministic checkers over live state the verifier's own process produced.

Imports are restricted to the allowlist: json, math, hashlib, pathlib, dataclasses,
typing. Nothing here reads a planted file, an environment secret, a clock, a random
source or a socket, and nothing here imports the submission. Every number a checker
compares is either a frozen byte the delivery unit carries or a value the verifier's
own harness run recorded; no value a submission printed or wrote is ever read on a
path that decides an outcome.

`hashlib` and `pathlib` are imported because the evidence assembly in tests/grade.py
hands digests and paths through this module's dataclass, and a checker re-derives a
digest rather than trusting one it was handed.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Floating-point identity for values both sides computed the same way. Not a
# tolerance on the metric: the metric's own tolerance is a frozen manifest field.
EXACT = 1e-12

BOUND_ROLE = "bound"
SUSTAIN_ROLE = "sustain"
PROGRESS_ROLE = "progress"


@dataclass(frozen=True)
class Evidence:
    """Everything a checker is allowed to see, assembled by tests/grade.py.

    `telemetry` is the record the verifier's own harness run produced.
    `manifest` is the frozen environment/manifest.json of the delivery unit.
    The two digest fields are taken by tests/runner.py from bytes on disk and from
    the list the submission handed back, in the verifier's process.
    """

    telemetry: Dict[str, Any]
    manifest: Dict[str, Any]
    submitted_vocab_digest: str
    eval_corpus_digest: str
    eval_corpus_bytes: int
    delivery_root: str = ""
    # Recomputed by tests/runner.py from the training corpus bytes the frozen
    # harness resolves, under the harness's own token-length cap. It is live state
    # and never a number a submission reported.
    train_merge_capacity: int = -1
    train_merge_capacity_digest: str = ""


@dataclass(frozen=True)
class Outcome:
    passed: bool
    reason: str
    detail: str

    def as_dict(self) -> Dict[str, Any]:
        return {"passed": self.passed, "reason": self.reason, "detail": self.detail}


def _ok(detail: str) -> Outcome:
    return Outcome(True, "", detail)


def _no(reason: str, detail: str) -> Outcome:
    return Outcome(False, reason, detail)


def _points(evidence: Evidence) -> List[Dict[str, Any]]:
    rows = evidence.telemetry.get("eval_points")
    return list(rows) if isinstance(rows, list) else []


def _role(evidence: Evidence, role: str) -> List[Dict[str, Any]]:
    return [row for row in _points(evidence) if row.get("role") == role]


def _graded(evidence: Evidence) -> Optional[Dict[str, Any]]:
    row = evidence.telemetry.get("graded")
    return row if isinstance(row, dict) else None


def _measured(evidence: Evidence) -> Dict[str, Any]:
    block = evidence.manifest.get("measured_operating_points")
    return block if isinstance(block, dict) else {}


def acceptance_bar(evidence: Evidence) -> Optional[float]:
    """The bar a submission must reach, derived from frozen bytes, never authored here."""
    measured = _measured(evidence)
    optimum = measured.get("default_construction_optimum_bpb")
    separation = measured.get("default_separation_bpb")
    if optimum is None or separation is None:
        return None
    return float(optimum) - float(separation)


# ---------------------------------------------------------------------------
# 1. ABSENCE. An early stop establishes nothing, and says so rather than going quiet.
# ---------------------------------------------------------------------------
def check_early_stop_not_an_established_metric(evidence: Evidence) -> Outcome:
    compute = evidence.telemetry.get("compute") or {}
    graded = _graded(evidence)
    bound = _role(evidence, BOUND_ROLE)
    points = _points(evidence)
    if compute.get("halted_before_bound_point"):
        return _no(
            "early-stop-no-established-metric",
            "the run halted before the bound evaluation point after "
            + str(compute.get("consumed_updates"))
            + " updates, so no metric was established",
        )
    if graded is None or len(bound) != 1:
        return _no(
            "early-stop-no-established-metric",
            "no bound evaluation point was reached, so the record carries no established reading",
        )
    if not points:
        return _no(
            "early-stop-no-established-metric",
            "the record carries no evaluation point at all",
        )
    last = max(row.get("updates", 0) for row in points)
    consumed = compute.get("consumed_updates")
    if not isinstance(consumed, int) or consumed < last:
        return _no(
            "early-stop-no-established-metric",
            "the schedule reaches " + str(last) + " updates and the counter recorded "
            + str(consumed),
        )
    return _ok(
        "no halt record, one bound point, and the counter reached the full schedule at "
        + str(consumed)
        + " updates"
    )


# ---------------------------------------------------------------------------
# 2. INVARIANT. Every frozen axis is where the delivery unit left it.
# ---------------------------------------------------------------------------
def check_frozen_axes_unmoved(evidence: Evidence) -> Outcome:
    frozen = evidence.telemetry.get("frozen") or {}
    expected = {
        "compute_budget_updates": evidence.manifest.get("compute_budget_updates"),
        "vocab_budget": evidence.manifest.get("vocab_budget"),
        "model_spec_digest": evidence.manifest.get("model_spec_digest"),
        "eval_corpus_digest": evidence.manifest.get("eval_corpus_digest"),
        "eval_corpus_bytes": evidence.manifest.get("eval_corpus_bytes"),
        "train_corpus_digest": evidence.manifest.get("train_corpus_digest"),
    }
    moved = [
        key
        for key, value in sorted(expected.items())
        if frozen.get(key) != value
    ]
    if moved:
        return _no(
            "frozen-axis-moved",
            "these frozen axes do not match environment/manifest.json: " + ", ".join(moved),
        )
    return _ok("all six frozen axis carriers match the manifest of the delivery unit")


# ---------------------------------------------------------------------------
# 3. VALUE. The MERGE CAPACITY of the frozen training corpus.
#
#    The corpus supports a fixed number of greedy most-frequent-pair merges before
#    no adjacent pair occurs twice. That count is a property of the corpus bytes
#    and the frozen token-length cap, it is published nowhere on the agent surface,
#    and the only way to obtain it is to run the construction over the live corpus.
#    tests/runner.py recomputes it in the verifier's own process; the expected count
#    is the authoring lane's measurement, carried verifier-side in tests/anchors.json
#    and overlaid by tests/grade.py, so a tampered environment/manifest.json cannot
#    move it. The comparison is exact: an off-by-one capacity is a different corpus.
# ---------------------------------------------------------------------------
def check_train_corpus_merge_capacity(evidence: Evidence) -> Outcome:
    expected = _measured(evidence).get("reference_merges")
    observed = evidence.train_merge_capacity
    if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
        return _no(
            "train-corpus-merge-capacity-moved",
            "the verifier holds no measured merge capacity to compare against, it reads "
            + repr(expected),
        )
    if not isinstance(observed, int) or isinstance(observed, bool) or observed < 0:
        return _no(
            "train-corpus-merge-capacity-moved",
            "the run record carries no recomputed merge capacity, it reads " + repr(observed),
        )
    frozen_digest = evidence.manifest.get("train_corpus_digest")
    run_digest = (evidence.telemetry.get("frozen") or {}).get("train_corpus_digest")
    if evidence.train_merge_capacity_digest != frozen_digest:
        return _no(
            "train-corpus-merge-capacity-moved",
            "the capacity was recomputed over a corpus digesting "
            + str(evidence.train_merge_capacity_digest)[:16]
            + ", not over the frozen " + str(frozen_digest)[:16],
        )
    if run_digest is not None and run_digest != frozen_digest:
        return _no(
            "train-corpus-merge-capacity-moved",
            "the graded run trained on a corpus digesting " + str(run_digest)[:16]
            + ", not on the frozen " + str(frozen_digest)[:16],
        )
    room = evidence.manifest.get("vocab_budget")
    if isinstance(room, int) and observed > max(0, room - 256):
        return _no(
            "train-corpus-merge-capacity-moved",
            "the capacity " + str(observed) + " exceeds the "
            + str(max(0, room - 256)) + " entries the frozen vocabulary budget leaves",
        )
    if observed != expected:
        return _no(
            "train-corpus-merge-capacity-moved",
            "the live training corpus supports " + str(observed)
            + " greedy pair merges against the measured " + str(expected)
            + ", so the corpus the graded run read is not the one that was measured",
        )
    return _ok(
        "the live training corpus supports exactly "
        + str(observed)
        + " greedy pair merges, the measured capacity, recomputed over the frozen corpus "
        + str(frozen_digest)[:16]
    )


# ---------------------------------------------------------------------------
# 4. ORDERING. The schedule runs forward and the bound point is the last charged one.
# ---------------------------------------------------------------------------
def check_eval_points_ordered_by_updates(evidence: Evidence) -> Outcome:
    points = _points(evidence)
    if len(points) < 2:
        return _no("eval-schedule-out-of-order", "fewer than two evaluation points were recorded")
    updates = [row.get("updates") for row in points]
    for index in range(1, len(updates)):
        left, right = updates[index - 1], updates[index]
        if not isinstance(left, int) or not isinstance(right, int) or right <= left:
            return _no(
                "eval-schedule-out-of-order",
                "point " + str(index) + " at " + str(right) + " updates does not follow "
                + str(left),
            )
    roles = [row.get("role") for row in points]
    if roles.count(BOUND_ROLE) != 1:
        return _no(
            "eval-schedule-out-of-order",
            "the schedule declares " + str(roles.count(BOUND_ROLE)) + " bound points, expected one",
        )
    cut = roles.index(BOUND_ROLE)
    if any(role != PROGRESS_ROLE for role in roles[:cut]):
        return _no(
            "eval-schedule-out-of-order",
            "a point before the bound point is not a progress point",
        )
    if any(role != SUSTAIN_ROLE for role in roles[cut + 1:]):
        return _no(
            "eval-schedule-out-of-order",
            "a point after the bound point is not a sustain point",
        )
    return _ok(
        "the "
        + str(len(points))
        + " evaluation points ascend strictly and the single bound point sits at index "
        + str(cut)
    )


# ---------------------------------------------------------------------------
# 5. VALUE. The compute budget as ACTUALLY SPENT, from the harness's own counter.
# ---------------------------------------------------------------------------
def check_compute_budget_respected_as_spent(evidence: Evidence) -> Outcome:
    compute = evidence.telemetry.get("compute") or {}
    budget = evidence.manifest.get("compute_budget_updates")
    if compute.get("budget_updates") != budget:
        return _no(
            "compute-budget-overspent",
            "the run recorded a budget of " + str(compute.get("budget_updates"))
            + " against a frozen budget of " + str(budget),
        )
    charged = compute.get("charged_updates")
    if not isinstance(charged, int) or charged > int(budget):
        return _no(
            "compute-budget-overspent",
            "the harness counter recorded " + str(charged)
            + " charged updates against a budget of " + str(budget),
        )
    graded = _graded(evidence)
    if graded is None or graded.get("updates") != budget:
        return _no(
            "compute-budget-overspent",
            "the bound evaluation point sits at "
            + str(None if graded is None else graded.get("updates"))
            + " updates, not at the budget " + str(budget),
        )
    for row in _points(evidence):
        if row.get("charged_to_budget") and int(row.get("updates", 0)) > int(budget):
            return _no(
                "compute-budget-overspent",
                "a charged evaluation point sits at " + str(row.get("updates"))
                + " updates, past the budget",
            )
    return _ok(
        "the harness counter spent " + str(charged) + " of " + str(budget) + " charged updates"
    )


# ---------------------------------------------------------------------------
# 6. VALUE. The DENOMINATOR is the frozen byte count of the evaluation corpus.
#    This is the headline readout attack for a tokenizer slot: a vocabulary that
#    shrinks the token count must not lower bits per byte, and it cannot unless
#    the denominator moves with it.
# ---------------------------------------------------------------------------
def check_denominator_is_frozen_eval_bytes(evidence: Evidence) -> Outcome:
    denominator = evidence.telemetry.get("denominator") or {}
    frozen_bytes = evidence.manifest.get("eval_corpus_bytes")
    frozen_digest = evidence.manifest.get("eval_corpus_digest")
    if evidence.eval_corpus_bytes != frozen_bytes or evidence.eval_corpus_digest != frozen_digest:
        return _no(
            "denominator-not-frozen-eval-bytes",
            "the evaluation corpus on disk is "
            + str(evidence.eval_corpus_bytes)
            + " bytes with digest "
            + str(evidence.eval_corpus_digest)[:16]
            + ", against a frozen "
            + str(frozen_bytes)
            + " bytes",
        )
    if denominator.get("source") != "frozen-eval-corpus-bytes":
        return _no(
            "denominator-not-frozen-eval-bytes",
            "the denominator names its source as " + repr(denominator.get("source")),
        )
    if denominator.get("bytes") != frozen_bytes or denominator.get("digest") != frozen_digest:
        return _no(
            "denominator-not-frozen-eval-bytes",
            "the denominator carries " + str(denominator.get("bytes"))
            + " bytes against the frozen " + str(frozen_bytes),
        )
    vocabulary = evidence.telemetry.get("vocabulary") or {}
    tokens = vocabulary.get("eval_tokens")
    if tokens is not None and denominator.get("bytes") == tokens and tokens != frozen_bytes:
        return _no(
            "denominator-not-frozen-eval-bytes",
            "the denominator equals the evaluation token count, which is a token normaliser",
        )
    for row in _points(evidence):
        if row.get("denominator_bytes") != frozen_bytes:
            return _no(
                "denominator-not-frozen-eval-bytes",
                "the point at " + str(row.get("updates")) + " updates divides by "
                + str(row.get("denominator_bytes")),
            )
    graded = _graded(evidence)
    if graded is not None and graded.get("denominator_bytes") != frozen_bytes:
        return _no(
            "denominator-not-frozen-eval-bytes",
            "the graded point divides by " + str(graded.get("denominator_bytes")),
        )
    return _ok(
        "every reading divides by the frozen "
        + str(frozen_bytes)
        + " evaluation bytes while the vocabulary expressed them in "
        + str(tokens)
        + " tokens"
    )


# ---------------------------------------------------------------------------
# 7. VALUE. The evaluated weights are the ones the harness held at the bound point.
# ---------------------------------------------------------------------------
def check_graded_state_is_harness_owned(evidence: Evidence) -> Outcome:
    readout = evidence.telemetry.get("readout") or {}
    graded = _graded(evidence)
    if readout.get("weights_origin") != "harness-owned-state-at-bound-point":
        return _no(
            "graded-state-not-harness-owned",
            "the record names the weights origin as " + repr(readout.get("weights_origin")),
        )
    if readout.get("graded_role") != BOUND_ROLE:
        return _no(
            "graded-state-not-harness-owned",
            "the record grades the point with role " + repr(readout.get("graded_role")),
        )
    bound = _role(evidence, BOUND_ROLE)
    if graded is None or len(bound) != 1:
        return _no(
            "graded-state-not-harness-owned",
            "there is no single bound evaluation point to take a state digest from",
        )
    if graded.get("state_digest") != bound[0].get("state_digest"):
        return _no(
            "graded-state-not-harness-owned",
            "the graded state digest is not the digest recorded at the bound point",
        )
    earlier = [
        row.get("state_digest") for row in _points(evidence) if row.get("role") == PROGRESS_ROLE
    ]
    if graded.get("state_digest") in earlier:
        return _no(
            "graded-state-not-harness-owned",
            "the graded state digest repeats a progress point, so the state did not advance",
        )
    return _ok(
        "the graded state digest "
        + str(graded.get("state_digest"))[:16]
        + " is the harness state at the bound point and differs from every progress point"
    )


# ---------------------------------------------------------------------------
# 8. VALUE. No smoothing on the graded path.
# ---------------------------------------------------------------------------
def check_graded_readout_unsmoothed(evidence: Evidence) -> Outcome:
    readout = evidence.telemetry.get("readout") or {}
    graded = _graded(evidence)
    if readout.get("filter") != "none":
        return _no(
            "graded-readout-smoothed",
            "the graded path declares readout filter " + repr(readout.get("filter")),
        )
    bound = _role(evidence, BOUND_ROLE)
    if graded is None or len(bound) != 1:
        return _no(
            "graded-readout-smoothed",
            "no single bound point exists, so the graded reading has no raw counterpart",
        )
    raw = bound[0].get("bits")
    taken = graded.get("bits")
    if not isinstance(raw, float) or not isinstance(taken, float):
        return _no("graded-readout-smoothed", "the raw bits at the bound point are not a float")
    if not math.isfinite(raw) or not math.isfinite(taken):
        return _no("graded-readout-smoothed", "a non-finite value reached the graded path")
    if abs(taken - raw) > EXACT:
        return _no(
            "graded-readout-smoothed",
            "the graded bits " + repr(taken) + " differ from the raw bits " + repr(raw)
            + " recorded at the bound point, so a filter was applied",
        )
    return _ok("the graded bits are the raw bits at the bound point, unfiltered")


# ---------------------------------------------------------------------------
# 9. INVARIANT. The reading holds across the verifier-scheduled points after it.
# ---------------------------------------------------------------------------
def check_reading_sustained_across_scheduled_points(evidence: Evidence) -> Outcome:
    schedule = evidence.manifest.get("evaluation_schedule") or {}
    tolerance = schedule.get("sustain_tolerance_bpb")
    expected = schedule.get("sustain_offsets") or []
    graded = _graded(evidence)
    sustain = _role(evidence, SUSTAIN_ROLE)
    if graded is None:
        return _no("reading-not-sustained", "no bound reading exists to sustain")
    if tolerance is None:
        return _no("reading-not-sustained", "the manifest binds no sustain tolerance")
    if len(sustain) != len(expected) or not sustain:
        return _no(
            "reading-not-sustained",
            "the record carries " + str(len(sustain)) + " sustain points against "
            + str(len(expected)) + " the verifier schedules",
        )
    ceiling = float(graded.get("bits_per_byte")) + float(tolerance)
    bar = acceptance_bar(evidence)
    for row in sustain:
        value = row.get("bits_per_byte")
        if not isinstance(value, float) or not math.isfinite(value):
            return _no(
                "reading-not-sustained",
                "the sustain point at " + str(row.get("updates")) + " updates carries no reading",
            )
        if row.get("charged_to_budget"):
            return _no(
                "reading-not-sustained",
                "a sustain point was charged to the agent's compute budget",
            )
        if value > ceiling:
            return _no(
                "reading-not-sustained",
                "the reading at " + str(row.get("updates")) + " updates is " + repr(value)
                + ", above the graded reading plus the bound tolerance " + repr(ceiling),
            )
        if bar is not None and value > bar:
            return _no(
                "reading-not-sustained",
                "the reading at " + str(row.get("updates")) + " updates is " + repr(value)
                + ", above the acceptance bar " + repr(bar),
            )
    return _ok(
        "the reading held at all "
        + str(len(sustain))
        + " verifier-scheduled points after the bound point"
    )


# ---------------------------------------------------------------------------
# 10. EFFECT. The graded run used the vocabulary the submission handed back.
# ---------------------------------------------------------------------------
def check_submission_vocabulary_took_effect(evidence: Evidence) -> Outcome:
    vocabulary = evidence.telemetry.get("vocabulary") or {}
    if vocabulary.get("submitted_digest") != evidence.submitted_vocab_digest:
        return _no(
            "submission-vocabulary-inert",
            "the graded run digested "
            + str(vocabulary.get("submitted_digest"))[:16]
            + " while the submission returned "
            + str(evidence.submitted_vocab_digest)[:16],
        )
    size = vocabulary.get("size")
    if not isinstance(size, int) or size <= 256:
        return _no(
            "submission-vocabulary-inert",
            "the graded vocabulary holds " + str(size)
            + " entries, so the submission contributed none beyond the single bytes",
        )
    tokens = vocabulary.get("eval_tokens")
    if not isinstance(tokens, int) or tokens >= evidence.eval_corpus_bytes:
        return _no(
            "submission-vocabulary-inert",
            "the evaluation corpus still encodes to " + str(tokens)
            + " tokens over " + str(evidence.eval_corpus_bytes) + " bytes",
        )
    digest = hashlib.sha256(
        json.dumps([vocabulary.get("submitted_digest")], separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return _ok(
        "the submitted vocabulary of "
        + str(size)
        + " entries reached the graded run, witness "
        + digest[:16]
    )


# ---------------------------------------------------------------------------
# 11. DIVERGENCE. The graded number diverges in provenance from anything the
#     submission reported: it is recomputed from verifier-owned raw quantities.
# ---------------------------------------------------------------------------
def check_metric_not_taken_from_submission_report(evidence: Evidence) -> Outcome:
    graded = _graded(evidence)
    if graded is None:
        return _no("metric-sourced-from-submission", "no verifier-computed graded reading exists")
    bits = graded.get("bits")
    denominator = graded.get("denominator_bytes")
    taken = graded.get("bits_per_byte")
    if not isinstance(bits, float) or not isinstance(denominator, int) or denominator <= 0:
        return _no(
            "metric-sourced-from-submission",
            "the graded reading cannot be recomputed from raw bits and a byte count",
        )
    recomputed = bits / denominator
    if not isinstance(taken, float) or abs(taken - recomputed) > EXACT:
        return _no(
            "metric-sourced-from-submission",
            "the graded value " + repr(taken) + " is not the verifier's own "
            + repr(recomputed) + ", so it came from somewhere else",
        )
    reported = evidence.telemetry.get("reported_by_submission")
    reported = reported if isinstance(reported, dict) else {}
    claimed = reported.get("bits_per_byte")
    if isinstance(claimed, (int, float)) and abs(float(claimed) - recomputed) > EXACT:
        return _ok(
            "the submission claimed "
            + repr(float(claimed))
            + " and the graded value is the verifier's own "
            + repr(recomputed)
            + "; the two diverge and the verifier's stands"
        )
    return _ok(
        "the graded value is recomputed as raw bits over the frozen byte count, "
        + repr(recomputed)
        + ", with no submission-reported value on the path"
    )


# ---------------------------------------------------------------------------
# 12. VALUE. The submission beat the DEFAULT CONSTRUCTION's measured optimum.
#     Accepting the handed option set scores zero here, by construction.
# ---------------------------------------------------------------------------
def check_beats_default_construction_optimum(evidence: Evidence) -> Outcome:
    graded = _graded(evidence)
    bar = acceptance_bar(evidence)
    measured = _measured(evidence)
    if graded is None or bar is None:
        return _no(
            "default-construction-plateau-not-beaten",
            "no graded reading or no measured default optimum to compare against",
        )
    value = graded.get("bits_per_byte")
    if not isinstance(value, float) or not math.isfinite(value):
        return _no(
            "default-construction-plateau-not-beaten",
            "the graded reading is not a finite number",
        )
    if value > bar:
        return _no(
            "default-construction-plateau-not-beaten",
            "the graded reading " + repr(value) + " does not reach the bar " + repr(bar)
            + ", which is the measured optimum of the handed construction "
            + repr(measured.get("default_construction_optimum_bpb"))
            + " less the bound separation "
            + repr(measured.get("default_separation_bpb")),
        )
    return _ok(
        "the graded reading "
        + repr(value)
        + " is at or below the bar "
        + repr(bar)
        + ", so the handed construction's optimum was beaten"
    )


# The gate order tests/checkers.yaml declares, restated for a reader. tests/grade.py
# reads the manifest and never this list. The sustain check sits last because it
# re-tests the acceptance bar, and while it stood earlier it emitted its own reason
# for submissions that submission_vocabulary_took_effect and
# beats_default_construction_optimum exist to name.
SELECTOR_ORDER = [
    "check_early_stop_not_an_established_metric",
    "check_frozen_axes_unmoved",
    "check_train_corpus_merge_capacity",
    "check_eval_points_ordered_by_updates",
    "check_compute_budget_respected_as_spent",
    "check_denominator_is_frozen_eval_bytes",
    "check_graded_state_is_harness_owned",
    "check_graded_readout_unsmoothed",
    "check_submission_vocabulary_took_effect",
    "check_metric_not_taken_from_submission_report",
    "check_beats_default_construction_optimum",
    "check_reading_sustained_across_scheduled_points",
]


def evidence_from_record(record: Dict[str, Any], delivery_root: str = "") -> Evidence:
    """Assemble evidence from the run record. Pure: it reads the mapping it is given."""
    _ = pathlib.PurePosixPath(delivery_root or ".")
    return Evidence(
        telemetry=record.get("telemetry") or {},
        manifest=record.get("manifest") or {},
        submitted_vocab_digest=str(record.get("submitted_vocab_digest") or ""),
        eval_corpus_digest=str(record.get("eval_corpus_digest") or ""),
        eval_corpus_bytes=int(record.get("eval_corpus_bytes") or 0),
        delivery_root=str(delivery_root or ""),
        train_merge_capacity=(
            record["train_merge_capacity"]
            if isinstance(record.get("train_merge_capacity"), int)
            and not isinstance(record.get("train_merge_capacity"), bool)
            else -1
        ),
        train_merge_capacity_digest=str(record.get("train_merge_capacity_digest") or ""),
    )


def score_from_reading(evidence: Evidence) -> float:
    """The bound reward formula, in its lower-is-better form, over measured carriers.

    baseline_metric and target_metric are ABSENT for family F10, so the two carriers
    are the substrate-measured operating points the manifest records and names as
    such. The shape of the formula is the bound one and is not re-derived here.
    """
    measured = _measured(evidence)
    graded = _graded(evidence)
    floor = measured.get("default_construction_optimum_bpb")
    ceiling = measured.get("reference_construction_optimum_bpb")
    if graded is None or floor is None or ceiling is None:
        return 0.0
    value = graded.get("bits_per_byte")
    if not isinstance(value, float) or not math.isfinite(value):
        return 0.0
    span = float(floor) - float(ceiling)
    if span <= 0.0:
        return 0.0
    raw = (float(floor) - value) / span
    return min(max(raw, 0.0), 1.0)
