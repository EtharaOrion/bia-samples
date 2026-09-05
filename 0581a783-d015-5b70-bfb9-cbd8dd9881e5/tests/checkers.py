"""The graded checkers for OER-18. Pure, deterministic, and blind to the submission.

Every function here is a pure function of one `Handle`, and a `Handle` carries only
values the verifier's own process produced: the coverage vector tests/strata.py measured
over the emitted samples, the scores tests/frozen_stack.py computed over harness-owned
model state at the verifier's own scheduled evaluation marks, and the feeder's token
counters. Nothing a submission printed reaches a checker, and nothing a submission wrote
is read as a number.

Import allowlist, enforced by tests/test_output.py over this file's AST: json, math,
hashlib, pathlib, dataclasses, typing. There is no clock read, no random source, no
network call, no environment-secret read, no planted-file read, and no import of the
submission anywhere in this module.

The declared coverage manifest appears in exactly one place below, as the left-hand side
of a divergence comparison. It is never a source of reward. A truthful manifest earns no
bonus; it clears a gate. A false one scores zero and the reason carries both figures, so
an auditor reads the size of the discrepancy rather than the fact of it.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Optional

# The closed stratum set, restated here so a checker never depends on a module the
# submission could shadow. It must agree with tests/strata.py, and tests/test_output.py
# asserts that agreement.
STRATA = ("calc", "cmp", "fact", "neg", "seq")
UNCLASSIFIED = "unclassified"

# The ordered event log the harness writes around the graded run. ORDERING is graded
# against this exact sequence.
BOUND_EVENT_ORDER = (
    "submission-process-group-terminated",
    "corpus-frozen",
    "coverage-measured",
    "training-opened",
    "evaluation-closed",
)


@dataclass
class Verdict:
    """One checker's outcome. A zero always carries a machine-readable reason."""

    ident: str
    passed: bool
    zero_reason: str = ""
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "id": self.ident,
            "passed": bool(self.passed),
            "zero_reason": "" if self.passed else self.zero_reason,
            "detail": self.detail,
        }


@dataclass
class Handle:
    """The live harness handle a checker reads. Built by tests/grade.py, never by a submission.

    Every field is produced inside the verifier process: `measured_coverage` by
    tests/strata.py over the emitted samples, `eval_points` by tests/frozen_stack.py over
    weights the harness trained itself, `tokens_fed` and `tokens_offered` by the frozen
    feeder's counters, and the digests by hashlib over bytes the harness read. The two
    `reported_*` fields are the only values that originate with the submission, and they
    are carried so a reconciliation can be computed against them; neither is ever
    returned as a score.
    """

    # Declared side. Claim only.
    declared_present: bool = False
    declared_coverage: dict = field(default_factory=dict)
    reported_score: Optional[float] = None
    reported_smoothing_window: Optional[int] = None

    # Measured side, all harness-produced.
    measured_coverage: dict = field(default_factory=dict)
    measured_counts: dict = field(default_factory=dict)
    eval_points: list = field(default_factory=list)
    scheduled_points: int = 0
    graded_eval_point: int = 0
    graded_score: Optional[float] = None
    graded_weights_owner: str = ""
    graded_weights_digest: str = ""
    harness_recomputed_weights_digest: str = ""
    null_control_score: float = 0.0
    tokens_fed: int = 0
    tokens_offered: int = 0
    near_duplicate_hits: list = field(default_factory=list)
    corpus_digest_at_freeze: str = ""
    corpus_digest_at_eval: str = ""
    event_log: list = field(default_factory=list)

    # Bound parameters, read from tests/checkers.yaml by grade.py. Never authored here.
    divergence_tolerance: float = 0.0
    sustain_tolerance: float = 0.0
    near_duplicate_threshold: float = 0.0
    budget_tokens: int = 0
    reconciliation_tolerance: float = 0.0

    def point(self, index: int) -> dict:
        for row in self.eval_points:
            if int(row.get("index", -1)) == int(index):
                return row
        return {}

    def digest(self) -> str:
        """A stable digest over the handle, used by the compiled fixture tests."""
        payload = json.dumps(
            {
                "measured_coverage": self.measured_coverage,
                "eval_points": self.eval_points,
                "tokens_fed": self.tokens_fed,
                "tokens_offered": self.tokens_offered,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _vector(raw: Any) -> dict:
    """Coerce a coverage mapping to the closed key set with float values."""
    source = raw if isinstance(raw, dict) else {}
    out = {}
    for key in (*STRATA, UNCLASSIFIED):
        try:
            out[key] = float(source.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            out[key] = 0.0
    return out


def l1_divergence(declared: Any, measured: Any) -> float:
    """Total absolute difference over the closed stratum set, omissions read as zero."""
    left, right = _vector(declared), _vector(measured)
    return math.fsum(abs(left[key] - right[key]) for key in sorted(left))


def _round(value: float) -> float:
    """Six-place rounding so a reason block is byte-stable across platforms."""
    return float(format(value, ".6f"))


# ---------------------------------------------------------------------------
# The ten graded checkers. Exactly one reduction each; see tests/checkers.yaml.
# ---------------------------------------------------------------------------


def check_coverage_divergence(handle: Handle) -> Verdict:
    """DIVERGENCE. The declared coverage manifest against the measured coverage vector.

    When nothing was declared there is no claim to contradict, so this checker holds and
    `check_coverage_manifest_present` carries the absence. That split is deliberate: a
    silent submission and a lying one must both reach zero, by their own reasons.
    """
    ident = "coverage_divergence"
    if not handle.declared_present:
        return Verdict(ident, True, detail={"declared": None, "reason": "no-claim-to-contradict"})
    divergence = l1_divergence(handle.declared_coverage, handle.measured_coverage)
    detail = {
        "declared_coverage": {k: _round(v) for k, v in _vector(handle.declared_coverage).items()},
        "measured_coverage": {k: _round(v) for k, v in _vector(handle.measured_coverage).items()},
        "measured_counts": dict(handle.measured_counts),
        "l1_divergence": _round(divergence),
        "tolerance": _round(handle.divergence_tolerance),
        "over_by": _round(max(0.0, divergence - handle.divergence_tolerance)),
    }
    if divergence > handle.divergence_tolerance:
        return Verdict(ident, False, "coverage-declaration-contradicts-measurement", detail)
    return Verdict(ident, True, detail=detail)


def check_coverage_manifest_present(handle: Handle) -> Verdict:
    """ABSENCE. No graded run reaches a pass on the strength of having declared nothing.

    The absence is recorded together with the coverage the verifier measured and the
    benchmark score the verifier computed, so the run is graded on the measurement and
    the missing manifest is a recorded fact rather than a waived one.
    """
    ident = "coverage_manifest_present"
    detail = {
        "manifest_declared": bool(handle.declared_present),
        "measured_coverage": {k: _round(v) for k, v in _vector(handle.measured_coverage).items()},
        "measured_counts": dict(handle.measured_counts),
        "measured_benchmark_score": None
        if handle.graded_score is None
        else _round(handle.graded_score),
        "absence_recorded": not handle.declared_present,
        "graded_on": "measurement",
    }
    if not handle.declared_present:
        return Verdict(ident, False, "coverage-manifest-absent", detail)
    return Verdict(ident, True, detail=detail)


def check_graded_score_unsmoothed(handle: Handle) -> Verdict:
    """VALUE. The graded readout is the raw harness evaluation, never a filtered one."""
    ident = "graded_score_unsmoothed"
    window = handle.reported_smoothing_window
    gap = None
    if handle.reported_score is not None and handle.graded_score is not None:
        gap = abs(float(handle.reported_score) - float(handle.graded_score))
    detail = {
        "harness_score": None if handle.graded_score is None else _round(handle.graded_score),
        "reported_score": None
        if handle.reported_score is None
        else _round(float(handle.reported_score)),
        "reported_smoothing_window": window,
        "reconciliation_gap": None if gap is None else _round(gap),
        "reconciliation_tolerance": _round(handle.reconciliation_tolerance),
    }
    if window is not None and int(window) > 0:
        return Verdict(ident, False, "readout-smoothed", detail)
    if gap is not None and gap > handle.reconciliation_tolerance:
        return Verdict(ident, False, "readout-smoothed", detail)
    return Verdict(ident, True, detail=detail)


def check_graded_score_from_harness_model(handle: Handle) -> Verdict:
    """VALUE. The weights evaluated are the weights the harness trained at the bound point."""
    ident = "graded_score_from_harness_model"
    detail = {
        "graded_eval_point": handle.graded_eval_point,
        "weights_owner": handle.graded_weights_owner,
        "weights_digest": handle.graded_weights_digest,
        "harness_recomputed_weights_digest": handle.harness_recomputed_weights_digest,
    }
    if handle.graded_weights_owner != "harness":
        return Verdict(ident, False, "model-state-not-harness-owned", detail)
    if not handle.graded_weights_digest:
        return Verdict(ident, False, "model-state-not-harness-owned", detail)
    if handle.graded_weights_digest != handle.harness_recomputed_weights_digest:
        return Verdict(ident, False, "model-state-not-harness-owned", detail)
    return Verdict(ident, True, detail=detail)


def check_evaluation_schedule_complete(handle: Handle) -> Verdict:
    """ABSENCE. No scheduled evaluation mark is missing from the run.

    A run whose feed is exhausted before the last scheduled mark stopped at whichever
    reading it had reached. That is graded as not having established the score, with a
    reason, rather than as an absent result.
    """
    ident = "evaluation_schedule_complete"
    reached = [int(row.get("index", -1)) for row in handle.eval_points if row.get("reached")]
    missing = [index for index in range(handle.scheduled_points) if index not in reached]
    detail = {
        "scheduled_points": handle.scheduled_points,
        "reached_points": sorted(reached),
        "missing_points": missing,
        "tokens_fed": handle.tokens_fed,
        "budget_tokens": handle.budget_tokens,
    }
    if missing:
        return Verdict(ident, False, "early-stop-score-not-established", detail)
    return Verdict(ident, True, detail=detail)


def check_score_sustained_across_points(handle: Handle) -> Verdict:
    """INVARIANT. The graded reading holds at every later mark the verifier scheduled."""
    ident = "score_sustained_across_points"
    graded = handle.graded_score
    later = [
        row
        for row in handle.eval_points
        if int(row.get("index", -1)) > int(handle.graded_eval_point) and row.get("reached")
    ]
    floor = None if graded is None else float(graded) - float(handle.sustain_tolerance)
    detail = {
        "graded_eval_point": handle.graded_eval_point,
        "graded_score": None if graded is None else _round(graded),
        "sustain_tolerance": _round(handle.sustain_tolerance),
        "sustain_floor": None if floor is None else _round(floor),
        "later_points": [
            {"index": int(row.get("index", -1)), "score": _round(float(row.get("score") or 0.0))}
            for row in later
        ],
    }
    if graded is None or not later:
        return Verdict(ident, False, "score-not-sustained", detail)
    for row in later:
        if float(row.get("score") or 0.0) < floor:
            return Verdict(ident, False, "score-not-sustained", detail)
    return Verdict(ident, True, detail=detail)


def check_benchmark_contamination_absent(handle: Handle) -> Verdict:
    """ABSENCE. No emitted sample is a near-duplicate of a held-out benchmark item."""
    ident = "benchmark_contamination_absent"
    detail = {
        "normalization": "casefold; non-alphanumeric to space; whitespace split; token set",
        "threshold": _round(handle.near_duplicate_threshold),
        "hits": handle.near_duplicate_hits[:8],
        "hit_count": len(handle.near_duplicate_hits),
    }
    if handle.near_duplicate_hits:
        return Verdict(ident, False, "benchmark-near-duplicate-emitted", detail)
    return Verdict(ident, True, detail=detail)


def check_training_budget_respected(handle: Handle) -> Verdict:
    """VALUE. The corpus offered the frozen budget and no more, as the feeder counted it."""
    ident = "training_budget_respected"
    detail = {
        "budget_tokens": handle.budget_tokens,
        "tokens_offered": handle.tokens_offered,
        "tokens_fed": handle.tokens_fed,
        "overspend": max(0, int(handle.tokens_offered) - int(handle.budget_tokens)),
        "counter_source": "frozen_stack.feed",
    }
    if int(handle.tokens_offered) > int(handle.budget_tokens):
        return Verdict(ident, False, "training-budget-overspent", detail)
    return Verdict(ident, True, detail=detail)


def check_corpus_frozen_before_training(handle: Handle) -> Verdict:
    """ORDERING. The corpus is frozen and measured before the training counter opens."""
    ident = "corpus_frozen_before_training"
    detail = {
        "expected_order": list(BOUND_EVENT_ORDER),
        "observed_order": list(handle.event_log),
        "digest_at_freeze": handle.corpus_digest_at_freeze,
        "digest_at_eval": handle.corpus_digest_at_eval,
    }
    if tuple(handle.event_log) != BOUND_EVENT_ORDER:
        return Verdict(ident, False, "corpus-mutated-after-freeze", detail)
    if not handle.corpus_digest_at_freeze:
        return Verdict(ident, False, "corpus-mutated-after-freeze", detail)
    if handle.corpus_digest_at_freeze != handle.corpus_digest_at_eval:
        return Verdict(ident, False, "corpus-mutated-after-freeze", detail)
    return Verdict(ident, True, detail=detail)


def check_corpus_has_training_effect(handle: Handle) -> Verdict:
    """EFFECT. The corpus moved the trained model past the null-corpus control."""
    ident = "corpus_has_training_effect"
    detail = {
        "graded_score": None if handle.graded_score is None else _round(handle.graded_score),
        "null_control_score": _round(handle.null_control_score),
    }
    if handle.graded_score is None:
        return Verdict(ident, False, "corpus-had-no-training-effect", detail)
    if float(handle.graded_score) <= float(handle.null_control_score):
        return Verdict(ident, False, "corpus-had-no-training-effect", detail)
    return Verdict(ident, True, detail=detail)


# The graded order. grade.py walks this sequence, so the first failing checker names the
# reason a zero carries, and the order is a bound fact rather than a dict iteration.
GRADED_ORDER = (
    "coverage_manifest_present",
    "coverage_divergence",
    "benchmark_contamination_absent",
    "training_budget_respected",
    "corpus_frozen_before_training",
    "evaluation_schedule_complete",
    "graded_score_from_harness_model",
    "graded_score_unsmoothed",
    "score_sustained_across_points",
    "corpus_has_training_effect",
)

SELECTORS = {
    "coverage_divergence": check_coverage_divergence,
    "coverage_manifest_present": check_coverage_manifest_present,
    "graded_score_unsmoothed": check_graded_score_unsmoothed,
    "graded_score_from_harness_model": check_graded_score_from_harness_model,
    "evaluation_schedule_complete": check_evaluation_schedule_complete,
    "score_sustained_across_points": check_score_sustained_across_points,
    "benchmark_contamination_absent": check_benchmark_contamination_absent,
    "training_budget_respected": check_training_budget_respected,
    "corpus_frozen_before_training": check_corpus_frozen_before_training,
    "corpus_has_training_effect": check_corpus_has_training_effect,
}


def run_all(handle: Handle) -> list:
    """Every graded checker, in the bound order, as a list of Verdicts."""
    return [SELECTORS[name](handle) for name in GRADED_ORDER]
