"""Pure, deterministic checkers for slot OER-03.

Every function here is a pure function of one telemetry mapping that the
VERIFIER'S OWN PROCESS produced. Nothing in this module reads a file, reads an
environment variable, reads a clock, opens a socket, consults a random source,
or imports the submission. The import set is exactly the allowlist:

    json, math, hashlib, pathlib, dataclasses, typing

The telemetry mapping is assembled by tests/grade.py out of live harness
handles: the verifier-owned run directory tests/runner.py created, the
harness-owned weight ledger the pinned harness wrote inside it, the verifier's
own evaluation of those weights on the frozen held-out split, and the process
group exit record the runner captured. No number in it was printed by the
submission, with one deliberate exception: `submission_report`, which is carried
ONLY so a checker can measure its divergence from the recomputed answer and is
never the graded quantity.

The centre of this module is one sentence. The graded step is the first
verifier-scheduled evaluation point whose unsmoothed verifier-computed loss
falls below the bound target and stays below it for the bound number of
subsequent verifier-scheduled points. It is not the schedule length, not a
smoothed readout, not a submission field, and a run that never reaches it is
graded a failure with a reason rather than reported as an absent result.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Kept so the allowlist import set is exercised rather than decorative: the
# manifest path is named as data, and no checker ever opens it.
MANIFEST_RELATIVE_PATH = pathlib.PurePosixPath("tests") / "checkers.yaml"

# Every machine-readable reason this module can emit. A zero that is not in this
# tuple is a defect in this module, never a silent score.
ZERO_REASONS: Tuple[str, ...] = (
    "non-convergence-ungraded",
    "run-produced-no-verifier-state",
    "frozen-axis-moved",
    "weights-not-harness-owned",
    "evaluation-order-violated",
    "readout-smoothed",
    "early-stop-without-sustained-crossing",
    "crossing-not-sustained",
    "graded-step-not-the-crossing",
    "reported-crossing-diverges",
)

# Terminal run statuses the verifier itself recognises. A status outside this
# set is not a weaker status, it is an unreadable record.
RUN_STATUSES: Tuple[str, ...] = ("completed", "terminated-early", "crashed", "timeout")

# The one basis an outcome document may carry. "submission-reported" is absent
# from this tuple on purpose: the graded outcome is recomputed or it does not
# exist.
OUTCOME_BASIS = "verifier-recomputed"

# The reason the derived outcome carries when nothing crossed. It is a reason,
# not a silence, and not an absent result.
NON_CONVERGENCE_REASON = "non-convergence-no-sustained-crossing"


@dataclass(frozen=True)
class Verdict:
    """One checker's decision, and the reason a zero is what it is."""

    ident: str
    passed: bool
    reason: str
    detail: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.ident,
            "passed": bool(self.passed),
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Evaluation:
    """One verifier-scheduled evaluation of harness-owned weights."""

    step: int
    loss: float
    filter_name: str
    weights_digest: str
    weights_source: str
    params_digest: str
    scheduled_by: str


def _ok(ident: str, detail: str) -> Verdict:
    return Verdict(ident, True, "", detail)


def _no(ident: str, reason: str, detail: str) -> Verdict:
    # A reason outside the closed set would be a bare zero wearing a label.
    if reason not in ZERO_REASONS:
        raise ValueError("zero reason outside the closed set: " + repr(reason))
    return Verdict(ident, False, reason, detail)


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def bound(telemetry: Dict[str, Any], key: str, fallback: Any = None) -> Any:
    """One verifier-owned bound value. Never a submission-supplied one."""
    return _mapping(telemetry.get("bound")).get(key, fallback)


def run_record(telemetry: Dict[str, Any]) -> Dict[str, Any]:
    return _mapping(telemetry.get("run"))


def outcome_document(telemetry: Dict[str, Any]) -> Dict[str, Any]:
    return _mapping(telemetry.get("outcome"))


def evaluations(telemetry: Dict[str, Any]) -> List[Evaluation]:
    """The verifier's own evaluation records, in the order the verifier wrote them.

    The order is preserved rather than normalised, because the ORDERING checker
    grades that order and a silently sorted copy would make a broken schedule
    unobservable.
    """
    rows: List[Evaluation] = []
    for raw in telemetry.get("evaluations") or []:
        row = _mapping(raw)
        try:
            step = int(row.get("step"))
            loss = float(row.get("loss"))
        except (TypeError, ValueError):
            continue
        if loss != loss or math.isinf(loss):
            continue
        rows.append(
            Evaluation(
                step=step,
                loss=loss,
                filter_name=str(row.get("filter", "")),
                weights_digest=str(row.get("weights_digest", "")),
                weights_source=str(row.get("weights_source", "")),
                params_digest=str(row.get("params_digest", "")),
                scheduled_by=str(row.get("scheduled_by", "")),
            )
        )
    return rows


def ascending(rows: Sequence[Evaluation]) -> List[Evaluation]:
    return sorted(rows, key=lambda item: item.step)


def harness_weights_digest(domain: str, step: int, params_digest: str) -> str:
    """The digest the harness binds for the weights it owns at one step.

    Recomputed here from the bound domain, so a digest minted under a superseded
    domain, or one the submission chose for itself, does not verify.
    """
    payload = str(domain) + "|" + str(int(step)) + "|" + str(params_digest)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def first_below(rows: Sequence[Evaluation], target: float) -> Optional[int]:
    """The first ascending evaluation step strictly below the bar. Unsustained."""
    for row in ascending(rows):
        if row.loss < target:
            return row.step
    return None


def first_sustained_below(
    rows: Sequence[Evaluation], target: float, sustain: int
) -> Optional[int]:
    """THE graded quantity: the first crossing that survives the sustain window.

    A single favourable evaluation is not a crossing. The target must hold at the
    candidate step and at `sustain` further verifier-scheduled points. A candidate
    with too few subsequent points has not established the condition, so it is not
    a crossing either; the run simply stopped before the evidence existed.
    """
    ordered = ascending(rows)
    for index, row in enumerate(ordered):
        if row.loss >= target:
            continue
        window = ordered[index + 1 : index + 1 + int(sustain)]
        if len(window) < int(sustain):
            return None
        if all(item.loss < target for item in window):
            return row.step
    return None


def derive_outcome(telemetry: Dict[str, Any]) -> Dict[str, Any]:
    """The outcome document the grader emits. Always attributed, never absent.

    Both branches carry the same keys. A non-converging run yields an explicit
    null graded step and a machine-readable reason, so downstream there is no
    shape difference between "did not cross" and "crossed", only a value
    difference. That is what stops a non-convergence from reading as a missing
    result.
    """
    target = float(bound(telemetry, "target_loss", 0.0))
    sustain = int(bound(telemetry, "sustain_points", 0))
    step = first_sustained_below(evaluations(telemetry), target, sustain)
    if step is None:
        return {
            "crossed": False,
            "graded_step": None,
            "reason": NON_CONVERGENCE_REASON,
            "basis": OUTCOME_BASIS,
        }
    return {
        "crossed": True,
        "graded_step": int(step),
        "reason": "",
        "basis": OUTCOME_BASIS,
    }


# ---------------------------------------------------------------------------
# The checkers. Declaration order below is gate order in tests/grade.py.
# ---------------------------------------------------------------------------


def absence_of_ungraded_non_convergence(telemetry: Dict[str, Any]) -> Verdict:
    """ABSENCE. No grading route emits an unattributed or absent outcome.

    This is the archetype applied to the grading path rather than to the run. The
    document must always carry the `crossed` flag, must always carry the
    `graded_step` key explicitly even when it is null, must always be recomputed
    rather than reported, and must carry a machine-readable reason whenever it did
    not cross. An outcome missing any of those is a route by which a
    non-converging run leaves the grader as an absent result.
    """
    ident = "non_convergence_graded_as_failure_with_a_reason"
    doc = outcome_document(telemetry)
    if "crossed" not in doc:
        return _no(ident, "non-convergence-ungraded", "the outcome document carries no crossed flag")
    if "graded_step" not in doc:
        return _no(
            ident,
            "non-convergence-ungraded",
            "the outcome document omits the graded_step key, so a failure is shaped like an absent result",
        )
    if doc.get("basis") != OUTCOME_BASIS:
        return _no(
            ident,
            "non-convergence-ungraded",
            "the outcome basis is " + repr(doc.get("basis")) + ", not " + OUTCOME_BASIS,
        )
    if bool(doc.get("crossed")):
        if doc.get("graded_step") is None:
            return _no(
                ident,
                "non-convergence-ungraded",
                "the outcome claims a crossing and names no step",
            )
        return _ok(ident, "crossed at step " + str(doc.get("graded_step")) + ", attributed")
    reason = str(doc.get("reason") or "")
    if not reason or not _is_kebab(reason):
        return _no(
            ident,
            "non-convergence-ungraded",
            "a non-converging run carries reason " + repr(reason) + ", which is not machine-readable",
        )
    if doc.get("graded_step") is not None:
        return _no(
            ident,
            "non-convergence-ungraded",
            "the outcome denies a crossing and still names a graded step",
        )
    return _ok(ident, "did not cross, graded a failure with reason " + reason)


KEBAB_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


def _is_kebab(text: str) -> bool:
    """Lowercase kebab, the only shape a downstream grader can branch on."""
    if not text:
        return False
    for part in text.split("-"):
        if not part:
            return False
        for char in part:
            if char not in KEBAB_ALPHABET:
                return False
    return True


def run_produced_verifier_owned_state(telemetry: Dict[str, Any]) -> Verdict:
    """EFFECT. Launching the submission changed live state the verifier owns."""
    ident = "run_produced_verifier_owned_state"
    run = run_record(telemetry)
    status = str(run.get("status", ""))
    if status not in RUN_STATUSES:
        return _no(
            ident,
            "run-produced-no-verifier-state",
            "run status " + repr(status) + " is outside " + ", ".join(RUN_STATUSES),
        )
    rows = evaluations(telemetry)
    if not rows:
        return _no(
            ident,
            "run-produced-no-verifier-state",
            "status " + status + " and zero verifier-owned evaluations, so nothing was measured",
        )
    empty = [row.step for row in rows if not row.weights_digest]
    if empty:
        return _no(
            ident,
            "run-produced-no-verifier-state",
            "steps carrying no harness weight digest: " + ", ".join(str(item) for item in empty[:5]),
        )
    try:
        executed = int(run.get("steps_executed"))
    except (TypeError, ValueError):
        return _no(ident, "run-produced-no-verifier-state", "steps_executed is unreadable")
    if executed <= 0:
        return _no(
            ident,
            "run-produced-no-verifier-state",
            "the runner observed " + str(executed) + " optimizer steps",
        )
    return _ok(ident, str(len(rows)) + " verifier-owned evaluations over " + str(executed) + " steps")


def frozen_axes_held(telemetry: Dict[str, Any]) -> Verdict:
    """INVARIANT. Dataset, batch size, architecture and one pass per step unmoved."""
    ident = "frozen_axes_held"
    observed = _mapping(telemetry.get("frozen_axes_observed"))
    pairs = (
        ("batch_size", bound(telemetry, "frozen_batch_size")),
        ("fwd_bwd_per_step", bound(telemetry, "frozen_fwd_bwd_per_step")),
        ("dataset_digest", bound(telemetry, "frozen_dataset_digest")),
        ("architecture_digest", bound(telemetry, "frozen_architecture_digest")),
    )
    for key, want in pairs:
        got = observed.get(key)
        if got != want:
            return _no(
                ident,
                "frozen-axis-moved",
                "frozen axis " + key + " is " + repr(got) + ", bound at " + repr(want),
            )
    return _ok(ident, "all four frozen axes match the bound values")


def evaluated_weights_owned_by_the_harness(telemetry: Dict[str, Any]) -> Verdict:
    """VALUE. The weights evaluated are the ones the run produced at that step.

    The digest is recomputed from the bound domain, the step, and the parameter
    digest the harness recorded. A checkpoint the submission selected carries a
    source other than the harness, and a digest minted under a superseded domain
    does not reproduce.
    """
    ident = "evaluated_weights_owned_by_the_harness"
    domain = str(bound(telemetry, "weights_digest_domain", ""))
    if not domain:
        return _no(ident, "weights-not-harness-owned", "no weights digest domain is bound")
    for row in evaluations(telemetry):
        if row.weights_source != "harness":
            return _no(
                ident,
                "weights-not-harness-owned",
                "step " + str(row.step) + " evaluated weights sourced from " + repr(row.weights_source),
            )
        want = harness_weights_digest(domain, row.step, row.params_digest)
        if row.weights_digest != want:
            return _no(
                ident,
                "weights-not-harness-owned",
                "step " + str(row.step) + " digest does not reproduce under domain " + domain,
            )
    return _ok(ident, "every evaluated state reproduces under " + domain)


def evaluation_points_ascend_on_the_verifier_schedule(telemetry: Dict[str, Any]) -> Verdict:
    """ORDERING. The evaluation grid is the verifier's, ascending, on its stride."""
    ident = "evaluation_points_ascend_on_the_verifier_schedule"
    try:
        stride = int(bound(telemetry, "eval_stride", 0))
    except (TypeError, ValueError):
        stride = 0
    if stride <= 0:
        return _no(ident, "evaluation-order-violated", "no positive evaluation stride is bound")
    previous: Optional[int] = None
    for row in evaluations(telemetry):
        if row.scheduled_by != "verifier":
            return _no(
                ident,
                "evaluation-order-violated",
                "step " + str(row.step) + " was scheduled by " + repr(row.scheduled_by),
            )
        if row.step <= 0 or row.step % stride != 0:
            return _no(
                ident,
                "evaluation-order-violated",
                "step " + str(row.step) + " is not a positive multiple of stride " + str(stride),
            )
        if previous is not None and row.step <= previous:
            return _no(
                ident,
                "evaluation-order-violated",
                "step " + str(row.step) + " does not follow " + str(previous),
            )
        previous = row.step
    if previous is None:
        return _no(ident, "evaluation-order-violated", "the evaluation grid is empty")
    return _ok(ident, "ascending on stride " + str(stride) + " through step " + str(previous))


def graded_readout_unsmoothed(telemetry: Dict[str, Any]) -> Verdict:
    """VALUE. The graded readout is the raw evaluation, never a filtered one.

    A submission may EMA-blend, average or otherwise filter its own reported loss
    for its own use. The graded quantity is recomputed unsmoothed by the verifier,
    and a filtered value on the graded path scores zero with this reason.
    """
    ident = "graded_readout_unsmoothed"
    admitted = str(bound(telemetry, "readout_filter_admitted", ""))
    if not admitted:
        return _no(ident, "readout-smoothed", "no admitted readout filter is bound")
    for row in evaluations(telemetry):
        if row.filter_name != admitted:
            return _no(
                ident,
                "readout-smoothed",
                "step " + str(row.step) + " carries filter " + repr(row.filter_name)
                + " on the graded path, admitted is " + repr(admitted),
            )
    return _ok(ident, "every graded readout is " + admitted)


def early_stop_is_not_a_crossing(telemetry: Dict[str, Any]) -> Verdict:
    """EFFECT. Halting at a favourable evaluation does not produce a crossing.

    The run must actually have executed far enough for the sustain window to
    exist. A run that stopped at its first favourable evaluation established
    nothing, and this is graded as a failure with a reason rather than as an
    absent result.
    """
    ident = "early_stop_is_not_a_crossing"
    run = run_record(telemetry)
    target = float(bound(telemetry, "target_loss", 0.0))
    sustain = int(bound(telemetry, "sustain_points", 0))
    stride = int(bound(telemetry, "eval_stride", 0))
    candidate = first_below(evaluations(telemetry), target)
    try:
        executed = int(run.get("steps_executed"))
    except (TypeError, ValueError):
        executed = -1
    if candidate is None:
        return _ok(ident, "no evaluation fell below the bar, so nothing was stopped early on")
    needed = candidate + sustain * stride
    if executed < needed:
        return _no(
            ident,
            "early-stop-without-sustained-crossing",
            "first favourable evaluation at " + str(candidate) + " needs " + str(needed)
            + " executed steps to sustain, the run executed " + str(executed)
            + " and stopped by " + repr(str(run.get("stopped_by", ""))),
        )
    return _ok(ident, "the run executed " + str(executed) + " steps, past the sustain window at " + str(needed))


def crossing_sustained_across_verifier_evaluations(telemetry: Dict[str, Any]) -> Verdict:
    """INVARIANT. The target holds at the graded step and at the bound follow-ups."""
    ident = "crossing_sustained_across_verifier_evaluations"
    target = float(bound(telemetry, "target_loss", 0.0))
    sustain = int(bound(telemetry, "sustain_points", 0))
    if sustain <= 0:
        return _no(ident, "crossing-not-sustained", "no positive sustain window is bound")
    rows = evaluations(telemetry)
    step = first_sustained_below(rows, target, sustain)
    if step is None:
        dipped = first_below(rows, target)
        detail = "no evaluation fell below " + str(target)
        if dipped is not None:
            detail = (
                "step " + str(dipped) + " fell below " + str(target)
                + " and the condition did not hold across " + str(sustain) + " further verifier evaluations"
            )
        return _no(ident, "crossing-not-sustained", detail)
    return _ok(ident, "crossing at " + str(step) + " sustained across " + str(sustain) + " further evaluations")


def graded_step_is_the_crossing_not_the_schedule(telemetry: Dict[str, Any]) -> Verdict:
    """VALUE. The graded step equals the recomputed crossing, not the schedule length.

    Recomputed here from the evaluation series and the bound bar alone. A schedule
    extended past the crossing changes the schedule length and cannot change this
    number, which is what makes the extension worth exactly nothing.
    """
    ident = "graded_step_is_the_crossing_not_the_schedule"
    target = float(bound(telemetry, "target_loss", 0.0))
    sustain = int(bound(telemetry, "sustain_points", 0))
    recomputed = first_sustained_below(evaluations(telemetry), target, sustain)
    doc = outcome_document(telemetry)
    graded = doc.get("graded_step")
    run = run_record(telemetry)
    if graded is None and recomputed is None:
        return _ok(ident, "no crossing, and the outcome names none")
    if graded != recomputed:
        return _no(
            ident,
            "graded-step-not-the-crossing",
            "the outcome names " + repr(graded) + " and the verifier recomputes "
            + repr(recomputed) + " from the bar " + str(target),
        )
    try:
        schedule = int(run.get("schedule_length"))
        executed = int(run.get("steps_executed"))
    except (TypeError, ValueError):
        return _no(ident, "graded-step-not-the-crossing", "the run record carries no readable step counts")
    if int(graded) > executed:
        return _no(
            ident,
            "graded-step-not-the-crossing",
            "the graded step " + str(graded) + " is past the " + str(executed) + " steps the runner observed",
        )
    if int(graded) == schedule and recomputed != schedule:
        return _no(
            ident,
            "graded-step-not-the-crossing",
            "the graded step equals the schedule length " + str(schedule),
        )
    return _ok(ident, "graded step " + str(graded) + " is the recomputed crossing, schedule length " + str(schedule))


def reported_crossing_matches_recomputed_crossing(telemetry: Dict[str, Any]) -> Verdict:
    """DIVERGENCE. What the submission reported and what the verifier recomputed agree.

    The recomputed number is the graded one either way. This checker exists so a
    divergence is loud rather than absorbed: a submission whose own readout says it
    crossed somewhere the verifier's evaluation does not is shaping a readout, and
    that is graded rather than tolerated.
    """
    ident = "reported_crossing_matches_recomputed_crossing"
    target = float(bound(telemetry, "target_loss", 0.0))
    sustain = int(bound(telemetry, "sustain_points", 0))
    recomputed = first_sustained_below(evaluations(telemetry), target, sustain)
    report = _mapping(telemetry.get("submission_report"))
    reported = report.get("crossing_step")
    if reported is None:
        return _ok(ident, "the submission reported no crossing, so nothing diverges")
    try:
        reported_int = int(reported)
    except (TypeError, ValueError):
        return _no(ident, "reported-crossing-diverges", "the reported crossing " + repr(reported) + " is unreadable")
    if recomputed is None or reported_int != int(recomputed):
        return _no(
            ident,
            "reported-crossing-diverges",
            "the submission reports " + str(reported_int) + " and the verifier recomputes " + repr(recomputed),
        )
    return _ok(ident, "reported and recomputed crossings agree at " + str(reported_int))


# The gate chain, in the order tests/grade.py runs it. The order is part of the
# design: the grading-path obligation is checked before anything about the run,
# so a defective outcome document can never hide behind a failing run.
CHECKERS: Tuple[Tuple[str, Callable[[Dict[str, Any]], Verdict]], ...] = (
    ("non_convergence_graded_as_failure_with_a_reason", absence_of_ungraded_non_convergence),
    ("run_produced_verifier_owned_state", run_produced_verifier_owned_state),
    ("frozen_axes_held", frozen_axes_held),
    ("evaluated_weights_owned_by_the_harness", evaluated_weights_owned_by_the_harness),
    (
        "evaluation_points_ascend_on_the_verifier_schedule",
        evaluation_points_ascend_on_the_verifier_schedule,
    ),
    ("graded_readout_unsmoothed", graded_readout_unsmoothed),
    ("early_stop_is_not_a_crossing", early_stop_is_not_a_crossing),
    (
        "crossing_sustained_across_verifier_evaluations",
        crossing_sustained_across_verifier_evaluations,
    ),
    (
        "graded_step_is_the_crossing_not_the_schedule",
        graded_step_is_the_crossing_not_the_schedule,
    ),
    (
        "reported_crossing_matches_recomputed_crossing",
        reported_crossing_matches_recomputed_crossing,
    ),
)

# Which bound state key each checker reads. seed/tasks/OER-03/drift.yaml binds
# every silent mutation to a checker, and seed/tasks/OER-03/adequacy.py proves
# causality mechanically against this mapping rather than against a comment.
STATE_KEYS_READ: Dict[str, Tuple[str, ...]] = {
    "non_convergence_graded_as_failure_with_a_reason": (),
    "run_produced_verifier_owned_state": (),
    "frozen_axes_held": (
        "frozen_batch_size",
        "frozen_fwd_bwd_per_step",
        "frozen_dataset_digest",
        "frozen_architecture_digest",
    ),
    "evaluated_weights_owned_by_the_harness": ("weights_digest_domain",),
    "evaluation_points_ascend_on_the_verifier_schedule": ("eval_stride",),
    "graded_readout_unsmoothed": ("readout_filter_admitted",),
    "early_stop_is_not_a_crossing": ("target_loss", "sustain_points", "eval_stride"),
    "crossing_sustained_across_verifier_evaluations": ("target_loss", "sustain_points"),
    "graded_step_is_the_crossing_not_the_schedule": (
        "target_loss",
        "sustain_points",
        "max_schedule_steps",
    ),
    "reported_crossing_matches_recomputed_crossing": ("target_loss", "sustain_points"),
}


def run_all(telemetry: Dict[str, Any]) -> List[Verdict]:
    """Every checker, in gate order. Pure: the same telemetry gives the same list."""
    return [selector(telemetry) for _, selector in CHECKERS]


def first_failure(verdicts: Sequence[Verdict]) -> Optional[Verdict]:
    for verdict in verdicts:
        if not verdict.passed:
            return verdict
    return None


def normalized_reward(graded_step: Optional[int], baseline: int, target: int) -> float:
    """The bounded-continuous score. Lower steps are better; the target is a bar."""
    if graded_step is None:
        return 0.0
    span = float(baseline) - float(target)
    if span <= 0.0:
        return 0.0
    raw = (float(baseline) - float(graded_step)) / span
    return min(max(raw, 0.0), 1.0)


def verdict_digest(verdicts: Sequence[Verdict]) -> str:
    """A digest over the whole gate outcome, so two runs can be compared exactly."""
    payload = json.dumps(
        [item.as_dict() for item in verdicts], sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
