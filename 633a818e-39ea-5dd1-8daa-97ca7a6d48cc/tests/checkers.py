"""The eleven graded checkers. Pure, deterministic, and read only harness telemetry.

Every function here is a pure function of one session telemetry document that
the verifier's own process wrote. Nothing in this file reads a planted answer,
an environment secret, a clock, a random source or the network, and nothing here
imports the submission. The imports are exactly the six the AST allowlist admits.

Every bound value a checker compares against arrives in the telemetry header,
which tests/runner.py wrote from tests/lab.py. No threshold is authored here.
"""

import json
import math
import pathlib
from dataclasses import dataclass, field
from typing import Optional

HEADER_KIND = "session_header"
ATTEMPT_KIND = "attempt"
FOOTER_KIND = "session_footer"


@dataclass
class Verdict:
    """One checker's answer: whether it held, and, when it did not, why."""

    ident: str
    passed: bool
    zero_reason: str = ""
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "id": self.ident,
            "passed": self.passed,
            "zero_reason": "" if self.passed else self.zero_reason,
            "detail": self.detail,
        }


@dataclass
class Session:
    """One session telemetry document, as the verifier's own process wrote it."""

    header: dict = field(default_factory=dict)
    attempts: list = field(default_factory=list)
    footer: dict = field(default_factory=dict)

    @property
    def graded(self) -> Optional[dict]:
        """The attempt the graded readout is taken from.

        Selection is best-of-k, which is the bound final_selection, so the graded
        attempt is the scored attempt with the lowest multi-seed mean. When no
        attempt scored at all the last recorded attempt is graded instead, so a
        session that halted every run short of the sustain window is graded on
        what it did rather than reported as absent.
        """
        scored = [row for row in self.attempts if row.get("mean_steps") is not None]
        if scored:
            return min(scored, key=lambda row: (row["mean_steps"], row["index"]))
        if self.attempts:
            return self.attempts[-1]
        return None


def load_session(path) -> Session:
    """Read the telemetry the harness wrote. The path is handed in, never guessed."""
    handle = pathlib.Path(path)
    session = Session()
    if not handle.is_file():
        return session
    for line in handle.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        kind = row.get("kind")
        if kind == HEADER_KIND:
            session.header = row
        elif kind == ATTEMPT_KIND:
            session.attempts.append(row)
        elif kind == FOOTER_KIND:
            session.footer = row
    return session


def _empty(ident: str, reason: str) -> Verdict:
    return Verdict(ident, False, reason, "the telemetry document carries no attempt record")


def check_attempt_sequence_contiguous(session: Session) -> Verdict:
    """ORDERING. Attempt indices ascend contiguously from zero across the session."""
    ident = "attempt_sequence_contiguous"
    if not session.attempts:
        return _empty(ident, "attempt-sequence-out-of-order")
    indices = [row.get("index") for row in session.attempts]
    expected = list(range(len(indices)))
    if indices != expected:
        return Verdict(
            ident, False, "attempt-sequence-out-of-order",
            "recorded indices " + json.dumps(indices[:8]) + " do not ascend contiguously from zero",
        )
    return Verdict(ident, True, "", "indices ascend contiguously across " + str(len(indices)) + " attempts")


def check_session_holds_attempt_budget(session: Session) -> Verdict:
    """ORDERING. The sequence runs to the bound terminator instead of stopping short."""
    ident = "session_holds_attempt_budget"
    terminator = session.header.get("terminator")
    if terminator is None:
        return Verdict(ident, False, "session-budget-not-held", "the header binds no attempt terminator")
    recorded = len(session.attempts)
    if recorded != int(terminator):
        return Verdict(
            ident, False, "session-budget-not-held",
            "the session recorded " + str(recorded) + " attempts against a terminator of " + str(terminator),
        )
    return Verdict(ident, True, "", "the session held all " + str(recorded) + " bound attempts")


def check_carried_state_monotone(session: Session) -> Verdict:
    """INVARIANT. Carried state is exactly the accumulation of the applied deltas."""
    ident = "carried_state_monotone"
    if not session.attempts:
        return _empty(ident, "carried-state-not-accumulated")
    carried = {"a": 0, "b": 0, "c": 0}
    for row in session.attempts:
        before = row.get("state_before") or {}
        after = row.get("state_after") or {}
        for axis in sorted(carried):
            if int(before.get(axis, -1)) != carried[axis]:
                return Verdict(
                    ident, False, "carried-state-not-accumulated",
                    "attempt " + str(row.get("index")) + " opened axis " + axis
                    + " at " + str(before.get(axis)) + " while the session carried " + str(carried[axis]),
                )
        moved = str(row.get("axis", ""))
        applied = int(row.get("applied_delta_units", 0) or 0)
        for axis in sorted(carried):
            gain = applied if axis == moved else 0
            if int(after.get(axis, -1)) != carried[axis] + gain:
                return Verdict(
                    ident, False, "carried-state-not-accumulated",
                    "attempt " + str(row.get("index")) + " closed axis " + axis
                    + " at " + str(after.get(axis)) + " rather than at " + str(carried[axis] + gain),
                )
            carried[axis] = carried[axis] + gain
    return Verdict(ident, True, "", "carried state accumulated to " + json.dumps(carried, sort_keys=True))


def check_reallocation_follows_flattening(session: Session) -> Verdict:
    """ORDERING. The session left the flattened direction once it stopped paying."""
    ident = "reallocation_follows_flattening"
    footer = session.footer
    if not footer:
        return _empty(ident, "flattened-direction-never-abandoned")
    onset = footer.get("flattening_onset_attempt")
    allowance = session.header.get("post_flattening_probe_allowance")
    if allowance is None:
        return Verdict(
            ident, False, "flattened-direction-never-abandoned",
            "the header binds no post-flattening probe allowance",
        )
    if onset is None:
        return Verdict(
            ident, True, "",
            "no direction reached its plateau in this session, so no reallocation was owed",
        )
    wasted = int(footer.get("post_flattening_attempts_on_flattened_axis", 0) or 0)
    reallocated = footer.get("first_reallocation_attempt")
    if wasted > int(allowance):
        return Verdict(
            ident, False, "flattened-direction-never-abandoned",
            "the session spent " + str(wasted) + " attempts on the flattened direction after attempt "
            + str(onset) + ", above the bound probe allowance of " + str(allowance),
        )
    if reallocated is None or int(reallocated) <= int(onset):
        return Verdict(
            ident, False, "flattened-direction-never-abandoned",
            "no attempt after the flattening onset at " + str(onset) + " moved a different direction",
        )
    return Verdict(
        ident, True, "",
        "reallocated at attempt " + str(reallocated) + " after the onset at " + str(onset),
    )


def check_graded_weights_are_run_weights(session: Session) -> Verdict:
    """VALUE. The evaluated weights are the weights the run held at the graded step."""
    ident = "graded_weights_are_run_weights"
    row = session.graded
    if row is None:
        return _empty(ident, "checkpoint-substituted")
    evaluated = str(row.get("weights_digest", ""))
    produced = str(row.get("run_weights_digest", ""))
    if evaluated != produced:
        return Verdict(
            ident, False, "checkpoint-substituted",
            "the graded attempt evaluated weights " + (evaluated[:16] or "<absent>")
            + " while the run produced " + (produced[:16] or "<absent>") + " at the graded step",
        )
    return Verdict(ident, True, "", "evaluated weights match the run weights at the graded step")


def check_graded_readout_unsmoothed(session: Session) -> Verdict:
    """VALUE. The graded readout is the verifier's raw evaluation, filtered by nothing."""
    ident = "graded_readout_unsmoothed"
    if not session.attempts:
        return _empty(ident, "readout-smoothing-on-graded-path")
    filtered = [row.get("index") for row in session.attempts if str(row.get("readout_filter", "none")) != "none"]
    if filtered:
        return Verdict(
            ident, False, "readout-smoothing-on-graded-path",
            "these attempts requested a filtered readout on the graded path: " + json.dumps(filtered[:8]),
        )
    row = session.graded
    target = session.header.get("target_loss")
    observed = row.get("graded_loss_at_crossing") if row else None
    if observed is not None and target is not None and not observed < float(target):
        return Verdict(
            ident, False, "readout-smoothing-on-graded-path",
            "the raw evaluation at the graded step is " + repr(observed) + ", not below " + repr(target),
        )
    return Verdict(ident, True, "", "the graded readout is the raw evaluation at the graded step")


def check_early_stop_is_not_a_crossing(session: Session) -> Verdict:
    """ABSENCE. No graded crossing is credited from a run halted inside the window."""
    ident = "early_stop_is_not_a_crossing"
    row = session.graded
    if row is None:
        return _empty(ident, "early-stop-not-a-crossing")
    halted = row.get("halted_at")
    if halted is None:
        return Verdict(ident, True, "", "the graded attempt carries no halt marker")
    crossing = row.get("graded_crossing_step")
    cadence = session.header.get("eval_every")
    sustain = session.header.get("sustain_points")
    if crossing is None or cadence is None or sustain is None:
        return Verdict(
            ident, False, "early-stop-not-a-crossing",
            "the graded attempt halted at step " + str(halted) + " with no sustained crossing recorded",
        )
    needed = int(crossing) + int(sustain) * int(cadence)
    if int(halted) < needed:
        return Verdict(
            ident, False, "early-stop-not-a-crossing",
            "the graded attempt halted at " + str(halted) + " before the sustain window closed at " + str(needed),
        )
    return Verdict(ident, True, "", "the halt at " + str(halted) + " falls outside the sustain window")


def check_crossing_sustained(session: Session) -> Verdict:
    """INVARIANT. The crossing holds at the graded step and at every further point."""
    ident = "crossing_sustained"
    row = session.graded
    if row is None:
        return _empty(ident, "crossing-not-sustained")
    crossing = row.get("graded_crossing_step")
    if crossing is None or not row.get("graded_crossing_sustained"):
        return Verdict(
            ident, False, "crossing-not-sustained",
            "no evaluation point held below target for the bound sustain window",
        )
    claimed = row.get("claimed_crossing_step")
    if claimed is not None:
        if not row.get("claimed_step_sustained"):
            return Verdict(
                ident, False, "crossing-not-sustained",
                "the session reported a crossing at step " + str(claimed)
                + ", which the verifier recomputed as not sustained",
            )
        if int(claimed) != int(crossing):
            return Verdict(
                ident, False, "crossing-not-sustained",
                "the session reported a crossing at step " + str(claimed)
                + " while the verifier recomputed it at " + str(crossing),
            )
    return Verdict(
        ident, True, "",
        "the crossing at step " + str(crossing) + " held across the bound sustain window",
    )


def check_target_band_reached_by_session(session: Session) -> Verdict:
    """EFFECT. The session moved the graded quantity off the baseline control."""
    ident = "target_band_reached_by_session"
    footer = session.footer
    best = footer.get("best_mean_steps") if footer else None
    baseline = session.header.get("baseline_control_mean")
    if best is None or baseline is None:
        return Verdict(
            ident, False, "no-measured-improvement",
            "the session recorded no scored attempt to compare against the baseline control",
        )
    if not float(best) < float(baseline):
        return Verdict(
            ident, False, "no-measured-improvement",
            "the best multi-seed mean is " + repr(best) + " against a baseline control of " + repr(baseline),
        )
    return Verdict(
        ident, True, "",
        "the best multi-seed mean " + repr(best) + " improves on the baseline control " + repr(baseline),
    )


def check_multi_seed_separation_established(session: Session) -> Verdict:
    """DIVERGENCE. No single seed carries the mean, and every seed separates."""
    ident = "multi_seed_separation_established"
    footer = session.footer
    header = session.header
    per_seed = (footer or {}).get("best_per_seed_steps") or {}
    baseline_seeds = header.get("baseline_control_per_seed") or {}
    target_seeds = header.get("target_control_per_seed") or {}
    band = header.get("seed_artifact_band")
    ladder = header.get("seed_ladder") or []
    if not per_seed or not baseline_seeds or not target_seeds or band is None or not ladder:
        return Verdict(
            ident, False, "separation-not-established",
            "the telemetry carries no per-seed ladder to compute divergence over",
        )
    normalized = []
    for seed in ladder:
        agent = per_seed.get(seed)
        base = baseline_seeds.get(seed)
        goal = target_seeds.get(seed)
        if agent is None or base is None or goal is None or base <= goal:
            return Verdict(
                ident, False, "separation-not-established",
                "seed " + str(seed) + " carries no comparable pair of control and session values",
            )
        normalized.append((float(base) - float(agent)) / (float(base) - float(goal)))
    if min(normalized) <= 0.0:
        return Verdict(
            ident, False, "separation-not-established",
            "seed-level separation is non-positive on at least one seed: "
            + json.dumps([round(value, 6) for value in normalized]),
        )
    centre = math.fsum(normalized) / float(len(normalized))
    spread = max(abs(value - centre) for value in normalized)
    if spread > float(band):
        return Verdict(
            ident, False, "separation-not-established",
            "per-seed separation diverges by " + str(round(spread, 6))
            + ", above the bound artifact band of " + str(band),
        )
    return Verdict(
        ident, True, "",
        "every seed separated, with a spread of " + str(round(spread, 6)) + " inside the bound band",
    )


def check_reference_operating_point_matches_final_state(session: Session) -> Verdict:
    """VALUE. The reward's reference control is the operating point the reference
    session's own final carried state produces, and the session's recorded final
    state is the state its last attempt actually closed on."""
    ident = "reference_operating_point_matches_final_state"
    header = session.header
    bound_state = header.get("reference_final_state")
    bound_mean = header.get("reference_final_state_control_mean")
    bound_seeds = header.get("reference_final_state_control_per_seed") or {}
    target_mean = header.get("target_control_mean")
    target_seeds = header.get("target_control_per_seed") or {}
    if not bound_state or bound_mean is None or target_mean is None:
        return Verdict(
            ident, False, "reference-operating-point-unheld",
            "the telemetry header carries no measured reference final state to read the operating point back from",
        )
    if float(bound_mean) != float(target_mean):
        return Verdict(
            ident, False, "reference-operating-point-unheld",
            "the reference final state " + json.dumps(bound_state, sort_keys=True)
            + " measures " + repr(bound_mean)
            + " while the reference-operating-point control the reward divides by is " + repr(target_mean),
        )
    diverged = sorted(
        seed for seed in target_seeds if bound_seeds.get(seed) != target_seeds.get(seed)
    )
    if not bound_seeds or not target_seeds or diverged:
        return Verdict(
            ident, False, "reference-operating-point-unheld",
            "the reference final state and the reference-operating-point control are not one operating"
            " point on seeds " + json.dumps(diverged),
        )
    recorded = (session.footer or {}).get("final_state")
    closing = session.attempts[-1].get("state_after") if session.attempts else None
    if not recorded or not closing:
        return Verdict(
            ident, False, "reference-operating-point-unheld",
            "the telemetry records no final carried state for the session",
        )
    if {axis: int(value) for axis, value in recorded.items()} != {
        axis: int(value) for axis, value in closing.items()
    }:
        return Verdict(
            ident, False, "reference-operating-point-unheld",
            "the recorded final carried state " + json.dumps(recorded, sort_keys=True)
            + " is not the state the last attempt closed on " + json.dumps(closing, sort_keys=True),
        )
    return Verdict(
        ident, True, "",
        "the reference-operating-point control " + repr(target_mean)
        + " is the operating point the reference final state "
        + json.dumps(bound_state, sort_keys=True) + " holds, and the session closed carrying "
        + json.dumps(recorded, sort_keys=True),
    )


ORDERED_CHECKS = (
    check_attempt_sequence_contiguous,
    check_session_holds_attempt_budget,
    check_carried_state_monotone,
    check_reallocation_follows_flattening,
    check_graded_weights_are_run_weights,
    check_graded_readout_unsmoothed,
    check_early_stop_is_not_a_crossing,
    check_crossing_sustained,
    check_target_band_reached_by_session,
    check_multi_seed_separation_established,
    check_reference_operating_point_matches_final_state,
)


def run_all(session: Session) -> list:
    """Every checker, in the manifest's order, over one telemetry document."""
    return [check(session) for check in ORDERED_CHECKS]
