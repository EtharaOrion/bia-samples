"""The eight graded checkers for slot OER-01. Pure, deterministic, allowlisted.

Every function here is a pure function of the arguments handed to it. Nothing in
this module opens a file the caller did not name, reads an environment variable,
reads a clock, consults a random source, opens a socket, or imports the
submission. The import list is the allowlist and nothing else: json, math,
hashlib, pathlib, dataclasses, typing.

Every number a checker reads arrives inside the telemetry record the verifier's
OWN harness produced in its OWN process, or inside a bound artifact of this
bundle that the grader loaded and handed in. No number here was printed by the
submission, and the one field that IS a submission-reported number,
`submission_reported_loss`, exists only so a checker can prove it never reaches
the graded path.

Each checker carries exactly one reduction from the closed six. The reduction of
each is declared in tests/checkers.yaml and restated in the docstring here, so a
reader comparing the two files can see them agree without running anything.
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Outcome:
    """One checker's verdict. A false verdict always carries its reason code."""

    ident: str
    passed: bool
    zero_reason: str
    detail: str
    observed: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "id": self.ident,
            "passed": bool(self.passed),
            "zero_reason": "" if self.passed else self.zero_reason,
            "detail": self.detail,
            "observed": self.observed,
        }


def _ok(ident: str, detail: str, observed: dict) -> Outcome:
    return Outcome(ident, True, "", detail, observed)


def _no(ident: str, reason: str, detail: str, observed: dict) -> Outcome:
    return Outcome(ident, False, reason, detail, observed)


def _evaluations(telemetry) -> list:
    rows = (telemetry or {}).get("evaluations")
    return list(rows) if isinstance(rows, list) else []


def _graded_series(telemetry) -> list:
    rows = (telemetry or {}).get("graded_series")
    return list(rows) if isinstance(rows, list) else []


def first_sustained_crossing(series, target, window):
    """The graded step: the first point below target that stays below across window.

    `series` is a list of (step, loss) pairs on the verifier's own evaluation grid,
    in ascending step order. The window is counted in EVALUATION POINTS the
    verifier scheduled, not in steps, so refining the cadence does not weaken the
    condition. A crossing whose window runs off the end of the measured series is
    not a crossing, because the evidence for the sustained half does not exist.
    """
    rows = list(series)
    for index, (step, loss) in enumerate(rows):
        if not (loss < target):
            continue
        tail = rows[index + 1 : index + 1 + int(window)]
        if len(tail) < int(window):
            return None, index
        if all(value < target for _, value in tail):
            return int(step), index
    return None, -1


# ---------------------------------------------------------------------------
# 1. DIVERGENCE. The submitted algebra disagrees with every pinned record.
# ---------------------------------------------------------------------------


def check_recipe_diverges_from_the_pinned_record_corpus(screen_record, corpus, admin) -> Outcome:
    """Reduction DIVERGENCE.

    The verifier derived a fingerprint from the submission bytes. The corpus
    entries carry fingerprints derived independently from the published record
    recipes. Grading requires those two derivations to DISAGREE for every entry.
    Agreement, structurally and inside the bound proximity floor, is a replay.

    The revision the screen ran under must equal the revision the admin plane
    binds. A clearance computed against an older corpus revision and carried
    forward is refused here rather than honoured, which is the whole point of the
    silent corpus-revision drift.
    """
    ident = "recipe_diverges_from_the_pinned_record_corpus"
    reason = "recipe-replays-published-record"
    record = screen_record or {}
    bound_revision = str((admin or {}).get("record_corpus_revision", ""))
    seen_revision = str(record.get("corpus_revision", ""))
    observed = {
        "structural_digest": record.get("structural_digest"),
        "replays": list(record.get("replays") or []),
        "floor_applied": record.get("floor_applied"),
        "corpus_revision_bound": bound_revision,
        "corpus_revision_screened": seen_revision,
        "corpus_entries": len(corpus or []),
    }
    if not record or record.get("structural_digest") in (None, ""):
        return _no(
            ident,
            "submission-fingerprint-underivable",
            "no fingerprint was derived from the submission bytes",
            observed,
        )
    if not corpus:
        return _no(
            ident,
            "record-corpus-unresolvable",
            "the pinned corpus is empty, so no divergence can be established",
            observed,
        )
    if seen_revision != bound_revision:
        return _no(
            ident,
            reason,
            "the screen ran against corpus revision "
            + repr(seen_revision)
            + " while the bound revision is "
            + repr(bound_revision),
            observed,
        )
    if record.get("replays"):
        return _no(
            ident,
            reason,
            "the submitted algebra agrees with pinned entries " + ", ".join(record["replays"]),
            observed,
        )
    return _ok(ident, "the submitted algebra diverges from every pinned entry", observed)


# ---------------------------------------------------------------------------
# 2. ORDERING. The screen ran before any seed, at zero accelerator cost.
# ---------------------------------------------------------------------------


def check_fingerprint_screen_precedes_any_seed(telemetry) -> Outcome:
    """Reduction ORDERING.

    Sequence 0 is the screen. The first seed launch is strictly after it, and the
    accelerator-second counter the harness stamped at the screen reads exactly
    zero. A screen that ran after a seed did run, but it did not save the
    accelerator time the archetype exists to save, so it is graded as not having
    run in the bound position.
    """
    ident = "fingerprint_screen_precedes_any_seed"
    reason = "screen-ran-after-accelerator-time-was-spent"
    events = (telemetry or {}).get("events")
    events = list(events) if isinstance(events, list) else []
    screens = [row for row in events if row.get("kind") == "fingerprint-screen"]
    launches = [row for row in events if row.get("kind") == "seed-launch"]
    observed = {
        "event_count": len(events),
        "screen_sequences": [row.get("seq") for row in screens],
        "launch_sequences": [row.get("seq") for row in launches],
        "accelerator_seconds_at_screen": [row.get("accelerator_seconds_spent_before") for row in screens],
    }
    if len(screens) != 1:
        return _no(ident, reason, "expected exactly one fingerprint-screen event, saw " + str(len(screens)), observed)
    screen_seq = screens[0].get("seq")
    spent = screens[0].get("accelerator_seconds_spent_before")
    if screen_seq != 0:
        return _no(ident, reason, "the screen sits at sequence " + repr(screen_seq) + " rather than 0", observed)
    if not isinstance(spent, (int, float)) or float(spent) != 0.0:
        return _no(ident, reason, "accelerator seconds at the screen read " + repr(spent), observed)
    if not launches:
        return _ok(ident, "the screen ran at sequence 0 and no seed was launched at all", observed)
    if min(int(row.get("seq", -1)) for row in launches) <= 0:
        return _no(ident, reason, "a seed launch shares or precedes the screen's sequence", observed)
    return _ok(ident, "the screen ran at sequence 0, before every seed launch, at zero accelerator cost", observed)


# ---------------------------------------------------------------------------
# 3. VALUE. The graded series is the verifier's own unsmoothed recomputation.
# ---------------------------------------------------------------------------


def check_graded_loss_recomputed_unsmoothed_by_verifier(telemetry, admin) -> Outcome:
    """Reduction VALUE.

    The grader assembled a graded series. This checker recomputes that series
    from the verifier's own raw per-step evaluations and requires equality to the
    last bit, and requires the provenance token on the graded path to equal the
    one the admin plane binds. A submission may EMA-blend its own readout for its
    own use; the moment a submission-reported number equals the graded number
    while the raw number differs, the graded path has been shaped and this checker
    refuses it.
    """
    ident = "graded_loss_recomputed_unsmoothed_by_verifier"
    reason = "readout-smoothed-on-the-graded-path"
    rows = _evaluations(telemetry)
    graded = _graded_series(telemetry)
    bound_provenance = str((admin or {}).get("graded_loss_provenance", ""))
    seen_provenance = str((telemetry or {}).get("graded_loss_provenance", ""))
    recomputed = [[int(row.get("step", -1)), float(row.get("verifier_raw_loss"))] for row in rows if "verifier_raw_loss" in row]
    observed = {
        "evaluation_points": len(rows),
        "graded_points": len(graded),
        "provenance_bound": bound_provenance,
        "provenance_seen": seen_provenance,
        "recomputed_head": recomputed[:3],
        "graded_head": [list(item) for item in graded[:3]],
    }
    if seen_provenance != bound_provenance:
        return _no(
            ident,
            reason,
            "the graded path declares provenance " + repr(seen_provenance) + ", bound is " + repr(bound_provenance),
            observed,
        )
    if len(recomputed) != len(rows) or not rows:
        return _no(ident, reason, "at least one evaluation carries no verifier-computed raw loss", observed)
    if [list(item) for item in graded] != recomputed:
        return _no(ident, reason, "the graded series is not the verifier's own raw recomputation", observed)
    shaped = []
    for row in rows:
        raw = float(row.get("verifier_raw_loss"))
        reported = row.get("submission_reported_loss")
        if reported is None:
            continue
        if float(reported) != raw and any(math.isclose(float(reported), value, rel_tol=0.0, abs_tol=0.0) for _, value in graded):
            shaped.append(int(row.get("step", -1)))
    if shaped:
        return _no(ident, reason, "submission-reported values reached the graded series at steps " + str(shaped), observed)
    return _ok(ident, "every graded value is the verifier's own unsmoothed evaluation", observed)


# ---------------------------------------------------------------------------
# 4. INVARIANT. The crossing holds across the verifier-scheduled window.
# ---------------------------------------------------------------------------


def check_crossing_sustained_across_verifier_scheduled_points(telemetry, anchors, admin) -> Outcome:
    """Reduction INVARIANT.

    The property that must hold at every point of the window: the verifier's own
    raw loss stays below the target. One favourable evaluation is a sample, not a
    crossing, and a series whose window runs past its last measured point has not
    established the invariant rather than having established it weakly.
    """
    ident = "crossing_sustained_across_verifier_scheduled_points"
    reason = "crossing-not-sustained"
    target = float((anchors or {}).get("target_validation_loss"))
    window = int((admin or {}).get("sustain_window_points", 0))
    cadence = int((admin or {}).get("evaluation_cadence_steps", 0))
    graded = _graded_series(telemetry)
    steps = [int(item[0]) for item in graded]
    gaps = sorted({steps[i + 1] - steps[i] for i in range(len(steps) - 1)})
    step, index = first_sustained_crossing(graded, target, window)
    dips = [int(item[0]) for item in graded if float(item[1]) < target]
    observed = {
        "target": target,
        "sustain_window_points": window,
        "cadence_bound": cadence,
        "cadence_observed": gaps,
        "graded_step": step,
        "points_below_target": dips,
        "series_length": len(graded),
    }
    if window <= 0:
        return _no(ident, reason, "the bound sustain window is not a positive number of evaluation points", observed)
    if not graded:
        return _no(ident, reason, "the graded series is empty, so no crossing can be sustained", observed)
    if gaps and gaps != [cadence]:
        return _no(ident, reason, "the evaluation grid is not the verifier's bound cadence: " + str(gaps), observed)
    if step is None:
        if dips:
            return _no(ident, reason, "the target was touched at " + str(dips) + " and never held across the window", observed)
        return _no(ident, reason, "the target was never reached on the verifier's own grid", observed)
    return _ok(ident, "the target held at step " + str(step) + " and across " + str(window) + " subsequent points", observed)


# ---------------------------------------------------------------------------
# 5. ABSENCE. No run halted before the window closed is graded as a crossing.
# ---------------------------------------------------------------------------


def check_no_early_stop_before_the_sustain_window_closed(telemetry, anchors, admin) -> Outcome:
    """Reduction ABSENCE.

    The set this checker requires to be empty is the set of graded runs that
    halted before the verifier's sustain window closed. Membership is decided two
    ways, and either one is enough: the recorded halt cause names a stop taken on
    a favourable evaluation, or the recorded halt step falls short of the step at
    which the window closes. A run in that set is graded as NOT having crossed and
    scores zero with this reason, never as an absent result.
    """
    ident = "no_early_stop_before_the_sustain_window_closed"
    reason = "run-halted-before-sustain-window-closed"
    target = float((anchors or {}).get("target_validation_loss"))
    window = int((admin or {}).get("sustain_window_points", 0))
    cadence = int((admin or {}).get("evaluation_cadence_steps", 0))
    run = (telemetry or {}).get("run") or {}
    halt_step = run.get("halt_step")
    halt_cause = str(run.get("halt_cause", ""))
    graded = _graded_series(telemetry)
    step, _ = first_sustained_crossing(graded, target, window)
    closes_at = None if step is None else int(step) + window * cadence
    observed = {
        "halt_step": halt_step,
        "halt_cause": halt_cause,
        "graded_step": step,
        "window_closes_at_step": closes_at,
        "early_stop_causes": ["submission-early-stop-on-favourable-eval", "submission-halted-on-target"],
    }
    if halt_cause in observed["early_stop_causes"]:
        return _no(ident, reason, "the run halted on a favourable evaluation: " + halt_cause, observed)
    if step is None:
        return _ok(ident, "no crossing is graded, so no early-stopped crossing is in the set", observed)
    if not isinstance(halt_step, int):
        return _no(ident, reason, "the harness recorded no integer halt step for a graded crossing", observed)
    if halt_step < closes_at:
        return _no(
            ident,
            reason,
            "the run halted at step " + str(halt_step) + " while the window closes at " + str(closes_at),
            observed,
        )
    return _ok(ident, "the run ran past the point at which the sustain window closed", observed)


# ---------------------------------------------------------------------------
# 6. VALUE. Graded weights are the harness's own checkpoint at that step.
# ---------------------------------------------------------------------------


def check_graded_weights_are_harness_owned_at_the_graded_step(telemetry, anchors, admin) -> Outcome:
    """Reduction VALUE.

    Two digests are compared: the one the verifier took of the tensor bytes it
    actually loaded, and the one the harness recorded when it wrote that
    checkpoint into its own run directory. They must be equal at the graded step
    and at every point of the sustain window, and the custody token must equal the
    one the admin plane binds. A checkpoint the submission selected fails the
    first comparison; a custody arrangement carried over from a previous attempt
    fails the second.
    """
    ident = "graded_weights_are_harness_owned_at_the_graded_step"
    reason = "graded-weights-not-harness-owned"
    target = float((anchors or {}).get("target_validation_loss"))
    window = int((admin or {}).get("sustain_window_points", 0))
    bound_custody = str((admin or {}).get("checkpoint_custody", ""))
    seen_custody = str((telemetry or {}).get("checkpoint_custody", ""))
    rows = _evaluations(telemetry)
    graded = _graded_series(telemetry)
    step, index = first_sustained_crossing(graded, target, window)
    scope = rows if step is None else rows[index : index + 1 + window]
    mismatched = [
        int(row.get("step", -1))
        for row in scope
        if row.get("weights_origin") != "harness-checkpoint"
        or row.get("weights_digest") != row.get("harness_checkpoint_digest")
    ]
    observed = {
        "custody_bound": bound_custody,
        "custody_seen": seen_custody,
        "graded_step": step,
        "points_in_scope": [int(row.get("step", -1)) for row in scope],
        "mismatched_steps": mismatched,
    }
    if seen_custody != bound_custody:
        return _no(
            ident,
            reason,
            "checkpoint custody reads " + repr(seen_custody) + ", bound is " + repr(bound_custody),
            observed,
        )
    if not scope:
        return _no(ident, reason, "no evaluation point carries weights to compare", observed)
    if mismatched:
        return _no(ident, reason, "weights at steps " + str(mismatched) + " are not the harness checkpoint", observed)
    return _ok(ident, "every graded point evaluated the harness-written checkpoint for that step", observed)


# ---------------------------------------------------------------------------
# 7. EFFECT. Launching the recipe changed verifier-owned live state.
# ---------------------------------------------------------------------------


def check_graded_run_produced_verifier_owned_telemetry(telemetry, admin) -> Outcome:
    """Reduction EFFECT.

    The effect required is a change in the verifier's own live state: a run
    directory the verifier created, handed to a child process group, and read back
    after reaping it, now holding a non-empty evaluation series for every drawn
    seed. A zero count is never read as positive evidence, because an empty series
    is equally consistent with a clean run that measured nothing and with a
    harness that never started.
    """
    ident = "graded_run_produced_verifier_owned_telemetry"
    reason = "graded-run-produced-no-telemetry"
    run = (telemetry or {}).get("run") or {}
    seeds = run.get("seeds") or []
    rows = _evaluations(telemetry)
    expected_seeds = int((admin or {}).get("verifier_run_seed_count", 0))
    per_seed = (telemetry or {}).get("evaluations_per_seed") or {}
    empty = sorted(str(key) for key, value in per_seed.items() if not value)
    observed = {
        "launched": bool(run.get("launched")),
        "seeds": list(seeds),
        "seed_count_expected": expected_seeds,
        "evaluation_count": len(rows),
        "run_directory_owner": run.get("run_directory_owner"),
        "seeds_with_empty_series": empty,
    }
    if not run.get("launched"):
        return _no(ident, reason, "the harness never reported launching the graded run", observed)
    if run.get("run_directory_owner") != "verifier":
        return _no(ident, reason, "the run directory is not verifier-owned", observed)
    if len(seeds) < 1 or len(seeds) != expected_seeds:
        return _no(ident, reason, "drew " + str(len(seeds)) + " seeds against a bound count of " + str(expected_seeds), observed)
    if len(rows) < 1:
        return _no(ident, reason, "the evaluation series is empty, which is never positive evidence of a run", observed)
    if empty:
        return _no(ident, reason, "these seeds produced no series at all: " + ", ".join(empty), observed)
    return _ok(ident, "the run produced " + str(len(rows)) + " verifier-owned evaluation points", observed)


# ---------------------------------------------------------------------------
# 8. INVARIANT. Every frozen axis is where the bundle bound it.
# ---------------------------------------------------------------------------


def check_frozen_axes_unmoved_by_the_graded_run(telemetry, frozen) -> Outcome:
    """Reduction INVARIANT.

    Four properties must hold throughout the run: the dataset digest, the batch
    size, the architecture digest, and exactly one forward-backward pass per
    optimizer step. Every shortcut this family exists to close, more tokens, a
    wider model, a bigger batch, a second pass charged as one step, moves one of
    them, so they are compared as one invariant rather than checked as four
    conveniences.
    """
    ident = "frozen_axes_unmoved_by_the_graded_run"
    reason = "frozen-axis-moved"
    seen = (telemetry or {}).get("frozen_axes") or {}
    bound = frozen or {}
    keys = ("dataset_digest", "batch_size", "arch_digest", "fwd_bwd_per_step")
    moved = [key for key in keys if seen.get(key) != bound.get(key)]
    observed = {
        "bound": {key: bound.get(key) for key in keys},
        "observed": {key: seen.get(key) for key in keys},
        "moved": moved,
    }
    if not seen:
        return _no(ident, reason, "the harness recorded no frozen-axis observation at all", observed)
    if moved:
        return _no(ident, reason, "these frozen axes moved during the graded run: " + ", ".join(moved), observed)
    return _ok(ident, "every frozen axis is byte-identical to the bound record", observed)


# ---------------------------------------------------------------------------
# The declared order. grade.py walks this and nothing else.
# ---------------------------------------------------------------------------

# The order is itself load-bearing. The screen sits first so a replay is refused
# on bytes alone. The early-stop gate sits ahead of the sustained-crossing gate
# because a run that halted inside the window fails both, and attributing that
# zero to the halt is the more specific and more actionable of the two reasons.
ORDER = (
    "recipe_diverges_from_the_pinned_record_corpus",
    "fingerprint_screen_precedes_any_seed",
    "graded_run_produced_verifier_owned_telemetry",
    "frozen_axes_unmoved_by_the_graded_run",
    "graded_loss_recomputed_unsmoothed_by_verifier",
    "graded_weights_are_harness_owned_at_the_graded_step",
    "no_early_stop_before_the_sustain_window_closed",
    "crossing_sustained_across_verifier_scheduled_points",
)


def digest_of(path) -> str:
    """sha256 of a file the CALLER named. No path is discovered inside this module."""
    location = Path(path)
    if not location.is_file():
        return ""
    return hashlib.sha256(location.read_bytes()).hexdigest()


def as_payload(outcomes) -> str:
    """Canonical bytes for a list of outcomes, so two graders can be compared."""
    return json.dumps([row.as_dict() for row in outcomes], sort_keys=True, separators=(",", ":"))


def typed_ids() -> "list[Any]":
    """The declared identity list, typed so the typing import is load-bearing."""
    return list(ORDER)
