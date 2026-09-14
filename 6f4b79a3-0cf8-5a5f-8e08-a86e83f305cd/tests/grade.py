#!/usr/bin/env python3
"""The gate chain for slot OER-01. Reaches every checker; imports no submission.

Order is load-bearing and is graded by a checker of its own. The fingerprint
screen runs FIRST, over bytes alone, before the harness is asked for a single
accelerator second. A replay therefore costs nothing to refuse. Only when the
screen clears does this module ask the harness for telemetry, and only then do the
remaining seven checkers run.

This module imports the checkers and the reference-independent fingerprint
canonicaliser. It NEVER imports the submission. The submission is bytes this
process reads and hands to tests/runner.py, which launches it as a separate
process group in a fresh scratch directory and reaps it. That boundary is the
reason no submitted code can reach the objects this file grades with.

Every exit path writes /logs/verifier/score.json. tests/test.sh owns the reward
contract and writes both bound reward artifacts from an EXIT trap, so an abort in
this file still produces an attributed zero rather than silence.
"""

import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fingerprint  # noqa: E402
import runner  # noqa: E402
from checkers import (  # noqa: E402
    ORDER,
    check_crossing_sustained_across_verifier_scheduled_points,
    check_fingerprint_screen_precedes_any_seed,
    check_frozen_axes_unmoved_by_the_graded_run,
    check_graded_loss_recomputed_unsmoothed_by_verifier,
    check_graded_run_produced_verifier_owned_telemetry,
    check_graded_weights_are_harness_owned_at_the_graded_step,
    check_no_early_stop_before_the_sustain_window_closed,
    check_recipe_diverges_from_the_pinned_record_corpus,
    first_sustained_crossing,
)

SCORE_PATH = os.environ.get("SCORE_PATH", "/logs/verifier/score.json")
REPORT_PATH = os.environ.get("BIA_REPORT", "/logs/verifier/report.json")
SUBMISSION = os.environ.get("BIA_SUBMISSION", "/workspace/submission/recipe.py")
TELEMETRY = os.environ.get("BIA_TELEMETRY", "/logs/verifier/telemetry.json")
RUBRICS = str(HERE / "rubrics.jsonl")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def anchors():
    return load_json(HERE / "anchors.json")


def corpus():
    return load_json(HERE / "corpus.json")


def admin_plane(bound):
    """The verifier-owned admin plane. Read from the bundle, never from the run."""
    return dict(bound.get("admin_plane") or {})


def frozen_axes(bound):
    return dict(bound.get("frozen_axes") or {})


def clip(value):
    number = float(value)
    if number != number:
        return 0.0
    return min(max(number, 0.0), 1.0)


def reward_for(graded_step, bound):
    """The bound lower-is-better formula. Reaching or beating the target scores 1.0."""
    baseline = float(bound["baseline_metric"])
    target = float(bound["target_metric"])
    if graded_step is None:
        return 0.0
    raw = (baseline - float(graded_step)) / (baseline - target)
    return clip(raw)


def screen_submission(path, bound):
    """The record-displacement screen. Bytes only, before any accelerator second."""
    text = ""
    location = Path(path)
    if location.is_file():
        text = location.read_text(encoding="utf-8", errors="replace")
    recipe = fingerprint.recipe_from_source(text)
    plane = admin_plane(bound)
    record = fingerprint.screen(recipe, corpus().get("entries") or [], plane.get("fingerprint_proximity_floor"))
    record["corpus_revision"] = str(corpus().get("revision", ""))
    record["submission_sha256"] = runner.sha256_of(location)
    record["recipe_readable"] = recipe is not None
    return record


def rubric_surface_declared():
    """True when this bundle declares a trajectory rubric surface at all."""
    path = Path(RUBRICS)
    if not path.is_file():
        return 0
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


def run_checkers(telemetry, screen_record, bound):
    """Every declared checker, in the declared order, over live verifier state."""
    plane = admin_plane(bound)
    frozen = frozen_axes(bound)
    anchor = bound
    by_id = {
        "recipe_diverges_from_the_pinned_record_corpus": lambda: check_recipe_diverges_from_the_pinned_record_corpus(
            screen_record, corpus().get("entries") or [], plane
        ),
        "fingerprint_screen_precedes_any_seed": lambda: check_fingerprint_screen_precedes_any_seed(telemetry),
        "graded_run_produced_verifier_owned_telemetry": lambda: check_graded_run_produced_verifier_owned_telemetry(
            telemetry, plane
        ),
        "frozen_axes_unmoved_by_the_graded_run": lambda: check_frozen_axes_unmoved_by_the_graded_run(telemetry, frozen),
        "graded_loss_recomputed_unsmoothed_by_verifier": lambda: check_graded_loss_recomputed_unsmoothed_by_verifier(
            telemetry, plane
        ),
        "graded_weights_are_harness_owned_at_the_graded_step": lambda: check_graded_weights_are_harness_owned_at_the_graded_step(
            telemetry, anchor, plane
        ),
        "crossing_sustained_across_verifier_scheduled_points": lambda: check_crossing_sustained_across_verifier_scheduled_points(
            telemetry, anchor, plane
        ),
        "no_early_stop_before_the_sustain_window_closed": lambda: check_no_early_stop_before_the_sustain_window_closed(
            telemetry, anchor, plane
        ),
    }
    return [by_id[ident]() for ident in ORDER]


def grade(telemetry, screen_record, bound):
    """The whole gate chain as one pure function. adequacy.py drives exactly this.

    required_pass: every declared checker is required, so one false verdict forces
    the score to exactly 0.0. The zero is attributed to the FIRST gate that refused
    in the declared order, and every other refusing gate is carried alongside under
    gate.all_failed_reasons, so a reader sees both the specific cause and the full
    picture. Only when every gate passes does the graded scalar become the clipped
    anchor-gap closure, which is a float and never a pass flag.
    """
    plane = admin_plane(bound)
    outcomes = run_checkers(telemetry, screen_record, bound)
    failed = [row for row in outcomes if not row.passed]
    graded_step, _ = first_sustained_crossing(
        list((telemetry or {}).get("graded_series") or []),
        float(bound["target_validation_loss"]),
        int(plane.get("sustain_window_points", 0)),
    )
    payload = {
        "schema": "bia.oer01.score/v1",
        "reward": 0.0,
        "reason": "",
        "metric": {
            "graded_step": graded_step,
            "baseline": int(bound["baseline_metric"]),
            "target": int(bound["target_metric"]),
            "target_validation_loss": float(bound["target_validation_loss"]),
            "direction": "lower is better",
        },
        "checkers": [row.as_dict() for row in outcomes],
        "gate": {
            "aggregation": "required_pass",
            "required": len(outcomes),
            "failed": [row.ident for row in failed],
            "threshold_met": False,
            "pass_threshold": float(bound["pass_threshold"]),
            "rubric_lines_declared": rubric_surface_declared(),
        },
    }
    if failed:
        payload["reason"] = failed[0].zero_reason
        payload["gate"]["all_failed_reasons"] = [row.zero_reason for row in failed]
        return payload
    if graded_step is None:
        payload["reason"] = "crossing-not-sustained"
        return payload
    payload["reward"] = reward_for(graded_step, bound)
    payload["reason"] = "crossing-sustained-and-recipe-displaced" if payload["reward"] > 0.0 else "no-improvement-over-baseline"
    payload["gate"]["threshold_met"] = payload["reward"] >= float(bound["pass_threshold"])
    return payload


def emit(payload):
    """Write the score document. The reward artifacts belong to tests/test.sh."""
    for path in (SCORE_PATH, REPORT_PATH):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"reward": payload["reward"], "reason": payload["reason"]}, sort_keys=True))


def main():
    bound = anchors()
    try:
        screen_record = screen_submission(SUBMISSION, bound)
    except Exception:
        traceback.print_exc()
        emit(
            {
                "schema": "bia.oer01.score/v1",
                "reward": 0.0,
                "reason": "recipe-replays-published-record",
                "metric": {"graded_step": None, "baseline": int(bound["baseline_metric"]), "target": int(bound["target_metric"])},
                "checkers": [],
                "gate": {"aggregation": "required_pass", "failed": ["recipe_diverges_from_the_pinned_record_corpus"]},
            }
        )
        return 0
    if not screen_record.get("recipe_readable"):
        # Refused on bytes alone at zero accelerator cost, like the replay path
        # below, but attributed apart from it: no recipe was read, so nothing was
        # compared. Calling this a replay asserts a corpus agreement never observed.
        emit(
            {
                "schema": "bia.oer01.score/v1",
                "reward": 0.0,
                "reason": "recipe-unreadable",
                "metric": {"graded_step": None, "baseline": int(bound["baseline_metric"]), "target": int(bound["target_metric"])},
                "checkers": [],
                "gate": {
                    "aggregation": "required_pass",
                    "failed": ["recipe_diverges_from_the_pinned_record_corpus"],
                    "accelerator_seconds_spent": 0.0,
                    "replays": [],
                },
            }
        )
        return 0
    if screen_record.get("replays"):
        # Refused on bytes alone. No seed has executed, so this costs zero
        # accelerator time, which is the whole point of running the screen first.
        emit(
            {
                "schema": "bia.oer01.score/v1",
                "reward": 0.0,
                "reason": "recipe-replays-published-record",
                "metric": {"graded_step": None, "baseline": int(bound["baseline_metric"]), "target": int(bound["target_metric"])},
                "checkers": [],
                "gate": {
                    "aggregation": "required_pass",
                    "failed": ["recipe_diverges_from_the_pinned_record_corpus"],
                    "accelerator_seconds_spent": 0.0,
                    "replays": list(screen_record.get("replays") or []),
                },
            }
        )
        return 0
    telemetry = runner.telemetry(TELEMETRY, SUBMISSION, screen_record, admin_plane(bound), frozen_axes(bound))
    emit(grade(telemetry, screen_record, bound))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # pragma: no cover - the attributed-crash path
        traceback.print_exc()
        emit(
            {
                "schema": "bia.oer01.score/v1",
                "reward": 0.0,
                "reason": "grader-internal-error",
                "metric": {"graded_step": None, "baseline": 3250, "target": 2690},
                "checkers": [],
                "gate": {"aggregation": "required_pass", "failed": ["grader"]},
            }
        )
        raise SystemExit(0)
