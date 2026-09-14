"""Drive the session, run every checker, and stage the score document.

This module imports the checkers and the frozen reference surface. It never
imports the submission: the submission is run out of process by tests/runner.py,
which copies it alone into a fresh temporary directory and kills its process
group on the way out.

Grading is a gate chain followed by a continuous magnitude. Every required
checker is a gate, and a failed gate scores exactly 0.0 carrying that checker's
machine-readable reason. When every gate holds, the reward is the bound
lower-is-better formula over the multi-seed mean, clipped to the closed
interval, with a continuous taper below the bound separation margin so the score
never snaps at the margin.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lab  # noqa: E402
import runner  # noqa: E402
from checkers import (  # noqa: E402
    Session,
    check_attempt_sequence_contiguous,
    check_carried_state_monotone,
    check_crossing_sustained,
    check_early_stop_is_not_a_crossing,
    check_graded_readout_unsmoothed,
    check_graded_weights_are_run_weights,
    check_multi_seed_separation_established,
    check_reallocation_follows_flattening,
    check_reference_operating_point_matches_final_state,
    check_session_holds_attempt_budget,
    check_target_band_reached_by_session,
    load_session,
    run_all,
)

DEFAULT_SUBMISSION = "/app/submission.py"
DEFAULT_STAGE = "/tmp/oer06-stage.json"
DEFAULT_SESSION_LOG = "/logs/verifier/session.jsonl"

GATES = (
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


def metric_block(session: Session) -> dict:
    header = session.header
    footer = session.footer
    graded = session.graded or {}
    return {
        "graded_step": graded.get("graded_crossing_step"),
        "graded_attempt": graded.get("index"),
        "attempts_recorded": len(session.attempts),
        "attempt_terminator": header.get("terminator"),
        "multi_seed_mean_steps": footer.get("best_mean_steps"),
        "per_seed_steps": footer.get("best_per_seed_steps") or {},
        "baseline_operating_point": header.get("baseline_control_mean"),
        "target_operating_point": header.get("target_control_mean"),
        "separation_margin": header.get("separation_margin"),
        "flattening_onset_attempt": footer.get("flattening_onset_attempt"),
        "post_flattening_attempts_on_flattened_axis": footer.get(
            "post_flattening_attempts_on_flattened_axis"
        ),
        "first_reallocation_attempt": footer.get("first_reallocation_attempt"),
        "baseline_metric": None,
        "target_metric": None,
        "anchors_state": "absent",
        "anchors_gap": "gap-oer-per-family-anchors-unmeasured",
    }


def score_session(session: Session) -> dict:
    """One score document: the reward, the reason it is that number, and the metric."""
    verdicts = run_all(session)
    rows = [item.as_dict() for item in verdicts]
    failed = [item for item in verdicts if not item.passed]
    block = metric_block(session)
    if failed:
        return {
            "reward": 0.0,
            "reason": failed[0].zero_reason,
            "reasons": [item.zero_reason for item in failed],
            "metric": block,
            "checkers": rows,
        }
    raw = lab.raw_reward(
        session.header.get("baseline_control_mean"),
        session.header.get("target_control_mean"),
        session.footer.get("best_mean_steps"),
    )
    reward = lab.margin_taper(raw)
    block["raw_ratio"] = raw
    block["margin_tapered"] = raw < lab.SEPARATION_MARGIN
    if reward <= 0.0:
        return {
            "reward": 0.0,
            "reason": "no-measured-improvement",
            "reasons": ["no-measured-improvement"],
            "metric": block,
            "checkers": rows,
        }
    return {
        "reward": round(reward, 6),
        "reason": "graded-crossing-sustained-across-held-budget",
        "reasons": [],
        "metric": block,
        "checkers": rows,
    }


def grade(submission: Path, session_log: Path) -> dict:
    runner.run_session(submission, session_log)
    return score_session(load_session(session_log))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="grade one OER-06 session")
    parser.add_argument("--submission", default=DEFAULT_SUBMISSION)
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--session-log", default=DEFAULT_SESSION_LOG)
    args = parser.parse_args(argv)

    document = grade(Path(args.submission), Path(args.session_log))
    stage = Path(args.stage)
    stage.parent.mkdir(parents=True, exist_ok=True)
    stage.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"reward": document["reward"], "reason": document["reason"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
