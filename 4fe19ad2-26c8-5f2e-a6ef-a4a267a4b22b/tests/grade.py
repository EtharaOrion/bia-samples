#!/usr/bin/env python3
"""The gate chain and the reward. Imports the checkers; never imports the submission.

Order of operations, and none of it is negotiable:

1. `session_driver.run_session` drives the whole fifty-attempt refinement loop in
   the verifier's own process, launching the submission through `runner.launch`
   as a separate process group. No submission byte is imported here.
2. Every checker in `checkers.CHAIN` runs. There is no short circuit, so the
   score document carries every reason, not just the first.
3. Aggregation is `required_pass` over the ten required checkers. One failure is
   a reward of exactly 0.0 carrying that checker's machine-readable reason.
4. Only when every required checker passes does the anchored magnitude apply, and
   the magnitude is computed from the consolidated metric this module recomputes
   through `checkers.consolidation`, never from a number the submission wrote.

A module-level handler converts any unhandled exception into an attributed zero,
so a crashed grader produces a reason rather than an unwritten artifact.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import checkers  # noqa: E402
import reward  # noqa: E402
import session_driver  # noqa: E402

from checkers import (  # noqa: E402  the declared selectors, named where the manifest can see them
    check_bound_envelope_respected,
    check_carried_summary_is_current,
    check_crossing_sustained_across_scheduled_evaluations,
    check_early_halt_yields_no_crossing,
    check_evaluated_weights_owned_by_the_harness,
    check_frozen_axes_untouched,
    check_graded_crossing_recomputed_unsmoothed,
    check_graded_metric_consolidated_over_distinct_recipes,
    check_refinement_frontier_carried_across_iterations,
    check_reported_crossing_reconciles_with_verifier_measurement,
)

SELECTORS = (
    check_graded_crossing_recomputed_unsmoothed,
    check_crossing_sustained_across_scheduled_evaluations,
    check_early_halt_yields_no_crossing,
    check_evaluated_weights_owned_by_the_harness,
    check_graded_metric_consolidated_over_distinct_recipes,
    check_refinement_frontier_carried_across_iterations,
    check_reported_crossing_reconciles_with_verifier_measurement,
    check_bound_envelope_respected,
    check_frozen_axes_untouched,
    check_carried_summary_is_current,
)

SUBMISSION = pathlib.Path(os.environ.get("BIA_SUBMISSION", "/workspace/refine.py"))


def session_document() -> dict:
    """The live session, or a recorded one when the harness hands one over.

    `BIA_SESSION_DOC` is how the checker-adequacy harness and the compiled tests
    drive the SAME checker chain over planted fixtures. It never changes what is
    graded; it only changes which recorded session the chain reads.
    """
    recorded = os.environ.get("BIA_SESSION_DOC")
    if recorded:
        return json.loads(pathlib.Path(recorded).read_text(encoding="utf-8"))
    attempts = os.environ.get("BIA_MAX_ATTEMPTS")
    return session_driver.run_session(
        SUBMISSION, max_attempts=int(attempts) if attempts else None)


def grade(session: dict) -> dict:
    rows = checkers.run_chain(session)
    report = [{"id": name, "passed": bool(verdict.ok), "reason": verdict.reason,
               "detail": verdict.detail} for name, verdict in rows]
    failed = [item for item in report if not item["passed"]]
    metric, indices = checkers.consolidation(session)
    anchors = session["anchors"]
    baseline = anchors.get("baseline_metric")
    target = anchors.get("target_metric")
    payload = {
        "graded_step": metric,
        "consolidation_indices": indices,
        "baseline": None if baseline is None else int(baseline),
        "target": None if target is None else int(target),
        "anchors_state": anchors.get("anchors_state"),
        "baseline_source": anchors.get("baseline_source"),
        "target_source": anchors.get("target_source"),
        "anchor_seconds": anchors.get("seconds"),
        "attempts": len(session.get("attempts") or []),
        "attempts_spent": session.get("attempts_spent"),
        "final_selection": session["bindings"]["final_selection"],
    }
    if failed:
        return {"reward": 0.0, "reason": failed[0]["reason"], "metric": payload, "checkers": report}
    if anchors.get("anchors_state") != "measured":
        # The verifier could not build a scale on this substrate. That is a fact
        # about the run, not about the submission, so it carries the probe's own
        # machine-readable reason rather than being reported as the submission
        # failing to improve. Inventing an endpoint here is exactly the defect
        # this bundle was repaired for.
        payload["anchors_gap"] = anchors.get("anchors_gap")
        return {"reward": 0.0, "reason": anchors.get("anchors_gap") or "anchors-unmeasurable",
                "metric": payload, "checkers": report}
    score = reward.anchored(metric, baseline, target)
    reason = "consolidated-crossing-graded" if score > 0.0 else "no-improvement-over-baseline"
    return {"reward": score, "reason": reason, "metric": payload, "checkers": report}


def main() -> int:
    if not os.environ.get("BIA_SESSION_DOC") and not SUBMISSION.is_file():
        reward.write(0.0, "submission-missing", {"graded_step": None, "submission": str(SUBMISSION)})
        return 0
    outcome = grade(session_document())
    reward.write(outcome["reward"], outcome["reason"], outcome["metric"], outcome["checkers"])
    print(json.dumps({"reward": outcome["reward"], "reason": outcome["reason"],
                      "metric": outcome["metric"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001  a crashed grader still owes an attributed zero
        traceback.print_exc()
        reward.write(0.0, "grader-internal-error", {"graded_step": None})
        raise SystemExit(0)
