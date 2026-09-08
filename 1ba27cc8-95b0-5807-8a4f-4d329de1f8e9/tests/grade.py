#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import sys
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import checkers  # noqa: E402
import score  # noqa: E402
import session_driver  # noqa: E402

from checkers import (  # noqa: E402
    check_bound_envelope_respected,
    check_carried_summary_is_current,
    check_crossing_sustained_across_scheduled_evaluations,
    check_early_halt_yields_no_crossing,
    check_evaluated_weights_owned_by_the_harness,
    check_frozen_axes_untouched,
    check_graded_crossing_recomputed_unsmoothed,
    check_reported_crossing_reconciles_with_verifier_measurement,
)

SUBMISSION = pathlib.Path(os.environ.get("BIA_SUBMISSION", "/workspace/refine.py"))

def session_document() -> dict:
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
    final_loss = checkers.best_final_loss(session)
    anchors = session["anchors"]
    payload = {
        "graded_step": metric,
        "consolidation_indices": indices,
        "baseline": int(anchors["baseline_metric"]),
        "target": int(anchors["target_metric"]),
        "target_loss": float(anchors["target_loss"]),
        "final_loss": final_loss,
        "no_progress_loss": score.NO_PROGRESS_LOSS,
        "attempts": len(session.get("attempts") or []),
        "final_selection": session["bindings"]["final_selection"],
    }

    blocking = failed
    if blocking:
        return {"score": 0.0, "reason": blocking[0]["reason"], "metric": payload,
                "checkers": report}

    value = score.anchored(metric, anchors["baseline_metric"], anchors["target_metric"],
                            final_loss=final_loss, target_loss=anchors["target_loss"])
    if metric is not None:
        reason = "consolidated-crossing-graded"
    elif value > 0.0:
        reason = "loss-band-graded-below-crossing"
    else:
        reason = "no-improvement-over-baseline"
    return {"score": value, "reason": reason, "metric": payload, "checkers": report}

def main() -> int:
    if not os.environ.get("BIA_SESSION_DOC") and not SUBMISSION.is_file():
        score.write(0.0, "submission-missing", {"graded_step": None, "submission": str(SUBMISSION)})
        return 0
    outcome = grade(session_document())
    score.write(outcome["score"], outcome["reason"], outcome["metric"], outcome["checkers"])
    print(json.dumps({"score": outcome["score"], "reason": outcome["reason"],
                      "metric": outcome["metric"]}, sort_keys=True))
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        score.write(0.0, "grader-internal-error", {"graded_step": None})
        raise SystemExit(0)
