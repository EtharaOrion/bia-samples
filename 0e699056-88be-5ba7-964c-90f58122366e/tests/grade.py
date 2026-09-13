#!/usr/bin/env python3
"""The gate chain for slot OER-10: run every checker, gate, then score the ladder.

This process imports the checkers and reads the reference bytes. It NEVER imports the
submission. The submission ran earlier, alone, in a fresh directory, as its own session
leader, under tests/runner.py, and the only thing that crosses back is the telemetry
record this verifier's own process wrote.

Two stages, and they are not the same thing:

  gate    every required checker must pass. This is `required_pass` over the closed
          checker set, and it can only zero a score. A failing gate emits that checker's
          machine-readable reason and stops; nothing downstream can raise it back.
  ladder  the continuous reward, computed from the graded validation loss against the two
          run-local measured ends the harness itself produced. This is what makes the
          reward a float on [0, 1] rather than a pass or fail bit.

The reward carrier is /logs/verifier/reward.txt, a bare float, which is the path the live
verifier binds. The reason and the metric block travel in /logs/verifier/score.json, and
both are written from the same call so the two can never disagree.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import checkers  # noqa: E402

BUNDLE = HERE.parent
REFERENCE = BUNDLE / "solution" / "reference.py"

REWARD_PATH = "/logs/verifier/reward.txt"
SCORE_DOCUMENT_PATH = "/logs/verifier/score.json"
TELEMETRY_PATH = "/logs/verifier/telemetry.json"

REASON_OK = "graded-loss-established"
REASON_ABSENT = "telemetry-absent"
REASON_LADDER = "ladder-unresolved"

# The selectors this carrier reaches, named literally so an outside reader can confirm
# from bytes that every checker declared in tests/checkers.yaml is actually invoked here.
SELECTORS = (
    "check_recipe_not_a_replay",
    "check_fingerprint_screen_precedes_training",
    "check_graded_loss_is_verifier_computed",
    "check_graded_readout_unsmoothed",
    "check_loss_sustained_at_scheduled_points",
    "check_early_stop_not_a_result",
    "check_evaluated_weights_harness_owned",
    "check_token_budget_respected_as_fed",
    "check_eval_split_not_trained_on",
    "check_control_arm_displaced",
)


def gate(outcomes):
    """required_pass over the closed checker set. Returns (passed, reason)."""
    rows = list(outcomes or [])
    if not rows:
        return False, REASON_ABSENT
    for row in rows:
        if not row.passed:
            return False, row.reason
    return True, REASON_OK


def ladder(telemetry):
    """The continuous score from the graded loss against the two run-local ends."""
    block = telemetry.get("ladder") if isinstance(telemetry, dict) else None
    if not isinstance(block, dict):
        return None, None, None
    try:
        control = float(block.get("control_arm_loss"))
        floor = float(block.get("reference_floor_loss"))
    except (TypeError, ValueError):
        return None, None, None
    view = checkers.HarnessView.from_mapping(telemetry)
    graded = view.graded_loss()
    if graded is None or control <= floor:
        return None, control, floor
    raw = (control - graded) / (control - floor)
    return min(max(raw, 0.0), 1.0), control, floor


def score_document(telemetry):
    """One score document. Every zero carries a machine-readable reason."""
    view = checkers.HarnessView.from_mapping(telemetry)
    if not view.present:
        return {
            "reward": 0.0,
            "reason": REASON_ABSENT,
            "metric": _metric(None, None, None, None),
            "checkers": [],
        }
    outcomes = checkers.run_all(view)
    rows = [row.as_dict() for row in outcomes]
    passed, reason = gate(outcomes)
    value, control, floor = ladder(telemetry)
    graded = view.graded_loss()
    if not passed:
        return {
            "reward": 0.0,
            "reason": reason,
            "metric": _metric(graded, control, floor, view.bound_step()),
            "checkers": rows,
        }
    if value is None:
        return {
            "reward": 0.0,
            "reason": REASON_LADDER,
            "metric": _metric(graded, control, floor, view.bound_step()),
            "checkers": rows,
        }
    return {
        "reward": round(value, 12),
        "reason": REASON_OK,
        "metric": _metric(graded, control, floor, view.bound_step()),
        "checkers": rows,
    }


def _metric(graded, control, floor, step):
    return {
        "quantity": "validation loss of the trained model",
        "direction": "lower",
        "graded_loss": graded,
        "graded_at_step": step,
        "computed_by": "verifier",
        "anchors_state": "absent",
        "baseline_metric": None,
        "target_metric": None,
        "anchors_gap": "gap-oer-per-family-anchors-unmeasured",
        "ladder_state": "run-local-measured",
        "ladder_gap": "gap-oer-10-run-local-ladder-is-not-a-family-anchor",
        "control_arm_loss": control,
        "reference_floor_loss": floor,
        "reference_sha256": _reference_digest(),
    }


def _reference_digest():
    import hashlib

    if not REFERENCE.is_file():
        return None
    return hashlib.sha256(REFERENCE.read_bytes()).hexdigest()


def emit(document, reward_path=REWARD_PATH, score_path=SCORE_DOCUMENT_PATH):
    """Write the bare float and the reason document from one call, so they never disagree."""
    reward = Path(reward_path)
    score = Path(score_path)
    reward.parent.mkdir(parents=True, exist_ok=True)
    score.parent.mkdir(parents=True, exist_ok=True)
    score.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    reward.write_text(repr(float(document["reward"])) + "\n", encoding="utf-8")
    return document


def load_telemetry(path):
    target = Path(path)
    if not target.is_file():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except ValueError:
        return None


def main(argv):
    telemetry_path = argv[0] if argv else TELEMETRY_PATH
    reward_path = argv[1] if len(argv) > 1 else REWARD_PATH
    score_path = argv[2] if len(argv) > 2 else SCORE_DOCUMENT_PATH
    document = score_document(load_telemetry(telemetry_path))
    emit(document, reward_path, score_path)
    print("reward " + repr(document["reward"]) + " reason " + document["reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
