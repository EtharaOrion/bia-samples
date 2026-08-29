#!/usr/bin/env python3
"""Reduce the harness telemetry for OER-09 to one float and one reason.

This module imports `checkers` and the `reference`. It NEVER imports the submission. The
submission is executed, in isolation, by tests/runner.py, and the only thing that crosses back
from it is bytes on stdout that nothing here grades.

Reward contract, as the live verifier binds it:

    /logs/verifier/reward.txt   the bare float, the carrier the runtime reads
    /logs/verifier/score.json   the machine-readable reason and the metric block

Both are written from one place so the instrument's binding and the reason carrier can never
disagree.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent

for _path in (str(HERE), str(BUNDLE / "solution")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import checkers  # noqa: E402

REWARD_PATH = Path("/logs/verifier/reward.txt")
SCORE_PATH = Path("/logs/verifier/score.json")
HARNESS_LOGS = Path("/logs/harness")

# Each graded section, and the harness-owned file it is read from. These are the real handles;
# no section is ever assembled from a value the submission printed.
SECTIONS = (
    ("register", "register.json"),
    ("pool", "pool_observed.json"),
    ("claim", "curation_claim.json"),
    ("feed", "feed_ledger.json"),
    ("curation", "curation_ledger.json"),
    ("split", "split_audit.json"),
    ("evaluation", "eval_record.json"),
    ("weights", "weights_ledger.json"),
    ("run", "run_ledger.json"),
)

# Named so verifier.py can see every selector this carrier reaches. The list is walked, not
# just declared: checkers.CHECKERS holds the same nine in the same order.
SELECTORS = (
    "check_graded_loss_unsmoothed",
    "check_graded_weights_harness_owned",
    "check_token_budget_respected_as_fed",
    "check_curated_pool_consumed_by_trainer",
    "check_filter_claim_matches_observed_pool",
    "check_improvement_sustained_across_points",
    "check_eval_split_never_trained_on",
    "check_early_stop_does_not_establish_loss",
    "check_curation_precedes_first_feed",
)

ANCHORS_GAP = "gap-oer-per-family-anchors-unmeasured"


def reference_binding() -> str:
    """The identity of the reference this grader was built against, for the score document."""
    try:
        import reference

        return str(getattr(reference, "REFERENCE_ID", "reference-id-absent"))
    except Exception:
        return "reference-unimportable"


def load_state(logs: Path = HARNESS_LOGS) -> dict:
    """Assemble the graded state from the harness's own records. Absent is absent, not empty."""
    state: dict = {}
    for name, filename in SECTIONS:
        path = logs / filename
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict):
            state[name] = payload
    return state


def evaluate(state: dict) -> dict:
    """The whole grading decision. Pure in `state`; writes nothing."""
    verdicts = checkers.run_all(state)
    rows = [item.as_dict() for item in verdicts]
    failed = [item for item in verdicts if not item.ok]

    evaluation = state.get("evaluation") if isinstance(state.get("evaluation"), dict) else {}
    metric = {
        "quantity": "validation loss of the trained model",
        "direction": "lower",
        "graded_loss": evaluation.get("graded_loss"),
        "control_loss": evaluation.get("control_loss"),
        "reference_loss": evaluation.get("reference_loss"),
        "bound_point": evaluation.get("bound_point"),
        "sustain_points": evaluation.get("sustain_points"),
        "baseline_metric": None,
        "target_metric": None,
        "anchors_state": "absent",
        "anchors_gap": ANCHORS_GAP,
        "formula_bound": "raw = (baseline_metric - agent_metric) / (baseline_metric - target_metric)",
        "formula_operated": "raw = (control_loss - graded_loss) / (control_loss - reference_loss)",
        "clip": "score = min(max(raw, 0.0), 1.0)",
    }

    if not state:
        return {
            "reward": 0.0,
            "reason": checkers.REASON_STATE_UNREADABLE,
            "metric": metric,
            "checkers": rows,
            "reference": reference_binding(),
        }

    if failed:
        # The gate. Every checker is required, so one failure is the whole reward, and it
        # carries that checker's own machine-readable reason rather than a bare zero.
        return {
            "reward": 0.0,
            "reason": failed[0].reason,
            "failed": [item.ident for item in failed],
            "metric": metric,
            "checkers": rows,
            "reference": reference_binding(),
        }

    margin, degenerate = checkers.normalised_margin(state)
    if margin is None:
        return {
            "reward": 0.0,
            "reason": degenerate,
            "metric": metric,
            "checkers": rows,
            "reference": reference_binding(),
        }

    return {
        "reward": float(margin),
        "reason": "normalised-improvement",
        "metric": metric,
        "checkers": rows,
        "reference": reference_binding(),
    }


def write_reward(document: dict, reward_path: Path = REWARD_PATH, score_path: Path = SCORE_PATH) -> float:
    """Write the bare float and the score document. The float is the last byte written."""
    value = float(document.get("reward", 0.0))
    value = min(max(value, 0.0), 1.0)
    document["reward"] = value
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    reward_path.write_text("{:.6f}".format(value) + "\n", encoding="utf-8")
    return value


def main() -> int:
    document = evaluate(load_state())
    value = write_reward(document)
    print("reward " + "{:.6f}".format(value) + " reason " + str(document.get("reason")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
