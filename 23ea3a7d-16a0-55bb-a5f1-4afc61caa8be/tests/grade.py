#!/usr/bin/env python3
"""The grading process for OER-16. It never imports the submission.

Chain, in order:

    runner.py    runs the submission ALONE in a fresh directory and returns raw
                 artifacts and a status. No verdict, no score.
    harness.py   builds the frozen model in THIS process, spends the frozen
                 budget on THIS process's counter, and evaluates over the frozen
                 corpus. Produces one telemetry record.
    checkers.py  grades that record against tests/config.json, which is the
                 admin-plane configuration the drift schedule moves.

The declared surface is tests/checkers.yaml; every selector named there is
listed in CHECKER_SELECTORS below and resolved out of the checkers module, so
the manifest and the code cannot drift apart without this file failing loudly.

The reward is written LAST: the bare float to /logs/verifier/reward.txt and the
machine-readable reason plus metric block to /logs/verifier/score.json. Every
zero carries a reason. `tests/test.sh` traps every exit path so an aborted
verifier still produces a reason rather than silence.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))

import checkers  # noqa: E402
import harness  # noqa: E402
import runner  # noqa: E402

BOUND_REWARD_PATH = "/logs/verifier/reward.txt"
BOUND_SCORE_PATH = "/logs/verifier/score.json"

TESTS_DIR = Path(__file__).resolve().parent
BUNDLE = TESTS_DIR.parent

# Every selector tests/checkers.yaml declares, resolved out of tests/checkers.py.
CHECKER_SELECTORS = (
    "measured_bits_per_byte_is_harness_measured",
    "graded_readout_is_unsmoothed",
    "denominator_is_frozen_corpus_bytes",
    "claim_matches_measurement",
    "reading_sustained_across_scheduled_points",
    "early_stop_establishes_no_metric",
    "claim_absence_is_recorded",
    "compute_budget_respected_as_spent",
    "claim_sealed_before_measurement",
    "frozen_axes_and_vocab_ceiling_hold",
)


def reward_root() -> Path:
    """The bound root, or a lane-private redirect for a local exercise.

    /logs/verifier/ is one shared host path and concurrent lanes collide on it,
    so a local exercise redirects the ROOT. The bound contract path is the one
    named above and it is what ships.
    """
    override = os.environ.get("OER16_REWARD_ROOT", "").strip()
    if override:
        return Path(override)
    return Path(str(PurePosixPath(BOUND_REWARD_PATH).parent))


def load_config() -> dict:
    return json.loads((TESTS_DIR / "config.json").read_text(encoding="utf-8"))


def resolve_selectors():
    missing = [name for name in CHECKER_SELECTORS if not hasattr(checkers, name)]
    if missing:
        raise RuntimeError("tests/checkers.yaml names selectors absent from checkers.py: "
                           + ", ".join(missing))
    return [(name, getattr(checkers, name)) for name in CHECKER_SELECTORS]


def aggregate_required_pass(verdicts):
    """A required_pass gate. Mirrors seed/forge/verifier.py aggregate() exactly.

    Every declared checker is required, so the gate is a universal quantifier
    over a non-empty set and can never report success for having checked
    nothing. adequacy.py drives the REAL verifier.aggregate over the same
    criteria and asserts this mirror agrees with it.
    """
    if not verdicts:
        return 0.0, "no-criteria"
    for verdict in verdicts:
        if not verdict.passed:
            return 0.0, verdict.reason
    return 1.0, "every-required-criterion-passed"


def metric_reward(measured, baseline, target) -> float:
    """The bound lower-is-better formula, clipped to the closed interval."""
    span = float(baseline) - float(target)
    if span == 0.0:
        return 0.0
    raw = (float(baseline) - float(measured)) / span
    return min(max(raw, 0.0), 1.0)


def score(record: dict, config: dict):
    verdicts = [selector(record, config) for _, selector in resolve_selectors()]
    gate, gate_reason = aggregate_required_pass(verdicts)
    graded = record.get("readout", {}).get("graded_bits_per_byte")
    claimed = record.get("claim", {}).get("claimed_bits_per_byte")
    anchors = config.get("anchors", {})

    numbers = {}
    for verdict in verdicts:
        if not verdict.passed:
            numbers = dict(verdict.numbers)
            break

    if gate < 1.0:
        reward, reason = 0.0, gate_reason
    elif anchors.get("anchors_state") != "measured":
        # Fail closed. A score computed from an invented denominator would be
        # worse than no score, so the absence is reported as the reason.
        reward, reason = 0.0, "anchors-absent"
        numbers = {"anchors_gap": anchors.get("gap")}
    else:
        reward = metric_reward(graded, anchors["baseline_metric"], anchors["target_metric"])
        reason = "scored" if reward > 0.0 else "no-improvement-over-baseline"

    document = {
        "slot": "OER-16",
        "reward": reward,
        "reason": reason,
        "metric": {
            "graded_bits_per_byte": graded,
            "claimed_bits_per_byte": claimed,
            "claim_absent": not record.get("claim", {}).get("present", False),
            "denominator_bytes": record.get("corpus", {}).get("denominator_bytes"),
            "spent_token_updates": record.get("compute", {}).get("spent_token_updates"),
            "points_completed": record.get("schedule", {}).get("points_completed"),
            "baseline_metric": anchors.get("baseline_metric"),
            "target_metric": anchors.get("target_metric"),
            "anchors_state": anchors.get("anchors_state"),
        },
        "reason_numbers": numbers,
        "gate": {"mode": "required_pass", "value": gate, "reason": gate_reason},
        "checkers": [verdict.as_dict() for verdict in verdicts],
    }
    return reward, document


def emit(reward: float, document: dict) -> None:
    root = reward_root()
    root.mkdir(parents=True, exist_ok=True)
    (root / PurePosixPath(BOUND_SCORE_PATH).name).write_text(
        json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    # The bare float, written LAST, is the carrier the live verifier binds.
    (root / PurePosixPath(BOUND_REWARD_PATH).name).write_text(
        format(float(reward), ".6f") + "\n", encoding="utf-8"
    )


def main() -> int:
    config = load_config()
    frozen_dir = Path(os.environ.get("OER16_FROZEN_DIR") or (BUNDLE / "environment" / "frozen"))
    submission = Path(os.environ.get("OER16_SUBMISSION") or "/app/submission.py")

    frozen = harness.load_frozen(frozen_dir)
    result = runner.run_submission(submission, frozen_dir)
    try:
        record = harness.measure(frozen, result.artifacts)
    finally:
        runner.cleanup(result)

    reward, document = score(record, config)
    document["submission"] = {
        "path": str(submission),
        "launched": result.launched,
        "returncode": result.returncode,
        "timed_out": result.timed_out,
    }
    emit(reward, document)
    print(json.dumps({"reward": reward, "reason": document["reason"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
