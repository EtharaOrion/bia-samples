"""Reduce the live telemetry to one float and one machine-readable reason.

This file is the carrier every checker declared in tests/checkers.yaml is
reached by. It imports the checkers and it NEVER imports the submission and
never imports the reference: the submission is executed in a separate process by
tests/runner.py, under its own session, and only the telemetry that run produced
crosses back here.

The gate is `required_pass`: every required checker must pass before any number
is computed. Past the gate the reward is continuous, and it is the bound formula
for a lower-is-better metric with a continuous separation ramp on top of it.
Nothing here is binary at the margin.

Writes, in this order and last of all:
    /logs/verifier/reward.txt   the bare float, the carrier the live verifier binds
    /logs/verifier/score.json   the reason and the metric block
"""

import json
import math
import pathlib
import sys

import checkers
from checkers import (
    check_compaction_dropped_entries_reconstructed,
    check_crossing_sustained_across_scheduled_points,
    check_early_stop_is_not_a_crossing,
    check_frozen_axes_unmoved,
    check_graded_crossing_recomputed_unsmoothed,
    check_graded_weights_are_harness_owned,
    check_iteration_sequence_ordered,
    check_multi_seed_mean_clears_margin,
    check_summary_agrees_with_durable_ledger,
)

# The runtime root Harbor mounts. Bound, never chosen here.
TELEMETRY = pathlib.Path("/logs/verifier")
REWARD_PATH = pathlib.Path("/logs/verifier/reward.txt")
SCORE_PATH = pathlib.Path("/logs/verifier/score.json")

# Bound by seed/staging/oer1/LANE_SPEC.md. The separation margin enters as a
# linear ramp rather than a cliff, so the reward degrades continuously through
# it. seed/contract.yaml carries null for this field; the divergence is recorded
# in solution/grounding.yaml and this file does not amend the batch field.
SEPARATION_MARGIN = 0.05

REWARD_FLOOR = 0.0
REWARD_CEILING = 1.0

# The declared graded set. Identity, selector, required flag and weight, in the
# same order tests/checkers.yaml declares them. The selectors are named here
# literally so an outside reader can see that the carrier reaches every one.
GRADED = (
    ("graded_crossing_recomputed_unsmoothed", check_graded_crossing_recomputed_unsmoothed, True, 3),
    ("crossing_sustained_across_scheduled_points", check_crossing_sustained_across_scheduled_points, True, 3),
    ("early_stop_is_not_a_crossing", check_early_stop_is_not_a_crossing, True, 2),
    ("graded_weights_are_harness_owned", check_graded_weights_are_harness_owned, True, 3),
    ("multi_seed_mean_clears_margin", check_multi_seed_mean_clears_margin, True, 3),
    ("iteration_sequence_ordered", check_iteration_sequence_ordered, True, 2),
    ("summary_agrees_with_durable_ledger", check_summary_agrees_with_durable_ledger, True, 3),
    ("compaction_dropped_entries_reconstructed", check_compaction_dropped_entries_reconstructed, True, 3),
    ("frozen_axes_unmoved", check_frozen_axes_unmoved, True, 2),
)

# checkers is imported as a module as well as by name so tests/checkers.py is
# reachable from this carrier by file name, and so an outside reader can see
# that the declared set here is the same tuple checkers.SELECTORS declares.
DECLARED_ELSEWHERE = tuple(ident for ident, _ in checkers.SELECTORS)


def clip(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return REWARD_FLOOR
    if number != number:
        return REWARD_FLOOR
    return max(REWARD_FLOOR, min(REWARD_CEILING, number))


def read_json(path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def seed_means(root):
    """The agent arm and the verifier's reference arm over the shared seed set."""
    agent = {}
    for directory in sorted((root / "eval").glob("seed-*")) if (root / "eval").is_dir() else []:
        record = read_json(directory / "evaluations.json")
        if not isinstance(record, dict) or record.get("graded_step") is None:
            continue
        agent[str(record.get("seed"))] = float(record["graded_step"])
    arm = read_json(root / "eval" / "reference_arm.json") or {}
    reference = {}
    for key, value in (arm.get("seeds") or {}).items():
        try:
            reference[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    shared = sorted(set(agent) & set(reference))
    if not shared:
        return None, None, []
    agent_mean = math.fsum(agent[key] for key in shared) / len(shared)
    reference_mean = math.fsum(reference[key] for key in shared) / len(shared)
    return agent_mean, reference_mean, shared


def normalized(metric, baseline, target):
    """The bound lower-is-better formula, clipped. Reaching the target is 1.0."""
    span = float(baseline) - float(target)
    if span <= 0.0 or not math.isfinite(span):
        return None
    return clip((float(baseline) - float(metric)) / span)


def final_submission_digest(root):
    timeline = read_json(root / "loop" / "iterations.json")
    rows = (timeline or {}).get("iterations") if isinstance(timeline, dict) else None
    if not isinstance(rows, list) or not rows:
        return None
    last = rows[-1]
    return str(last.get("submission_digest")) if isinstance(last, dict) else None


def carriers(root):
    """Where this run's reward and score document are written.

    Under the bound runtime root the answer is the bound pair and nothing else.
    A grade driven over a fixture root writes beside that fixture instead, so an
    adequacy run exercises this exact reduction without writing into the runtime
    path it does not own. The bound literal is never recomputed from the root.
    """
    if root == TELEMETRY:
        return REWARD_PATH, SCORE_PATH
    return root / "reward.txt", root / "score.json"


def emit(root, reward, reason, metric, criteria):
    """The last thing that happens. The float first, then the reason document."""
    reward_path, score_path = carriers(root)
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "reward": round(float(reward), 6),
        "reason": reason,
        "metric": metric,
        "criteria": criteria,
        "aggregation": "required_pass gate, then the continuous bound formula",
        "separation_margin": SEPARATION_MARGIN,
        "anchors_state": "absent",
        "anchors_gap": "gap-oer-per-family-anchors-unmeasured",
        "slot": "OER-07",
    }
    # The reason document goes first and the bare float goes last, so the
    # carrier existing implies the reason for it already exists. The EXIT trap
    # in tests/test.sh keys on the carrier, and that ordering is what makes the
    # trap's test sound rather than merely present.
    score_path.write_text(json.dumps(document, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    reward_path.write_text("{:.6f}\n".format(float(reward)), encoding="utf-8")
    return document


def grade(root):
    """The whole reduction: gate, then the continuous number, then the write."""
    criteria = []
    failed = None
    for ident, selector, required, weight in GRADED:
        result = selector(root)
        criteria.append(
            {
                "id": ident,
                "ok": bool(result.ok),
                "reason": result.reason,
                "detail": result.detail,
                "required": bool(required),
                "weight": int(weight),
            }
        )
        if required and not result.ok and failed is None:
            failed = result

    metric = {
        "graded_step": None,
        "baseline": None,
        "target": None,
        "agent_mean_steps": None,
        "reference_mean_steps": None,
        "seeds": [],
        "separation": None,
        "margin_factor": None,
        "final_submission_digest": final_submission_digest(root),
    }

    if failed is not None:
        return emit(root, REWARD_FLOOR, failed.reason, metric, criteria)

    anchors = read_json(root / "anchors.json")
    if not isinstance(anchors, dict) or not bool(anchors.get("resolved")):
        return emit(root, REWARD_FLOOR, "anchors-unresolved", metric, criteria)
    baseline = anchors.get("baseline_metric")
    target = anchors.get("target_metric")
    if baseline is None or target is None:
        return emit(root, REWARD_FLOOR, "anchors-unresolved", metric, criteria)

    agent_mean, reference_mean, shared = seed_means(root)
    if agent_mean is None:
        return emit(root, REWARD_FLOOR, "multi-seed-separation-not-cleared", metric, criteria)

    agent_norm = normalized(agent_mean, baseline, target)
    reference_norm = normalized(reference_mean, baseline, target)
    if agent_norm is None or reference_norm is None:
        return emit(root, REWARD_FLOOR, "anchors-unresolved", metric, criteria)

    separation = agent_norm - reference_norm
    margin_factor = clip(separation / SEPARATION_MARGIN)
    reward = clip(agent_norm * margin_factor)

    metric.update(
        {
            "graded_step": agent_mean,
            "baseline": baseline,
            "target": target,
            "agent_mean_steps": agent_mean,
            "reference_mean_steps": reference_mean,
            "seeds": shared,
            "separation": separation,
            "margin_factor": margin_factor,
        }
    )
    if reward <= REWARD_FLOOR:
        return emit(root, REWARD_FLOOR, "multi-seed-separation-not-cleared", metric, criteria)
    return emit(root, reward, "graded", metric, criteria)


def main():
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else TELEMETRY
    document = grade(root)
    sys.stdout.write(json.dumps(document, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
