#!/usr/bin/env python3
"""Reduce one verifier telemetry record to one float, and say why it is that float.

Import discipline, stated first because it is the point. This module imports
`checkers`, `probe`, `corpus.loader`, `corpus_rules` and `runner`. It NEVER
imports the submission. The submission runs, if it runs at all, inside the
subprocess `tests/runner.py` opens, and its only output channel is a transcript
this process reads as bytes. There is no module here the submission can rebind,
no hook it can install, and no attribute it can shadow.

Truth is computed only after that subprocess has exited and its process group
has been reaped. That ordering is recorded into the verifier's own order log and
is itself graded, by `check_truth_computed_after_submission_exit`, so it is a
measured property of the run rather than a promise this docstring makes.

The reward:

    every required checker must pass                      -> gate chain
    magnitude = mean over seeds of clipped raw            -> continuous
    confidence = clip(separation / 0.05, 0, 1)            -> continuous ramp
    reward = clip(magnitude * confidence, 0, 1)

with `raw = (baseline_metric - agent_metric) / (baseline_metric - target_metric)`
for this lower-is-better metric. Reaching the target scores exactly 1.0 and
beating it also scores 1.0, because the target is a bar rather than a point on a
gradient. Every zero carries a machine-readable reason.

Anchors. F15's anchors are UNMEASURED, so `baseline_metric` and `target_metric`
are absent from this bundle. A run whose telemetry carries no anchor pair is
refused with the reason `anchor-pair-absent`; no number is substituted for a
measurement nobody made.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import checkers  # noqa: E402
import corpus_rules  # noqa: E402
import probe  # noqa: E402
import runner  # noqa: E402
from corpus import loader  # noqa: E402

REWARD_PATH = "/logs/verifier/reward.txt"
SCORE_DOCUMENT_PATH = "/logs/verifier/score.json"

# The twelve selectors this module reaches, named so an outside reader can match
# tests/checkers.yaml against the bytes that actually run. checkers.CHAIN is the
# single ordering; this tuple exists to name it and is asserted equal to it.
SELECTORS = (
    checkers.check_truth_computed_after_submission_exit,
    checkers.check_grading_tree_not_writable_from_submission,
    checkers.check_no_submission_number_on_graded_metric,
    checkers.check_graded_crossing_recomputed_unsmoothed,
    checkers.check_early_stop_yields_no_crossing,
    checkers.check_crossing_sustained_across_scheduled_evals,
    checkers.check_evaluated_weights_are_harness_owned,
    checkers.check_behavioural_probe_deterministic,
    checkers.check_probe_transcript_bound_to_training_deltas,
    checkers.check_corpus_pinned_before_novelty_verdict,
    checkers.check_behavioural_novelty_clears_corpus,
    checkers.check_multi_seed_separation_clears_margin,
)

assert SELECTORS == checkers.CHAIN, "the named selectors and the graded chain disagree"

# The same twelve selectors written as bare names, so tests/checkers.yaml can be
# matched against these bytes by a reader or by an analyser that follows names
# rather than resolving attribute access.
SELECTOR_NAMES = (
    "check_truth_computed_after_submission_exit",
    "check_grading_tree_not_writable_from_submission",
    "check_no_submission_number_on_graded_metric",
    "check_graded_crossing_recomputed_unsmoothed",
    "check_early_stop_yields_no_crossing",
    "check_crossing_sustained_across_scheduled_evals",
    "check_evaluated_weights_are_harness_owned",
    "check_behavioural_probe_deterministic",
    "check_probe_transcript_bound_to_training_deltas",
    "check_corpus_pinned_before_novelty_verdict",
    "check_behavioural_novelty_clears_corpus",
    "check_multi_seed_separation_clears_margin",
)

assert SELECTOR_NAMES == tuple(fn.__name__ for fn in checkers.CHAIN), "selector names drifted"


def clip(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:
        return 0.0
    return max(0.0, min(1.0, number))


def _anchor_state(ctx) -> str:
    anchors = (ctx or {}).get("anchors")
    anchors = anchors if isinstance(anchors, dict) else {}
    return str(anchors.get("state", "absent"))


def _metric_block(ctx, found) -> dict:
    anchors = (ctx or {}).get("anchors")
    anchors = anchors if isinstance(anchors, dict) else {}
    return {
        "graded_steps_per_seed": list(found.get("derived_steps") or []),
        "mean_raw": found.get("mean_raw"),
        "half_spread": found.get("half_spread"),
        "separation": found.get("separation"),
        "separation_margin": checkers.SEPARATION_MARGIN,
        "confidence": found.get("confidence"),
        "seeds": found.get("seeds"),
        "baseline_metric": anchors.get("baseline_metric"),
        "target_metric": anchors.get("target_metric"),
        "anchors_state": _anchor_state(ctx),
        "anchors_gap": "gap-oer-per-family-anchors-unmeasured",
        "direction": "lower",
    }


def grade(ctx) -> dict:
    """One telemetry record in, one score document out. Pure over `ctx`."""
    verdicts = [row.as_dict() for row in checkers.run_all(ctx)]

    if _anchor_state(ctx) == "absent":
        return {
            "reward": 0.0,
            "reason": "anchor-pair-absent",
            "verdicts": verdicts,
            "metric": _metric_block(ctx, {}),
            "note": "F15's baseline and target are unmeasured. The reward SCHEMA is bound and no number is substituted for a measurement nobody made.",
        }

    for row in verdicts:
        if not row["ok"]:
            return {
                "reward": 0.0,
                "reason": row["reason"],
                "failed_checker": row["id"],
                "verdicts": verdicts,
                "metric": _metric_block(ctx, {}),
            }

    found = checkers.separation(ctx)
    if not found.get("resolved"):
        return {
            "reward": 0.0,
            "reason": str(found.get("reason", "separation-unresolvable")),
            "verdicts": verdicts,
            "metric": _metric_block(ctx, {}),
        }

    reward = clip(found["magnitude"])
    document = {
        "reward": reward,
        "reason": "graded" if reward > 0.0 else "separation-below-noise-floor",
        "verdicts": verdicts,
        "metric": _metric_block(ctx, found),
    }
    document["metric"]["per_seed_raw"] = found["per_seed_raw"]
    return document


def load_context(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def recompute_novelty(ctx) -> dict:
    """Recompute the corpus side of the verdict from the frozen bytes.

    Called on the live path so the corpus digest a verdict is stamped against is
    the one this tree carries right now, rather than one recorded earlier. The
    submission's transcript is NOT recomputed here; it arrives from the isolated
    subprocess `runner.run_probe` opened, and it is held to the harness deltas by
    `check_probe_transcript_bound_to_training_deltas`.
    """
    state = (ctx or {}).get("drift_state") or {}
    set_name = str(state.get("probe_input_set", "probe-set-a"))
    rows = loader.signatures(corpus_rules.build, probe, set_name)
    return {"corpus_pinned_digest": loader.pinned_digest(), "entries": len(rows), "probe_set": set_name}


def write_reward(reward: float, reason: str, document: dict, reward_path: Path, score_path: Path) -> None:
    """The bare float goes to the bound carrier; the reason goes beside it."""
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(document)
    payload["reward"] = clip(reward)
    payload["reason"] = reason
    score_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    reward_path.write_text(repr(clip(reward)) + "\n", encoding="utf-8")


def main(argv) -> int:
    context_path = Path(argv[1]) if len(argv) > 1 else Path("/logs/harness/telemetry.json")
    reward_path = Path(argv[2]) if len(argv) > 2 else Path(REWARD_PATH)
    score_path = Path(argv[3]) if len(argv) > 3 else Path(SCORE_DOCUMENT_PATH)

    ctx = load_context(context_path)
    if ctx is None:
        write_reward(
            0.0,
            "telemetry-absent",
            {"metric": {"graded_step": None, "anchors_state": "absent"}, "verdicts": []},
            reward_path,
            score_path,
        )
        return 0

    document = grade(ctx)
    write_reward(document["reward"], document["reason"], document, reward_path, score_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
