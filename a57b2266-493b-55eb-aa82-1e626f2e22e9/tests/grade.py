#!/usr/bin/env python3
"""Drive the live checkers over one run and write the bound reward contract.

This module imports the checkers and the reference. It never imports the
submission: the submission is launched out of process by tests/runner.py and is
only ever read back as data.

The graded quantity is the nanoGPT validation loss the verifier computes itself,
raw and unsmoothed, on its own held-out FineWeb validation shards, from a forward
pass of the parameter snapshot the harness-owned frozen train stage wrote at the
bound evaluation point. No number the submission reported, printed or wrote
reaches this path, and no number an in-container harness printed is the grade.

F13 carries no published anchors. `baseline_metric` and `target_metric` stay null
under gap-oer-per-family-anchors-unmeasured, and the two slots of the bound
lower-is-better formula are filled by in-run verifier measurement of the pristine
delivered composition and of the reference composition. The anchor constants of
the retired surrogate path did not survive the re-base onto the nanoGPT substrate
and were not carried across: no anchor number is invented here or anywhere else
in this bundle.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
for extra in (HERE, BUNDLE / "solution"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import harness  # noqa: E402
import reference  # noqa: E402
from checkers import (  # noqa: E402
    check_end_to_end_gain_not_stage_local,
    check_eval_split_not_trained_on,
    check_evaluation_point_is_bound_not_selected,
    check_floor_anchor_matches_bound_reference,
    check_frozen_axes_unmodified,
    check_full_chain_completed_no_early_stop,
    check_graded_loss_from_harness_weights,
    check_graded_parameters_bind_frozen_architecture,
    check_holdout_split_is_verifier_owned,
    check_loss_sustained_across_bound_evals,
    check_no_smoothing_on_graded_path,
    check_session_state_carried_forward,
    check_stage_chain_order_observed,
    check_stage_claim_matches_next_stage_consumption,
    check_token_budget_respected_as_fed,
    recomputed_loss,
)

# id -> (selector, required, weight). The order is the order defects surface in.
GRADED = (
    ("frozen_axes_unmodified", check_frozen_axes_unmodified, True, 1),
    ("stage_chain_order_observed", check_stage_chain_order_observed, True, 1),
    ("stage_claim_matches_next_stage_consumption", check_stage_claim_matches_next_stage_consumption, True, 1),
    ("token_budget_respected_as_fed", check_token_budget_respected_as_fed, True, 1),
    # The split's ownership is graded BEFORE the leakage absence and before any
    # checker that reads a loss, so a run whose graded split did not resolve to
    # the verifier's own copy surfaces under its own reason instead of being
    # reported as a composition that leaked or failed to improve.
    ("holdout_split_is_verifier_owned", check_holdout_split_is_verifier_owned, True, 1),
    ("eval_split_not_trained_on", check_eval_split_not_trained_on, True, 1),
    ("full_chain_completed_no_early_stop", check_full_chain_completed_no_early_stop, True, 1),
    ("evaluation_point_is_bound_not_selected", check_evaluation_point_is_bound_not_selected, True, 1),
    # The architecture binding is graded BEFORE the loss is read from the
    # snapshot. A graded artifact that is not the frozen decoder is not a worse
    # model, it is not this model, and it must say so rather than be scored.
    ("graded_parameters_bind_frozen_architecture", check_graded_parameters_bind_frozen_architecture, True, 1),
    ("graded_loss_from_harness_weights", check_graded_loss_from_harness_weights, True, 1),
    ("no_smoothing_on_graded_path", check_no_smoothing_on_graded_path, True, 1),
    # The floor anchor is graded BEFORE the two checkers that compare against it,
    # so a span resting on an anchor the environment did not establish surfaces
    # under its own reason instead of being reported as a composition that failed
    # to improve.
    ("floor_anchor_matches_bound_reference", check_floor_anchor_matches_bound_reference, True, 1),
    # The end-to-end gain is graded BEFORE the sustain invariant. Both refuse a
    # composition that fails to improve overall, but only one of them names that
    # as what happened: a submission that never improved is a stage-local optimum,
    # not an improvement that failed to hold across folds. Grading the sustain
    # invariant first would attribute every no-op to improvement-not-sustained and
    # tell the agent to chase fold stability it never had.
    ("end_to_end_gain_not_stage_local", check_end_to_end_gain_not_stage_local, True, 1),
    ("loss_sustained_across_bound_evals", check_loss_sustained_across_bound_evals, True, 1),
    ("session_state_carried_forward", check_session_state_carried_forward, True, 1),
)

DEFAULT_REWARD_ROOT = "/logs/verifier"


def run_checkers(handle) -> list:
    rows = []
    for ident, selector, required, weight in GRADED:
        try:
            outcome = selector(handle)
            passed, reason, detail = outcome.passed, outcome.reason, outcome.detail
        except (OSError, ValueError, KeyError, IndexError, TypeError, ZeroDivisionError, RuntimeError) as exc:
            passed, reason, detail = False, _zero_reason(ident), "the checker could not resolve live state: " + repr(exc)
        rows.append(
            {
                "id": ident,
                "passed": bool(passed),
                "reason": reason,
                "detail": detail,
                "required": required,
                "weight": weight,
            }
        )
    return rows


ZERO_REASONS = {
    "frozen_axes_unmodified": "frozen-axis-modified",
    "stage_chain_order_observed": "stage-chain-order-violated",
    "stage_claim_matches_next_stage_consumption": "stage-claim-contradicted-by-chain",
    "token_budget_respected_as_fed": "token-budget-not-respected-as-fed",
    "holdout_split_is_verifier_owned": "holdout-split-not-verifier-owned",
    "eval_split_not_trained_on": "evaluation-split-leaked-into-training",
    "full_chain_completed_no_early_stop": "early-stop-loss-not-established",
    "evaluation_point_is_bound_not_selected": "evaluation-point-submission-selected",
    "graded_parameters_bind_frozen_architecture": "frozen-architecture-not-instantiated",
    "graded_loss_from_harness_weights": "graded-loss-not-from-harness-weights",
    "no_smoothing_on_graded_path": "readout-smoothing-on-graded-path",
    "floor_anchor_matches_bound_reference": "floor-anchor-not-the-bound-reference",
    "loss_sustained_across_bound_evals": "improvement-not-sustained",
    "session_state_carried_forward": "session-state-broken",
    "end_to_end_gain_not_stage_local": "stage-local-optimum-degrades-composition",
}


def _zero_reason(ident: str) -> str:
    return ZERO_REASONS.get(ident, "checker-unresolved")


def score_run(handle) -> dict:
    """Aggregate. required_pass over the gate, then the continuous metric term."""
    rows = run_checkers(handle)
    telemetry = handle.telemetry
    base = float((telemetry.get("baseline") or {}).get("val_loss", 0.0))
    floor = float((telemetry.get("floor") or {}).get("val_loss", 0.0))
    try:
        agent = recomputed_loss(handle)
    except (OSError, ValueError, KeyError, IndexError, RuntimeError):
        agent = None

    substrate = handle.substrate or {}
    architecture = substrate.get("architecture") or {}
    metric = {
        "graded_validation_loss": None if agent is None else round(agent, 9),
        "unit": "mean cross entropy in nats per token on the held-out FineWeb validation split",
        "baseline_measured_in_run": round(base, 9),
        "floor_measured_in_run": round(floor, 9),
        "anchors_state": "absent",
        "anchors_gap": "gap-oer-per-family-anchors-unmeasured",
        "baseline_metric": None,
        "target_metric": None,
        "direction": "lower",
        "substrate_target_val_loss": (substrate.get("run") or {}).get("target_val_loss"),
        "architecture": {
            "vocab_size": architecture.get("vocab_size"),
            "num_layers": architecture.get("num_layers"),
            "model_dim": architecture.get("model_dim"),
            "head_dim": architecture.get("head_dim"),
            "seq_len": architecture.get("seq_len"),
        },
        "bound_evaluation_point": handle.bound.get("graded_evaluation_point"),
        "step_budget_steps": handle.bound.get("step_budget_steps"),
        "eval_fold_schedule": handle.bound.get("eval_fold_schedule"),
        "holdout_root": handle.bound.get("holdout_root"),
        "smoothing_on_graded_path": False,
    }

    failed = [row for row in rows if row["required"] and not row["passed"]]
    if failed:
        return {"reward": 0.0, "reason": failed[0]["reason"], "metric": metric, "checkers": rows}

    span = base - floor
    if span <= 0.0 or agent is None:
        return {"reward": 0.0, "reason": "anchor-span-degenerate", "metric": metric, "checkers": rows}
    raw = (base - agent) / span
    reward = min(max(raw, 0.0), 1.0)
    if reward <= 0.0:
        return {"reward": 0.0, "reason": "no-end-to-end-improvement", "metric": metric, "checkers": rows}
    return {"reward": reward, "reason": "graded-on-recomputed-validation-loss", "metric": metric, "checkers": rows}


def grade(bundle: Path, submission: Path, workspace: Path) -> dict:
    floor_source = reference.materialize(bundle / "environment", workspace / "reference_source")
    handle = harness.build(bundle, submission, floor_source, workspace / "run")
    return score_run(handle)


def emit(document: dict, reward_root: Path) -> None:
    """The score document lands first; the bare float is the last byte written."""
    reward_root.mkdir(parents=True, exist_ok=True)
    (reward_root / "score.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reward_root / "reward.txt").write_text("%.6f" % float(document["reward"]), encoding="utf-8")


def main() -> int:
    bundle = Path(os.environ.get("OER13_BUNDLE", str(BUNDLE)))
    submission = Path(os.environ.get("OER13_SUBMISSION", str(bundle / "environment")))
    reward_root = Path(os.environ.get("OER13_REWARD_ROOT", DEFAULT_REWARD_ROOT))
    workspace = Path(os.environ.get("OER13_WORKSPACE", "") or tempfile.mkdtemp(prefix="oer13-grade-"))
    document = grade(bundle, submission, workspace)
    emit(document, reward_root)
    print(json.dumps({"reward": document["reward"], "reason": document["reason"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
