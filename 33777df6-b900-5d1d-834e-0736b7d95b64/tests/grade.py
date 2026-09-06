#!/usr/bin/env python3
"""Reduce one telemetry record to one reward, one reason, and one metric block.

This file imports the checkers and, when a reference is handed to it, the
reference solution. It never imports the submission: the submission ran out of
process inside `tests/runner.py` and left an allocation behind, and everything
graded here was computed by the verifier from that allocation and the frozen
inputs.

Aggregation is `required_pass` over the gating reductions, composed with the
continuous separation ramp. Stated plainly, because a mode name is not a
description: if any required reduction is short of full, the reward is 0.0 and
the score document carries that reduction's own reason. If every required
reduction is full, the reward is the separation ramp, which is a continuous
value on [0, 1] and is emphatically not the boolean the mode name suggests. The
mode describes the gate; the ramp describes the result.

The three outcomes never merge. `separation-margin-cleared` is a measured win.
`degradation-past-bar` is a measured loss. `significance-unestablished-at-ceiling`
is neither: the run completed, every gate held, and the evaluation budget was the
binding limit, so the reward reports how far the band got without ever claiming
the allocation was shown to be better.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkers import (  # noqa: E402
    REASON_DEGRADED,
    REASON_ESTABLISHED,
    REASON_UNPROVEN,
    REQUIRED,
    SELECTORS,
    check_allocation_wellformed,
    check_bit_budget,
    check_calibration_band,
    check_calibration_noise_band,
    check_evaluation_schedule_order,
    check_frozen_inputs_intact,
    check_graded_metric_harness_sourced,
    check_graded_perplexity_unsmoothed,
    check_no_early_stop,
    check_not_degrading_past_bar,
    check_reading_sustained,
    check_separation_margin,
    classify_separation,
)

# Every selector the manifest names, resolved here so a manifest row and a
# graded reduction can never drift apart without this file failing to import.
BOUND_SELECTORS = {
    "submission_allocation_wellformed": check_allocation_wellformed,
    "bit_budget_respected_as_allocated": check_bit_budget,
    "frozen_corpus_and_reference_intact": check_frozen_inputs_intact,
    "graded_metric_harness_sourced": check_graded_metric_harness_sourced,
    "graded_perplexity_unsmoothed": check_graded_perplexity_unsmoothed,
    "evaluation_points_verifier_ordered": check_evaluation_schedule_order,
    "no_early_stop_on_favourable_evaluation": check_no_early_stop,
    "reading_sustained_across_scheduled_points": check_reading_sustained,
    "allocation_not_degrading_past_bar": check_not_degrading_past_bar,
    "separation_margin_cleared": check_separation_margin,
    "calibration_separation_within_band": check_calibration_band,
    "calibration_noise_band_width_held": check_calibration_noise_band,
}

# A telemetry record that never arrived is not a zero-scoring submission, it is a
# verifier that did not run. It still scores, and it still says why.
REASON_NO_TELEMETRY = "telemetry-absent"

# The refusals tests/runner.py records in telemetry["refusal"] when its own graded
# substrate is absent. Read here because the zero they produce is a zero about the
# VERIFIER's inputs and not about the submission's: with no checkpoint and no
# held-out split there is no evaluation point, and the gating reductions then fail
# for want of one and attribute the loss to the submission that supplied a
# perfectly well formed allocation. The reward is unchanged at 0.0, every required
# reduction still fails and every row is still written; only the attribution is
# taken from the producer that recorded the cause.
SUBSTRATE_REFUSALS = (
    "evaluation-substrate-absent-checkpoint",
    "evaluation-substrate-absent-heldout-split",
)

# The three required reductions that are computable with no evaluation point at
# all: two read the submitted allocation and its spend, one digests the frozen
# inputs. If any of them failed, the submission was rejected on grounds a missing
# checkpoint does not explain, and THAT rejection is the attribution, refusal or
# no refusal. A no-op submission must keep reading allocation-malformed. Only when
# all three held is the remaining loss wholly attributable to the absent
# substrate, which is when the refusal above is allowed to name the zero.
POINT_INDEPENDENT_REQUIRED = (
    "submission_allocation_wellformed",
    "bit_budget_respected_as_allocated",
    "frozen_corpus_and_reference_intact",
)


def load_reference(path: Path):
    """Import the reference solution by path. Never the submission, ever."""
    spec = importlib.util.spec_from_file_location("oer20_reference", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def grade(telemetry) -> dict:
    if not isinstance(telemetry, dict) or telemetry.get("schema") != "oer20.telemetry/v1":
        return {
            "reward": 0.0,
            "reason": REASON_NO_TELEMETRY,
            "verdict": "unscored",
            "checkers": [],
            "metric": {
                "separation_mean": None,
                "noise_half_width": None,
                "separation_margin": 0.05,
                "points_used": 0,
                "anchors_state": "absent",
                "baseline_metric": None,
                "target_metric": None,
            },
        }

    rows = []
    for name, _ in SELECTORS:
        verdict = BOUND_SELECTORS[name](telemetry)
        rows.append(
            {
                "id": name,
                "value": round(float(verdict.value), 6),
                "reason": verdict.reason,
                "detail": verdict.detail,
                "required": name in REQUIRED,
            }
        )

    failed = [row for row in rows if row["required"] and row["value"] < 1.0]
    refusal = str(telemetry.get("refusal") or "")
    measure = telemetry.get("measurement") or {}
    label, reason, value = classify_separation(measure)
    account = telemetry.get("accounting") or {}
    points = telemetry.get("points") or []

    metric = {
        "graded_quantity": "perplexity degradation at a fixed bit budget, against the frozen unquantized reference",
        "direction": "lower",
        "separation_mean": round(float(measure.get("separation_mean", 0.0)), 6),
        "noise_half_width": round(float(measure.get("noise_half_width", 0.0)), 6),
        "lower_edge": round(
            float(measure.get("separation_mean", 0.0)) - float(measure.get("noise_half_width", 0.0)), 6
        ),
        "upper_edge": round(
            float(measure.get("separation_mean", 0.0)) + float(measure.get("noise_half_width", 0.0)), 6
        ),
        "separation_margin": float(measure.get("separation_margin", 0.05)),
        "points_used": int(measure.get("points_used", 0)),
        "evaluation_ceiling_points": len(points),
        "mean_bits_per_param": round(float(account.get("mean_bits_per_param", 0.0)), 6),
        "allocated_bits": int(account.get("allocated_bits", 0)),
        "budget_bits": int(account.get("budget_bits", 0)),
        "mean_delta_agent": round(
            sum(float(row.get("delta_agent", 0.0)) for row in points) / len(points), 6
        )
        if points
        else None,
        "mean_delta_control": round(
            sum(float(row.get("delta_control", 0.0)) for row in points) / len(points), 6
        )
        if points
        else None,
        "anchors_state": "absent",
        "anchors_gap": "gap-oer-per-family-anchors-unmeasured",
        "baseline_metric": None,
        "target_metric": None,
    }

    graded_on_its_own_terms = [
        row for row in failed if row["id"] in POINT_INDEPENDENT_REQUIRED
    ]
    if refusal in SUBSTRATE_REFUSALS and not graded_on_its_own_terms:
        return {
            "reward": 0.0,
            "reason": refusal,
            "verdict": "unscored",
            "checkers": rows,
            "metric": metric,
        }

    if failed:
        return {
            "reward": 0.0,
            "reason": failed[0]["reason"],
            "verdict": "failed" if failed[0]["reason"] == REASON_DEGRADED else "rejected",
            "checkers": rows,
            "metric": metric,
        }

    reward = 1.0 if reason == REASON_ESTABLISHED else float(value)
    return {
        "reward": max(0.0, min(1.0, reward)),
        "reason": REASON_ESTABLISHED if reason == REASON_ESTABLISHED else REASON_UNPROVEN,
        "verdict": label,
        "checkers": rows,
        "metric": metric,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="OER-20 grader")
    parser.add_argument("--telemetry", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference", default="")
    args = parser.parse_args(argv)

    try:
        telemetry = json.loads(Path(args.telemetry).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        telemetry = None

    document = grade(telemetry)
    if args.reference:
        module = load_reference(Path(args.reference))
        if module is not None and hasattr(module, "REFERENCE_SHA256"):
            document["reference_sha256"] = module.REFERENCE_SHA256

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
