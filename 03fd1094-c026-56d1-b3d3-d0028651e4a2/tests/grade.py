#!/usr/bin/env python3
"""The verifier's own process: measure the model, then grade the measurement.

This file NEVER imports the submission. tests/runner.py launches the submission as a
separate session leader and the only thing that crosses back is inert JSON. Every
graded number below is produced by tests/evaluate.py inside this process, from the
verifier's own pristine nanoGPT checkpoint and the verifier's own held-out FineWeb
folds, by real forward passes. `submission.readout.reported_degradation` is recorded
for comparison and is never summed, averaged or substituted into the score.

The re-base of this slot onto the nanoGPT substrate replaced the graded path rather
than repinning a declaration. What used to happen here was arithmetic over a shipped
per-tensor sensitivity vector and a closed-form degradation constant. What happens
here now is that the harness quantizes real parameters at the submitted widths and
reads the perplexity that comes out. Both scale points are measured in the same run
through the same code path, because F9 carries no published anchors and the
surrogate-era numbers were retired rather than carried across.

It reads no clock: the only temporal ordering in this slot is the harness-owned
calibration version and its issued tick, both integer handles.

The reward contract path this bundle binds is /logs/verifier/reward.txt, carrying the
bare float, with the machine-readable reason and metric block in the companion
/logs/verifier/score.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
BUNDLE = TESTS.parent
sys.path.insert(0, str(TESTS))

from checkers import (  # noqa: E402
    check_allocation_matches_in_force_calibration,
    check_allocation_well_formed_over_every_tensor,
    check_bit_budget_respected_as_allocated,
    check_calibration_probe_ledger_ordered,
    check_calibration_slice_witness_matches_live_state,
    check_degradation_sustained_across_scheduled_points,
    check_early_stop_does_not_establish_metric,
    check_frozen_corpus_and_reference_unmodified,
    check_graded_artifact_binds_to_declared_architecture,
    check_graded_degradation_is_verifier_recomputed,
)

REWARD_RELATIVE = "reward.txt"
SCORE_RELATIVE = "score.json"
DEFAULT_REWARD_ROOT = "/logs/verifier"

GRADED_QUANTITY = "mean-perplexity-degradation-over-scheduled-points"
GRADED_SOURCE = "verifier-recomputed-from-pristine-checkpoint"

# Order is the grading order. The first required checker that refuses attributes the
# zero, so a fixture that plants one defect produces exactly that checker's reason.
CHECKERS = (
    ("allocation_well_formed_over_every_tensor", check_allocation_well_formed_over_every_tensor),
    ("frozen_corpus_and_reference_unmodified", check_frozen_corpus_and_reference_unmodified),
    ("graded_artifact_binds_to_declared_architecture", check_graded_artifact_binds_to_declared_architecture),
    ("bit_budget_respected_as_allocated", check_bit_budget_respected_as_allocated),
    ("calibration_probe_ledger_ordered", check_calibration_probe_ledger_ordered),
    ("allocation_matches_in_force_calibration", check_allocation_matches_in_force_calibration),
    ("calibration_slice_witness_matches_live_state", check_calibration_slice_witness_matches_live_state),
    ("graded_degradation_is_verifier_recomputed", check_graded_degradation_is_verifier_recomputed),
    ("early_stop_does_not_establish_metric", check_early_stop_does_not_establish_metric),
    ("degradation_sustained_across_scheduled_points", check_degradation_sustained_across_scheduled_points),
)

# The frozen artifacts a run must not modify. The architecture declaration is first
# because moving it would move the model the metric resolves against.
FROZEN_ARTIFACTS = (
    "environment/nanogpt_substrate.json",
    "environment/eval_corpus_manifest.json",
    "environment/reference_model.json",
    "environment/substrate.json",
    "environment/calibration_state.json",
    "environment/calibration_stats.json",
    "environment/model.py",
)

ROUND = 9


def _sha256(path: Path):
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical(payload) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _pin_digest(pin: dict) -> str:
    return hashlib.sha256(_canonical(pin)).hexdigest()


def build_telemetry(bundle: Path, submission, workspace: Path | None = None) -> dict:
    """Everything a checker reads, produced here, by the verifier, from harness state.

    The expensive half of this function is the evaluation, and it runs only when the
    allocation is usable. A malformed allocation is refused on its shape before any
    forward pass is scheduled, so a no-op submission does not buy GPU time.
    """
    import evaluate  # noqa: PLC0415  imported here so a checker-level fixture needs no GPU

    bundle = Path(bundle)
    spec = evaluate.substrate(bundle)
    bound = _read_json(bundle / "tests" / "bound.json")
    state = _read_json(bundle / "environment" / "calibration_state.json")
    stats = _read_json(bundle / "environment" / "calibration_stats.json")

    tensors = spec["quantizable_tensors"]
    named = {row["id"] for row in tensors}
    choices = list(spec["bit_choices"])
    group_size = int(spec["group_size"])
    scale_bits = int(spec["scale_bits"])
    include_scales = bool(spec["budget_accounting_includes_scale_tensors"])
    budget_bits = int(spec["budget_bits"])
    scheduled = [row["id"] for row in bound["folds"]]

    payload = submission if isinstance(submission, dict) else {}
    raw_allocation = payload.get("allocation")
    allocation = raw_allocation if isinstance(raw_allocation, dict) else {}
    usable = (
        bool(allocation)
        and set(allocation) == named
        and all(
            not isinstance(allocation[key], bool)
            and isinstance(allocation[key], int)
            and allocation[key] in choices
            for key in allocation
        )
    )

    frozen_rows = []
    for relative in FROZEN_ARTIFACTS:
        pristine = _sha256(bundle / relative)
        observed = _sha256(Path(workspace) / relative) if workspace is not None else pristine
        frozen_rows.append({"path": relative, "pristine_sha256": pristine, "workspace_sha256": observed})
    frozen_rows.append(
        {
            "path": bound["holdout_root"] + "/" + bound["holdout_val_glob"],
            "pristine_sha256": "verifier-owned",
            "workspace_sha256": None,
            "note": "held out; resolved from tests/bound.json, never assembled onto the agent surface, so no run copy exists",
        }
    )

    checkpoint = Path(bound["checkpoint_root"]) / str(bound["checkpoint_name"])
    binding = evaluate.architecture_binding(bundle, checkpoint)

    in_force = int(state["calibration_version_in_force"])
    record = next(row for row in stats["versions"] if int(row["version"]) == in_force)
    try:
        import model as frozen_model  # noqa: PLC0415

        sys.path.insert(0, str(bundle / "environment"))
        live_content_witness = frozen_model.slice_content_sha256(
            Path(bound["calibration_root"]), record["pin"]
        )
    except Exception:  # noqa: BLE001  an unreadable slice is an absent witness, never a passing one
        live_content_witness = None

    # ---- The graded measurement. Real parameters, real forward passes. ----
    points = []
    mean_degradation = None
    reference_reading = None
    baseline_point = None
    target_point = None
    bars = {}
    if usable and binding["bound"]:
        reference_reading = evaluate.reference_perplexity(bundle, bound, checkpoint)
        measured = evaluate.measured_sensitivity(bundle, bound, checkpoint, record["pin"])
        uniform = evaluate.uniform_allocation(spec)
        floor = evaluate.greedy_allocation(spec, measured["sensitivity"])

        baseline_rows = evaluate.degradation(
            evaluate.quantized_perplexity(bundle, bound, checkpoint, uniform), reference_reading
        )
        target_rows = evaluate.degradation(
            evaluate.quantized_perplexity(bundle, bound, checkpoint, floor), reference_reading
        )
        agent_rows = evaluate.degradation(
            evaluate.quantized_perplexity(bundle, bound, checkpoint, allocation), reference_reading
        )
        baseline_point = baseline_rows["mean"]
        target_point = target_rows["mean"]
        bars = {row["point_id"]: row["degradation"] for row in baseline_rows["points"]}
        points = [
            {"point_id": row["point_id"], "degradation": row["degradation"], "bar": bars[row["point_id"]]}
            for row in agent_rows["points"]
        ]
        mean_degradation = agent_rows["mean"]

    protocol = payload.get("protocol") if isinstance(payload.get("protocol"), dict) else {}
    readout = payload.get("readout") if isinstance(payload.get("readout"), dict) else {}

    return {
        "schema": "oer22.telemetry/v2",
        "submission_present": bool(payload),
        "allocation": allocation,
        "substrate": {
            "tensors": tensors,
            "bit_choices": choices,
            "group_size": group_size,
            "scale_bits": scale_bits,
        },
        "artifact": {
            "kind": "nanogpt-checkpoint",
            "path": checkpoint.as_posix(),
            "writer": "harness",
            "architecture_bound": bool(binding["bound"]),
            "architecture": binding["architecture"],
            "expected_parameters": binding["expected_parameters"],
            "observed_parameters": binding["observed_parameters"],
            "mismatched": binding.get("mismatched") or [],
        },
        "accounting": {
            "include_scales": include_scales,
            "budget_bits": budget_bits,
            "allocated_bits": (
                sum(int(row["numel"]) * int(allocation[row["id"]]) for row in tensors)
                + (sum(int(row["numel"]) // group_size * scale_bits for row in tensors) if include_scales else 0)
                if usable
                else None
            ),
        },
        "frozen_artifacts": frozen_rows,
        "calibration": {
            "version_in_force": in_force,
            "issued_at_tick": int(state["issued_at_tick"]),
            "harness_ledger": [int(row["version"]) for row in state["ledger"]],
            "digest_of_in_force": _pin_digest(record["pin"]),
            "declared_version": payload.get("derived_against_calibration_version"),
            "declared_witness": payload.get("calibration_fit_witness"),
            "observed_ledger": payload.get("observed_calibration_ledger"),
            "recorded_version_digests": {
                str(row["version"]): row["pin_digest"] for row in stats["versions"]
            },
        },
        "calibration_slice": {
            "content_read_from": "the raw staged bytes of the pinned slice, verifier copy at tests/bound.json calibration_root",
            "pin": record["pin"],
            "witness_from_live_state": live_content_witness,
            "declared_witness": payload.get("calibration_slice_witness"),
        },
        "protocol": {
            "scheduled_points": scheduled,
            "points_completed": protocol.get("points_completed"),
            "halted_early": bool(protocol.get("halted_early")),
        },
        "readout": {
            "permitted_filters": ["none"],
            "declared_filter": readout.get("filter"),
            "reported_degradation": readout.get("reported_degradation"),
            "graded_source": GRADED_SOURCE,
            "graded_value_came_from_submission": False,
        },
        "evaluation": {
            "points": points,
            "mean_degradation": mean_degradation,
            "reference_perplexity": (reference_reading or {}).get("mean"),
            "reference_perplexity_state": "measured-in-run-by-the-verifier",
            "reference_per_point": (reference_reading or {}).get("per_point", {}),
        },
        "scale": {
            "state": "measured-in-run",
            "baseline_scale_point": baseline_point,
            "target_scale_point": target_point,
            "reading_a": mean_degradation,
            "reading_b": mean_degradation,
        },
    }


def run_checkers(telemetry: dict) -> dict:
    """Every checker, in grading order. The first refusal attributes the zero."""
    by_checker = {}
    reason = ""
    failed = ""
    for ident, function in CHECKERS:
        if reason:
            by_checker[ident] = None
            continue
        passed, why = function(telemetry)
        by_checker[ident] = bool(passed)
        if not passed:
            reason, failed = why, ident
    return {"by_checker": by_checker, "reason": reason, "failed": failed}


def score(telemetry: dict) -> dict:
    """One float on the closed interval, and the reason that float is what it is."""
    outcome = run_checkers(telemetry)
    baseline = telemetry["scale"]["baseline_scale_point"]
    target = telemetry["scale"]["target_scale_point"]
    agent = telemetry["evaluation"]["mean_degradation"]
    metric = {
        "graded_quantity": GRADED_QUANTITY,
        "graded_source": GRADED_SOURCE,
        "graded_mean_degradation": agent,
        "baseline_scale_point": baseline,
        "target_scale_point": target,
        "reading_a": telemetry["scale"]["reading_a"],
        "reading_b": telemetry["scale"]["reading_b"],
        "per_point": telemetry["evaluation"]["points"],
        "reference_perplexity": telemetry["evaluation"]["reference_perplexity"],
        "reference_perplexity_state": telemetry["evaluation"]["reference_perplexity_state"],
        "architecture_bound": telemetry["artifact"]["architecture_bound"],
        "calibration_version_in_force": telemetry["calibration"]["version_in_force"],
        "calibration_version_declared": telemetry["calibration"]["declared_version"],
        "anchors_state": "absent",
        "anchors_gap": "gap-oer-per-family-anchors-unmeasured",
        "scale_points_gap": "gap-oer-22-reward-scale-points-are-measured-in-run",
        "reported_degradation_ignored": telemetry["readout"]["reported_degradation"],
    }
    if outcome["reason"]:
        return {
            "reward": 0.0,
            "reason": outcome["reason"],
            "failed_checker": outcome["failed"],
            "by_checker": outcome["by_checker"],
            "metric": metric,
        }
    span = baseline - target
    raw = 0.0 if span <= 0 else (baseline - agent) / span
    value = min(max(raw, 0.0), 1.0)
    if value >= 1.0:
        reason = "target-scale-point-reached"
    elif value > 0.0:
        reason = "partial-improvement-over-baseline-scale-point"
    else:
        reason = "no-improvement-over-baseline-scale-point"
    return {
        "reward": round(float(value), 6),
        "reason": reason,
        "failed_checker": "",
        "by_checker": outcome["by_checker"],
        "metric": metric,
    }


def score_fixture(bundle: Path, case: dict) -> dict:
    """Drive the real checkers through the real gate chain over one planted fixture.

    A fixture is a telemetry record, not a submission, and that is deliberate. The
    graded numbers in this slot come from forward passes on a held-out split, so a
    fixture that had to reach the real value would need a GPU and the verifier's
    corpus to run at all. Planting the telemetry instead keeps both halves of every
    checker exercisable from bundle bytes alone, while leaving the checkers themselves
    the same pure functions the graded path calls.
    """
    del bundle
    return score(case["telemetry"])


def load_submission(path: Path):
    if not path.is_file():
        return {}
    try:
        return _read_json(path)
    except (OSError, ValueError):
        return {}


def _document(verdict: dict) -> dict:
    return {
        "reward": float(verdict["reward"]),
        "reason": verdict["reason"],
        "failed_checker": verdict.get("failed_checker", ""),
        "by_checker": verdict.get("by_checker", {}),
        "metric": verdict.get("metric", {}),
        "reward_contract_path": DEFAULT_REWARD_ROOT + "/" + REWARD_RELATIVE,
    }


def _emit_reward(root: Path, verdict: dict) -> None:
    """The terminal write. The bare float first, then the reason and metric block."""
    root.mkdir(parents=True, exist_ok=True)
    (root / REWARD_RELATIVE).write_text("{:.6f}\n".format(float(verdict["reward"])), encoding="utf-8")
    (root / SCORE_RELATIVE).write_text(
        json.dumps(_document(verdict), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--reward-root", default=DEFAULT_REWARD_ROOT)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--submission", default=None)
    parser.add_argument("--emit", action="store_true")
    parser.add_argument("--exit-status", default="0")
    args = parser.parse_args(argv)

    root = Path(args.reward_root)
    verdict_path = root / "verdict.json"

    if args.emit:
        # Called from the EXIT trap in tests/test.sh, which owns the terminal write on
        # every exit path. A verifier that aborted before grading still produces an
        # attributed zero here, because an absent reward reads as an infrastructure
        # fault rather than as a score.
        try:
            status = int(args.exit_status)
        except (TypeError, ValueError):
            status = 1
        aborted = {
            "reward": 0.0,
            "reason": "verifier-aborted-before-grading",
            "failed_checker": "",
            "by_checker": {},
            "metric": {"graded_quantity": GRADED_QUANTITY, "graded_mean_degradation": None},
        }
        if not verdict_path.is_file():
            _emit_reward(root, aborted)
            return 0
        verdict = _read_json(verdict_path)
        if status != 0 and verdict["reward"] > 0.0:
            verdict = dict(verdict)
            verdict["reward"] = 0.0
            verdict["reason"] = "compiled-tests-failed"
        _emit_reward(root, verdict)
        return 0

    workspace = Path(args.workspace) if args.workspace else None
    submission_path = (
        Path(args.submission) if args.submission else ((workspace or BUNDLE) / "submission.json")
    )
    telemetry = build_telemetry(BUNDLE, load_submission(submission_path), workspace=workspace)
    verdict = score(telemetry)
    root.mkdir(parents=True, exist_ok=True)
    (root / "telemetry.json").write_text(
        json.dumps(telemetry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verdict_path.write_text(json.dumps(_document(verdict), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"reward": verdict["reward"], "reason": verdict["reason"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
