#!/usr/bin/env python3
"""The verifier's own process: build telemetry from harness state, then grade it.

This file NEVER imports the submission. tests/runner.py launches the submission as a
separate session leader and the only thing that crosses back is inert JSON. Every
graded number below is computed here, inside the verifier, from the verifier's own
pristine copy of the substrate and from the held-out evaluation payload the verifier
owns. `submission.readout.reported_degradation` is recorded for comparison and is
never summed, averaged or substituted into the score.

It imports the checkers and the reference. It reads no clock: the only temporal
ordering in this slot is the harness-owned calibration version and its issued tick,
both integer handles.

The reward contract path this bundle binds is /logs/verifier/reward.txt, carrying the
bare float, with the machine-readable reason and metric block in the companion
/logs/verifier/score.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

TESTS = Path(__file__).resolve().parent
BUNDLE = TESTS.parent
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(BUNDLE / "solution"))

from checkers import (  # noqa: E402
    check_allocation_matches_in_force_calibration,
    check_allocation_well_formed_over_every_tensor,
    check_bit_budget_respected_as_allocated,
    check_calibration_probe_ledger_ordered,
    check_degradation_sustained_across_scheduled_points,
    check_early_stop_does_not_establish_metric,
    check_frozen_corpus_and_reference_unmodified,
    check_graded_degradation_is_verifier_recomputed,
    check_surrogate_scale_witness_matches_live_substrate,
)

try:
    import reference
except ImportError:  # pragma: no cover - the oracle tree is absent from a shipped verifier
    reference = None

REWARD_RELATIVE = "reward.txt"
SCORE_RELATIVE = "score.json"
DEFAULT_REWARD_ROOT = "/logs/verifier"

GRADED_QUANTITY = "mean-perplexity-degradation-over-scheduled-points"
GRADED_SOURCE = "verifier-recomputed-from-pristine-substrate"

# Order is the grading order. The first required checker that refuses attributes the
# zero, so a fixture that plants one defect produces exactly that checker's reason.
CHECKERS = (
    ("allocation_well_formed_over_every_tensor", check_allocation_well_formed_over_every_tensor),
    ("frozen_corpus_and_reference_unmodified", check_frozen_corpus_and_reference_unmodified),
    ("bit_budget_respected_as_allocated", check_bit_budget_respected_as_allocated),
    ("calibration_probe_ledger_ordered", check_calibration_probe_ledger_ordered),
    ("allocation_matches_in_force_calibration", check_allocation_matches_in_force_calibration),
    ("surrogate_scale_witness_matches_live_substrate", check_surrogate_scale_witness_matches_live_substrate),
    ("graded_degradation_is_verifier_recomputed", check_graded_degradation_is_verifier_recomputed),
    ("early_stop_does_not_establish_metric", check_early_stop_does_not_establish_metric),
    ("degradation_sustained_across_scheduled_points", check_degradation_sustained_across_scheduled_points),
)

# The frozen artifacts a run must not modify. The first two are the ones the
# obligation names: the evaluation corpus, whose payload the verifier owns outright
# and whose agent-visible manifest is listed here, and the unquantized reference.
FROZEN_ARTIFACTS = (
    "environment/eval_corpus_manifest.json",
    "environment/reference_model.json",
    "environment/substrate.json",
    "environment/calibration_state.json",
    "environment/calibration_stats.json",
)
HELD_OUT_PAYLOAD = "tests/eval_corpus.json"

ROUND = 9

# The degradation constant is established only in harness-built environment state, at
# environment/substrate.json degradation_constant_K, and sits on no agent-visible byte.
SURROGATE_SCALE_PROBE_TAG = "oer22.surrogate-scale-probe/v1"


def _sha256(path: Path):
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _allocated_bits(tensors, allocation, group_size, scale_bits, include_scales) -> int:
    weight_bits = 0
    for row in tensors:
        weight_bits += int(row["numel"]) * int(allocation[row["id"]])
    if not include_scales:
        return weight_bits
    return weight_bits + sum(int(row["numel"]) // int(group_size) * int(scale_bits) for row in tensors)


def _degradation(tensors, allocation, sensitivity, constant) -> float:
    total = sum(int(row["numel"]) for row in tensors)
    accumulated = 0.0
    for index, row in enumerate(tensors):
        accumulated += float(sensitivity[index]) * int(row["numel"]) * (2.0 ** (-2 * int(allocation[row["id"]])))
    return round(float(constant) * accumulated / total, ROUND)


def _surrogate_scale_witness(probe_value: float) -> str:
    payload = [SURROGATE_SCALE_PROBE_TAG, probe_value]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_telemetry(bundle: Path, submission, workspace: Path | None = None) -> dict:
    """Everything a checker reads, produced here, by the verifier, from harness state."""
    bundle = Path(bundle)
    substrate = _read_json(bundle / "environment" / "substrate.json")
    scale = _read_json(bundle / "tests" / "scale_points.json")
    corpus = _read_json(bundle / HELD_OUT_PAYLOAD)
    state = _read_json(bundle / "environment" / "calibration_state.json")
    stats = _read_json(bundle / "environment" / "calibration_stats.json")

    tensors = substrate["tensors"]
    named = {row["id"] for row in tensors}
    choices = list(substrate["bit_choices"])
    group_size = int(substrate["group_size"])
    scale_bits = int(substrate["scale_bits"])
    constant = float(substrate["degradation_constant_K"])
    include_scales = bool(scale["budget_accounting_includes_scale_tensors"])
    budget_bits = int(scale["budget_bits"])
    scheduled = list(scale["scheduled_points"])
    bars = dict(scale["per_point_bar"])

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
        observed = None
        if workspace is not None:
            observed = _sha256(Path(workspace) / relative)
        else:
            observed = pristine
        frozen_rows.append(
            {"path": relative, "pristine_sha256": pristine, "workspace_sha256": observed}
        )
    frozen_rows.append(
        {
            "path": HELD_OUT_PAYLOAD,
            "pristine_sha256": _sha256(bundle / HELD_OUT_PAYLOAD),
            "workspace_sha256": None,
            "note": "held out; never assembled onto the agent surface, so no run copy exists",
        }
    )

    shards = {row["id"]: row for row in corpus["shards"]}
    points = []
    if usable:
        for point_id in scheduled:
            points.append(
                {
                    "point_id": point_id,
                    "degradation": _degradation(tensors, allocation, shards[point_id]["sensitivity"], constant),
                    "bar": round(float(bars[point_id]), ROUND),
                }
            )
    mean_degradation = (
        round(sum(row["degradation"] for row in points) / len(points), ROUND) if points else None
    )
    reference_perplexity = float(
        _read_json(bundle / "environment" / "reference_model.json")["unquantized_reference_perplexity"]
    )
    reading_b = (
        round(
            sum(reference_perplexity + row["degradation"] for row in points) / len(points)
            - reference_perplexity,
            ROUND,
        )
        if points
        else None
    )

    protocol = payload.get("protocol") if isinstance(payload.get("protocol"), dict) else {}
    readout = payload.get("readout") if isinstance(payload.get("readout"), dict) else {}
    digests = {int(key): value for key, value in scale["calibration_digests"].items()}
    in_force = int(state["calibration_version_in_force"])

    uniform_bits = int(substrate["uniform_reference_bits"])
    uniform_allocation = {row["id"]: uniform_bits for row in tensors}
    in_force_sensitivity = next(
        row["sensitivity"] for row in stats["versions"] if int(row["version"]) == in_force
    )
    probe_value = _degradation(tensors, uniform_allocation, in_force_sensitivity, constant)

    return {
        "schema": "oer22.telemetry/v1",
        "submission_present": bool(payload),
        "allocation": allocation,
        "substrate": {
            "tensors": tensors,
            "bit_choices": choices,
            "group_size": group_size,
            "scale_bits": scale_bits,
        },
        "accounting": {
            "include_scales": include_scales,
            "budget_bits": budget_bits,
            "allocated_bits": (
                _allocated_bits(tensors, allocation, group_size, scale_bits, include_scales)
                if usable
                else None
            ),
        },
        "frozen_artifacts": frozen_rows,
        "calibration": {
            "version_in_force": in_force,
            "issued_at_tick": int(state["issued_at_tick"]),
            "harness_ledger": [int(row["version"]) for row in state["ledger"]],
            "digest_of_in_force": digests.get(in_force),
            "declared_version": payload.get("derived_against_calibration_version"),
            "declared_witness": payload.get("calibration_fit_witness"),
            "observed_ledger": payload.get("observed_calibration_ledger"),
            "recorded_version_digests": {
                str(row["version"]): row["sensitivity_digest"] for row in stats["versions"]
            },
        },
        "surrogate_scale": {
            "constant_read_from": "environment/substrate.json degradation_constant_K, verifier pristine copy",
            "probe_tag": SURROGATE_SCALE_PROBE_TAG,
            "probe_allocation_bits": uniform_bits,
            "probe_calibration_version": in_force,
            "probe_value": probe_value,
            "witness_from_live_substrate": _surrogate_scale_witness(probe_value),
            "declared_witness": payload.get("surrogate_scale_witness"),
        },
        "protocol": {
            "scheduled_points": scheduled,
            "points_completed": protocol.get("points_completed"),
            "halted_early": bool(protocol.get("halted_early")),
        },
        "readout": {
            "permitted_filters": list(scale["permitted_readout_filters"]),
            "declared_filter": readout.get("filter"),
            "reported_degradation": readout.get("reported_degradation"),
            "graded_source": GRADED_SOURCE,
            "graded_value_came_from_submission": False,
        },
        "evaluation": {
            "points": points,
            "mean_degradation": mean_degradation,
            "reference_perplexity": reference_perplexity,
        },
        "scale": {
            "baseline_scale_point": round(float(scale["baseline_scale_point"]), ROUND),
            "target_scale_point": round(float(scale["target_scale_point"]), ROUND),
            "reading_a": mean_degradation,
            "reading_b": reading_b,
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
        "calibration_version_in_force": telemetry["calibration"]["version_in_force"],
        "calibration_version_declared": telemetry["calibration"]["declared_version"],
        "anchors_state": "absent",
        "anchors_gap": "gap-oer-per-family-anchors-unmeasured",
        "scale_points_gap": "gap-oer-22-reward-scale-points-are-substrate-internal",
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
    """Drive the real checkers through the real gate chain over one planted fixture."""
    bundle = Path(bundle)
    mutation = case.get("workspace_mutation")
    if not mutation:
        return score(build_telemetry(bundle, case["submission"]))
    holder = tempfile.mkdtemp(prefix="oer22-fixture-")
    try:
        workspace = Path(holder) / "workspace"
        (workspace / "environment").mkdir(parents=True)
        for relative in FROZEN_ARTIFACTS:
            shutil.copy2(bundle / relative, workspace / relative)
        target = workspace / mutation
        payload = _read_json(target)
        payload["unquantized_reference_perplexity"] = 0.0
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return score(build_telemetry(bundle, case["submission"], workspace=workspace))
    finally:
        shutil.rmtree(holder, ignore_errors=True)


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
        Path(args.submission)
        if args.submission
        else ((workspace or BUNDLE) / "submission.json")
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
