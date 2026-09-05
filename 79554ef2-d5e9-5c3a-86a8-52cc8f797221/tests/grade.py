#!/usr/bin/env python3
"""The gate chain and the reward write. Imports the checkers and the reference, never the submission.

Aggregation is `required_pass`, and this file is what that mode describes: every
declared checker is required, they run in the order `tests/checkers.yaml` declares
them, and the FIRST one that fails ends the grading with that checker's own
machine-readable `zero_reason`. Nothing after a failure is allowed to average a
zero away.

What `required_pass` does NOT describe is the number written when the gate passes.
A gate that emitted 1.0 on pass and 0.0 on fail would be a binary reward, and the
delivery contract forbids one. So the gate decides admission and the bound reward
formula decides the number: the graded bits-per-byte reading is placed against the
paired baseline this pass measured, over the declared full-credit separation, and
clipped into [0.0, 1.0]. Reaching the separation scores exactly 1.0 and beating it
also scores exactly 1.0, because the target is a bar and not a point on a gradient.

The reward carrier is the bare float at /logs/verifier/reward.txt and the reason
and metric block travel in /logs/verifier/score.json. Both are written from the
same call, so the instrument's binding and the reason carrier can never disagree.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkers import (  # noqa: E402
    Evidence,
    acceptance_bar,
    check_beats_default_construction_optimum,
    check_compute_budget_respected_as_spent,
    check_denominator_is_held_out_slice_bytes,
    check_early_stop_not_an_established_metric,
    check_eval_points_ordered_by_steps,
    check_frozen_axes_unmoved,
    check_graded_readout_unsmoothed,
    check_graded_state_is_harness_owned,
    check_metric_not_taken_from_submission_report,
    check_reading_sustained_across_scheduled_points,
    check_snapshot_shapes_match_substrate,
    check_submission_vocabulary_took_effect,
    evidence_from_record,
    score_from_reading,
)

SELECTORS = {
    "check_early_stop_not_an_established_metric": check_early_stop_not_an_established_metric,
    "check_frozen_axes_unmoved": check_frozen_axes_unmoved,
    "check_snapshot_shapes_match_substrate": check_snapshot_shapes_match_substrate,
    "check_eval_points_ordered_by_steps": check_eval_points_ordered_by_steps,
    "check_compute_budget_respected_as_spent": check_compute_budget_respected_as_spent,
    "check_denominator_is_held_out_slice_bytes": check_denominator_is_held_out_slice_bytes,
    "check_graded_state_is_harness_owned": check_graded_state_is_harness_owned,
    "check_graded_readout_unsmoothed": check_graded_readout_unsmoothed,
    "check_reading_sustained_across_scheduled_points": check_reading_sustained_across_scheduled_points,
    "check_submission_vocabulary_took_effect": check_submission_vocabulary_took_effect,
    "check_metric_not_taken_from_submission_report": check_metric_not_taken_from_submission_report,
    "check_beats_default_construction_optimum": check_beats_default_construction_optimum,
}

# Reasons this file can emit that no checker owns, because they describe a run that
# never reached the checkers at all. Each is still machine-readable.
CARRIER_REASONS = (
    "submission-absent",
    "submission-crashed",
    "submission-timeout",
    "submission-returned-no-vocabulary",
    "run-record-absent",
    "verifier-aborted",
)

ROW = re.compile(r"^\s*-\s*id:\s*(\S+)\s*$")
FIELD = re.compile(r"^\s{4}([a-z_]+):\s*(.*)$")

# The verifier's own copy of the reward-scale separation, the pin on the held-out
# slice, and the frozen architecture. See ANCHORS_PATH's file for why they are here
# and not in environment/manifest.json.
ANCHORS_PATH = Path(__file__).resolve().parent / "anchors.json"


def load_anchors(path: Path = ANCHORS_PATH) -> dict:
    """The reward-scale span the verifier owns, read from tests/ and not from the bundle.

    WHY THIS FUNCTION EXISTS
    -----------------------
    The reward is

        floor = the paired baseline reading this pass measured
        raw   = (floor - agent_metric) / full_credit_separation_bpb

    so the `full_credit_separation_bpb` is the distance below the paired baseline
    at which a submission scores 1.0. Publishing it on the agent surface would hand
    the solver the exact reading it needs, so it lives here. `tests/` is never
    assembled onto the agent surface.

    The FLOOR is not in this file and is not in any file: it is measured live by
    tests/runner.py, which trains a second decoder on the handed construction at
    the same seed, shards, step budget and held-out slice. There is therefore no
    literal reading of the graded metric anywhere in this bundle to recall.

    A missing or malformed file is a LOUD verifier fault, never a silent zero: an
    absent separation is an instrument that cannot grade, not a submission that
    scored nothing. `main` turns the raised error into the `verifier-aborted` reason.
    """
    if not path.is_file():
        raise SystemExit(
            "verifier anchors absent: " + str(path) + " carries the span of the "
            "reward scale and grading cannot proceed without it"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    block = payload.get("verifier_operating_points")
    if not isinstance(block, dict):
        raise SystemExit(
            "verifier anchors malformed: " + str(path) + " carries no verifier_operating_points"
        )
    for key in ("acceptance_separation_bpb", "full_credit_separation_bpb"):
        value = block.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise SystemExit(
                "verifier anchors malformed: " + str(path) + " carries no positive "
                "verifier_operating_points." + key
            )
    architecture = block.get("architecture")
    if not isinstance(architecture, dict) or "num_layers" not in architecture:
        raise SystemExit(
            "verifier anchors malformed: " + str(path) + " carries no "
            "verifier_operating_points.architecture, which is the shape "
            "check_snapshot_shapes_match_substrate compares the graded snapshot against"
        )
    return block


def merge_verifier_anchors(record: dict, block: dict) -> dict:
    """Overlay the verifier-owned endpoints onto the run record's manifest block.

    The verifier's copy WINS over anything the delivery unit carries under the same
    key, so a tampered `environment/manifest.json` cannot move the reward scale or
    the architecture the graded snapshot is checked against. The record is not
    mutated in place; the caller grades the returned copy.
    """
    merged = dict(record)
    manifest = dict(merged.get("manifest") or {})
    points = dict(manifest.get("verifier_operating_points") or {})
    points.update(block)
    manifest["verifier_operating_points"] = points
    merged["manifest"] = manifest
    return merged


def load_manifest(path: Path) -> list:
    """Checker rows in declared order. PyYAML when present, a shape reader when not.

    The fallback exists because the verifier image is pinned by digest and cannot
    reach the network to install anything. It reads the same file, in the same
    order, so the gate order can never diverge from the declared one.
    """
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return list(payload.get("checkers") or [])
    except ImportError:
        rows, current = [], None
        for line in path.read_text(encoding="utf-8").splitlines():
            head = ROW.match(line)
            if head:
                current = {"id": head.group(1)}
                rows.append(current)
                continue
            body = FIELD.match(line)
            if body and current is not None:
                value = body.group(2).strip().strip('"')
                if value in ("true", "false"):
                    current[body.group(1)] = value == "true"
                elif value.isdigit():
                    current[body.group(1)] = int(value)
                else:
                    current[body.group(1)] = value
        return rows


def gate(evidence: Evidence, rows: list) -> dict:
    """Run every declared checker in declared order and stop at the first failure."""
    observed = []
    for row in rows:
        selector = SELECTORS.get(str(row.get("selector", "")))
        if selector is None:
            return {
                "passed": False,
                "reason": "verifier-aborted",
                "detail": "no selector implements " + str(row.get("id")),
                "observed": observed,
            }
        outcome = selector(evidence)
        observed.append(
            {"id": row.get("id"), "passed": outcome.passed, "reason": outcome.reason,
             "detail": outcome.detail}
        )
        if not outcome.passed:
            reason = outcome.reason or str(row.get("zero_reason") or "verifier-aborted")
            return {"passed": False, "reason": reason, "detail": outcome.detail,
                    "observed": observed}
    return {"passed": True, "reason": "every-required-criterion-passed", "detail": "",
            "observed": observed}


def emit(reward_root: Path, value: float, reason: str, metric: dict, detail: str = "") -> None:
    """Write the bare float and the score document from one call, in that order."""
    reward_root.mkdir(parents=True, exist_ok=True)
    document = {
        "reward": float(value),
        "reason": reason,
        "detail": detail,
        "metric": metric,
    }
    (reward_root / "score.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # The bound reward carrier. Default root /logs/verifier, so the bound contract
    # path is /logs/verifier/reward.txt carrying one bare float.
    (reward_root / "reward.txt").write_text(repr(float(value)) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Grade one run record into one reward.")
    parser.add_argument("--record", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reward-root", required=True)
    parser.add_argument("--delivery-root", default="")
    args = parser.parse_args()

    reward_root = Path(args.reward_root)
    record_path = Path(args.record)
    rows = load_manifest(Path(args.manifest))

    if not record_path.is_file():
        emit(reward_root, 0.0, "run-record-absent", {"bits_per_byte": None},
             "the runner produced no record to grade")
        return 1

    record = json.loads(record_path.read_text(encoding="utf-8"))
    try:
        record = merge_verifier_anchors(record, load_anchors())
    except (SystemExit, ValueError) as exc:
        emit(reward_root, 0.0, "verifier-aborted", {"bits_per_byte": None}, str(exc))
        return 1
    block = (record.get("manifest") or {}).get("verifier_operating_points") or {}
    held_out = record.get("held_out") or {}
    baseline = record.get("paired_baseline") or {}
    metric = {
        "bits_per_byte": None,
        "denominator_bytes": held_out.get("slice_bytes"),
        "denominator_source": "held-out-fineweb-slice-bytes",
        "held_out_slice": held_out.get("source"),
        "held_out_shard_digest_pinned": bool(held_out.get("shard_sha256_pinned")),
        "held_out_shard_digest_gap": None
        if held_out.get("shard_sha256_pinned")
        else block.get("held_out_shard_sha256_gap"),
        "paired_baseline_bpb": baseline.get("bits_per_byte"),
        "acceptance_separation_bpb": block.get("acceptance_separation_bpb"),
        "acceptance_bar_bpb": None,
        "baseline_metric": None,
        "target_metric": None,
        "anchors_state": "absent",
        "anchors_gap": "gap-oer-per-family-anchors-unmeasured",
        "separations_gap": block.get("separations_gap"),
    }

    if record.get("error"):
        reason = str(record["error"])
        reason = reason if reason in CARRIER_REASONS else "verifier-aborted"
        emit(reward_root, 0.0, reason, metric, str(record.get("detail") or ""))
        return 1

    evidence = evidence_from_record(record, args.delivery_root)
    graded = (record.get("telemetry") or {}).get("graded") or {}
    metric["bits_per_byte"] = graded.get("bits_per_byte")
    metric["acceptance_bar_bpb"] = acceptance_bar(evidence)

    verdict = gate(evidence, rows)
    if not verdict["passed"]:
        emit(reward_root, 0.0, verdict["reason"], metric, verdict["detail"])
        return 1

    emit(reward_root, score_from_reading(evidence), "graded", metric, verdict["detail"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
