"""The ten graded checkers. Pure, deterministic, live state only.

Every function here is a pure function of one telemetry record. That record is
produced by the verifier's own process in tests/grade.py from harness-owned state: the
verifier's pristine nanoGPT checkpoint, the verifier's own held-out FineWeb folds, and
the harness calibration state. If a number reaches a checker, the verifier measured it
by running forward passes. Nothing a submission printed, reported or wrote is ever
summed, averaged or substituted into a graded quantity.

Two checkers changed shape when this slot was re-based onto the nanoGPT substrate, and
they changed because the surrogate they were written against no longer exists.
`graded_artifact_binds_to_declared_architecture` is new and it is the one that makes
the first clause of the simulator test enforceable: the thing quantized has to be a
real parameter snapshot of the declared architecture, shape for shape.
`calibration_slice_witness_matches_live_state` replaces the retired surrogate scale
witness and carries the same property forward, that a discovery value established only
in built environment state gates the graded path, over a value that is now a digest of
real staged FineWeb bytes rather than of an arithmetic constant.

Surviving the AST allowlist:
  - imports are json, math, hashlib and typing only. No pathlib use is needed here
    because no checker opens a file; grade.py does every read.
  - no planted file read, no environment-secret read, no random source, no network.
  - NO CLOCK IS READ. Temporal ordering in this slot is carried entirely by the
    harness-owned calibration version and its issued tick, both integer handles.
  - the submission is never imported. It arrives as inert data.
"""

from __future__ import annotations

import hashlib  # noqa: F401  bound for digest comparison helpers below
import json  # noqa: F401  bound for canonical comparison helpers below
import math
from typing import Any, Dict, List, Tuple

Verdict = Tuple[bool, str]

TOLERANCE = 1e-9

GRADED_SOURCE = "verifier-recomputed-from-pristine-checkpoint"


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def check_allocation_well_formed_over_every_tensor(telemetry: Dict[str, Any]) -> Verdict:
    """VALUE. Exactly one admissible integer bit width per quantizable tensor. No default."""
    allocation = telemetry.get("allocation")
    tensors = telemetry.get("substrate", {}).get("tensors") or []
    choices = set(telemetry.get("substrate", {}).get("bit_choices") or [])
    if not telemetry.get("submission_present"):
        return False, "allocation-malformed"
    if not isinstance(allocation, dict) or not allocation or not tensors or not choices:
        return False, "allocation-malformed"
    named = {row["id"] for row in tensors}
    if set(allocation) != named:
        return False, "allocation-malformed"
    for key in sorted(allocation):
        value = allocation[key]
        if isinstance(value, bool) or not isinstance(value, int) or value not in choices:
            return False, "allocation-malformed"
    return True, ""


def check_frozen_corpus_and_reference_unmodified(telemetry: Dict[str, Any]) -> Verdict:
    """ABSENCE. No modification of a frozen artifact is present on the run surface.

    The architecture declaration is one of the artifacts covered here, so a run that
    edited its local copy of environment/nanogpt_substrate.json to move a frozen axis
    is refused rather than graded against the axis it moved.
    """
    rows: List[Dict[str, Any]] = telemetry.get("frozen_artifacts") or []
    if not rows:
        return False, "frozen-artifact-modified"
    for row in rows:
        pristine = row.get("pristine_sha256")
        workspace = row.get("workspace_sha256")
        if not isinstance(pristine, str) or not pristine:
            return False, "frozen-artifact-modified"
        if workspace is not None and workspace != pristine:
            return False, "frozen-artifact-modified"
    return True, ""


def check_graded_artifact_binds_to_declared_architecture(telemetry: Dict[str, Any]) -> Verdict:
    """VALUE. The thing quantized IS a real snapshot of the frozen architecture.

    The verifier compares the checkpoint's state dict shape for shape against the map
    that environment/nanogpt_substrate.json implies, and this checker refuses unless
    every parameter binds. A weight table, an arithmetic cost model and a statistics
    blob each fail here, which is what stops the graded path being reinstated over a
    stand-in by anything short of editing the declaration itself.
    """
    artifact = telemetry.get("artifact") or {}
    if artifact.get("kind") != "nanogpt-checkpoint":
        return False, "graded-artifact-not-the-declared-architecture"
    if artifact.get("writer") != "harness":
        return False, "graded-artifact-not-the-declared-architecture"
    if not artifact.get("architecture_bound"):
        return False, "graded-artifact-not-the-declared-architecture"
    expected = artifact.get("expected_parameters")
    observed = artifact.get("observed_parameters")
    if not isinstance(expected, int) or not isinstance(observed, int) or expected <= 0:
        return False, "graded-artifact-not-the-declared-architecture"
    if expected != observed or artifact.get("mismatched"):
        return False, "graded-artifact-not-the-declared-architecture"
    return True, ""


def check_bit_budget_respected_as_allocated(telemetry: Dict[str, Any]) -> Verdict:
    """VALUE. Harness accounting over the quantized tensors fits the frozen budget."""
    accounting = telemetry.get("accounting") or {}
    allocated = accounting.get("allocated_bits")
    budget = accounting.get("budget_bits")
    if not accounting.get("include_scales"):
        return False, "bit-budget-overspent"
    if not isinstance(allocated, int) or not isinstance(budget, int):
        return False, "bit-budget-overspent"
    if allocated > budget:
        return False, "bit-budget-overspent"
    return True, ""


def check_calibration_probe_ledger_ordered(telemetry: Dict[str, Any]) -> Verdict:
    """ORDERING. The observed calibration version sequence is ascending and current.

    This is the checker that grades WHEN each calibration statistic was observed to be
    true. It reads two harness-owned handles, the calibration ledger and the version in
    force, and the sequence the submission recorded through the probe. A run that
    probed once and treated the answer as timeless terminates its sequence at the
    pre-drift version, which is what this refuses.
    """
    block = telemetry.get("calibration") or {}
    observed = block.get("observed_ledger")
    harness = block.get("harness_ledger") or []
    in_force = block.get("version_in_force")
    if not isinstance(observed, list) or not observed or not harness:
        return False, "calibration-version-sequence-disordered"
    for item in observed:
        if isinstance(item, bool) or not isinstance(item, int):
            return False, "calibration-version-sequence-disordered"
    if any(observed[i] >= observed[i + 1] for i in range(len(observed) - 1)):
        return False, "calibration-version-sequence-disordered"
    remaining = list(harness)
    for item in observed:
        while remaining and remaining[0] != item:
            remaining.pop(0)
        if not remaining:
            return False, "calibration-version-sequence-disordered"
        remaining.pop(0)
    if observed[-1] != in_force:
        return False, "calibration-version-sequence-disordered"
    return True, ""


def check_allocation_matches_in_force_calibration(telemetry: Dict[str, Any]) -> Verdict:
    """DIVERGENCE. The version the allocation was derived against IS the version in force.

    The declared integer alone is not accepted, because a declaration is free. The
    witness digest is taken over the canonical bytes of the calibration pin the
    submission actually fitted against, so a run that declares the current version
    while carrying a pre-drift fit diverges on the bytes and is refused here.
    """
    block = telemetry.get("calibration") or {}
    declared = block.get("declared_version")
    witness = block.get("declared_witness")
    in_force = block.get("version_in_force")
    expected = block.get("digest_of_in_force")
    if isinstance(declared, bool) or not isinstance(declared, int):
        return False, "allocation-derived-against-stale-calibration"
    if not isinstance(witness, str) or not witness:
        return False, "allocation-derived-against-stale-calibration"
    if declared != in_force or witness != expected:
        return False, "allocation-derived-against-stale-calibration"
    return True, ""


def check_calibration_slice_witness_matches_live_state(telemetry: Dict[str, Any]) -> Verdict:
    """DIVERGENCE. The slice the submission read IS the one live state actually holds.

    The content digest of the calibration slice is established only in built
    environment state, being sha256 over the raw staged FineWeb token bytes the image
    put on disk, and it appears on no agent-visible byte of this bundle. tests/grade.py
    recomputes it from the verifier's own copy of the same pin. A run that never read
    the built state cannot reproduce the digest, and a run that read the wrong slice
    diverges here even when every other declared field is correct.
    """
    block = telemetry.get("calibration_slice") or {}
    expected = block.get("witness_from_live_state")
    declared = block.get("declared_witness")
    if not isinstance(expected, str) or not expected:
        return False, "calibration-slice-content-not-established"
    if not isinstance(declared, str) or not declared:
        return False, "calibration-slice-content-not-established"
    if declared != expected:
        return False, "calibration-slice-content-not-established"
    return True, ""


def check_graded_degradation_is_verifier_recomputed(telemetry: Dict[str, Any]) -> Verdict:
    """VALUE. The graded readout is the verifier's own unsmoothed recomputation.

    Two things must hold. The graded number must carry the verifier's own provenance
    tag, so a number lifted from the submission can never be the graded one. And the
    readout filter the submission declares must be inside the registry in force, which
    admits the single value "none": an EMA-blended, averaged or otherwise filtered
    readout scores zero with reason readout-smoothed rather than being quietly
    recomputed behind the submission's back.
    """
    readout = telemetry.get("readout") or {}
    permitted = readout.get("permitted_filters") or []
    declared = readout.get("declared_filter")
    source = readout.get("graded_source")
    if source != GRADED_SOURCE:
        return False, "readout-smoothed"
    if readout.get("graded_value_came_from_submission"):
        return False, "readout-smoothed"
    if not isinstance(declared, str) or declared not in permitted:
        return False, "readout-smoothed"
    return True, ""


def check_early_stop_does_not_establish_metric(telemetry: Dict[str, Any]) -> Verdict:
    """EFFECT. Halting on a favourable evaluation point establishes nothing.

    A run that stops short of the verifier-scheduled protocol has not established the
    metric. That is graded as a zero with a reason, never as an absent result, so the
    difference between "did not establish it" and "was not measured" stays readable.
    """
    protocol = telemetry.get("protocol") or {}
    scheduled = protocol.get("scheduled_points") or []
    completed = protocol.get("points_completed")
    if protocol.get("halted_early"):
        return False, "early-stop-metric-not-established"
    if isinstance(completed, bool) or not isinstance(completed, int):
        return False, "early-stop-metric-not-established"
    if completed < len(scheduled):
        return False, "early-stop-metric-not-established"
    return True, ""


def check_degradation_sustained_across_scheduled_points(telemetry: Dict[str, Any]) -> Verdict:
    """INVARIANT. The reading holds at EVERY verifier-scheduled evaluation point.

    A single favourable fold is noise, not a result. The bar must hold at every fold of
    the held-out split the verifier schedules, so an allocation tuned to whichever fold
    is cheapest to beat is refused with reason degradation-not-sustained.
    """
    protocol = telemetry.get("protocol") or {}
    scheduled = list(protocol.get("scheduled_points") or [])
    points = (telemetry.get("evaluation") or {}).get("points") or []
    if not scheduled or len(points) != len(scheduled):
        return False, "degradation-not-sustained"
    seen = []
    for row in points:
        value = row.get("degradation")
        bar = row.get("bar")
        if not _finite(value) or not _finite(bar):
            return False, "degradation-not-sustained"
        if float(value) > float(bar) + TOLERANCE:
            return False, "degradation-not-sustained"
        seen.append(row.get("point_id"))
    if seen != scheduled:
        return False, "degradation-not-sustained"
    return True, ""
