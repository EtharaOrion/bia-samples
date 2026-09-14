"""The nine graded checkers for slot OER-21. Pure, deterministic, live-state only.

Every function here is a total function of the telemetry record the verifier's own process
produced in tests/evaluate.py. Nothing in this file opens a file, reads an environment
variable, reads a clock, touches the network, consults a random source or imports the
submission. The import set is the AST allowlist and nothing else: json, math, hashlib,
pathlib, dataclasses, typing.

A checker that needed a number the submission printed would be grading the submission's
report rather than the run. There is no such checker here: every number below is one
tests/evaluate.py computed from harness-owned quantized parameter state, by running real
forward passes of the frozen decoder over held-out FineWeb slices the solver cannot see.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

EVAL_PREFIX = "eval-slice-"
SEAL = "state-sealed"
EMITTED = "submission-emitted"
GRADED = "graded"
RAW_SOURCE = "verifier-raw-recompute"
NO_FILTER = "none"
HARNESS = "harness"
VERIFIER_OWNED = "verifier-owned"

REASON_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"

ARCHITECTURE_KEYS = ("vocab_size", "num_layers", "model_dim", "head_dim")


@dataclass(frozen=True)
class Result:
    ident: str
    passed: bool
    reason: str
    observed: str


def is_reason_code(text: Any) -> bool:
    """Lowercase kebab, no leading, trailing or doubled separator. No regex module."""
    if not isinstance(text, str) or not text:
        return False
    for part in text.split("-"):
        if not part:
            return False
        for character in part:
            if character not in REASON_ALPHABET:
                return False
    return True


def _ok(ident: str, observed: str) -> Result:
    return Result(ident, True, "", observed)


def _zero(ident: str, reason: str, observed: str) -> Result:
    return Result(ident, False, reason, observed)


def _points(telemetry: dict) -> list:
    block = telemetry.get("evaluation") or {}
    rows = block.get("points")
    return list(rows) if isinstance(rows, list) else []


def _scaling(telemetry: dict) -> dict:
    return ((telemetry.get("anchors") or {}).get("local_scaling")) or {}


def _substrate(telemetry: dict) -> dict:
    return telemetry.get("substrate") or {}


def digest_of(payload: Any) -> str:
    """A digest a checker computes for itself, so a recorded one can be re-derived."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def check_frozen_checkpoint_and_holdout_unmodified(telemetry: dict) -> Result:
    """The graded parameters and the graded split are the verifier's own, unmoved.

    The checkpoint digest is recomputed by the verifier from the file it actually opened.
    Where tests/anchors.json pins a value it must match; where the pin is recorded as
    absent under a declared gap this checker still requires that a digest was taken and
    that the parameters came from the verifier's own mount, so an absent pin never becomes
    an absent observation.
    """
    ident = "frozen_checkpoint_and_holdout_unmodified"
    block = _substrate(telemetry)
    holdout = telemetry.get("holdout") or {}
    if block.get("checkpoint_source") != VERIFIER_OWNED:
        return _zero(ident, "frozen-inputs-modified", "the graded parameters did not come from the verifier's own copy")
    observed = block.get("checkpoint_digest")
    if not isinstance(observed, str) or len(observed) != 64:
        return _zero(ident, "frozen-inputs-modified", "the verifier recorded no digest over the parameters it graded")
    bound = block.get("checkpoint_digest_bound")
    state = block.get("checkpoint_digest_state")
    if state == "pinned":
        if bound != observed:
            return _zero(
                ident,
                "frozen-inputs-modified",
                "the checkpoint digests to " + observed + " and the bound pin is " + repr(bound),
            )
    elif state != "absent":
        return _zero(ident, "frozen-inputs-modified", "the checkpoint digest pin is in the unknown state " + repr(state))
    if holdout.get("source") != VERIFIER_OWNED:
        return _zero(ident, "frozen-inputs-modified", "the evaluation split did not come from the verifier's own copy")
    if holdout.get("present_in_environment"):
        return _zero(ident, "frozen-inputs-modified", "the evaluation split is reachable from the agent container")
    rows = holdout.get("slices") or []
    if not rows:
        return _zero(ident, "frozen-inputs-modified", "no held-out slice stands behind the record")
    keys = [(row.get("shard"), row.get("token_offset")) for row in rows]
    if len(set(keys)) != len(keys):
        return _zero(ident, "frozen-inputs-modified", "two held-out slices name the same shard and offset")
    return _ok(
        ident,
        "parameters and " + str(len(rows)) + " held-out slice(s) are the verifier's own, checkpoint digest " + observed[:16],
    )


def check_graded_parameters_are_the_canonical_decoder(telemetry: dict) -> Result:
    """The graded artifact is a real parameter snapshot of the frozen architecture.

    Shape-bound to the declared vocab_size, num_layers, model_dim and head_dim, with the
    shard count and the quantizable parameter count the architecture implies. A weight
    table of some other shape, an arithmetic cost model and a tick counter each fail here
    rather than being graded as if they were a model.
    """
    ident = "graded_parameters_are_the_canonical_decoder"
    block = _substrate(telemetry)
    declared = block.get("declared_architecture") or {}
    observed = block.get("observed_architecture") or {}
    missing = [key for key in ARCHITECTURE_KEYS if key not in declared or key not in observed]
    if missing:
        return _zero(ident, "parameters-are-not-the-frozen-decoder", "the record does not carry " + ", ".join(missing))
    moved = [key for key in ARCHITECTURE_KEYS if int(declared[key]) != int(observed[key])]
    if moved or not block.get("architecture_agrees"):
        return _zero(
            ident,
            "parameters-are-not-the-frozen-decoder",
            "these declared axes disagree with the loaded parameters: " + ", ".join(moved or ["architecture_agrees"]),
        )
    shards = block.get("shards")
    quantizable = block.get("quantizable_parameters")
    if not isinstance(shards, int) or not isinstance(quantizable, int):
        return _zero(ident, "parameters-are-not-the-frozen-decoder", "the record carries no shard or parameter count")
    vocab = int(observed["vocab_size"])
    layers = int(observed["num_layers"])
    dim = int(observed["model_dim"])
    expected_shards = 2 + 6 * layers
    expected_parameters = 2 * vocab * dim + layers * (4 * dim * dim + 2 * 4 * dim * dim)
    if shards != expected_shards:
        return _zero(
            ident,
            "parameters-are-not-the-frozen-decoder",
            "the architecture implies " + str(expected_shards) + " shards and the record carries " + str(shards),
        )
    if quantizable != expected_parameters:
        return _zero(
            ident,
            "parameters-are-not-the-frozen-decoder",
            "the architecture implies " + str(expected_parameters)
            + " quantizable parameters and the record carries " + str(quantizable),
        )
    if int(dim) // int(observed["head_dim"]) <= 0:
        return _zero(ident, "parameters-are-not-the-frozen-decoder", "model_dim // head_dim is not a positive head count")
    rows = _points(telemetry)
    if not rows or any("perplexity_reference" not in row for row in rows):
        return _zero(ident, "parameters-are-not-the-frozen-decoder", "no forward pass reading stands behind the record")
    return _ok(
        ident,
        str(shards) + " shards over " + str(quantizable) + " parameters of a "
        + str(layers) + "-layer " + str(dim) + "-dim decoder at vocab " + str(vocab),
    )


def check_evaluation_follows_state_seal(telemetry: dict) -> Result:
    ident = "evaluation_follows_state_seal"
    phases = telemetry.get("phase_log")
    if not isinstance(phases, list) or SEAL not in phases or GRADED not in phases:
        return _zero(ident, "evaluation-precedes-state-seal", "the phase log carries no seal or no graded phase")
    if not phases or phases[0] != EMITTED:
        return _zero(ident, "evaluation-precedes-state-seal", "the phase log does not open at " + EMITTED)
    seal = phases.index(SEAL)
    graded = phases.index(GRADED)
    evaluations = [index for index, name in enumerate(phases) if str(name).startswith(EVAL_PREFIX)]
    early = [phases[index] for index in evaluations if index < seal]
    if early:
        return _zero(
            ident,
            "evaluation-precedes-state-seal",
            "these evaluation points read state before it was sealed: " + ", ".join(early),
        )
    if evaluations and graded < max(evaluations):
        return _zero(ident, "evaluation-precedes-state-seal", "the graded phase precedes an evaluation point")
    if graded != len(phases) - 1:
        return _zero(ident, "evaluation-precedes-state-seal", "the graded phase is not the last phase in the log")
    return _ok(ident, "seal at " + str(seal) + ", " + str(len(evaluations)) + " evaluation point(s) after it, graded last")


def check_submitted_scheme_changes_quantized_state(telemetry: dict) -> Result:
    ident = "submitted_scheme_changes_quantized_state"
    state = telemetry.get("state") or {}
    if not state.get("sealed"):
        return _zero(ident, "scheme-produced-no-effect", "the quantized parameter state was never sealed")
    default = state.get("default_scheme_state_digest")
    submitted = state.get("proposed_scheme_state_digest")
    if not default or not submitted:
        return _zero(ident, "scheme-produced-no-effect", "the harness recorded no comparable parameter state digest")
    if default == submitted:
        return _zero(
            ident,
            "scheme-produced-no-effect",
            "the submitted scheme quantizes to byte-identical parameter state as the handed default configuration",
        )
    return _ok(ident, "submitted state digest differs from the default configuration's")


def check_bit_budget_respected(telemetry: dict) -> Result:
    ident = "bit_budget_respected"
    budget = telemetry.get("budget") or {}
    if budget.get("accounted_by") != HARNESS:
        return _zero(ident, "bit-budget-overspent", "the allocation was not accounted by the harness")
    bound = budget.get("bound_total_bits")
    payload = budget.get("payload_bits")
    overhead = budget.get("scale_overhead_bits")
    allocated = budget.get("allocated_bits")
    values = [bound, payload, overhead, allocated]
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        return _zero(ident, "bit-budget-overspent", "the harness accounting is not an integer record")
    rows = budget.get("per_shard") or []
    summed = sum(int(row.get("payload_bits", 0)) for row in rows)
    summed_overhead = sum(int(row.get("overhead_bits", 0)) for row in rows)
    if summed != payload or summed_overhead != overhead:
        return _zero(ident, "bit-budget-overspent", "the per-shard accounting does not sum to the recorded totals")
    if payload + overhead != allocated:
        return _zero(ident, "bit-budget-overspent", "payload plus scale overhead does not equal the allocated total")
    if allocated > bound:
        return _zero(
            ident,
            "bit-budget-overspent",
            "the scheme allocates " + str(allocated) + " bits against a bound budget of " + str(bound),
        )
    return _ok(ident, str(allocated) + " bits allocated against a bound budget of " + str(bound))


def check_graded_readout_is_raw_recompute(telemetry: dict) -> Result:
    ident = "graded_readout_is_raw_recompute"
    readout = telemetry.get("readout") or {}
    graded = telemetry.get("graded") or {}
    if readout.get("graded_source") != RAW_SOURCE or graded.get("source") != RAW_SOURCE:
        return _zero(ident, "readout-smoothed-not-raw", "the graded number did not come from the verifier's own recomputation")
    if readout.get("graded_filter") != NO_FILTER:
        return _zero(ident, "readout-smoothed-not-raw", "a filter sits on the graded path")
    requested = readout.get("requested_graded_filter")
    if requested != NO_FILTER:
        return _zero(
            ident,
            "readout-smoothed-not-raw",
            "the submission asked for the graded readout to be filtered by " + repr(requested),
        )
    reported = readout.get("scheme_reported_metric")
    if reported is not None:
        return _zero(
            ident,
            "readout-smoothed-not-raw",
            "the submitted artifact carries its own metric field " + repr(reported.get("key")),
        )
    rows = _points(telemetry)
    if not rows:
        return _zero(ident, "readout-smoothed-not-raw", "no evaluation point stands behind the graded number")
    for row in rows:
        if row.get("filter") != NO_FILTER:
            return _zero(ident, "readout-smoothed-not-raw", "evaluation point " + str(row.get("slice")) + " is filtered")
    recomputed = max(float(row["degradation"]) for row in rows)
    recorded = graded.get("metric")
    if not isinstance(recorded, (int, float)) or math.isnan(float(recorded)):
        return _zero(ident, "readout-smoothed-not-raw", "the graded metric is not a real number")
    if abs(float(recorded) - recomputed) > 0.0:
        return _zero(
            ident,
            "readout-smoothed-not-raw",
            "the graded number " + repr(recorded) + " is not the raw worst-slice reading " + repr(recomputed),
        )
    return _ok(ident, "graded number equals the unfiltered worst-slice reading " + repr(recomputed))


def check_evaluation_not_halted_early(telemetry: dict) -> Result:
    ident = "evaluation_not_halted_early"
    block = telemetry.get("evaluation") or {}
    halt = block.get("halt_requested_after")
    if halt is not None:
        return _zero(
            ident,
            "evaluation-halted-early",
            "the submission asked to halt after " + str(halt) + " evaluation point(s), so the metric was never established",
        )
    scheduled = block.get("scheduled_points")
    completed = block.get("completed_points")
    if not isinstance(scheduled, int) or scheduled <= 0:
        return _zero(ident, "evaluation-halted-early", "the verifier scheduled no evaluation point")
    if completed != scheduled or len(_points(telemetry)) != scheduled:
        return _zero(
            ident,
            "evaluation-halted-early",
            str(completed) + " of " + str(scheduled) + " scheduled evaluation points completed",
        )
    return _ok(ident, "all " + str(scheduled) + " scheduled evaluation points completed")


def check_degradation_sustained_across_scheduled_points(telemetry: dict) -> Result:
    ident = "degradation_sustained_across_scheduled_points"
    rows = _points(telemetry)
    if not rows:
        return _zero(ident, "degradation-not-sustained", "no evaluation point carries a reading")
    values = [float(row["degradation"]) for row in rows]
    negative = [row["slice"] for row, value in zip(rows, values) if value < 0.0]
    if negative:
        return _zero(
            ident,
            "degradation-not-sustained",
            "these slices report a negative degradation, which the cross-entropy definition forbids: "
            + ", ".join(str(item) for item in negative),
        )
    block = telemetry.get("evaluation") or {}
    tolerance = block.get("sustain_tolerance")
    if not isinstance(tolerance, (int, float)):
        return _zero(ident, "degradation-not-sustained", "no sustain tolerance was measured for this run")
    if block.get("sustain_tolerance_source") != "verifier-measured-over-the-handed-option-space":
        return _zero(ident, "degradation-not-sustained", "the sustain tolerance was not measured by the verifier itself")
    spread = max(values) - min(values)
    if spread > float(tolerance):
        return _zero(
            ident,
            "degradation-not-sustained",
            "the reading spans " + repr(round(spread, 9)) + " across scheduled points against a measured tolerance of "
            + repr(tolerance) + ", so it was harvested from a favourable slice rather than sustained",
        )
    return _ok(
        ident,
        "reading sustained across " + str(len(values)) + " scheduled points, spread " + repr(round(spread, 9)),
    )


def scaled_improvement(telemetry: dict):
    """The bounded scaling of the graded metric. None when the record cannot carry one."""
    scaling = _scaling(telemetry)
    graded = (telemetry.get("graded") or {}).get("metric")
    baseline = scaling.get("default_optimum")
    target = scaling.get("wider_probe_optimum")
    if not all(isinstance(value, (int, float)) for value in (graded, baseline, target)):
        return None
    span = float(baseline) - float(target)
    if span <= 0.0:
        return None
    return (float(baseline) - float(graded)) / span


def check_beats_default_scheme_optimum(telemetry: dict) -> Result:
    ident = "beats_default_scheme_optimum"
    scaling = _scaling(telemetry)
    if scaling.get("default_optimum_source") != "verifier-search-over-handed-option-space":
        return _zero(ident, "default-scheme-plateau", "the bar was not measured by the verifier's own search")
    if scaling.get("wider_probe_source") != "verifier-widening-probe":
        return _zero(ident, "default-scheme-plateau", "the target endpoint was not measured by the verifier's own probe")
    baseline = scaling.get("default_optimum")
    target = scaling.get("wider_probe_optimum")
    if isinstance(baseline, (int, float)) and isinstance(target, (int, float)) and float(baseline) - float(target) <= 0.0:
        # A substrate condition and not a submission defect: the verifier's widening probe
        # found nothing below the handed optimum on these parameters, so the two endpoints
        # of the bounded scaling coincide and no submission can be separated from any other.
        # It carries its own reason so it is never misreported as the submission plateauing.
        return _zero(
            ident,
            "scaling-span-nonpositive",
            "the handed optimum " + repr(baseline) + " and the widening probe " + repr(target)
            + " leave no positive span to scale between, which is a property of the frozen"
            + " parameters rather than of this submission",
        )
    raw = scaled_improvement(telemetry)
    if raw is None:
        return _zero(ident, "default-scheme-plateau", "the record carries no comparable scaling endpoints")
    margin = scaling.get("separation_margin")
    if not isinstance(margin, (int, float)):
        return _zero(ident, "default-scheme-plateau", "no separation margin is bound in the record")
    if raw < float(margin):
        return _zero(
            ident,
            "default-scheme-plateau",
            "the submitted scheme improves on the handed option set's measured optimum "
            + repr(scaling.get("default_optimum"))
            + " by a scaled "
            + repr(round(raw, 9))
            + ", which is inside the separation margin "
            + repr(margin)
            + ", so it sits in the option set it was handed",
        )
    return _ok(ident, "scaled improvement over the handed optimum is " + repr(round(raw, 9)))


def relative_path(name: str) -> str:
    """Bundle-relative posix form, used when a checker names a carrier in its output."""
    return PurePosixPath(name).as_posix()
