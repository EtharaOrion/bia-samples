"""Pure, deterministic checkers over live state the verifier's own process produced.

Imports are restricted to the allowlist: json, math, hashlib, pathlib, dataclasses,
typing. Nothing here reads a planted file, an environment secret, a clock, a random
source or a socket, and nothing here imports the submission. Every number a checker
compares is either a frozen byte the delivery unit carries, a value the verifier's
own training run recorded, or a value the verifier holds under tests/; no value a
submission printed or wrote is ever read on a path that decides an outcome.

`hashlib` and `pathlib` are imported because the evidence assembly in tests/grade.py
hands digests and paths through this module's dataclass, and a checker re-derives a
digest rather than trusting one it was handed.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Floating-point identity for values both sides computed the same way. Not a
# tolerance on the metric: the metric's own tolerance is a frozen manifest field.
EXACT = 1e-12

BOUND_ROLE = "bound"
SUSTAIN_ROLE = "sustain"
PROGRESS_ROLE = "progress"

DENOMINATOR_SOURCE = "held-out-fineweb-slice-bytes"
WEIGHTS_ORIGIN = "harness-owned-parameters-at-bound-step"


@dataclass(frozen=True)
class Evidence:
    """Everything a checker is allowed to see, assembled by tests/grade.py.

    `telemetry` is the record the verifier's own training run produced.
    `manifest` is the frozen environment/manifest.json of the delivery unit, with
    the verifier's own operating points overlaid on top of it.
    `paired_baseline` is the second training run the verifier performed on the
    handed default construction, which is the floor of the reward scale.
    `held_out` is what tests/runner.py recorded about the slice it resolved.
    """

    telemetry: Dict[str, Any]
    manifest: Dict[str, Any]
    submitted_vocab_digest: str
    paired_baseline: Dict[str, Any] = field(default_factory=dict)
    held_out: Dict[str, Any] = field(default_factory=dict)
    substrate: Dict[str, Any] = field(default_factory=dict)
    delivery_root: str = ""


@dataclass(frozen=True)
class Outcome:
    passed: bool
    reason: str
    detail: str

    def as_dict(self) -> Dict[str, Any]:
        return {"passed": self.passed, "reason": self.reason, "detail": self.detail}


def _ok(detail: str) -> Outcome:
    return Outcome(True, "", detail)


def _no(reason: str, detail: str) -> Outcome:
    return Outcome(False, reason, detail)


def _points(evidence: Evidence) -> List[Dict[str, Any]]:
    rows = evidence.telemetry.get("eval_points")
    return list(rows) if isinstance(rows, list) else []


def _role(evidence: Evidence, role: str) -> List[Dict[str, Any]]:
    return [row for row in _points(evidence) if row.get("role") == role]


def _graded(evidence: Evidence) -> Optional[Dict[str, Any]]:
    row = evidence.telemetry.get("graded")
    return row if isinstance(row, dict) else None


def _verifier(evidence: Evidence) -> Dict[str, Any]:
    block = evidence.manifest.get("verifier_operating_points")
    return block if isinstance(block, dict) else {}


def _floor(evidence: Evidence) -> Optional[float]:
    """The reward-scale floor: the paired baseline reading, measured in this pass."""
    value = (evidence.paired_baseline or {}).get("bits_per_byte")
    return float(value) if isinstance(value, (int, float)) else None


def acceptance_bar(evidence: Evidence) -> Optional[float]:
    """The bar a submission must reach: the live floor less the declared separation."""
    floor = _floor(evidence)
    separation = _verifier(evidence).get("acceptance_separation_bpb")
    if floor is None or not isinstance(separation, (int, float)):
        return None
    return floor - float(separation)


def expected_shapes(architecture: Dict[str, Any]) -> Dict[str, List[int]]:
    """The parameter shapes the frozen architecture implies, derived not transcribed."""
    vocab = int(architecture["vocab_size"])
    model_dim = int(architecture["model_dim"])
    width = int(architecture["num_heads"]) * int(architecture["head_dim"])
    shapes = {
        "embed.weight": [vocab, model_dim],
        "head.weight": [vocab, model_dim],
    }
    for layer in range(int(architecture["num_layers"])):
        prefix = "blocks." + str(layer) + "."
        shapes[prefix + "attention.qkv.weight"] = [3 * width, model_dim]
        shapes[prefix + "attention.proj.weight"] = [model_dim, width]
        shapes[prefix + "mlp.up.weight"] = [4 * model_dim, model_dim]
        shapes[prefix + "mlp.down.weight"] = [model_dim, 4 * model_dim]
    return shapes


# ---------------------------------------------------------------------------
# 1. ABSENCE. An early stop establishes nothing, and says so rather than going quiet.
# ---------------------------------------------------------------------------
def check_early_stop_not_an_established_metric(evidence: Evidence) -> Outcome:
    compute = evidence.telemetry.get("compute") or {}
    graded = _graded(evidence)
    bound = _role(evidence, BOUND_ROLE)
    points = _points(evidence)
    if compute.get("halted_before_bound_point"):
        return _no(
            "early-stop-no-established-metric",
            "the run halted before the bound evaluation point after "
            + str(compute.get("consumed_steps"))
            + " steps, so no metric was established",
        )
    if graded is None or len(bound) != 1:
        return _no(
            "early-stop-no-established-metric",
            "no bound evaluation point was reached, so the record carries no established reading",
        )
    if not points:
        return _no(
            "early-stop-no-established-metric",
            "the record carries no evaluation point at all",
        )
    last = max(row.get("steps", 0) for row in points)
    consumed = compute.get("consumed_steps")
    if not isinstance(consumed, int) or consumed < last:
        return _no(
            "early-stop-no-established-metric",
            "the schedule reaches " + str(last) + " steps and the counter recorded "
            + str(consumed),
        )
    return _ok(
        "no halt record, one bound point, and the counter reached the full schedule at "
        + str(consumed)
        + " optimizer steps"
    )


# ---------------------------------------------------------------------------
# 2. INVARIANT. Every frozen axis is where the substrate declaration left it.
# ---------------------------------------------------------------------------
def check_frozen_axes_unmoved(evidence: Evidence) -> Outcome:
    frozen = evidence.telemetry.get("frozen") or {}
    run = evidence.manifest.get("run") or {}
    architecture = _verifier(evidence).get("architecture")
    if not isinstance(architecture, dict):
        return _no(
            "frozen-axis-moved",
            "the verifier holds no architecture to compare the run against",
        )
    expected = {
        "compute_budget_steps": run.get("compute_budget_steps"),
        "batch_tokens_per_step": run.get("batch_tokens_per_step"),
        "forward_passes_per_step": run.get("forward_passes_per_step"),
        "backward_passes_per_step": run.get("backward_passes_per_step"),
        "vocab_budget": (evidence.manifest.get("tokenizer") or {}).get("vocab_budget"),
    }
    moved = [key for key, value in sorted(expected.items()) if frozen.get(key) != value]
    observed = frozen.get("architecture")
    if not isinstance(observed, dict) or {
        key: observed.get(key) for key in sorted(architecture)
    } != {key: architecture[key] for key in sorted(architecture)}:
        moved.append("architecture")
    declared = (evidence.substrate or {}).get("architecture") or {}
    trimmed = {key: value for key, value in declared.items() if not key.startswith("_")}
    if trimmed and trimmed != {key: architecture[key] for key in sorted(architecture)}:
        moved.append("nanogpt_substrate.json:architecture")
    if moved:
        return _no(
            "frozen-axis-moved",
            "these frozen axes do not match the canonical substrate: " + ", ".join(sorted(moved)),
        )
    return _ok(
        "the run carries the substrate architecture, the frozen batch of "
        + str(expected["batch_tokens_per_step"])
        + " tokens per step and one forward and one backward pass per step"
    )


# ---------------------------------------------------------------------------
# 3. VALUE. WEIGHTS IN THE LOOP. The graded artifact is a real parameter snapshot
#    of the frozen architecture, shape for shape.
#
#    This is the checker that refuses a simulator. A vocab-8 weight table, an
#    arithmetic cost model and a tick counter each fail here, because none of
#    them carries 162201600 parameters in the exact shapes vocab_size 50304,
#    num_layers 12, model_dim 768 and head_dim 128 imply. The shapes are DERIVED
#    from the verifier's own copy of the architecture rather than transcribed, so
#    a tampered environment/ cannot move what the snapshot is checked against.
# ---------------------------------------------------------------------------
def check_snapshot_shapes_match_substrate(evidence: Evidence) -> Outcome:
    model = evidence.telemetry.get("model") or {}
    architecture = _verifier(evidence).get("architecture")
    if not isinstance(architecture, dict):
        return _no(
            "snapshot-shape-off-substrate",
            "the verifier holds no architecture to check the snapshot against",
        )
    observed = model.get("parameter_shapes")
    if not isinstance(observed, dict) or not observed:
        return _no(
            "snapshot-shape-off-substrate",
            "the run record carries no parameter shapes, so no weights were in the loop",
        )
    expected = expected_shapes(architecture)
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if missing or extra:
        return _no(
            "snapshot-shape-off-substrate",
            "the snapshot is missing " + str(len(missing)) + " declared tensors and carries "
            + str(len(extra)) + " undeclared ones, first missing "
            + (missing[0] if missing else "none"),
        )
    wrong = [
        name
        for name in sorted(expected)
        if [int(value) for value in observed[name]] != expected[name]
    ]
    if wrong:
        return _no(
            "snapshot-shape-off-substrate",
            "these tensors do not carry the shape the frozen architecture implies: "
            + ", ".join(wrong[:4]),
        )
    total = sum(
        _product(expected[name]) for name in expected
    )
    declared = _verifier(evidence).get("expected_parameter_count")
    if isinstance(declared, int) and declared != total:
        return _no(
            "snapshot-shape-off-substrate",
            "the shapes sum to " + str(total) + " parameters against the verifier's "
            + str(declared),
        )
    counted = model.get("parameter_count")
    if not isinstance(counted, int) or counted != total:
        return _no(
            "snapshot-shape-off-substrate",
            "the run counted " + str(counted) + " parameters against the "
            + str(total) + " the frozen architecture implies",
        )
    if not isinstance(model.get("snapshot_digest"), str) or len(model["snapshot_digest"]) != 64:
        return _no(
            "snapshot-shape-off-substrate",
            "the run record carries no digest over the parameter bytes",
        )
    return _ok(
        "the graded snapshot carries " + str(counted)
        + " parameters in the " + str(len(expected))
        + " tensors the frozen 12-layer 768-dimension decoder implies"
    )


def _product(shape: List[int]) -> int:
    out = 1
    for value in shape:
        out *= int(value)
    return out


# ---------------------------------------------------------------------------
# 4. ORDERING. The schedule runs forward and the bound point is the last charged one.
# ---------------------------------------------------------------------------
def check_eval_points_ordered_by_steps(evidence: Evidence) -> Outcome:
    points = _points(evidence)
    if len(points) < 2:
        return _no("eval-schedule-out-of-order", "fewer than two evaluation points were recorded")
    steps = [row.get("steps") for row in points]
    for index in range(1, len(steps)):
        left, right = steps[index - 1], steps[index]
        if not isinstance(left, int) or not isinstance(right, int) or right <= left:
            return _no(
                "eval-schedule-out-of-order",
                "point " + str(index) + " at " + str(right) + " steps does not follow "
                + str(left),
            )
    roles = [row.get("role") for row in points]
    if roles.count(BOUND_ROLE) != 1:
        return _no(
            "eval-schedule-out-of-order",
            "the schedule declares " + str(roles.count(BOUND_ROLE)) + " bound points, expected one",
        )
    cut = roles.index(BOUND_ROLE)
    if any(role != PROGRESS_ROLE for role in roles[:cut]):
        return _no(
            "eval-schedule-out-of-order",
            "a point before the bound point is not a progress point",
        )
    if any(role != SUSTAIN_ROLE for role in roles[cut + 1:]):
        return _no(
            "eval-schedule-out-of-order",
            "a point after the bound point is not a sustain point",
        )
    return _ok(
        "the "
        + str(len(points))
        + " evaluation points ascend strictly and the single bound point sits at index "
        + str(cut)
    )


# ---------------------------------------------------------------------------
# 5. VALUE. The compute budget as ACTUALLY SPENT, from the harness's own counter,
#    at exactly one forward and one backward pass per step.
# ---------------------------------------------------------------------------
def check_compute_budget_respected_as_spent(evidence: Evidence) -> Outcome:
    compute = evidence.telemetry.get("compute") or {}
    budget = (evidence.manifest.get("run") or {}).get("compute_budget_steps")
    if compute.get("budget_steps") != budget:
        return _no(
            "compute-budget-overspent",
            "the run recorded a budget of " + str(compute.get("budget_steps"))
            + " against a frozen budget of " + str(budget),
        )
    charged = compute.get("charged_steps")
    if not isinstance(charged, int) or charged > int(budget):
        return _no(
            "compute-budget-overspent",
            "the harness counter recorded " + str(charged)
            + " charged steps against a budget of " + str(budget),
        )
    graded = _graded(evidence)
    if graded is None or graded.get("steps") != budget:
        return _no(
            "compute-budget-overspent",
            "the bound evaluation point sits at "
            + str(None if graded is None else graded.get("steps"))
            + " steps, not at the budget " + str(budget),
        )
    consumed = compute.get("consumed_steps")
    for key in ("forward_passes", "backward_passes"):
        if compute.get(key) != consumed:
            return _no(
                "compute-budget-overspent",
                "the run recorded " + str(compute.get(key)) + " " + key + " against "
                + str(consumed) + " steps, so it is not one gradient per step",
            )
    for row in _points(evidence):
        if row.get("charged_to_budget") and int(row.get("steps", 0)) > int(budget):
            return _no(
                "compute-budget-overspent",
                "a charged evaluation point sits at " + str(row.get("steps"))
                + " steps, past the budget",
            )
    return _ok(
        "the harness counter spent " + str(charged) + " of " + str(budget)
        + " charged steps at one forward and one backward pass each"
    )


# ---------------------------------------------------------------------------
# 6. VALUE. The DENOMINATOR is the byte count of the VERIFIER-HELD slice.
#    This is the headline readout attack for a tokenizer slot: a vocabulary that
#    shrinks the token count must not lower bits per byte, and it cannot unless
#    the denominator moves with it. The same checker holds the second half of the
#    simulator test: the split is the verifier's, is absent from environment/, and
#    was resolved by the caller rather than from anywhere a solver can reach.
# ---------------------------------------------------------------------------
def check_denominator_is_held_out_slice_bytes(evidence: Evidence) -> Outcome:
    denominator = evidence.telemetry.get("denominator") or {}
    readout = evidence.telemetry.get("readout") or {}
    slice_bytes = (evidence.held_out or {}).get("slice_bytes")
    slice_digest = (evidence.held_out or {}).get("slice_sha256")
    if not isinstance(slice_bytes, int) or slice_bytes <= 0:
        return _no(
            "denominator-not-held-out-slice-bytes",
            "the verifier recorded no held-out slice length, it reads " + repr(slice_bytes),
        )
    if (evidence.held_out or {}).get("resolved_from") != "tests/anchors.json":
        return _no(
            "denominator-not-held-out-slice-bytes",
            "the evaluation slice was resolved from "
            + repr((evidence.held_out or {}).get("resolved_from")),
        )
    if (evidence.held_out or {}).get("reachable_from_environment"):
        return _no(
            "denominator-not-held-out-slice-bytes",
            "the evaluation shard is reachable from environment/, so the solver can see the split",
        )
    if readout.get("eval_split_resolved_by") != "caller":
        return _no(
            "denominator-not-held-out-slice-bytes",
            "the harness resolved its own evaluation split, it names "
            + repr(readout.get("eval_split_resolved_by")),
        )
    if denominator.get("source") != DENOMINATOR_SOURCE:
        return _no(
            "denominator-not-held-out-slice-bytes",
            "the denominator names its source as " + repr(denominator.get("source")),
        )
    if denominator.get("bytes") != slice_bytes or denominator.get("digest") != slice_digest:
        return _no(
            "denominator-not-held-out-slice-bytes",
            "the denominator carries " + str(denominator.get("bytes"))
            + " bytes against the " + str(slice_bytes) + " the verifier resolved",
        )
    vocabulary = evidence.telemetry.get("vocabulary") or {}
    tokens = vocabulary.get("eval_tokens")
    if tokens is not None and denominator.get("bytes") == tokens and tokens != slice_bytes:
        return _no(
            "denominator-not-held-out-slice-bytes",
            "the denominator equals the evaluation token count, which is a token normaliser",
        )
    for row in _points(evidence):
        if row.get("denominator_bytes") != slice_bytes:
            return _no(
                "denominator-not-held-out-slice-bytes",
                "the point at " + str(row.get("steps")) + " steps divides by "
                + str(row.get("denominator_bytes")),
            )
    graded = _graded(evidence)
    if graded is not None and graded.get("denominator_bytes") != slice_bytes:
        return _no(
            "denominator-not-held-out-slice-bytes",
            "the graded point divides by " + str(graded.get("denominator_bytes")),
        )
    return _ok(
        "every reading divides by the "
        + str(slice_bytes)
        + " bytes of the verifier-held slice while the vocabulary expressed them in "
        + str(tokens)
        + " tokens"
    )


# ---------------------------------------------------------------------------
# 7. VALUE. The evaluated weights are the ones the harness held at the bound step.
# ---------------------------------------------------------------------------
def check_graded_state_is_harness_owned(evidence: Evidence) -> Outcome:
    readout = evidence.telemetry.get("readout") or {}
    graded = _graded(evidence)
    if readout.get("weights_origin") != WEIGHTS_ORIGIN:
        return _no(
            "graded-state-not-harness-owned",
            "the record names the weights origin as " + repr(readout.get("weights_origin")),
        )
    if readout.get("graded_role") != BOUND_ROLE:
        return _no(
            "graded-state-not-harness-owned",
            "the record grades the point with role " + repr(readout.get("graded_role")),
        )
    bound = _role(evidence, BOUND_ROLE)
    if graded is None or len(bound) != 1:
        return _no(
            "graded-state-not-harness-owned",
            "there is no single bound evaluation point to take a snapshot digest from",
        )
    if graded.get("snapshot_digest") != bound[0].get("snapshot_digest"):
        return _no(
            "graded-state-not-harness-owned",
            "the graded snapshot digest is not the digest recorded at the bound point",
        )
    earlier = [
        row.get("snapshot_digest") for row in _points(evidence) if row.get("role") == PROGRESS_ROLE
    ]
    if graded.get("snapshot_digest") in earlier:
        return _no(
            "graded-state-not-harness-owned",
            "the graded snapshot digest repeats a progress point, so the parameters did not move",
        )
    return _ok(
        "the graded snapshot digest "
        + str(graded.get("snapshot_digest"))[:16]
        + " is the harness parameter state at the bound step and differs from every progress point"
    )


# ---------------------------------------------------------------------------
# 8. VALUE. No smoothing on the graded path.
# ---------------------------------------------------------------------------
def check_graded_readout_unsmoothed(evidence: Evidence) -> Outcome:
    readout = evidence.telemetry.get("readout") or {}
    graded = _graded(evidence)
    if readout.get("filter") != "none":
        return _no(
            "graded-readout-smoothed",
            "the graded path declares readout filter " + repr(readout.get("filter")),
        )
    bound = _role(evidence, BOUND_ROLE)
    if graded is None or len(bound) != 1:
        return _no(
            "graded-readout-smoothed",
            "no single bound point exists, so the graded reading has no raw counterpart",
        )
    raw = bound[0].get("bits")
    taken = graded.get("bits")
    if not isinstance(raw, float) or not isinstance(taken, float):
        return _no("graded-readout-smoothed", "the raw bits at the bound point are not a float")
    if not math.isfinite(raw) or not math.isfinite(taken):
        return _no("graded-readout-smoothed", "a non-finite value reached the graded path")
    if abs(taken - raw) > EXACT:
        return _no(
            "graded-readout-smoothed",
            "the graded bits " + repr(taken) + " differ from the raw bits " + repr(raw)
            + " recorded at the bound point, so a filter was applied",
        )
    return _ok("the graded bits are the raw bits at the bound point, unfiltered")


# ---------------------------------------------------------------------------
# 9. EFFECT. The graded run used the vocabulary the submission handed back.
# ---------------------------------------------------------------------------
def check_submission_vocabulary_took_effect(evidence: Evidence) -> Outcome:
    vocabulary = evidence.telemetry.get("vocabulary") or {}
    slice_bytes = (evidence.held_out or {}).get("slice_bytes")
    if vocabulary.get("submitted_digest") != evidence.submitted_vocab_digest:
        return _no(
            "submission-vocabulary-inert",
            "the graded run digested "
            + str(vocabulary.get("submitted_digest"))[:16]
            + " while the submission returned "
            + str(evidence.submitted_vocab_digest)[:16],
        )
    size = vocabulary.get("size")
    if not isinstance(size, int) or size <= 256:
        return _no(
            "submission-vocabulary-inert",
            "the graded vocabulary holds " + str(size)
            + " entries, so the submission contributed none beyond the single bytes",
        )
    tokens = vocabulary.get("eval_tokens")
    if not isinstance(tokens, int) or not isinstance(slice_bytes, int) or tokens >= slice_bytes:
        return _no(
            "submission-vocabulary-inert",
            "the held-out slice still encodes to " + str(tokens)
            + " tokens over " + str(slice_bytes) + " bytes",
        )
    digest = hashlib.sha256(
        json.dumps([vocabulary.get("submitted_digest")], separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return _ok(
        "the submitted vocabulary of "
        + str(size)
        + " entries reached the graded run, witness "
        + digest[:16]
    )


# ---------------------------------------------------------------------------
# 10. DIVERGENCE. The graded number diverges in provenance from anything the
#     submission reported: it is recomputed from verifier-owned raw quantities.
# ---------------------------------------------------------------------------
def check_metric_not_taken_from_submission_report(evidence: Evidence) -> Outcome:
    graded = _graded(evidence)
    if graded is None:
        return _no("metric-sourced-from-submission", "no verifier-computed graded reading exists")
    bits = graded.get("bits")
    denominator = graded.get("denominator_bytes")
    taken = graded.get("bits_per_byte")
    if not isinstance(bits, float) or not isinstance(denominator, int) or denominator <= 0:
        return _no(
            "metric-sourced-from-submission",
            "the graded reading cannot be recomputed from raw bits and a byte count",
        )
    recomputed = bits / denominator
    if not isinstance(taken, float) or abs(taken - recomputed) > EXACT:
        return _no(
            "metric-sourced-from-submission",
            "the graded value " + repr(taken) + " is not the verifier's own "
            + repr(recomputed) + ", so it came from somewhere else",
        )
    reported = evidence.telemetry.get("reported_by_submission")
    reported = reported if isinstance(reported, dict) else {}
    claimed = reported.get("bits_per_byte")
    if isinstance(claimed, (int, float)) and abs(float(claimed) - recomputed) > EXACT:
        return _ok(
            "the submission claimed "
            + repr(float(claimed))
            + " and the graded value is the verifier's own "
            + repr(recomputed)
            + "; the two diverge and the verifier's stands"
        )
    return _ok(
        "the graded value is recomputed as raw bits over the held-out byte count, "
        + repr(recomputed)
        + ", with no submission-reported value on the path"
    )


# ---------------------------------------------------------------------------
# 11. VALUE. The submission beat the PAIRED DEFAULT CONSTRUCTION, trained in the
#     same pass on the same seed, shards, step budget and held-out slice.
#     Accepting the handed option set scores zero here, by construction.
# ---------------------------------------------------------------------------
def check_beats_default_construction_optimum(evidence: Evidence) -> Outcome:
    graded = _graded(evidence)
    floor = _floor(evidence)
    bar = acceptance_bar(evidence)
    baseline = evidence.paired_baseline or {}
    if graded is None or floor is None or bar is None:
        return _no(
            "default-construction-plateau-not-beaten",
            "no graded reading, or no paired baseline run to place it against",
        )
    if baseline.get("steps") != graded.get("steps"):
        return _no(
            "default-construction-plateau-not-beaten",
            "the paired baseline ran to " + str(baseline.get("steps"))
            + " steps against the graded run's " + str(graded.get("steps"))
            + ", so the two are not comparable",
        )
    if baseline.get("denominator_bytes") != graded.get("denominator_bytes"):
        return _no(
            "default-construction-plateau-not-beaten",
            "the paired baseline was read over a different slice length",
        )
    value = graded.get("bits_per_byte")
    if not isinstance(value, float) or not math.isfinite(value):
        return _no(
            "default-construction-plateau-not-beaten",
            "the graded reading is not a finite number",
        )
    if value > bar:
        return _no(
            "default-construction-plateau-not-beaten",
            "the graded reading " + repr(value) + " does not reach the bar " + repr(bar)
            + ", which is the paired default construction's own reading " + repr(floor)
            + " less the bound separation "
            + repr(_verifier(evidence).get("acceptance_separation_bpb")),
        )
    return _ok(
        "the graded reading "
        + repr(value)
        + " is at or below the bar "
        + repr(bar)
        + ", so the handed construction was beaten by more than the bound separation"
    )


# ---------------------------------------------------------------------------
# 12. INVARIANT. The reading holds across the verifier-scheduled points after it.
# ---------------------------------------------------------------------------
def check_reading_sustained_across_scheduled_points(evidence: Evidence) -> Outcome:
    schedule = evidence.manifest.get("evaluation_schedule") or {}
    tolerance = schedule.get("sustain_tolerance_bpb")
    expected = schedule.get("sustain_offsets") or []
    graded = _graded(evidence)
    sustain = _role(evidence, SUSTAIN_ROLE)
    if graded is None:
        return _no("reading-not-sustained", "no bound reading exists to sustain")
    if tolerance is None:
        return _no("reading-not-sustained", "the manifest binds no sustain tolerance")
    if len(sustain) != len(expected) or not sustain:
        return _no(
            "reading-not-sustained",
            "the record carries " + str(len(sustain)) + " sustain points against "
            + str(len(expected)) + " the verifier schedules",
        )
    ceiling = float(graded.get("bits_per_byte")) + float(tolerance)
    bar = acceptance_bar(evidence)
    for row in sustain:
        value = row.get("bits_per_byte")
        if not isinstance(value, float) or not math.isfinite(value):
            return _no(
                "reading-not-sustained",
                "the sustain point at " + str(row.get("steps")) + " steps carries no reading",
            )
        if row.get("charged_to_budget"):
            return _no(
                "reading-not-sustained",
                "a sustain point was charged to the agent's compute budget",
            )
        if value > ceiling:
            return _no(
                "reading-not-sustained",
                "the reading at " + str(row.get("steps")) + " steps is " + repr(value)
                + ", above the graded reading plus the bound tolerance " + repr(ceiling),
            )
        if bar is not None and value > bar:
            return _no(
                "reading-not-sustained",
                "the reading at " + str(row.get("steps")) + " steps is " + repr(value)
                + ", above the acceptance bar " + repr(bar),
            )
    return _ok(
        "the reading held at all "
        + str(len(sustain))
        + " verifier-scheduled points after the bound point"
    )


# The gate order tests/checkers.yaml declares, restated for a reader. tests/grade.py
# reads the manifest and never this list. The sustain check sits last because it
# re-tests the acceptance bar, and while it stood earlier it emitted its own reason
# for submissions that submission_vocabulary_took_effect and
# beats_default_construction_optimum exist to name.
SELECTOR_ORDER = [
    "check_early_stop_not_an_established_metric",
    "check_frozen_axes_unmoved",
    "check_snapshot_shapes_match_substrate",
    "check_eval_points_ordered_by_steps",
    "check_compute_budget_respected_as_spent",
    "check_denominator_is_held_out_slice_bytes",
    "check_graded_state_is_harness_owned",
    "check_graded_readout_unsmoothed",
    "check_submission_vocabulary_took_effect",
    "check_metric_not_taken_from_submission_report",
    "check_beats_default_construction_optimum",
    "check_reading_sustained_across_scheduled_points",
]


def evidence_from_record(record: Dict[str, Any], delivery_root: str = "") -> Evidence:
    """Assemble evidence from the run record. Pure: it reads the mapping it is given."""
    _ = pathlib.PurePosixPath(delivery_root or ".")
    return Evidence(
        telemetry=record.get("telemetry") or {},
        manifest=record.get("manifest") or {},
        submitted_vocab_digest=str(record.get("submitted_vocab_digest") or ""),
        paired_baseline=record.get("paired_baseline") or {},
        held_out=record.get("held_out") or {},
        substrate=record.get("substrate") or {},
        delivery_root=str(delivery_root or ""),
    )


def score_from_reading(evidence: Evidence) -> float:
    """The bound reward formula, in its lower-is-better form, over live carriers.

    baseline_metric and target_metric are ABSENT for family F10, so the floor is
    the paired baseline reading this pass measured and the span is the declared
    full-credit separation the verifier holds. The shape of the formula is the
    bound one and is not re-derived here.
    """
    graded = _graded(evidence)
    floor = _floor(evidence)
    span = _verifier(evidence).get("full_credit_separation_bpb")
    if graded is None or floor is None or not isinstance(span, (int, float)):
        return 0.0
    value = graded.get("bits_per_byte")
    if not isinstance(value, float) or not math.isfinite(value):
        return 0.0
    if float(span) <= 0.0:
        return 0.0
    raw = (floor - value) / float(span)
    return min(max(raw, 0.0), 1.0)
