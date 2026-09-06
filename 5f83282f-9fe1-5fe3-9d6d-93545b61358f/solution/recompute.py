# FORGE-CANARY-BEGIN
# 0: c1bf62f21c72544c873dcfe066156e05675c2da8eb21db7e7c3496e4841a491b
# 1: 47ddb938c35da9dc6c2bdc768fb3ba5af39edb0dd65f970410c9ec566768f66a
# 2: ff720119754a3f6898ae3347decbaaa02ab17446aab067ec201923003bbe19ea
# 3: 24730eaae72f531fb73769a53815b0e50eb8483ab3e4482893db7fee1a9ead30
# FORGE-CANARY-END
#!/usr/bin/env python3
"""Derive every generated artifact in this slot from solution/grounding.yaml alone.

One private grounding file is the single derivation source. This module turns it into
the substrate the agent works on, the checker fixtures the verifier owns, the golden
trajectory, solve.sh, TRUTH.md, rubrics.json and tests/test_output.py. Nothing here
invokes a model, opens a socket, reads a clock, consults a locale or draws from a
random source, so running it twice over frozen bytes produces byte-identical output.

RE-BASED 2026-09-06. The predecessor of this file was written against the retired
statistics surrogate and called `reference.groups_of`, `reference.uniform_allocation`,
`reference.allocate`, `reference.evaluate`, `reference.degradation`,
`reference.sensitivity_digest` and `reference.mean_degradation`. Every one of those
names went with the surrogate, so the module raised AttributeError on import-time use
and no derived artifact in this bundle regenerated. This file is re-based onto the
bytes the bundle actually carries.

Two rules govern what this module writes.

It DERIVES from `solution/grounding.yaml` and from the vendored architecture
declaration that file names. It never restates an architectural number: the 48
quantizable tensors, their element counts, their group counts and the bit budget are
computed from `environment/nanogpt_substrate.json`, so this file cannot drift away
from the declaration.

It VERIFIES the vendored corpus carrier and never writes it. `tests/eval_corpus.json`
holds 1048576 real FineWeb10B token ids this bundle did not author. There is no
construction to re-run over upstream bytes, so a carrier that is missing, truncated or
overwritten stops the whole recompute here rather than being silently rebuilt over the
top of a real collection. The carrier is deliberately absent from the artifact map, so
no emit path in this module can reach it.

No graded number is written by this module. Every graded number in this slot is a
forward pass the verifier runs, and a value for one written down here would be a
fabricated measurement. Where a number cannot be recovered from bundle bytes it is
published as absent under a named gap.

Usage:
    python3 solution/recompute.py            # write every generated artifact
    python3 solution/recompute.py --check    # exit 1 if any committed byte drifted
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

BUNDLE = Path(__file__).resolve().parents[1]
GROUNDING = BUNDLE / "solution" / "grounding.yaml"
BANNER = "GENERATED SECTION. DO NOT HAND-EDIT."
SOURCE = "solution/grounding.yaml"
STAMP = BANNER + " source: " + SOURCE


def load():
    with GROUNDING.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def json_text(payload) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def canonical(payload) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def digest(payload) -> str:
    return hashlib.sha256(canonical(payload)).hexdigest()


# ---------------------------------------------------------------------------
# The vendored carrier, VERIFIED and never written
# ---------------------------------------------------------------------------


def verify_carriers(corpus) -> dict:
    """Read each vendored carrier and assert it is the bytes grounding.yaml declares.

    These are upstream FineWeb10B bytes this bundle did not author, so there is no
    construction to re-run and a mismatch is never repaired by regeneration. A carrier
    that is absent, truncated or overwritten by a generator stops the recompute here.
    """
    out = {}
    for row in corpus["carriers"]:
        relative = str(row["path"])
        path = BUNDLE / relative
        if not path.is_file():
            raise SystemExit(
                "corpus drift: the declared carrier " + relative + " is absent. It carries "
                "vendored upstream bytes and nothing here can regenerate it; recover it from "
                + str(row["shard"]) + " at token offset " + str(row["token_offset"])
                + " by the recipe in grounding.yaml corpus.carriers.recovery_recipe."
            )
        payload = path.read_bytes()
        measured = hashlib.sha256(payload).hexdigest()
        if len(payload) != int(row["length_bytes"]) or measured != str(row["sha256"]):
            raise SystemExit(
                "corpus drift: " + relative + " measures " + str(len(payload)) + " bytes at "
                + measured + " and grounding.yaml corpus.carriers declares "
                + str(row["length_bytes"]) + " bytes at " + str(row["sha256"])
            )
        out[relative] = payload
    return out


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


def architecture(declaration: dict) -> dict:
    keys = ("vocab_size", "num_layers", "model_dim", "head_dim", "num_heads", "seq_len")
    block = declaration["architecture"]
    return {key: int(block[key]) for key in keys}


def parameter_shapes(arch: dict) -> dict:
    """Every parameter shape the declaration implies. The same map environment/model.py builds."""
    vocab, layers = arch["vocab_size"], arch["num_layers"]
    dim = arch["model_dim"]
    inner = arch["head_dim"] * arch["num_heads"]
    shapes = {
        "embed.weight": [vocab, dim],
        "position.weight": [arch["seq_len"], dim],
        "head.weight": [vocab, dim],
        "final_norm.weight": [dim],
    }
    for i in range(layers):
        shapes["blocks." + str(i) + ".norm_attention.weight"] = [dim]
        shapes["blocks." + str(i) + ".attention.qkv.weight"] = [3 * inner, dim]
        shapes["blocks." + str(i) + ".attention.projection.weight"] = [dim, inner]
        shapes["blocks." + str(i) + ".norm_mlp.weight"] = [dim]
        shapes["blocks." + str(i) + ".mlp.up.weight"] = [4 * dim, dim]
        shapes["blocks." + str(i) + ".mlp.down.weight"] = [dim, 4 * dim]
    return shapes


def quantizable(arch: dict, group_size: int) -> list:
    """The matmul weights the allocation owns, in the order the layers declare them.

    Derived from the architecture, never listed. The embeddings, the position table,
    the output head and the norm gains are excluded, which is what grounding.yaml
    substrate.excluded_tensor_ids names.
    """
    dim = arch["model_dim"]
    inner = arch["head_dim"] * arch["num_heads"]
    rows = []
    for i in range(arch["num_layers"]):
        prefix = "blocks." + str(i) + "."
        for suffix, shape in (
            ("attention.qkv.weight", [3 * inner, dim]),
            ("attention.projection.weight", [dim, inner]),
            ("mlp.up.weight", [4 * dim, dim]),
            ("mlp.down.weight", [dim, 4 * dim]),
        ):
            numel = shape[0] * shape[1]
            if numel % group_size != 0:
                raise SystemExit(
                    "the declared group_size " + str(group_size) + " does not divide "
                    + prefix + suffix + ", which carries " + str(numel) + " elements"
                )
            rows.append({"groups": numel // group_size, "id": prefix + suffix,
                         "numel": numel, "shape": list(shape)})
    return rows


def derive(ground) -> dict:
    sub = ground["substrate"]
    declaration_relative = str(sub["declaration"])
    declaration = json.loads((BUNDLE / declaration_relative).read_text(encoding="utf-8"))
    arch = architecture(declaration)

    group_size = int(sub["group_size"])
    scale_bits = int(sub["scale_bits"])
    uniform_bits = int(sub["uniform_reference_bits"])
    bit_min, bit_max = int(sub["bit_min"]), int(sub["bit_max"])
    tensors = quantizable(arch, group_size)

    weight_numel = sum(row["numel"] for row in tensors)
    scale_total = sum(row["groups"] * scale_bits for row in tensors)
    budget_bits = weight_numel * uniform_bits + scale_total

    calibration = ground["calibration"]
    versions = []
    for row in calibration["versions"]:
        pin = {key: row["pin"][key] for key in sorted(row["pin"])}
        versions.append({
            "version": int(row["version"]),
            "tick": int(row["tick"]),
            "label": str(row["label"]),
            "pin": pin,
            "pin_digest": digest(pin),
        })
    in_force = int(calibration["in_force_version"])
    by_version = {row["version"]: row for row in versions}
    if in_force not in by_version:
        raise SystemExit("calibration.in_force_version names a version this file does not declare")

    protocol = ground["evaluation_protocol"]
    scheduled = [str(point) for point in protocol["scheduled_points"]]

    admissible_floors = [
        width for width in range(bit_min, bit_max + 1)
        if weight_numel * width + scale_total <= budget_bits
    ]

    return {
        "declaration_relative": declaration_relative,
        "architecture": arch,
        "parameter_count": len(parameter_shapes(arch)),
        "tensors": tensors,
        "group_size": group_size,
        "scale_bits": scale_bits,
        "uniform_bits": uniform_bits,
        "bit_min": bit_min,
        "bit_max": bit_max,
        "bit_choices": list(range(bit_min, bit_max + 1)),
        "weight_numel": weight_numel,
        "scale_total": scale_total,
        "budget_bits": budget_bits,
        "admissible_floors": admissible_floors,
        "versions": versions,
        "in_force": in_force,
        "in_force_pin": by_version[in_force]["pin"],
        "in_force_digest": by_version[in_force]["pin_digest"],
        "ledger": [int(value) for value in calibration["ledger"]],
        "scheduled": scheduled,
        "tokens_per_point": int(protocol["tokens_per_point"]),
        "include_scales": bool(ground["accounting"]["includes_scale_tensors"]),
        "permitted_filters": [str(value) for value in ground["readout"]["permitted_filters"]],
    }


# ---------------------------------------------------------------------------
# Fixtures. One accepted telemetry base, mutated by exactly one named defect each.
# ---------------------------------------------------------------------------


def accepted_telemetry(ground, model) -> dict:
    planted = ground["fixtures"]["planted"]
    allocation = {row["id"]: model["uniform_bits"] for row in model["tensors"]}
    bar = float(planted["baseline_degradation"])
    target = float(planted["target_degradation"])
    reference = float(planted["reference_perplexity"])
    return {
        "accounting": {
            "allocated_bits": model["budget_bits"],
            "budget_bits": model["budget_bits"],
            "include_scales": model["include_scales"],
        },
        "allocation": dict(sorted(allocation.items())),
        "artifact": {
            "architecture": dict(model["architecture"], _note=str(ground["substrate"]["architecture_note"])),
            "architecture_bound": True,
            "expected_parameters": model["parameter_count"],
            "kind": str(ground["substrate"]["kind"]),
            "mismatched": [],
            "observed_parameters": model["parameter_count"],
            "path": str(planted["checkpoint_path"]),
            "writer": "harness",
        },
        "calibration": {
            "declared_version": model["in_force"],
            "declared_witness": model["in_force_digest"],
            "digest_of_in_force": model["in_force_digest"],
            "harness_ledger": list(model["ledger"]),
            "issued_at_tick": next(
                row["tick"] for row in model["versions"] if row["version"] == model["in_force"]
            ),
            "observed_ledger": list(model["ledger"]),
            "recorded_version_digests": {
                str(row["version"]): row["pin_digest"] for row in model["versions"]
            },
            "version_in_force": model["in_force"],
        },
        "calibration_slice": {
            "content_read_from": str(planted["content_read_from"]),
            "declared_witness": str(planted["slice_witness_placeholder"]),
            "pin": dict(model["in_force_pin"]),
            "witness_from_live_state": str(planted["slice_witness_placeholder"]),
        },
        "evaluation": {
            "mean_degradation": target,
            "points": [
                {"bar": bar, "degradation": target, "point_id": point} for point in model["scheduled"]
            ],
            "reference_per_point": {point: reference for point in model["scheduled"]},
            "reference_perplexity": reference,
            "reference_perplexity_state": "measured-in-run-by-the-verifier",
        },
        "frozen_artifacts": [
            {
                "path": str(planted["frozen_artifact_path"]),
                "pristine_sha256": str(planted["frozen_pristine_digest"]),
                "workspace_sha256": str(planted["frozen_pristine_digest"]),
            }
        ],
        "protocol": {
            "halted_early": False,
            "points_completed": len(model["scheduled"]),
            "scheduled_points": list(model["scheduled"]),
        },
        "readout": {
            "declared_filter": "none",
            "graded_source": "verifier-recomputed-from-pristine-checkpoint",
            "graded_value_came_from_submission": False,
            "permitted_filters": list(model["permitted_filters"]),
            "reported_degradation": None,
        },
        "scale": {
            "baseline_scale_point": bar,
            "reading_a": target,
            "reading_b": target,
            "state": "measured-in-run",
            "target_scale_point": target,
        },
        "schema": "oer22.telemetry/v2",
        "submission_present": True,
        "substrate": {
            "bit_choices": list(model["bit_choices"]),
            "group_size": model["group_size"],
            "scale_bits": model["scale_bits"],
            "tensors": [dict(row) for row in model["tensors"]],
        },
    }


def mutate(name, telemetry, ground, model) -> dict:
    """Apply exactly one named defect. An unknown name is refused, never ignored."""
    planted = ground["fixtures"]["planted"]
    row = json.loads(json.dumps(telemetry))
    if name == "none":
        return row
    if name == "drop-allocation":
        row["allocation"] = {}
        row["accounting"]["allocated_bits"] = None
        return row
    if name == "modify-frozen-artifact":
        row["frozen_artifacts"][0]["workspace_sha256"] = str(planted["frozen_modified_digest"])
        return row
    if name == "unbind-architecture":
        row["artifact"]["architecture_bound"] = False
        row["artifact"]["mismatched"] = [str(planted["mismatched_parameter"])]
        row["artifact"]["observed_parameters"] = int(planted["mismatched_observed_parameters"])
        return row
    if name == "overspend-budget":
        tensor = str(planted["overspend_tensor"])
        bits = int(planted["overspend_bits"])
        if tensor not in row["allocation"]:
            raise SystemExit("fixtures.planted.overspend_tensor names a tensor the architecture does not carry")
        row["allocation"][tensor] = bits
        numel = next(entry["numel"] for entry in model["tensors"] if entry["id"] == tensor)
        row["accounting"]["allocated_bits"] = (
            model["budget_bits"] + numel * (bits - model["uniform_bits"])
        )
        return row
    if name == "truncate-ledger":
        row["calibration"]["observed_ledger"] = [model["ledger"][0]]
        return row
    if name == "declare-stale-calibration":
        stale = min(entry["version"] for entry in model["versions"])
        row["calibration"]["declared_version"] = stale
        row["calibration"]["declared_witness"] = next(
            entry["pin_digest"] for entry in model["versions"] if entry["version"] == stale
        )
        return row
    if name == "wrong-slice-witness":
        row["calibration_slice"]["declared_witness"] = str(planted["slice_witness_wrong"])
        return row
    if name == "absent-slice-witness":
        row["calibration_slice"]["declared_witness"] = None
        return row
    if name == "smooth-readout":
        row["readout"]["declared_filter"] = str(planted["smoothed_filter"])
        return row
    if name == "lift-graded-value":
        row["readout"]["graded_source"] = str(planted["lifted_graded_source"])
        return row
    if name == "halt-early":
        row["protocol"]["halted_early"] = True
        row["protocol"]["points_completed"] = int(planted["early_points_completed"])
        return row
    if name == "miss-the-bar-on-two-folds":
        missed = float(planted["missed_degradation"])
        half = len(row["evaluation"]["points"]) // 2
        for index, point in enumerate(row["evaluation"]["points"]):
            if index >= half:
                point["degradation"] = missed
        row["evaluation"]["mean_degradation"] = round(
            sum(point["degradation"] for point in row["evaluation"]["points"])
            / len(row["evaluation"]["points"]), 9
        )
        row["scale"]["reading_a"] = row["evaluation"]["mean_degradation"]
        row["scale"]["reading_b"] = row["evaluation"]["mean_degradation"]
        return row
    raise SystemExit("grounding.yaml fixtures declares an unknown mutation: " + str(name))


def fixtures(ground, model) -> dict:
    block = ground["fixtures"]
    base = accepted_telemetry(ground, model)
    cases = []
    for row in block["cases"]:
        cases.append({
            "expect_reason": str(row["expect_reason"]),
            "expect_reward": float(row["expect_reward"]),
            "id": str(row["id"]),
            "note": " ".join(str(row["note"]).split()),
            "telemetry": mutate(str(row["mutation"]), base, ground, model),
        })
    return {
        "_source": " ".join(str(block["source_note"]).split()),
        "cases": cases,
        "schema": str(block["schema"]),
        "statement": " ".join(str(block["statement"]).split()),
        "witness_placeholder_note": " ".join(str(block["witness_placeholder_note"]).split()),
    }


# ---------------------------------------------------------------------------
# Artifact bodies
# ---------------------------------------------------------------------------


def artifacts(ground, model) -> dict:
    sub = ground["substrate"]
    protocol = ground["evaluation_protocol"]
    source = " ".join(str(sub["source_note"]).split())
    out = {}

    out["environment/substrate.json"] = json_text({
        "_source": source,
        "architecture": dict(model["architecture"], _note=" ".join(str(sub["architecture_note"]).split())),
        "bit_choices": list(model["bit_choices"]),
        "budget_accounting": " ".join(str(ground["accounting"]["formula"]).split()),
        "budget_accounting_includes_scale_tensors": model["include_scales"],
        "budget_bits": model["budget_bits"],
        "checkpoint": {
            "_note": " ".join(str(sub["checkpoint_note"]).split()),
            "builder": " ".join(str(sub["checkpoint_builder"]).split()),
            "content_sha256_state": " ".join(str(sub["checkpoint_digest_state"]).split()),
            "parameter_count_quantizable": model["weight_numel"],
            "path": str(sub["checkpoint_path"]),
            "schema": str(sub["checkpoint_schema"]),
        },
        "declaration": model["declaration_relative"],
        "evaluation": {key: " ".join(str(value).split()) for key, value in sorted(sub["evaluation"].items())},
        "excluded_tensors": {
            "ids": [str(value) for value in sub["excluded_tensor_ids"]],
            "reason": " ".join(str(sub["excluded_reason"]).split()),
        },
        "group_size": model["group_size"],
        "kind": str(sub["kind"]),
        "quantizable_tensors": [dict(row) for row in model["tensors"]],
        "quantizer": " ".join(str(sub["quantizer"]).split()),
        "scale_bits": model["scale_bits"],
        "schema": str(sub["schema"]),
        "statement": " ".join(str(sub["statement"]).split()),
        "uniform_reference_bits": model["uniform_bits"],
    })

    reference_model = ground["reference_model"]
    out["environment/reference_model.json"] = json_text({
        "_source": source,
        "_state_note": " ".join(str(reference_model["retired_scalar_note"]).split()),
        "architecture_declaration": model["declaration_relative"],
        "frozen": [" ".join(str(value).split()) for value in ground["frozen"]],
        "reference_artifact": str(reference_model["reference_artifact"]),
        "schema": str(reference_model["schema"]),
        "statement": " ".join(str(reference_model["statement"]).split()),
        "unquantized_reference_perplexity": None,
        "unquantized_reference_perplexity_state": str(reference_model["perplexity_state"]),
    })

    out["environment/eval_corpus_manifest.json"] = json_text({
        "_points_note": " ".join(str(protocol["points_note"]).split()),
        "_source": source,
        "corpus": str(protocol["corpus"]),
        "payload_present_in_this_container": bool(protocol["payload_present_in_the_agent_container"]),
        "points": [
            {"point_id": point, "tokens": model["tokens_per_point"]} for point in model["scheduled"]
        ],
        "resolved_by": str(protocol["resolved_by"]),
        "scheduled_points": list(model["scheduled"]),
        "schema": str(protocol["schema_manifest"]),
        "statement": " ".join(str(protocol["manifest_statement"]).split()),
        "val_glob": str(protocol["val_glob"]),
    })

    calibration = ground["calibration"]
    out["environment/calibration_state.json"] = json_text({
        "_source": source,
        "calibration_version_in_force": model["in_force"],
        "issued_at_tick": next(
            row["tick"] for row in model["versions"] if row["version"] == model["in_force"]
        ),
        "ledger": [
            {"label": row["label"], "tick": row["tick"], "version": row["version"]}
            for row in model["versions"]
        ],
        "pin_digest": model["in_force_digest"],
        "schema": str(calibration["schema_state"]),
        "statement": " ".join(str(calibration["state_statement"]).split()),
    })

    out["environment/calibration_stats.json"] = json_text({
        "_source": source,
        "content_digest_state": " ".join(str(calibration["content_digest_state"]).split()),
        "corpus_root": str(calibration["corpus_root"]),
        "pin_digest_form": " ".join(str(calibration["pin_digest_form"]).split()),
        "reader": " ".join(str(calibration["reader"]).split()),
        "schema": str(calibration["schema_stats"]),
        "statement": " ".join(str(calibration["stats_statement"]).split()),
        "versions": [
            {
                "label": row["label"],
                "pin": row["pin"],
                "pin_digest": row["pin_digest"],
                "tick": row["tick"],
                "version": row["version"],
            }
            for row in model["versions"]
        ],
    })

    scale = ground["scale_points"]
    out["tests/scale_points.json"] = json_text({
        "_source": source + " Verifier-owned; never assembled onto the agent surface.",
        "baseline_recipe": " ".join(str(scale["baseline_recipe"]).split()),
        "baseline_scale_point": None,
        "declared_under": [
            "gap-oer-per-family-anchors-unmeasured",
            "gap-oer-22-reward-scale-points-are-measured-in-run",
        ],
        "per_point_bar": None,
        "per_point_bar_recipe": " ".join(str(scale["per_point_bar_recipe"]).split()),
        "permitted_readout_filters": list(model["permitted_filters"]),
        "reference_perplexity_recipe": " ".join(str(scale["reference_perplexity_recipe"]).split()),
        "scale_points_are_not_family_anchors": not bool(scale["is_a_family_anchor"]),
        "scheduled_points": list(model["scheduled"]),
        "schema": "oer22.scale_points/v2",
        "state": str(scale["state"]),
        "statement": " ".join(str(scale["substitution_disclosure"]).split()),
        "target_recipe": " ".join(str(scale["target_recipe"]).split()),
        "target_scale_point": None,
    })

    out["tests/fixtures.json"] = json_text(fixtures(ground, model))
    out["solution/golden_trajectory.json"] = json_text(golden(ground, model))
    out["solution/rubrics.json"] = json_text(rubrics_json(ground, model))
    out["solution/solve.sh"] = solve_sh(model)
    out["solution/TRUTH.md"] = truth_md(ground, model)
    out["tests/test_output.py"] = test_output_py(ground)
    return out


def golden(ground, model) -> dict:
    block = ground["golden_trajectory"]
    rows = []
    for step in block["steps"]:
        observed = int(step["calibration_version_observed"])
        record = next(row for row in model["versions"] if row["version"] == observed)
        rows.append({
            "action": " ".join(str(step["action"]).split()),
            "calibration_fit_witness": record["pin_digest"],
            "calibration_version_observed": observed,
            "pin_observed": dict(record["pin"]),
            "step": int(step["step"]),
        })
    return {
        "_generated": STAMP,
        "graded_mean_degradation": None,
        "graded_points": None,
        "graded_reward": None,
        "graded_state": str(block["graded_state"]),
        "graded_state_note": (
            "Every graded number in this slot is a forward pass tests/evaluate.py runs "
            "inside the verifier over its own pristine checkpoint and its own held-out "
            "folds. None is recoverable from these bytes, so each is published as absent "
            "under gap-oer-22-reward-scale-points-are-measured-in-run rather than "
            "written down here."
        ),
        "schema": str(block["schema"]),
        "steps": rows,
        "submitted": {
            "allocation_state": "fitted-in-run-by-environment/model.py-fit_allocation",
            "calibration_fit_witness": model["in_force_digest"],
            "calibration_slice_witness_state": (
                "established only in built environment state; on no byte of this bundle"
            ),
            "derived_against_calibration_version": model["in_force"],
            "observed_calibration_ledger": list(model["ledger"]),
            "protocol": {"halted_early": False, "points_completed": len(model["scheduled"])},
            "readout": {"filter": "none", "reported_degradation": None},
            "schema": "oer22.submission/v2",
        },
    }


def rubrics_json(ground, model) -> dict:
    del ground
    return {
        "_generated": STAMP,
        "criteria": [
            {
                "expectation": (
                    "the submitted allocation equals the allocation environment/model.py "
                    "fit_allocation returns over the calibration slice in force, tensor for tensor"
                ),
                "id": "allocation-matches-reference",
                "reference_value_state": "fitted-in-run; no allocation table is published in this bundle",
            },
            {
                "expectation": "calibration_fit_witness equals the digest of the in-force calibration pin",
                "id": "fit-witness-matches-in-force-calibration",
                "reference_value": model["in_force_digest"],
            },
            {
                "expectation": (
                    "calibration_slice_witness equals sha256 over the raw staged token bytes of the "
                    "pinned slice, which the run must read back out of built environment state "
                    "because it is on no agent-visible byte"
                ),
                "id": "calibration-slice-witness-matches-live-state",
                "reference_value_state": "established only in built environment state; on no byte of this bundle",
            },
            {
                "expectation": "derived_against_calibration_version equals the version in force at the graded evaluation",
                "id": "declared-calibration-version-in-force",
                "reference_value": model["in_force"],
            },
            {
                "expectation": "observed_calibration_ledger is strictly ascending and ends at the in-force version",
                "id": "observed-ledger-terminates-at-in-force",
                "reference_value": list(model["ledger"]),
            },
            {
                "expectation": "allocated bits including scale tensors do not exceed the frozen budget",
                "id": "budget-respected-under-accounting-in-force",
                "reference_value": model["budget_bits"],
            },
            {
                "expectation": "every scheduled evaluation point completed and halted_early is false",
                "id": "protocol-completed",
                "reference_value": {"halted_early": False, "points_completed": len(model["scheduled"])},
            },
            {
                "expectation": "the declared readout filter is none",
                "id": "readout-unfiltered",
                "reference_value": "none",
            },
            {
                "expectation": "the verifier-recomputed mean degradation reaches the target scale point",
                "id": "graded-mean-degradation",
                "reference_value_state": "measured-in-run-by-the-verifier; absent from every byte of this bundle",
            },
        ],
        "judged": "the solution against its reference answer",
        "schema": "forge.solution_rubrics/v1",
    }


def solve_sh(model) -> str:
    body = [
        "#!/usr/bin/env bash",
        "# " + STAMP,
        "#",
        "# The reference solution entry point. It does the thing the slot is about: it RE-PROBES",
        "# the calibration handle before submitting rather than reusing the answer it fitted",
        "# earlier in the session, and it records the version it fitted against together with two",
        "# witnesses over bytes it actually touched, the pin and the slice content.",
        "#",
        "# It also does the thing the re-base is about: it loads the real nanoGPT checkpoint and",
        "# measures the per-tensor cost curve with real forward passes over the calibration slice",
        "# in force. There is no shipped sensitivity vector to read and no closed-form degradation",
        "# to evaluate, so this script has no path to an allocation that does not run the model.",
        "set -euo pipefail",
        "",
        "# tests/runner.py launches a submission by COPYING its entry point alone into a",
        "# fresh temporary directory, so ${BASH_SOURCE[0]} does not sit in the bundle when",
        "# this runs under the verifier and a bundle path derived from it resolves into the",
        "# temporary directory. The workspace the runner hands over is authoritative, and the",
        "# script's own location is used only when nothing handed one over.",
        'SOLUTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'WORKSPACE="${OER22_WORKSPACE:-$(cd "${SOLUTION_DIR}/.." && pwd)}"',
        'BUNDLE_DIR="${WORKSPACE}"',
        'if [ ! -f "${BUNDLE_DIR}/environment/substrate.json" ]; then',
        '  BUNDLE_DIR="$(cd "${SOLUTION_DIR}/.." && pwd)"',
        "fi",
        'if [ ! -f "${BUNDLE_DIR}/environment/substrate.json" ]; then',
        '  echo "cannot resolve the bundle: no environment/substrate.json under ${BUNDLE_DIR}" >&2',
        "  exit 1",
        "fi",
        'OUT="${OER22_SUBMISSION:-${WORKSPACE}/submission.json}"',
        'DEVICE="${OER22_DEVICE:-cuda}"',
        'CALIBRATION_ROWS="${OER22_CALIBRATION_ROWS:-8}"',
        "",
        "# The checkpoint. environment/substrate.json checkpoint.path is the agent surface's own",
        "# path and it is read from that file rather than restated here. OER22_CHECKPOINT",
        "# overrides it, which is how the verifier points this same entry point at the pristine",
        "# copy tests/bound.json pins without either surface loading the other's file.",
        'CHECKPOINT="${OER22_CHECKPOINT:-$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))[\'checkpoint\'][\'path\'])" "${BUNDLE_DIR}/environment/substrate.json")}"',
        "",
        "# First probe, recorded. The reference agent does not stop here.",
        'python3 "${BUNDLE_DIR}/environment/calibration_probe.py" --workspace "${WORKSPACE}" >/dev/null',
        "",
        'OER22_BUNDLE="${BUNDLE_DIR}" \\',
        'OER22_WORKSPACE="${WORKSPACE}" \\',
        'OER22_OUT="${OUT}" \\',
        'OER22_CHECKPOINT="${CHECKPOINT}" \\',
        'OER22_DEVICE="${DEVICE}" \\',
        'OER22_CALIBRATION_ROWS="${CALIBRATION_ROWS}" \\',
        "python3 - <<'PYTHON'",
        "import json, os, sys",
        "from pathlib import Path",
        "",
        'bundle = Path(os.environ["OER22_BUNDLE"])',
        'workspace = Path(os.environ["OER22_WORKSPACE"])',
        'sys.path.insert(0, str(bundle / "solution"))',
        "import reference",
        "",
        "# The second probe happens inside reference.solve, which re-reads the calibration state",
        "# at fitting time instead of carrying the first answer forward as a timeless fact.",
        "payload = reference.solve(",
        "    workspace,",
        '    Path(os.environ["OER22_CHECKPOINT"]),',
        '    os.environ["OER22_DEVICE"],',
        '    int(os.environ["OER22_CALIBRATION_ROWS"]),',
        ")",
        'if len(payload["allocation"]) != ' + str(len(model["tensors"])) + ":",
        '    raise SystemExit("the fit did not cover every quantizable tensor")',
        'with open(os.environ["OER22_OUT"], "w", encoding="utf-8") as handle:',
        "    json.dump(payload, handle, indent=2, sort_keys=True)",
        '    handle.write("\\n")',
        "PYTHON",
        "",
        "printf 'wrote %s\\n' \"${OUT}\"",
        "",
    ]
    return "\n".join(body)


def truth_md(ground, model) -> str:
    arch = model["architecture"]
    lines = [
        "<!-- " + STAMP + " -->",
        "",
        "# TRUTH, slot OER-22",
        "",
        "## What is graded",
        "",
        "Perplexity degradation at a fixed bit budget, being `ppl(quantized) - ppl(unquantized reference)` over the held-out FineWeb folds, averaged over every scheduled point. Lower is better. Both perplexities are produced by `tests/evaluate.py` inside the verifier process, from the verifier's own pristine checkpoint, in the same run through the same code path. No number the submission reported or wrote ever enters the graded path.",
        "",
        "## The thing the slot is about",
        "",
        "The calibration set the allocation is fitted against **moves** between allocation time and evaluation time, and nothing announces the move. A calibration version is a pin into the frozen FineWeb10B train shards rather than a table of numbers, and the per-tensor cost curve is measured by real forward passes on whichever slice is in force, so a slice that moved is genuinely different activations and genuinely a different answer about which tensor deserves the next bit. An allocation tuned to the slice in force at iteration three is mis-tuned against the slice in force at the graded evaluation. `environment/calibration_probe.py` is the first-class handle for asking; it is a skill under test, not a trap with no handle.",
        "",
        "## The discovery value the graded path depends on",
        "",
        "The **content digest of the calibration slice in force**, being sha256 over the raw staged little-endian uint16 token bytes of the pinned slice. It is established only in built environment state, because it is a digest of FineWeb bytes the image put on disk rather than of anything written down in this bundle, and it appears on no agent-visible byte. A run must read it back out of that state through `environment/model.py slice_content_sha256` or the probe, and carry it as `calibration_slice_witness`. The checker `calibration_slice_witness_matches_live_state` refuses any other reading with `calibration-slice-content-not-established`. No value for that digest is published in this bundle, by construction, and none is invented here.",
        "",
        "## The frozen substrate",
        "",
        "| quantity | value |",
        "|---|---|",
        "| architecture | vocab_size " + str(arch["vocab_size"]) + ", num_layers " + str(arch["num_layers"])
        + ", model_dim " + str(arch["model_dim"]) + ", head_dim " + str(arch["head_dim"])
        + ", num_heads " + str(arch["num_heads"]) + ", seq_len " + str(arch["seq_len"]) + " |",
        "| quantizable tensors | " + str(len(model["tensors"])) + ", being 4 matmul weights per layer |",
        "| quantizable parameters | " + str(model["weight_numel"]) + " |",
        "| excluded from allocation and budget | embeddings, position table, output head, norm gains |",
        "| bit choices | " + ", ".join(str(bit) for bit in model["bit_choices"]) + " |",
        "| group size, scale width | " + str(model["group_size"]) + ", " + str(model["scale_bits"]) + " |",
        "| bit budget, bits | " + str(model["budget_bits"]) + " |",
        "| budget accounting includes scale tensors | " + str(model["include_scales"]) + " |",
        "| unquantized reference perplexity | measured in run, never declared |",
        "| scheduled evaluation points | " + ", ".join(model["scheduled"])
        + ", being four disjoint folds of the held-out validation slice |",
        "| calibration version in force | " + str(model["in_force"]) + " |",
        "",
        "The budget is exactly what the uniform " + str(model["uniform_bits"]) + "-bit allocation costs: `"
        + str(model["weight_numel"]) + " * " + str(model["uniform_bits"]) + " + "
        + str(sum(row["groups"] for row in model["tensors"])) + " * " + str(model["scale_bits"]) + " = "
        + str(model["budget_bits"]) + "`. The uniform allocation therefore saturates it exactly, which is what makes it the baseline scale point. The same arithmetic decides which floors a fit may consider: a uniform-w allocation fits the budget for w in "
        + ", ".join(str(width) for width in model["admissible_floors"]) + " and busts it for every wider w.",
        "",
        "## The reference allocation procedure",
        "",
        "`environment/model.py fit_allocation` is the procedure, expressed once and called by `solution/reference.py` on the agent surface and by `tests/evaluate.py` on the verifier surface. It cuts the pinned calibration slice into one window per scheduled evaluation point, measures the per-tensor cost curve at every admissible width on the FIRST window, reads that curve as its tightest non-increasing non-negative envelope, builds a candidate family over two derived knobs, and then selects on the windows it did not fit on.",
        "",
        "The family is one candidate per admissible floor width, plus one per demotion cap, the cap being how many tensors may go below the uniform reference width and walked over the powers of two up to the tensor count, plus the uniform reference allocation itself. A candidate qualifies only if it is at least as good as the uniform allocation on EVERY selection window, which is the calibration-side reading of the per-fold invariant `check_degradation_sustained_across_scheduled_points` enforces, and among the qualifying candidates the lowest mean wins.",
        "",
        "Three properties follow, and they are the three the predecessor procedure did not have. The fit is anchored to a candidate the budget admits for free, so it can never ship a fit its own measurement ranks below the baseline. The curve is measured at the widths the fit spends at rather than at one width and extrapolated, so the ordering that decides which tensor is starved comes from a reading with signal in it. And selection runs on tokens the curve never saw, because a candidate scored on the tokens it was fitted to is scored on the tokens it overfits.",
        "",
        "## The two scale points",
        "",
        "No allocation table and no degradation number is published here, and that is a consequence of the re-base rather than an omission. Under a real checkpoint they are the output of forward passes on a GPU over a corpus this bundle does not carry, so publishing a table of them would be publishing fabricated measurements. The verifier measures all three in run through one code path:",
        "",
        "- baseline scale point: the uniform allocation at " + str(model["uniform_bits"]) + " bits, which saturates the budget exactly",
        "- target scale point: the allocation `environment/model.py fit_allocation` returns over the calibration slice in force",
        "- per-point bar: the baseline allocation's degradation at each fold",
        "",
        "`raw = (baseline - agent) / (baseline - target)`, then `score = min(max(raw, 0.0), 1.0)`. The reference procedure reaches the target scale point by construction, because the target IS what that procedure measures, so it scores exactly 1.0. Beating it also scores 1.0; the target is a bar.",
        "",
        "Every required checker gates the score. A failing required checker yields 0.0 with its machine-readable reason in `/logs/verifier/score.json`.",
        "",
        "## The simulator test",
        "",
        "First, weights in the loop. The graded artifact is a real parameter snapshot of the frozen architecture, and `check_graded_artifact_binds_to_declared_architecture` compares its state dict shape for shape against the map `environment/nanogpt_substrate.json` implies before anything is evaluated. A weight table, a cost model or a statistics blob fails that comparison. PASSES.",
        "",
        "Second, the verifier recomputes. The graded scalar is produced by `tests/evaluate.py` inside the verifier process on a held-out FineWeb validation slice that `environment/Dockerfile` explicitly refuses to stage into the agent image and that is resolved only from `tests/bound.json`, which no solver-reachable call touches. Everything the in-container harness prints is telemetry. PASSES.",
        "",
        "Third, no forward pass means no score. Delete `GPT.forward` in `environment/model.py` and every perplexity in this bundle raises. There is no closed-form degradation, no shipped sensitivity vector, no count table and no tick counter left anywhere that could keep emitting a number. PASSES.",
        "",
        "## The regeneration chain",
        "",
        "`solution/recompute.py` derives every artifact above from `solution/grounding.yaml` and from the architecture declaration that file names, and it VERIFIES rather than writes the one vendored carrier this bundle holds. `tests/eval_corpus.json` carries "
        + str(ground["corpus"]["carriers"][0]["token_count"]) + " real FineWeb10B token ids and the vendored record corpus; it is absent from the artifact map, so no emit path can overwrite it, and a length or sha256 that does not match the declaration stops the recompute rather than rebuilding over the top of real bytes.",
        "",
        "## Declared gaps",
        "",
    ]
    for row in ground["gaps"]:
        lines.append("- `" + str(row["id"]) + "`: " + " ".join(str(row["effect_here"]).split()))
    lines.append("")
    return "\n".join(lines)


def test_output_py(ground) -> str:
    rejecting = {str(row["id"]): str(row["rejecting_control"]) for row in ground["checkers"]}
    identifiers = [str(row["id"]) for row in ground["checkers"]]
    lines = [
        "#!/usr/bin/env python3",
        '"""' + STAMP,
        "",
        "Compiled tests over the checker fixtures. Verifier-owned.",
        "",
        "One test per declared checker, each carrying that checker's accepting half and its",
        "rejecting half, plus the statement-ambiguity test proving exactly one graded outcome.",
        "",
        "Every case here is a telemetry record rather than a submission, because after the",
        "re-base of this slot onto the nanoGPT substrate the graded numbers are forward passes",
        "over a held-out split. Planting telemetry keeps both halves of every checker",
        "exercisable from bundle bytes alone, with no GPU and no corpus, while leaving the",
        "checkers themselves the same pure functions the graded path calls.",
        '"""',
        "from __future__ import annotations",
        "",
        "import json",
        "import sys",
        "from pathlib import Path",
        "",
        "TESTS = Path(__file__).resolve().parent",
        "sys.path.insert(0, str(TESTS))",
        "",
        "import grade",
        "",
        "FIXTURES = json.loads((TESTS / 'fixtures.json').read_text(encoding='utf-8'))",
        "CASES = {row['id']: row for row in FIXTURES['cases']}",
        "",
        "REJECTING = {",
    ]
    for ident in identifiers:
        lines.append("    '" + ident + "': '" + rejecting[ident] + "',")
    del identifiers
    lines += [
        "}",
        "",
        "",
        "def _score(case_id):",
        "    return grade.score_fixture(TESTS.parent, CASES[case_id])",
        "",
        "",
        "def test_reference_accepted():",
        "    row = _score('reference-accepted')",
        "    assert row['reward'] == 1.0, row",
        "    assert row['reason'] == 'target-scale-point-reached', row",
        "",
        "",
        "def test_every_checker_has_both_halves():",
        '    """One accepting half and one rejecting half for each checker in the gate chain."""',
        "    declared = [ident for ident, _ in grade.CHECKERS]",
        "    assert sorted(declared) == sorted(REJECTING), declared",
        "    accepting = _score('reference-accepted')",
        "    for ident in declared:",
        "        assert accepting['by_checker'][ident] is True, (ident, accepting)",
        "        control = REJECTING[ident]",
        "        rejecting = _score(control)",
        "        assert rejecting['reward'] == 0.0, (ident, rejecting)",
        "        assert rejecting['reason'] == CASES[control]['expect_reason'], (ident, rejecting)",
        "        assert rejecting['by_checker'][ident] is False, (ident, rejecting)",
        "",
        "",
        "def test_every_fixture_reaches_its_declared_outcome():",
        "    for case_id, case in sorted(CASES.items()):",
        "        row = _score(case_id)",
        "        assert row['reward'] == case['expect_reward'], (case_id, row)",
        "        assert row['reason'] == case['expect_reason'], (case_id, row)",
        "",
        "",
        "def test_graded_artifact_must_be_the_declared_decoder():",
        '    """The re-base control. A parameter file that is not the frozen architecture scores zero.',
        "",
        "    This is the checker that makes the first clause of the simulator test enforceable",
        "    from bundle bytes: a weight table, a cost model or a statistics blob cannot present",
        "    the shape map that environment/nanogpt_substrate.json implies.",
        '    """',
        "    row = _score('ctl-graded-artifact-is-not-the-decoder')",
        "    assert row['reward'] == 0.0, row",
        "    assert row['reason'] == 'graded-artifact-not-the-declared-architecture', row",
        "    assert row['by_checker']['graded_artifact_binds_to_declared_architecture'] is False, row",
        "    accepted = _score('reference-accepted')",
        "    assert accepted['metric']['architecture_bound'] is True, accepted",
        "",
        "",
        "def test_calibration_slice_witness_is_sensitive_to_content():",
        '    """A run that never read built environment state cannot produce the content digest."""',
        "    for case_id in ('ctl-calibration-slice-witness-wrong', 'ctl-calibration-slice-witness-absent'):",
        "        row = _score(case_id)",
        "        assert row['reward'] == 0.0, row",
        "        assert row['reason'] == 'calibration-slice-content-not-established', row",
        "        assert row['by_checker']['calibration_slice_witness_matches_live_state'] is False, row",
        "",
        "",
        "def test_no_graded_number_comes_from_the_submission():",
        "    accepted = _score('reference-accepted')",
        "    assert accepted['metric']['graded_source'] == 'verifier-recomputed-from-pristine-checkpoint', accepted",
        "    assert accepted['metric']['reported_degradation_ignored'] is None, accepted",
        "    lifted = _score('ctl-graded-value-lifted-from-submission')",
        "    assert lifted['reward'] == 0.0, lifted",
        "    assert lifted['reason'] == 'readout-smoothed', lifted",
        "",
        "",
        "def test_reference_perplexity_is_measured_not_declared():",
        "    accepted = _score('reference-accepted')",
        "    assert accepted['metric']['reference_perplexity_state'] == 'measured-in-run-by-the-verifier', accepted",
        "",
        "",
        "def test_statement_admits_exactly_one_graded_outcome():",
        "    row = _score('reference-accepted')",
        "    assert row['metric']['graded_quantity'] == 'mean-perplexity-degradation-over-scheduled-points', row",
        "    assert row['metric']['reading_a'] == row['metric']['reading_b'], row",
        "",
        "",
        "def main():",
        "    failures = []",
        "    for name, function in sorted(globals().items()):",
        "        if not name.startswith('test_') or not callable(function):",
        "            continue",
        "        try:",
        "            function()",
        "        except AssertionError as problem:",
        "            failures.append(name + ': ' + str(problem)[:400])",
        "    for row in failures:",
        "        print('FAIL ' + row)",
        "    print('compiled tests: ' + str(len(failures)) + ' failed')",
        "    return 1 if failures else 0",
        "",
        "",
        "if __name__ == '__main__':",
        "    raise SystemExit(main())",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv) -> int:
    check = "--check" in argv
    ground = load()
    carriers = verify_carriers(ground["corpus"])
    model = derive(ground)
    generated = artifacts(ground, model)
    for relative in generated:
        if relative in carriers:
            raise SystemExit(
                "refusing to emit over the vendored carrier " + relative
                + ": it holds upstream bytes this bundle did not author"
            )
    drifted = []
    for relative, text in sorted(generated.items()):
        path = BUNDLE / relative
        if check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                drifted.append(relative)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")
        if relative.endswith(".sh"):
            path.chmod(0o755)
    if check:
        for relative in drifted:
            print("DRIFTED " + relative)
        print("recompute --check: " + str(len(drifted)) + " drifted of " + str(len(generated)))
        print("carriers verified: " + str(len(carriers)))
        return 1 if drifted else 0
    print("recompute: " + str(len(generated)) + " artifacts derived from " + SOURCE)
    print("carriers verified and not written: " + ", ".join(sorted(carriers)))
    return 0


# FORGE-SCREENING-CARRIER-BEGIN
# GENERATED SECTION. DO NOT HAND-EDIT.
# Generated by seed/forge/screenfreeze.py. Derives the contamination-screening provenance carrier
# from the frozen `screening` block in solution/grounding.yaml and nothing else. It opens no
# connection, reads no wall clock, consults no host language setting, draws no entropy, starts no
# child process, and imports nothing outside this tree.
import hashlib as _forge_hashlib
import json as _forge_json
import pathlib as _forge_pathlib
import sys as _forge_sys

import yaml as _forge_yaml

_FORGE_CARRIER_KEYS = (
    "schema",
    "unit_uuid",
    "screening_roots",
    "authority_mode",
    "source_identifiers",
    "fork_ancestry_snapshot",
    "base_commit_sha",
    "applicable_dates",
    "instrument_versions",
    "atom_result_digests",
    "applicability",
    "sanitization_closure",
    "empty_submission_result",
    "attestations",
    "binding_block",
    "keyid",
    "trust_root_public_key_hex",
    "namespace",
    "normalization_domain_version",
    "signer_identity",
)

_FORGE_BINDING_KEYS = (
    "canonical_bundle_hash",
    "pinned_image_digest",
    "binding_envelope",
)

_FORGE_SCREENING_KEY = "screening"
_FORGE_GROUNDING = "grounding.yaml"
_FORGE_CARRIER = "provenance.yaml"
_FORGE_BANNER = "# GENERATED SECTION. DO NOT HAND-EDIT."


def _forge_here():
    return _forge_pathlib.Path(__file__).resolve().parent


def _forge_sorted(value):
    """Sort every container so two runs over the same frozen bytes emit identical bytes."""
    if isinstance(value, dict):
        return {key: _forge_sorted(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_forge_sorted(item) for item in value]
    return value


def _forge_frozen_screening():
    """Read the frozen screening block. Absence is refused rather than defaulted."""
    path = _forge_here() / _FORGE_GROUNDING
    with path.open("r", encoding="utf-8") as handle:
        document = _forge_yaml.safe_load(handle)
    block = (document or {}).get(_FORGE_SCREENING_KEY)
    if not isinstance(block, dict):
        raise SystemExit(
            "solution/grounding.yaml carries no frozen `screening` block, so the provenance "
            "carrier cannot be derived. Refusing to emit a carrier over values nobody froze."
        )
    missing = [key for key in _FORGE_CARRIER_KEYS if key not in block]
    unknown = [key for key in sorted(block) if key not in _FORGE_CARRIER_KEYS]
    if missing or unknown:
        raise SystemExit(
            "the frozen `screening` block does not mirror the closed carrier schema: "
            "missing " + repr(missing) + ", unknown " + repr(unknown)
        )
    return block


def _forge_canonical_bytes(payload):
    return _forge_json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _forge_carrier_payload():
    """Assemble the carrier as exactly the closed key set, in the order the schema fixes.

    The binding block is attached AFTER the canonical payload is hashed and never enters the
    preimage, because a payload that contained a hash of itself would have no acyclic ordering.
    """
    block = _forge_frozen_screening()
    payload = {}
    for key in _FORGE_CARRIER_KEYS:
        if key == "binding_block":
            continue
        payload[key] = _forge_sorted(block[key])
    digest = _forge_hashlib.sha256(_forge_canonical_bytes(payload)).hexdigest()

    binding = _forge_sorted(block["binding_block"]) or {}
    shaped = {key: binding.get(key) for key in _FORGE_BINDING_KEYS}
    ordered = {}
    for key in _FORGE_CARRIER_KEYS:
        ordered[key] = shaped if key == "binding_block" else payload[key]
    return ordered, digest


def _forge_carrier_text():
    payload, digest = _forge_carrier_payload()
    header = (
        _FORGE_BANNER + "\n"
        + "# Derived from solution/grounding.yaml `screening` by solution/recompute.py.\n"
        + "# canonical payload sha256 (binding_block excluded from the preimage): " + digest + "\n"
    )
    body = _forge_yaml.safe_dump(
        payload, sort_keys=False, default_flow_style=False, allow_unicode=False, width=100
    )
    return header + body


def _forge_emit_carrier():
    """Write the carrier, or in check mode compare and report drift. Never both."""
    argv = list(_forge_sys.argv[1:])
    check = "--check" in argv
    path = _forge_here() / _FORGE_CARRIER
    text = _forge_carrier_text()
    if check:
        current = path.read_text(encoding="utf-8") if path.is_file() else ""
        if current == text:
            return 0
        _forge_sys.stderr.write(
            "drift: " + _FORGE_CARRIER + " does not match the carrier derived from the frozen "
            "`screening` block in " + _FORGE_GROUNDING + "\n"
        )
        return 1
    path.write_text(text, encoding="utf-8")
    return 0


_FORGE_INNER_MAIN = main


def main(*args, **kwargs):
    """Run the host generator, then derive the provenance carrier from the frozen block."""
    status = _FORGE_INNER_MAIN(*args, **kwargs)
    drift = _forge_emit_carrier()
    if drift and not status:
        return drift
    return status

# FORGE-SCREENING-CARRIER-END


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
