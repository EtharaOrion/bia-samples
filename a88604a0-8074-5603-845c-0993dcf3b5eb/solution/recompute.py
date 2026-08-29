# FORGE-CANARY-BEGIN
# 0: 6cd6d4095b8c46edb19844d9a47b8bed30e9f18c9c08e6a2f6e00e50efb536c7
# 1: da9ddbd74935fb6553ce0e09af0fa47966411c5d0c0d325b5ac90692b3fe10d4
# 2: 3edff244794cb33456487492145c4e0d8cc1beec9b7c011807e4b6469ff7f42e
# 3: 5d2d36feaeda32df545c2e90cb95ab59a03cb2ee42d5dac76cd28116a82b4baf
# FORGE-CANARY-END
#!/usr/bin/env python3
"""Derive every generated artifact in this slot from solution/grounding.yaml alone.

One private grounding file is the single derivation source. This module turns it
into the substrate the agent works on, the held-out evaluation payload the verifier
owns, the checker fixtures, the golden trajectory, solve.sh, TRUTH.md, rubrics.json
and tests/test_output.py. Nothing here invokes a model, opens a socket, reads a
clock, consults a locale or draws from a random source, so running it twice over
frozen bytes produces byte-identical output.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))

import reference  # noqa: E402

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


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


def derive(ground) -> dict:
    sub = ground["substrate"]
    tensors = [{"id": row["id"], "numel": int(row["numel"])} for row in sub["tensors"]]
    group_size = int(sub["group_size"])
    scale_bits = int(sub["scale_bits"])
    constant = float(sub["degradation_constant_K"])
    bit_min, bit_max = int(sub["bit_min"]), int(sub["bit_max"])
    uniform_bits = int(sub["uniform_reference_bits"])

    for row in tensors:
        row["groups"] = reference.groups_of(row["numel"], group_size)

    uniform = reference.uniform_allocation(tensors, uniform_bits)
    budget_bits = reference.allocated_bits(tensors, uniform, group_size, scale_bits, True)

    base = [float(value) for value in ground["evaluation_corpus"]["base_sensitivity"]]
    shards = []
    for row in ground["evaluation_corpus"]["shards"]:
        vector = [round(base[i] * float(row["multiplier"][i]), reference.ROUND) for i in range(len(base))]
        shards.append({"id": row["id"], "tokens": int(row["tokens"]), "sensitivity": vector})

    mean_vector = [
        round(sum(shard["sensitivity"][i] for shard in shards) / len(shards), reference.ROUND)
        for i in range(len(base))
    ]

    versions = []
    for row in ground["calibration"]["versions"]:
        if row["sensitivity_source"] == "base_sensitivity":
            vector = [round(value, reference.ROUND) for value in base]
        elif row["sensitivity_source"] == "mean_over_shards":
            vector = list(mean_vector)
        else:
            raise ValueError("unknown sensitivity_source: " + str(row["sensitivity_source"]))
        versions.append(
            {
                "version": int(row["version"]),
                "tick": int(row["tick"]),
                "label": str(row["label"]),
                "sensitivity": vector,
                "sensitivity_digest": reference.sensitivity_digest(vector),
            }
        )

    in_force = int(ground["calibration"]["in_force_version"])
    by_version = {row["version"]: row for row in versions}
    scheduled = list(ground["evaluation_protocol"]["scheduled_points"])
    ordered_shards = [next(s for s in shards if s["id"] == point) for point in scheduled]

    def evaluate(allocation):
        return reference.evaluate(tensors, allocation, ordered_shards, constant)

    bars = {row["point_id"]: row["degradation"] for row in evaluate(uniform)}
    reference_allocation = reference.allocate(
        tensors, by_version[in_force]["sensitivity"], budget_bits, group_size, scale_bits,
        bit_min, bit_max, True,
    )
    stale_allocation = reference.allocate(
        tensors, by_version[1]["sensitivity"], budget_bits, group_size, scale_bits,
        bit_min, bit_max, True,
    )
    overspend_allocation = reference.allocate(
        tensors, by_version[in_force]["sensitivity"], budget_bits, group_size, scale_bits,
        bit_min, bit_max, False,
    )
    starved_vector = list(by_version[in_force]["sensitivity"])
    for index, row in enumerate(tensors):
        if base[index] < 0.2:
            starved_vector[index] = 0.0
    starved_allocation = reference.allocate(
        tensors, starved_vector, budget_bits, group_size, scale_bits, bit_min, bit_max, True,
    )

    uniform_points = evaluate(uniform)
    reference_points = evaluate(reference_allocation)
    baseline_scale_point = reference.mean_degradation(uniform_points)
    target_scale_point = reference.mean_degradation(reference_points)

    return {
        "tensors": tensors,
        "group_size": group_size,
        "scale_bits": scale_bits,
        "constant": constant,
        "bit_min": bit_min,
        "bit_max": bit_max,
        "uniform_bits": uniform_bits,
        "uniform_allocation": uniform,
        "budget_bits": budget_bits,
        "shards": shards,
        "ordered_shards": ordered_shards,
        "scheduled": scheduled,
        "bars": bars,
        "versions": versions,
        "in_force": in_force,
        "ledger": [int(v) for v in ground["calibration"]["ledger"]],
        "reference_allocation": reference_allocation,
        "stale_allocation": stale_allocation,
        "overspend_allocation": overspend_allocation,
        "starved_allocation": starved_allocation,
        "reference_points": reference_points,
        "uniform_points": uniform_points,
        "baseline_scale_point": baseline_scale_point,
        "target_scale_point": target_scale_point,
        "reference_perplexity": float(ground["substrate"]["reference_perplexity"]),
        "permitted_filters": list(ground["readout"]["permitted_filters"]),
        "include_scales": bool(ground["accounting"]["includes_scale_tensors"]),
    }


# ---------------------------------------------------------------------------
# Submissions
# ---------------------------------------------------------------------------


def submission(model, allocation, version, witness_vector, ledger, points_completed,
               halted_early, readout_filter, reported):
    return {
        "schema": "oer22.submission/v1",
        "allocation": {key: int(value) for key, value in sorted(allocation.items())},
        "quantization_scheme": {"group_size": model["group_size"], "scale_bits": model["scale_bits"]},
        "derived_against_calibration_version": int(version),
        "calibration_fit_witness": reference.sensitivity_digest(witness_vector),
        "observed_calibration_ledger": [int(v) for v in ledger],
        "protocol": {
            "points_completed": int(points_completed),
            "halted_early": bool(halted_early),
        },
        "readout": {"filter": readout_filter, "reported_degradation": reported},
    }


def reference_submission(model):
    version = model["in_force"]
    vector = next(row["sensitivity"] for row in model["versions"] if row["version"] == version)
    return submission(
        model,
        model["reference_allocation"],
        version,
        vector,
        model["ledger"],
        len(model["scheduled"]),
        False,
        "none",
        reference.mean_degradation(model["reference_points"]),
    )


def fixtures(model) -> dict:
    version = model["in_force"]
    current = next(row["sensitivity"] for row in model["versions"] if row["version"] == version)
    stale = next(row["sensitivity"] for row in model["versions"] if row["version"] == 1)
    accepted = reference_submission(model)

    def case(ident, expect_reward, expect_reason, payload, mutate=None, note=""):
        return {
            "id": ident,
            "expect_reward": expect_reward,
            "expect_reason": expect_reason,
            "workspace_mutation": mutate,
            "note": note,
            "submission": payload,
        }

    empty = json.loads(json.dumps(accepted))
    empty["allocation"] = {}

    modified = json.loads(json.dumps(accepted))

    overspend = submission(
        model, model["overspend_allocation"], version, current, model["ledger"],
        len(model["scheduled"]), False, "none", None,
    )
    truncated = submission(
        model, model["reference_allocation"], version, current, [1],
        len(model["scheduled"]), False, "none", None,
    )
    frozen_answer = submission(
        model, model["stale_allocation"], 1, stale, model["ledger"],
        len(model["scheduled"]), False, "none", None,
    )
    smoothed = submission(
        model, model["reference_allocation"], version, current, model["ledger"],
        len(model["scheduled"]), False, "ema", None,
    )
    early = submission(
        model, model["reference_allocation"], version, current, model["ledger"],
        1, True, "none", None,
    )
    starved = submission(
        model, model["starved_allocation"], version, current, model["ledger"],
        len(model["scheduled"]), False, "none", None,
    )
    starved["calibration_fit_witness"] = reference.sensitivity_digest(current)

    cases = [
        case("reference-accepted", 1.0, "target-scale-point-reached", accepted,
             note="the accepting half: the live checkers accept THIS reference, bound by sha256 below"),
        case("ctl-no-op-submission", 0.0, "allocation-malformed", empty,
             note="a no-op submission carrying no allocation at all"),
        case("ctl-frozen-artifact-modified", 0.0, "frozen-artifact-modified", modified,
             mutate="environment/reference_model.json",
             note="the evaluation reference edited in the run workspace"),
        case("ctl-budget-accounted-weights-only", 0.0, "bit-budget-overspent", overspend,
             note="stale control for silent mutation bit-budget-accounting-includes-scale-tensors"),
        case("ctl-probe-ledger-truncated-at-v1", 0.0, "calibration-version-sequence-disordered", truncated,
             note="stale control for silent mutation calibration-probe-ledger-appends-v2"),
        case("ctl-allocation-frozen-at-calibration-v1", 0.0, "allocation-derived-against-stale-calibration",
             frozen_answer,
             note="stale control for the headline silent mutation calibration-set-drifts-to-v2"),
        case("ctl-readout-ema-blended", 0.0, "readout-smoothed", smoothed,
             note="stale control for silent mutation readout-filter-registry-closes-ema"),
        case("ctl-early-stop-on-favourable-point", 0.0, "early-stop-metric-not-established", early,
             note="halted at the first favourable evaluation point"),
        case("ctl-sustained-claim-over-two-points", 0.0, "degradation-not-sustained", starved,
             note="stale control for silent mutation evaluation-point-schedule-extends"),
    ]
    return {
        "_generated": STAMP,
        "reference_module_sha256": hashlib.sha256(
            (BUNDLE / "solution" / "reference.py").read_bytes()
        ).hexdigest(),
        "reference_submission_sha256": hashlib.sha256(reference.canonical(accepted)).hexdigest(),
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# Artifact bodies
# ---------------------------------------------------------------------------


def artifacts(ground, model) -> dict:
    out = {}

    out["environment/substrate.json"] = json_text(
        {
            "_generated": STAMP,
            "schema": "oer22.substrate/v1",
            "kind": "deterministic-surrogate",
            "surrogate_disclosure": ground["substrate"]["surrogate_disclosure"],
            "degradation_form": ground["substrate"]["degradation_form"],
            "degradation_constant_K": model["constant"],
            "bit_choices": list(range(model["bit_min"], model["bit_max"] + 1)),
            "uniform_reference_bits": model["uniform_bits"],
            "group_size": model["group_size"],
            "scale_bits": model["scale_bits"],
            "budget_bits": model["budget_bits"],
            "budget_accounting": ground["accounting"]["formula"],
            "budget_accounting_includes_scale_tensors": model["include_scales"],
            "tensors": model["tensors"],
        }
    )

    out["environment/reference_model.json"] = json_text(
        {
            "_generated": STAMP,
            "schema": "oer22.reference_model/v1",
            "unquantized_reference_perplexity": model["reference_perplexity"],
            "frozen": ground["frozen"],
            "statement": (
                "This is the unquantized reference the graded degradation is measured "
                "against. It is frozen. The verifier grades from its own pristine copy."
            ),
        }
    )

    out["environment/eval_corpus_manifest.json"] = json_text(
        {
            "_generated": STAMP,
            "schema": "oer22.eval_corpus_manifest/v1",
            "statement": (
                "The evaluation corpus payload is held out and never appears on the agent "
                "surface. This manifest names the evaluation points and their sizes only."
            ),
            "scheduled_points": model["scheduled"],
            "points": [{"point_id": row["id"], "tokens": row["tokens"]} for row in model["ordered_shards"]],
        }
    )

    out["environment/calibration_state.json"] = json_text(
        {
            "_generated": STAMP,
            "schema": "oer22.calibration_state/v1",
            "statement": (
                "Harness-owned calibration state. The version in force is the one an "
                "allocation is graded against. Temporal ordering is carried by the "
                "harness tick, a monotonically issued integer handle. No clock is read."
            ),
            "calibration_version_in_force": model["in_force"],
            "issued_at_tick": next(
                row["tick"] for row in model["versions"] if row["version"] == model["in_force"]
            ),
            "ledger": [
                {"version": row["version"], "tick": row["tick"], "label": row["label"]}
                for row in model["versions"]
            ],
            "sensitivity_digest": next(
                row["sensitivity_digest"] for row in model["versions"] if row["version"] == model["in_force"]
            ),
        }
    )

    out["environment/calibration_stats.json"] = json_text(
        {
            "_generated": STAMP,
            "schema": "oer22.calibration_stats/v1",
            "statement": (
                "Per-version estimated per-tensor sensitivity. A version that is not in "
                "force is retained for the record and is NOT the version an allocation is "
                "graded against."
            ),
            "tensor_order": [row["id"] for row in model["tensors"]],
            "versions": [
                {
                    "version": row["version"],
                    "tick": row["tick"],
                    "sensitivity": row["sensitivity"],
                    "sensitivity_digest": row["sensitivity_digest"],
                }
                for row in model["versions"]
            ],
        }
    )

    out["tests/eval_corpus.json"] = json_text(
        {
            "_generated": STAMP,
            "schema": "oer22.eval_corpus/v1",
            "statement": "Held out. Never assembled onto the non-oracle agent surface.",
            "tensor_order": [row["id"] for row in model["tensors"]],
            "shards": model["shards"],
        }
    )

    out["tests/scale_points.json"] = json_text(
        {
            "_generated": STAMP,
            "schema": "oer22.scale_points/v1",
            "statement": (
                "Substrate-internal scale points, recomputed from frozen bytes. These are "
                "NOT family anchors and they do not close "
                "gap-oer-per-family-anchors-unmeasured."
            ),
            "baseline_scale_point": model["baseline_scale_point"],
            "target_scale_point": model["target_scale_point"],
            "per_point_bar": model["bars"],
            "scheduled_points": model["scheduled"],
            "permitted_readout_filters": model["permitted_filters"],
            "budget_bits": model["budget_bits"],
            "budget_accounting_includes_scale_tensors": model["include_scales"],
            "calibration_version_in_force": model["in_force"],
            "calibration_ledger": model["ledger"],
            "calibration_digests": {
                str(row["version"]): row["sensitivity_digest"] for row in model["versions"]
            },
        }
    )

    out["tests/fixtures.json"] = json_text(fixtures(model))

    out["solution/golden_trajectory.json"] = json_text(golden(ground, model))
    out["solution/solve.sh"] = solve_sh(model)
    out["solution/TRUTH.md"] = truth_md(ground, model)
    out["solution/rubrics.json"] = json_text(rubrics_json(model))
    out["tests/test_output.py"] = test_output_py(model)
    return out


def golden(ground, model) -> dict:
    rows = []
    for step in ground["golden_trajectory"]["steps"]:
        observed = int(step["calibration_version_observed"])
        vector = next(row["sensitivity"] for row in model["versions"] if row["version"] == observed)
        rows.append(
            {
                "step": int(step["step"]),
                "action": str(step["action"]),
                "calibration_version_observed": observed,
                "calibration_fit_witness": reference.sensitivity_digest(vector),
                "allocation_if_fitted_here": reference.allocate(
                    model["tensors"], vector, model["budget_bits"], model["group_size"],
                    model["scale_bits"], model["bit_min"], model["bit_max"], True,
                ),
            }
        )
    return {
        "_generated": STAMP,
        "schema": "oer22.golden_trajectory/v1",
        "steps": rows,
        "submitted": reference_submission(model),
        "graded_points": model["reference_points"],
        "graded_mean_degradation": model["target_scale_point"],
        "graded_reward": 1.0,
    }


def solve_sh(model) -> str:
    body = [
        "#!/usr/bin/env bash",
        "# " + STAMP,
        "#",
        "# The reference solution entry point. It does the thing the slot is about: it",
        "# RE-PROBES the calibration handle before submitting rather than reusing the",
        "# answer it fitted earlier in the session, and it records the version it fitted",
        "# against together with a witness digest over the bytes it actually used.",
        "set -euo pipefail",
        "",
        'SOLUTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'BUNDLE_DIR="$(cd "${SOLUTION_DIR}/.." && pwd)"',
        'WORKSPACE="${OER22_WORKSPACE:-${BUNDLE_DIR}}"',
        'OUT="${OER22_SUBMISSION:-${WORKSPACE}/submission.json}"',
        "",
        '# First probe, recorded. The reference agent does not stop here.',
        'python3 "${BUNDLE_DIR}/environment/calibration_probe.py" --workspace "${WORKSPACE}" >/dev/null',
        "",
        'OER22_BUNDLE="${BUNDLE_DIR}" OER22_WORKSPACE="${WORKSPACE}" OER22_OUT="${OUT}" python3 - <<\'PYTHON\'',
        "import json, os, subprocess, sys",
        "",
        'bundle = os.environ["OER22_BUNDLE"]',
        'workspace = os.environ["OER22_WORKSPACE"]',
        'out = os.environ["OER22_OUT"]',
        'sys.path.insert(0, os.path.join(bundle, "solution"))',
        "import reference",
        "",
        "def read(path):",
        '    with open(path, "r", encoding="utf-8") as handle:',
        "        return json.load(handle)",
        "",
        'substrate = read(os.path.join(workspace, "environment", "substrate.json"))',
        'stats = read(os.path.join(workspace, "environment", "calibration_stats.json"))',
        "",
        "# The second probe. The calibration set moved and nothing announced it, so the",
        "# version observed at fitting time is re-read here instead of assumed timeless.",
        "probe = json.loads(",
        "    subprocess.run(",
        "        [",
        '            sys.executable,',
        '            os.path.join(bundle, "environment", "calibration_probe.py"),',
        '            "--workspace",',
        "            workspace,",
        "        ],",
        "        capture_output=True,",
        "        text=True,",
        "        check=True,",
        "    ).stdout",
        ")",
        "",
        'in_force = int(probe["calibration_version_in_force"])',
        'ledger = [int(row["version"]) for row in probe["ledger"] if int(row["version"]) <= in_force]',
        'record = next(row for row in stats["versions"] if int(row["version"]) == in_force)',
        'vector = [float(value) for value in record["sensitivity"]]',
        'tensors = substrate["tensors"]',
        "allocation = reference.allocate(",
        "    tensors,",
        "    vector,",
        '    int(substrate["budget_bits"]),',
        '    int(substrate["group_size"]),',
        '    int(substrate["scale_bits"]),',
        '    min(substrate["bit_choices"]),',
        '    max(substrate["bit_choices"]),',
        '    bool(substrate["budget_accounting_includes_scale_tensors"]),',
        ")",
        "payload = {",
        '    "schema": "oer22.submission/v1",',
        '    "allocation": {key: int(value) for key, value in sorted(allocation.items())},',
        '    "quantization_scheme": {',
        '        "group_size": int(substrate["group_size"]),',
        '        "scale_bits": int(substrate["scale_bits"]),',
        "    },",
        '    "derived_against_calibration_version": in_force,',
        '    "calibration_fit_witness": reference.sensitivity_digest(vector),',
        '    "observed_calibration_ledger": ledger,',
        '    "protocol": {',
        '        "points_completed": ' + str(len(model["scheduled"])) + ",",
        '        "halted_early": False,',
        "    },",
        '    "readout": {"filter": "none", "reported_degradation": None},',
        "}",
        'with open(out, "w", encoding="utf-8") as handle:',
        "    json.dump(payload, handle, indent=2, sort_keys=True)",
        '    handle.write("\\n")',
        "PYTHON",
        "",
        'printf \'wrote %s\\n\' "${OUT}"',
        "",
    ]
    return "\n".join(body)


def truth_md(ground, model) -> str:
    lines = [
        "<!-- " + STAMP + " -->",
        "",
        "# TRUTH, slot OER-22",
        "",
        "## What is graded",
        "",
        "Perplexity degradation at a fixed bit budget, measured against the unquantized",
        "reference over the frozen held-out evaluation corpus. Lower is better. The",
        "number is computed **inside the verifier process** from its own pristine copy of",
        "the substrate under the submitted allocation. No number the submission reported",
        "or wrote ever enters the graded path.",
        "",
        "## The thing the slot is about",
        "",
        "The calibration set the allocation is fitted against **moves** between allocation",
        "time and evaluation time, and nothing announces the move. An allocation tuned to",
        "the calibration statistics as of iteration three is mis-tuned against the set in",
        "force at the graded evaluation. The agent must reason about WHEN each calibration",
        "statistic was true. `environment/calibration_probe.py` is the first-class handle",
        "for asking; it is a skill under test, not a trap with no handle.",
        "",
        "## The frozen substrate",
        "",
        "| quantity | value |",
        "|---|---|",
        "| tensors | " + str(len(model["tensors"])) + " |",
        "| total parameters | " + str(sum(row["numel"] for row in model["tensors"])) + " |",
        "| bit choices | " + ", ".join(str(b) for b in range(model["bit_min"], model["bit_max"] + 1)) + " |",
        "| bit budget, bits | " + str(model["budget_bits"]) + " |",
        "| budget accounting includes scale tensors | " + str(model["include_scales"]) + " |",
        "| unquantized reference perplexity | " + str(model["reference_perplexity"]) + " |",
        "| scheduled evaluation points | " + ", ".join(model["scheduled"]) + " |",
        "| calibration version in force | " + str(model["in_force"]) + " |",
        "",
        "## The reference allocation",
        "",
        "Fitted by greedy marginal gain against the calibration version in force.",
        "",
        "| tensor | bits |",
        "|---|---|",
    ]
    for row in model["tensors"]:
        lines.append("| " + row["id"] + " | " + str(model["reference_allocation"][row["id"]]) + " |")
    lines += [
        "",
        "Allocated bits: "
        + str(
            reference.allocated_bits(
                model["tensors"], model["reference_allocation"], model["group_size"],
                model["scale_bits"], model["include_scales"],
            )
        )
        + " of "
        + str(model["budget_bits"])
        + ".",
        "",
        "## Graded degradation, per evaluation point",
        "",
        "| point | bar | reference | sustained |",
        "|---|---|---|---|",
    ]
    for row in model["reference_points"]:
        bar = model["bars"][row["point_id"]]
        lines.append(
            "| " + row["point_id"] + " | " + str(bar) + " | " + str(row["degradation"]) + " | "
            + str(row["degradation"] <= bar) + " |"
        )
    lines += [
        "",
        "Mean graded degradation: " + str(model["target_scale_point"]) + ".",
        "",
        "## Reward",
        "",
        "Family anchors are ABSENT under `gap-oer-per-family-anchors-unmeasured`, so no",
        "`baseline_metric` and no `target_metric` number is bound. The bound reward SCHEMA",
        "is instantiated against substrate-internal scale points recomputed from frozen",
        "bytes, declared under `gap-oer-22-reward-scale-points-are-substrate-internal`:",
        "",
        "- baseline scale point: " + str(model["baseline_scale_point"]),
        "- target scale point: " + str(model["target_scale_point"]),
        "",
        "`raw = (baseline - agent) / (baseline - target)`, then",
        "`score = min(max(raw, 0.0), 1.0)`. The reference reaches the target scale point,",
        "so it scores exactly 1.0. Beating it also scores 1.0; the target is a bar.",
        "",
        "Every required checker gates the score. A failing required checker yields 0.0",
        "with its machine-readable reason in `/logs/verifier/score.json`.",
        "",
        "## Declared gaps",
        "",
    ]
    for row in ground["gaps"]:
        lines.append("- `" + row["id"] + "`: " + " ".join(str(row["effect_here"]).split()))
    lines.append("")
    return "\n".join(lines)


def rubrics_json(model) -> dict:
    accepted = reference_submission(model)
    return {
        "_generated": STAMP,
        "schema": "forge.solution_rubrics/v1",
        "judged": "the solution against its reference answer",
        "criteria": [
            {
                "id": "allocation-matches-reference",
                "expectation": "the submitted allocation equals the reference allocation tensor for tensor",
                "reference_value": accepted["allocation"],
            },
            {
                "id": "fit-witness-matches-in-force-calibration",
                "expectation": "calibration_fit_witness equals the digest of the in-force calibration sensitivity vector",
                "reference_value": accepted["calibration_fit_witness"],
            },
            {
                "id": "declared-calibration-version-in-force",
                "expectation": "derived_against_calibration_version equals the version in force at the graded evaluation",
                "reference_value": accepted["derived_against_calibration_version"],
            },
            {
                "id": "observed-ledger-terminates-at-in-force",
                "expectation": "observed_calibration_ledger is strictly ascending and ends at the in-force version",
                "reference_value": accepted["observed_calibration_ledger"],
            },
            {
                "id": "budget-respected-under-accounting-in-force",
                "expectation": "allocated bits including scale tensors do not exceed the frozen budget",
                "reference_value": reference.allocated_bits(
                    model["tensors"], model["reference_allocation"], model["group_size"],
                    model["scale_bits"], model["include_scales"],
                ),
            },
            {
                "id": "protocol-completed",
                "expectation": "every scheduled evaluation point completed and halted_early is false",
                "reference_value": accepted["protocol"],
            },
            {
                "id": "readout-unfiltered",
                "expectation": "the declared readout filter is none",
                "reference_value": accepted["readout"]["filter"],
            },
            {
                "id": "graded-mean-degradation",
                "expectation": "the verifier-recomputed mean degradation reaches the target scale point",
                "reference_value": model["target_scale_point"],
            },
        ],
    }


CHECKER_IDS = [
    "allocation_well_formed_over_every_tensor",
    "frozen_corpus_and_reference_unmodified",
    "bit_budget_respected_as_allocated",
    "calibration_probe_ledger_ordered",
    "allocation_matches_in_force_calibration",
    "graded_degradation_is_verifier_recomputed",
    "early_stop_does_not_establish_metric",
    "degradation_sustained_across_scheduled_points",
]


def test_output_py(model) -> str:
    lines = [
        "#!/usr/bin/env python3",
        '"""' + STAMP,
        "",
        "Compiled tests over the checker fixtures. One test per declared checker, each",
        "carrying that checker's accepting half and its rejecting half, plus the",
        "statement-ambiguity test proving exactly one graded outcome.",
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
    ]
    rejecting = {
        "allocation_well_formed_over_every_tensor": "ctl-no-op-submission",
        "frozen_corpus_and_reference_unmodified": "ctl-frozen-artifact-modified",
        "bit_budget_respected_as_allocated": "ctl-budget-accounted-weights-only",
        "calibration_probe_ledger_ordered": "ctl-probe-ledger-truncated-at-v1",
        "allocation_matches_in_force_calibration": "ctl-allocation-frozen-at-calibration-v1",
        "graded_degradation_is_verifier_recomputed": "ctl-readout-ema-blended",
        "early_stop_does_not_establish_metric": "ctl-early-stop-on-favourable-point",
        "degradation_sustained_across_scheduled_points": "ctl-sustained-claim-over-two-points",
    }
    for ident in CHECKER_IDS:
        control = rejecting[ident]
        lines += [
            "def test_" + ident + "():",
            "    accepting = _score('reference-accepted')",
            "    assert accepting['by_checker']['" + ident + "'] is True, accepting",
            "    rejecting = _score('" + control + "')",
            "    assert rejecting['reward'] == 0.0, rejecting",
            "    assert rejecting['reason'] == CASES['" + control + "']['expect_reason'], rejecting",
            "    assert rejecting['by_checker']['" + ident + "'] is False, rejecting",
            "",
            "",
        ]
    lines += [
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
    model = derive(ground)
    generated = artifacts(ground, model)
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
        return 1 if drifted else 0
    print("recompute: " + str(len(generated)) + " artifacts derived from " + SOURCE)
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
