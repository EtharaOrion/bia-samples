# FORGE-CANARY-BEGIN
# 0: 11823f27948760ef159fce0af80264ba2f455fa2bac4b1efedca11f48e00d85b
# 1: ef44e416203ed287097f45927456dcbe8ba5f50130362e65a70b89322a03503a
# 2: 766d91786c288769ec37748c3540c05010de62ccfa9f06ecd631cca36d2a8f64
# 3: a84fe43a20fa280e54c00500d4c9972789732c14b1083847b2e871d9b0d3db7a
# FORGE-CANARY-END
#!/usr/bin/env python3
"""Derive every generated artifact in slot OER-03 from solution/grounding.yaml.

One source, one deriver. This module invokes NO model, NO network, NO clock, NO
locale and NO random source. Every value it writes is either read out of
grounding.yaml or computed from it by arithmetic and sha256 over frozen bytes, so
running it twice over frozen bytes produces byte-identical output.

Derived artifacts, each carrying the generated banner and naming its source:

    tests/golden.json        the bound state, the anchors, the golden telemetry,
                             the extension control, and one fixture per checker
    tests/test_output.py     the compiled surface, both halves of every checker
    solution/solve.sh        the oracle entry point
    solution/TRUTH.md        the private answer record
    solution/rubrics.json    the solution-against-reference rubric

Usage:  python3 solution/recompute.py [--check]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional

import yaml

HERE = pathlib.Path(__file__).resolve().parent
BUNDLE = HERE.parent
GROUNDING = HERE / "grounding.yaml"

BANNER = "GENERATED SECTION. DO NOT HAND-EDIT."
SOURCE = "solution/grounding.yaml"


def load() -> Dict[str, Any]:
    with GROUNDING.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def params_digest(seed: str, step: int) -> str:
    """A stand-in for the harness parameter digest. Fixed by the seed and step."""
    return hashlib.sha256((str(seed) + "|params|" + str(int(step))).encode("utf-8")).hexdigest()


def weights_digest(domain: str, step: int, digest: str) -> str:
    """The same binding tests/checkers.py recomputes. Written here, never guessed."""
    payload = str(domain) + "|" + str(int(step)) + "|" + str(digest)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Telemetry construction
# ---------------------------------------------------------------------------


def evaluation_row(
    step: int,
    loss: float,
    state: Dict[str, Any],
    seed: str,
    domain: Optional[str] = None,
    filter_name: Optional[str] = None,
    source: str = "harness",
) -> Dict[str, Any]:
    digest = params_digest(seed, step)
    used_domain = domain or str(state["weights_digest_domain"])
    return {
        "step": int(step),
        "loss": float(loss),
        "filter": filter_name or str(state["readout_filter_admitted"]),
        "weights_digest": weights_digest(used_domain, step, digest),
        "weights_source": source,
        "params_digest": digest,
        "scheduled_by": "verifier",
    }


def observed_axes(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "batch_size": int(state["frozen_batch_size"]),
        "fwd_bwd_per_step": int(state["frozen_fwd_bwd_per_step"]),
        "dataset_digest": str(state["frozen_dataset_digest"]),
        "architecture_digest": str(state["frozen_architecture_digest"]),
    }


def bound_block(payload: Dict[str, Any]) -> Dict[str, Any]:
    state = dict(payload["bound_state"])
    anchors = payload["anchors"]
    state["baseline_metric"] = int(anchors["baseline_metric"])
    state["target_metric"] = int(anchors["target_metric"])
    return state


def golden_telemetry(payload: Dict[str, Any]) -> Dict[str, Any]:
    state = bound_block(payload)
    golden = payload["golden"]
    seed = str(golden["params_digest_seed"])
    series = {int(key): float(value) for key, value in golden["series"].items()}
    rows = [evaluation_row(step, series[step], state, seed) for step in sorted(series)]
    return {
        "bound": state,
        "run": {
            "status": str(golden["run_status"]),
            "exit_code": 0,
            "stopped_by": str(golden["stopped_by"]),
            "schedule_length": int(golden["schedule_length"]),
            "steps_executed": int(golden["steps_executed"]),
        },
        "evaluations": rows,
        "frozen_axes_observed": observed_axes(state),
        "submission_report": {},
    }


def extension_telemetry(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The negative control: the same run with a schedule extended past the crossing."""
    state = bound_block(payload)
    golden = payload["golden"]
    control = golden["extension_control"]
    seed = str(golden["params_digest_seed"])
    series = {int(key): float(value) for key, value in golden["series"].items()}
    telemetry = golden_telemetry(payload)
    last_step = max(series)
    last_loss = series[last_step]
    stride = int(state["eval_stride"])
    decrement = float(control["decrement_per_point"])
    rows = list(telemetry["evaluations"])
    for index in range(1, int(control["extra_points"]) + 1):
        step = last_step + index * stride
        loss = last_loss - index * decrement
        rows.append(evaluation_row(step, loss, state, seed))
    telemetry = copy.deepcopy(telemetry)
    telemetry["evaluations"] = rows
    telemetry["run"]["schedule_length"] = int(control["schedule_length"])
    telemetry["run"]["steps_executed"] = last_step + int(control["extra_points"]) * stride
    return telemetry


# ---------------------------------------------------------------------------
# Planted defect fixtures. One per checker, both halves per invariant 17.
# ---------------------------------------------------------------------------


def _mutate(name: str, base: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    state = bound_block(payload)
    pre = payload["pre_drift_state"]
    golden = payload["golden"]
    seed = str(golden["params_digest_seed"])
    telemetry = copy.deepcopy(base)

    if name == "no_evaluations":
        telemetry["evaluations"] = []
        telemetry["run"]["status"] = "crashed"
        telemetry["run"]["exit_code"] = 1
        telemetry["run"]["stopped_by"] = "nonzero-exit"
        telemetry["run"]["steps_executed"] = 0
        return telemetry

    if name == "stale_weights_domain":
        stale = str(pre["weights_digest_domain"])
        telemetry["evaluations"] = [
            evaluation_row(row["step"], row["loss"], state, seed, domain=stale)
            for row in base["evaluations"]
        ]
        return telemetry

    if name == "submission_selected_weights":
        rows = copy.deepcopy(base["evaluations"])
        rows[-1]["weights_source"] = "submission"
        telemetry["evaluations"] = rows
        return telemetry

    if name == "descending_evaluations":
        telemetry["evaluations"] = list(reversed(copy.deepcopy(base["evaluations"])))
        return telemetry

    if name == "ema_filter":
        telemetry["evaluations"] = [
            evaluation_row(row["step"], row["loss"], state, seed, filter_name="ema")
            for row in base["evaluations"]
        ]
        return telemetry

    if name == "unsustained_dip":
        bar = float(state["target_loss"])
        rows = copy.deepcopy(base["evaluations"])
        # Lift the third point after the crossing back above the bar. Under the
        # pre-drift sustain window of two this would have been harvested; under
        # the bound window of three it is a dip and not a crossing.
        crossing = _first_below(rows, bar)
        stride = int(state["eval_stride"])
        lifted = crossing + 3 * stride
        for row in rows:
            if row["step"] == lifted:
                row["loss"] = bar + 0.00060
        telemetry["evaluations"] = [
            evaluation_row(row["step"], row["loss"], state, seed) for row in rows
        ]
        return telemetry

    if name == "early_stop":
        bar = float(state["target_loss"])
        crossing = _first_below(base["evaluations"], bar)
        rows = [row for row in copy.deepcopy(base["evaluations"]) if row["step"] <= crossing]
        telemetry["evaluations"] = rows
        telemetry["run"]["status"] = "terminated-early"
        telemetry["run"]["stopped_by"] = "submission-early-stop"
        telemetry["run"]["steps_executed"] = crossing
        return telemetry

    if name == "outcome_without_reason":
        telemetry["outcome"] = {"crossed": False}
        return telemetry

    if name == "graded_step_from_schedule":
        telemetry["outcome"] = {
            "crossed": True,
            "graded_step": int(telemetry["run"]["schedule_length"]),
            "reason": "",
            "basis": "verifier-recomputed",
        }
        return telemetry

    if name == "stale_target_bar":
        stale_bar = float(pre["target_loss"])
        stale_step = _first_sustained(base["evaluations"], stale_bar, int(pre["sustain_points"]))
        telemetry["outcome"] = {
            "crossed": True,
            "graded_step": stale_step,
            "reason": "",
            "basis": "verifier-recomputed",
        }
        return telemetry

    if name == "reported_crossing_early":
        telemetry["submission_report"] = {"crossing_step": 2000, "filter": "ema"}
        return telemetry

    if name == "stale_batch_size":
        axes = dict(base["frozen_axes_observed"])
        axes["batch_size"] = int(pre["frozen_batch_size"])
        telemetry["frozen_axes_observed"] = axes
        return telemetry

    raise ValueError("unknown mutation: " + name)


def _first_below(rows: List[Dict[str, Any]], bar: float) -> int:
    for row in sorted(rows, key=lambda item: item["step"]):
        if float(row["loss"]) < bar:
            return int(row["step"])
    raise ValueError("the golden series never falls below " + str(bar))


def _first_sustained(rows: List[Dict[str, Any]], bar: float, sustain: int) -> Optional[int]:
    ordered = sorted(rows, key=lambda item: item["step"])
    for index, row in enumerate(ordered):
        if float(row["loss"]) >= bar:
            continue
        window = ordered[index + 1 : index + 1 + sustain]
        if len(window) < sustain:
            return None
        if all(float(item["loss"]) < bar for item in window):
            return int(row["step"])
    return None


# ---------------------------------------------------------------------------
# Artifact derivation
# ---------------------------------------------------------------------------


def build_golden(payload: Dict[str, Any]) -> Dict[str, Any]:
    state = bound_block(payload)
    base = golden_telemetry(payload)
    fixtures = []
    for row in payload["fixtures"]["planted"]:
        fixtures.append(
            {
                "id": str(row["id"]),
                "checker": str(row["checker"]),
                "zero_reason": str(row["zero_reason"]),
                "stale_control": str(row.get("stale_control", "")),
                "statement": " ".join(str(row["statement"]).split()),
                "telemetry": _mutate(str(row["mutate"]), base, payload),
            }
        )
    return {
        "banner": BANNER,
        "source": SOURCE,
        "slot": str(payload["slot"]),
        "provisional_identifier": str(payload["provisional_identifier"]),
        "anchors": {
            "state": str(payload["anchors"]["state"]),
            "baseline_metric": int(payload["anchors"]["baseline_metric"]),
            "target_metric": int(payload["anchors"]["target_metric"]),
            "direction": str(payload["anchors"]["direction"]),
            "pass_threshold": float(payload["anchors"]["pass_threshold"]),
            "authority": str(payload["anchors"]["authority"]),
        },
        "bound_state": state,
        "pre_drift_state": dict(payload["pre_drift_state"]),
        "reference": {
            "carrier": str(payload["golden"]["reference_carrier"]),
            "sha256": sha256_file(BUNDLE / str(payload["golden"]["reference_carrier"])),
        },
        "golden": {
            "id": str(payload["fixtures"]["accepting"]["id"]),
            "expected_graded_step": int(payload["golden"]["expected_graded_step"]),
            "expected_reward": float(payload["golden"]["expected_reward"]),
            "telemetry": base,
        },
        "extension_control": {
            "id": "ctl-schedule-extended-past-crossing",
            "expected_graded_step": int(payload["golden"]["extension_control"]["expected_graded_step"]),
            "expected_reward": float(payload["golden"]["extension_control"]["expected_reward"]),
            "telemetry": extension_telemetry(payload),
        },
        "fixtures": fixtures,
        "provenance": {
            "kind": str(payload["golden_provenance"]["kind"]),
            "measured": bool(payload["golden_provenance"]["measured"]),
            "gap_id": str(payload["golden_provenance"]["gap_id"]),
            "statement": " ".join(str(payload["golden_provenance"]["statement"]).split()),
        },
    }


def render_test_output(payload: Dict[str, Any], golden: Dict[str, Any]) -> str:
    lines: List[str] = []
    add = lines.append
    add('"""' + BANNER)
    add("")
    add("Source: " + SOURCE + ". Regenerate with: python3 solution/recompute.py")
    add("")
    add("The compiled surface for slot OER-03. One accepting test and one rejecting")
    add("test for every checker in tests/checkers.yaml, plus the reference-accept test")
    add("and the negative control proving that a schedule extended past the sustained")
    add("crossing scores identically to one that is not extended.")
    add("")
    add("These tests import tests/checkers.py and tests/grade.py. They never import a")
    add("submission, and they read no state a submission produced.")
    add('"""')
    add("")
    add("from __future__ import annotations")
    add("")
    add("import json")
    add("import pathlib")
    add("import sys")
    add("")
    add("HERE = pathlib.Path(__file__).resolve().parent")
    add("sys.path.insert(0, str(HERE))")
    add("")
    add("import checkers  # noqa: E402")
    add("import grade  # noqa: E402")
    add("")
    add('GOLDEN = json.loads((HERE / "golden.json").read_text(encoding="utf-8"))')
    add('ACCEPTING = GOLDEN["golden"]["telemetry"]')
    add('FIXTURES = {row["id"]: row for row in GOLDEN["fixtures"]}')
    add("")
    add("")
    add("def _reason(telemetry):")
    add("    return grade.grade_telemetry(telemetry)")
    add("")
    add("")
    add("def test_reference_accepted_end_to_end():")
    add("    result = _reason(ACCEPTING)")
    add('    assert result["reward"] == GOLDEN["golden"]["expected_reward"]')
    add('    assert result["reason"] == ""')
    add('    assert result["metric"]["graded_step"] == GOLDEN["golden"]["expected_graded_step"]')
    add("")
    add("")
    add("def test_schedule_extension_past_crossing_scores_identically():")
    add('    plain = _reason(ACCEPTING)')
    add('    extended = _reason(GOLDEN["extension_control"]["telemetry"])')
    add('    assert extended["metric"]["graded_step"] == plain["metric"]["graded_step"]')
    add('    assert extended["reward"] == plain["reward"]')
    add('    assert extended["metric"]["schedule_length"] != plain["metric"]["schedule_length"]')
    add("")
    for row in payload["fixtures"]["planted"]:
        ident = str(row["id"]).replace("-", "_")
        add("")
        add("def test_" + ident + "_is_rejected():")
        add('    result = _reason(FIXTURES["' + str(row["id"]) + '"]["telemetry"])')
        add("    assert result[\"reward\"] == 0.0")
        add('    assert result["reason"] == "' + str(row["zero_reason"]) + '"')
        add("")
    seen = []
    for name, _ in _checker_rows(payload):
        if name in seen:
            continue
        seen.append(name)
        add("")
        add("def test_" + name + "():")
        add('    verdicts = {item.ident: item for item in checkers.run_all(_outcome(ACCEPTING))}')
        add('    assert verdicts["' + name + '"].passed')
        add("")
    add("")
    add("def _outcome(telemetry):")
    add("    filled = dict(telemetry)")
    add('    filled["outcome"] = checkers.derive_outcome(filled)')
    add("    return filled")
    return "\n".join(lines) + "\n"


def _checker_rows(payload: Dict[str, Any]):
    manifest = yaml.safe_load((BUNDLE / "tests" / "checkers.yaml").read_text(encoding="utf-8"))
    for row in manifest["checkers"]:
        yield str(row["id"]), str(row["zero_reason"])


def render_solve(payload: Dict[str, Any]) -> str:
    golden = payload["golden"]
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "# " + BANNER,
            "# Source: " + SOURCE + ". Regenerate with: python3 solution/recompute.py",
            "#",
            "# Oracle entry point for slot " + str(payload["slot"]) + ".",
            "#",
            "# Installs the reference solution as the submission and runs it under the",
            "# pinned harness. The reference crosses at step "
            + str(golden["expected_graded_step"])
            + " on the verifier's",
            "# cadence and carries a schedule of "
            + str(golden["schedule_length"])
            + " steps, so "
            + str(int(golden["schedule_length"]) - int(golden["expected_graded_step"]))
            + " steps of schedule sit",
            "# past the graded point and buy exactly nothing. That is the point of the",
            "# reference, not an accident in it.",
            "set -euo pipefail",
            "",
            'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
            'BUNDLE="$(dirname "$HERE")"',
            "",
            'mkdir -p /app',
            'cp "$HERE/reference.py" /app/submission.py',
            'cp "$BUNDLE"/environment/bia_harness.py "$BUNDLE"/environment/bia_loader.py "$BUNDLE"/environment/shape.json /app/env/ 2>/dev/null || true',
            "",
            'cd /app',
            'python3 /app/submission.py',
            "",
            "# The reference never decides when it crossed and never reports a crossing.",
            "# The verifier recomputes the graded step from its own evaluations.",
        ]
    ) + "\n"


def render_truth(payload: Dict[str, Any], golden: Dict[str, Any]) -> str:
    anchors = payload["anchors"]
    g = payload["golden"]
    rows = []
    add = rows.append
    add("<!-- " + BANNER + " -->")
    add("<!-- Source: " + SOURCE + ". Regenerate with: python3 solution/recompute.py -->")
    add("")
    add("# TRUTH, slot " + str(payload["slot"]))
    add("")
    add("Private. Never on the agent surface Harbor assembles.")
    add("")
    add("## The graded quantity")
    add("")
    add("`" + str(payload["graded_quantity"]["token"]) + "`. "
        + " ".join(str(payload["graded_quantity"]["definition"]).split()))
    add("")
    add("It is " + " ".join(str(payload["graded_quantity"]["never"]).split()))
    add("")
    add("## The reference answer")
    add("")
    add("| Quantity | Value |")
    add("|---|---|")
    add("| Reference carrier | `" + str(g["reference_carrier"]) + "` |")
    add("| Reference sha256 | `" + str(golden["reference"]["sha256"]) + "` |")
    add("| Schedule length | " + str(g["schedule_length"]) + " |")
    add("| Steps executed | " + str(g["steps_executed"]) + " |")
    add("| Graded step, the sustained crossing | **" + str(g["expected_graded_step"]) + "** |")
    add("| Crossing under the pre-drift bar | " + str(g["expected_crossing_under_pre_drift_bar"]) + " |")
    add("| Reward | " + str(g["expected_reward"]) + " |")
    add("| Baseline anchor | " + str(anchors["baseline_metric"]) + " |")
    add("| Target anchor | " + str(anchors["target_metric"]) + " |")
    add("")
    add(" ".join(str(g["expected_reward_derivation"]).split()))
    add("")
    add("## Why the schedule tail is worth nothing")
    add("")
    add("The reference schedule runs to " + str(g["schedule_length"]) + " steps and the sustained crossing lands at "
        + str(g["expected_graded_step"]) + ". The negative control `ctl-schedule-extended-past-crossing` carries the same run with a "
        + str(g["extension_control"]["schedule_length"]) + "-step schedule and "
        + str(g["extension_control"]["extra_points"])
        + " further evaluation points below the bar, and it scores byte-identically. A schedule extended past the crossing changes the schedule length and cannot change the graded step.")
    add("")
    add("## What a non-converging run produces")
    add("")
    add(" ".join(str(payload["graded_quantity"]["non_convergence_is_graded"]).split()))
    add("")
    add(" ".join(str(payload["graded_quantity"]["verifier_aborting_is_also_graded"]).split()))
    add("")
    add("## Provenance of the golden series")
    add("")
    add(" ".join(str(payload["golden_provenance"]["statement"]).split()))
    add("")
    add("Gap: `" + str(payload["golden_provenance"]["gap_id"]) + "`.")
    add("")
    add("## Tier")
    add("")
    add("Target tier " + str(payload["tier"]["target_tier"]) + ", tier_exemption_granted "
        + str(payload["tier"]["tier_exemption_granted"]) + ", anchorable tier "
        + str(payload["tier"]["anchorable_tier"]) + ".")
    add("")
    add(" ".join(str(payload["tier"]["perception_axis_exemption"]).split()))
    add("")
    add(" ".join(str(payload["tier"]["no_difficulty_claim"]).split()))
    return "\n".join(rows) + "\n"


def render_rubrics(payload: Dict[str, Any]) -> str:
    document = {
        "banner": BANNER,
        "source": SOURCE,
        "slot": str(payload["slot"]),
        "judged": "the solution against its reference answer",
        "not_this_file": "tests/rubrics.jsonl is a different artifact judged against the trajectory; neither substitutes for the other",
        "rows": [
            {
                "id": str(row["id"]),
                "criterion": " ".join(str(row["criterion"]).split()),
                "reference_value": row["reference_value"],
            }
            for row in payload["solution_rubric_rows"]
        ],
    }
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


ARTIFACTS = ("tests/golden.json", "tests/test_output.py", "solution/solve.sh", "solution/TRUTH.md", "solution/rubrics.json")


def derive() -> Dict[str, str]:
    payload = load()
    golden = build_golden(payload)
    return {
        "tests/golden.json": json.dumps(golden, indent=2, sort_keys=True) + "\n",
        "tests/test_output.py": render_test_output(payload, golden),
        "solution/solve.sh": render_solve(payload),
        "solution/TRUTH.md": render_truth(payload, golden),
        "solution/rubrics.json": render_rubrics(payload),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Derive the generated artifacts of slot OER-03.")
    parser.add_argument("--check", action="store_true", help="report drift instead of writing")
    args = parser.parse_args()

    outputs = derive()
    drifted = []
    for relative, text in sorted(outputs.items()):
        path = BUNDLE / relative
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current == text:
            continue
        drifted.append(relative)
        if not args.check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    if args.check:
        print(json.dumps({"drifted": drifted, "artifacts": list(ARTIFACTS)}, indent=2, sort_keys=True))
        return 1 if drifted else 0
    for relative in ("solution/solve.sh",):
        (BUNDLE / relative).chmod(0o755)
    print(json.dumps({"written": drifted, "artifacts": list(ARTIFACTS)}, indent=2, sort_keys=True))
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
    sys.exit(main())
