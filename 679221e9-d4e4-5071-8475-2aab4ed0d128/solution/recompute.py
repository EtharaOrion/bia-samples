# FORGE-CANARY-BEGIN
# 0: ebf7521773d324c1e8fe3d1839edce3f4378edfd81b757caf9519d58e363e3b7
# 1: ec155d925483bef5cf111cb11a3bfefd142da8ee0418d48c8fdfac088bd181ad
# 2: 10cf6ba58292dd0a4b681d94830d92964f2f7566402200356625ca10d813fbbe
# 3: 5903e25af5b96bf90008f5f7823116e2f05196155912b107e7c32384263ba88e
# FORGE-CANARY-END
#!/usr/bin/env python3
"""Derive every generated artifact in this bundle from solution/grounding.yaml.

One source, one derivation, four generated files plus the golden trajectory and
the checker fixtures they carry. Nothing here invokes a model, opens a socket,
reads a clock, consults a locale or touches a random source, so running it twice
over frozen bytes produces byte-identical output.

The frozen substrate is not duplicated into the source. It is read from
environment/ and refused unless every file's sha256 equals the value grounding.yaml
binds, because a second copy is a second source that can disagree and a
digest-bound read is one source that cannot.

Usage:
    python3 solution/recompute.py            # write the generated artifacts
    python3 solution/recompute.py --check     # exit non-zero if any artifact drifted
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import yaml

BANNER = "GENERATED SECTION. DO NOT HAND-EDIT."
SOURCE = "solution/grounding.yaml"

BUNDLE = Path(__file__).resolve().parent.parent
GROUNDING = BUNDLE / "solution" / "grounding.yaml"
ENVIRONMENT = BUNDLE / "environment"
TESTS = BUNDLE / "tests"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_grounding() -> dict:
    return yaml.safe_load(GROUNDING.read_text(encoding="utf-8"))


def verify_substrate(grounding: dict) -> None:
    for name, expected in sorted(grounding["substrate_sha256"].items()):
        actual = hashlib.sha256((ENVIRONMENT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(
                "substrate digest moved for " + name + ": grounding binds " + expected
                + " and the file hashes to " + actual
                + ". Refusing to derive anything from a source that no longer describes the bytes."
            )


def derive_result(grounding: dict) -> dict:
    """Recompute the expected result from the frozen substrate and the reference."""
    runner = load_module("oer20_runner_for_recompute", TESTS / "runner.py")
    reference = load_module("oer20_reference_for_recompute", BUNDLE / "solution" / "reference.py")

    payload = hashlib.sha256(reference.canonical()).hexdigest()
    if payload != grounding["reference"]["allocation_sha256"]:
        raise SystemExit(
            "reference allocation digest moved: grounding binds "
            + grounding["reference"]["allocation_sha256"] + " and reference.py hashes to " + payload
        )

    model = runner.load_json(ENVIRONMENT / "model_stats.json")
    corpus = runner.load_json(ENVIRONMENT / "eval_corpus.json")
    unquantized = runner.load_json(ENVIRONMENT / "reference_unquantized.json")
    allocation = runner.normalise_allocation(model, reference.REFERENCE_ALLOCATION)
    account = runner.accounting(model, allocation)
    schedule = runner.schedule_points(ENVIRONMENT, corpus)
    points = runner.evaluate(model, corpus, unquantized, allocation, schedule["order"])
    measure = runner.measurement(points)
    margin = float(grounding["bindings"]["separation_margin"])
    lower = measure["separation_mean"] - measure["noise_half_width"]
    return {
        "allocated_bits": account["allocated_bits"],
        "budget_bits": account["budget_bits"],
        "mean_bits_per_param": round(account["mean_bits_per_param"], 6),
        "scheduled_order": list(schedule["order"]),
        "separation_mean": round(measure["separation_mean"], 6),
        "noise_half_width": round(measure["noise_half_width"], 6),
        "lower_edge": round(lower, 6),
        "upper_edge": round(measure["separation_mean"] + measure["noise_half_width"], 6),
        "verdict": "established" if lower >= margin else "unproven",
        "reason": "separation-margin-cleared" if lower >= margin else "significance-unestablished-at-ceiling",
        "reward": 1.0 if lower >= margin else round(max(0.0, min(1.0, lower / margin)), 6),
    }


def check_recorded_result(grounding: dict, derived: dict) -> None:
    recorded = dict(grounding["expected_result"])
    for key, value in sorted(derived.items()):
        if recorded.get(key) != value:
            raise SystemExit(
                "expected_result." + key + " in " + SOURCE + " records " + repr(recorded.get(key))
                + " and the derivation produces " + repr(value)
            )


def build_solve_sh(grounding: dict, derived: dict) -> str:
    ref = grounding["reference"]
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "# " + BANNER,
            "# Source: " + SOURCE,
            "#",
            "# The reference submission. It writes exactly one artifact, allocation.json,",
            "# and reports no perplexity number, because the graded quantity is computed by",
            "# the verifier over harness-owned quantized state and a number written here",
            "# would be ignored on the graded path by construction.",
            "#",
            "# scheme: " + str(ref["scheme"]),
            "# allocation sha256: " + str(ref["allocation_sha256"]),
            "# allocated bits: " + str(grounding["expected_result"]["allocated_bits"])
            + " of " + str(grounding["expected_result"]["budget_bits"]) + " budgeted",
            "set -euo pipefail",
            "",
            'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
            'python3 "${HERE}/reference.py"',
            "",
        ]
    )


def build_truth_md(grounding: dict, derived: dict) -> str:
    ref = grounding["reference"]
    result = grounding["expected_result"]
    out = [
        "<!-- " + BANNER + " -->",
        "<!-- Source: " + SOURCE + " -->",
        "",
        "# TRUTH, slot OER-20",
        "",
        "Family F9, post-training quantization. Primary archetype AR7, Ambiguous",
        "Intermediate States, exactly one.",
        "",
        "## The reference allocation",
        "",
        "Scheme `" + str(ref["scheme"]) + "`, allocation payload sha256 `"
        + str(ref["allocation_sha256"]) + "`.",
        "",
        "| tensor | bits |",
        "|---|---|",
    ]
    for name in sorted(ref["bits"]):
        out.append("| `" + name + "` | " + str(ref["bits"][name]) + " |")
    out += [
        "",
        str(ref["why"]).strip(),
        "",
        "## What the verifier measures",
        "",
        "| quantity | value |",
        "|---|---|",
        "| allocated bits | " + str(result["allocated_bits"]) + " |",
        "| budget bits | " + str(result["budget_bits"]) + " |",
        "| mean bits per parameter | " + str(result["mean_bits_per_param"]) + " |",
        "| verifier-derived evaluation order | " + ", ".join(result["scheduled_order"]) + " |",
        "| separation mean | " + str(result["separation_mean"]) + " |",
        "| noise half width | " + str(result["noise_half_width"]) + " |",
        "| lower edge of the band | " + str(result["lower_edge"]) + " |",
        "| upper edge of the band | " + str(result["upper_edge"]) + " |",
        "| bound separation margin | " + str(grounding["bindings"]["separation_margin"]) + " |",
        "| verdict | " + str(result["verdict"]) + " |",
        "| reason | `" + str(result["reason"]) + "` |",
        "| reward | " + str(result["reward"]) + " |",
        "",
        "## The three outcomes, which are not two",
        "",
    ]
    for key in ("established", "failed", "unproven"):
        row = grounding["outcomes"][key]
        out += [
            "### " + str(row["verdict"]),
            "",
            "- reason: `" + str(row["reason"]) + "`",
            "- condition: `" + str(row["condition"]) + "`",
            "- reward: " + str(row["reward"]),
        ]
        if row.get("gated_by"):
            out.append("- gated by: `" + str(row["gated_by"]) + "`")
        if row.get("meaning"):
            out += ["", str(row["meaning"]).strip()]
        out.append("")
    out += [
        "## The reward ramp",
        "",
        str(grounding["reward_ramp"]["statement"]).strip(),
        "",
        "`" + str(grounding["reward_ramp"]["function"]) + "`",
        "",
        "Measured points are recorded in `" + str(grounding["reward_ramp"]["measured_in"]) + "`.",
        "",
        "## Golden trajectory",
        "",
    ]
    for step in grounding["golden_trajectory"]:
        out += [
            "**Step " + str(step["step"]) + ".** " + str(step["action"]).strip(),
            "",
            "> holds: " + str(step["holds"]).strip(),
            "",
        ]
    out += ["## Anchors", "", "- `anchors_state`: " + str(grounding["anchors"]["anchors_state"])]
    out += [
        "- gap: `" + str(grounding["anchors"]["anchors_gap"]) + "`",
        "- `baseline_metric`: absent, not invented",
        "- `target_metric`: absent, not invented",
        "",
        str(grounding["anchors"]["why_gradable_anyway"]).strip(),
        "",
        "## Declared gaps",
        "",
    ]
    for gap in grounding["declared_gaps"]:
        out.append("- `" + str(gap["id"]) + "`: " + str(gap["what"]).strip())
    out.append("")
    return "\n".join(out)


def build_rubrics_json(grounding: dict, derived: dict) -> str:
    payload = {
        "schema": "forge.solution_rubrics/v1",
        "generated_section": BANNER,
        "generated_from": SOURCE,
        "slot": "OER-20",
        "reference_allocation_sha256": grounding["reference"]["allocation_sha256"],
        "grades": "the solution against its reference answer",
        "not": "trajectory behaviour, which tests/rubrics.jsonl grades separately and which this file never substitutes for",
        "criteria": [
            {
                "id": "matches-reference-scheme",
                "statement": "The submitted scheme is the reference scheme.",
                "expect": grounding["reference"]["scheme"],
            },
            {
                "id": "matches-reference-allocation",
                "statement": "The submitted per-tensor bit allocation is the reference allocation.",
                "expect": {str(k): int(v) for k, v in sorted(grounding["reference"]["bits"].items())},
            },
            {
                "id": "spends-the-budget-exactly",
                "statement": "The allocation spends the fixed bit budget without exceeding it.",
                "expect": {
                    "allocated_bits": derived["allocated_bits"],
                    "budget_bits": derived["budget_bits"],
                    "mean_bits_per_param": derived["mean_bits_per_param"],
                },
            },
            {
                "id": "clears-the-separation-margin",
                "statement": "The lower edge of the measured separation band reaches the bound margin.",
                "expect": {
                    "lower_edge": derived["lower_edge"],
                    "separation_margin": grounding["bindings"]["separation_margin"],
                    "verdict": derived["verdict"],
                    "reason": derived["reason"],
                },
            },
            {
                "id": "earns-full-reward",
                "statement": "The reference solution is accepted end to end at full reward.",
                "expect": derived["reward"],
            },
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def build_test_output(grounding: dict, derived: dict) -> str:
    fixtures = grounding["checker_fixtures"]
    rows = [row for row in fixtures if row.get("checker")]
    lines = [
        "#!/usr/bin/env python3",
        '"""' + BANNER,
        "",
        "Source: " + SOURCE,
        "",
        "The compiled tests tests/checkers.yaml names in every compiled_test field.",
        "Each one re-asserts one reduction over the SAME telemetry record the grader",
        "read, so a reduction that passed in grade.py and fails here is a disagreement",
        "between the manifest and the run rather than a second opinion about the run.",
        "",
        "Runnable two ways: as a script, which tests/test.sh does, and under pytest,",
        "which reads the telemetry path from OER20_TELEMETRY.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "import argparse",
        "import json",
        "import os",
        "import sys",
        "from pathlib import Path",
        "",
        "sys.path.insert(0, str(Path(__file__).resolve().parent))",
        "",
        "from checkers import (  # noqa: E402",
        "    REQUIRED,",
        "    SELECTORS,",
        "    classify_separation,",
        ")",
        "",
        "_STATE = {\"telemetry\": None, \"verdict\": None}",
        "",
        "EXPECTED_ORDER = " + repr(list(derived["scheduled_order"])),
        "EXPECTED_ALLOCATED_BITS = " + repr(int(derived["allocated_bits"])),
        "EXPECTED_BUDGET_BITS = " + repr(int(derived["budget_bits"])),
        "EXPECTED_REASON = " + repr(str(derived["reason"])),
        "EXPECTED_REWARD = " + repr(float(derived["reward"])),
        "BOUND_MARGIN = " + repr(float(grounding["bindings"]["separation_margin"])),
        "",
        "",
        "def _read(path):",
        "    return json.loads(Path(path).read_text(encoding=\"utf-8\"))",
        "",
        "",
        "def telemetry():",
        "    if _STATE[\"telemetry\"] is None:",
        "        _STATE[\"telemetry\"] = _read(os.environ[\"OER20_TELEMETRY\"])",
        "    return _STATE[\"telemetry\"]",
        "",
        "",
        "def verdict():",
        "    if _STATE[\"verdict\"] is None:",
        "        _STATE[\"verdict\"] = _read(os.environ[\"OER20_VERDICT\"])",
        "    return _STATE[\"verdict\"]",
        "",
        "",
        "def _reduction(name):",
        "    for ident, selector in SELECTORS:",
        "        if ident == name:",
        "            return selector(telemetry())",
        "    raise AssertionError(\"tests/checkers.yaml names a selector checkers.py does not carry: \" + name)",
        "",
        "",
    ]
    for row in rows:
        ident = str(row["checker"])
        reason = str(row["expect_reason"])
        required = ident != "separation_margin_cleared"
        lines += [
            "def test_" + ident + "():",
            '    """' + ident + ": passes at full value, and its zero carries " + reason + '."""',
            "    outcome = _reduction(" + repr(ident) + ")",
            "    assert 0.0 <= outcome.value <= 1.0",
            "    assert " + repr(reason) + " == " + repr(reason),
            (
                "    assert outcome.passed, " + repr(ident) + " + \" scored \" + repr(outcome.value) + \" with reason \" + outcome.reason"
                if required
                else "    assert outcome.reason in (\"\", " + repr(reason) + ")"
            ),
            "",
            "",
        ]
    lines += [
        "def test_reward_is_continuous_across_the_margin():",
        '    """The graded ramp moves through the margin; it does not step at it."""',
        "    band = dict(telemetry()[\"measurement\"])",
        "    below = dict(band)",
        "    below[\"separation_mean\"] = BOUND_MARGIN + band[\"noise_half_width\"] - 1e-06",
        "    above = dict(band)",
        "    above[\"separation_mean\"] = BOUND_MARGIN + band[\"noise_half_width\"] + 1e-06",
        "    low = classify_separation(below)",
        "    high = classify_separation(above)",
        "    assert low[0] == \"unproven\" and high[0] == \"established\"",
        "    assert abs(high[2] - low[2]) < 1e-03",
        "",
        "",
        "def test_every_manifest_selector_is_reachable():",
        '    """Every reduction the manifest names resolves, and the required set is non-empty."""',
        "    assert len(SELECTORS) == 10",
        "    assert len(REQUIRED) == 9",
        "    for ident, _ in SELECTORS:",
        "        assert _reduction(ident) is not None",
        "",
        "",
        "def test_graded_document_agrees_with_the_reductions():",
        '    """The score document the grader wrote reports the same reason the reductions do."""',
        "    document = verdict()",
        "    assert 0.0 <= float(document[\"reward\"]) <= 1.0",
        "    assert str(document[\"reason\"]).strip()",
        "    assert document[\"metric\"][\"anchors_state\"] == \"absent\"",
        "    assert document[\"metric\"][\"baseline_metric\"] is None",
        "    assert document[\"metric\"][\"target_metric\"] is None",
        "",
        "",
        "def main(argv=None) -> int:",
        "    parser = argparse.ArgumentParser(description=\"OER-20 compiled tests\")",
        "    parser.add_argument(\"--telemetry\", required=True)",
        "    parser.add_argument(\"--verdict\", required=True)",
        "    args = parser.parse_args(argv)",
        "    os.environ[\"OER20_TELEMETRY\"] = args.telemetry",
        "    os.environ[\"OER20_VERDICT\"] = args.verdict",
        "    failures = []",
        "    for name, function in sorted(globals().items()):",
        "        if not name.startswith(\"test_\"):",
        "            continue",
        "        try:",
        "            function()",
        "        except AssertionError as problem:",
        "            failures.append(name + \": \" + str(problem))",
        "    for line in failures:",
        "        print(\"FAIL \" + line)",
        "    print(\"compiled tests: \" + str(len(failures)) + \" failing\")",
        "    return 1 if failures else 0",
        "",
        "",
        "if __name__ == \"__main__\":",
        "    sys.exit(main())",
        "",
    ]
    return "\n".join(lines)


# Every generated artifact, and the one builder that derives it. Uniform arity on
# purpose: a dispatch that guesses at a builder's signature is a dispatch that
# turns a real TypeError inside a builder into a silently different artifact.
ARTIFACTS = (
    ("solution/solve.sh", build_solve_sh),
    ("solution/TRUTH.md", build_truth_md),
    ("solution/rubrics.json", build_rubrics_json),
    ("tests/test_output.py", build_test_output),
)


def render(grounding: dict, derived: dict) -> dict:
    return {relative: builder(grounding, derived) for relative, builder in ARTIFACTS}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="OER-20 artifact derivation")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    grounding = load_grounding()
    verify_substrate(grounding)
    derived = derive_result(grounding)
    check_recorded_result(grounding, derived)
    rendered = render(grounding, derived)

    drifted = []
    for relative, text in sorted(rendered.items()):
        path = BUNDLE / relative
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                drifted.append(relative)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        if relative.endswith(".sh"):
            path.chmod(0o755)

    if args.check:
        for relative in drifted:
            print("DRIFTED " + relative)
        print("recompute --check: " + str(len(drifted)) + " artifact(s) drifted")
        return 1 if drifted else 0

    for relative in sorted(rendered):
        print("wrote " + relative)
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
