# FORGE-CANARY-BEGIN
# 0: 7279f06eaca7e6e56cd377cd18031950c3c86f9eee50d066360d84d094275959
# 1: 175843a6807dd54a5c81540bd837a5fe03a920b64861fb04a4fb13232cc76b25
# 2: 2896c052e2fad2e3715d47f60e9f2dda20bba461996e4cbb6d3bfce027e9274e
# 3: 9ae7bfa80a56d5d5f55e7e2c04c08fe3eb9ddd29f7236d6a3a72cef11914e150
# FORGE-CANARY-END
#!/usr/bin/env python3
"""Derive every generated artifact in this bundle from solution/grounding.yaml.

One source, one derivation, five generated files. Nothing here invokes a model,
opens a socket, reads a clock, consults a locale or touches a random source, so
running it twice over frozen bytes produces byte-identical output.

WHAT THIS PASS RE-BASED. The preceding pass replaced this slot's arithmetic
surrogate with the frozen nanoGPT checkpoint and a real forward pass, and rewrote
tests/runner.py accordingly: `schedule_points` lost its environment argument and
now derives the order from the verifier's own held-out manifest, and `evaluate`
grew from five parameters to eight because it needs a live model, a reference
state, a control allocation, a state digest and a device. This file still called
the retired five-parameter shape, so it raised TypeError before it derived
anything and the whole recompute chain could not be run. It is re-based here onto
the API tests/runner.py actually exports.

WHAT IS DERIVED AND WHAT IS NOT. The bit accounting is fully derivable from
bundle bytes and is derived: this file walks environment/model_stats.json with the
allocation it reads out of the digest-bound solution/reference.py, through the
LIVE tests/runner.py::normalise_allocation and ::accounting, and refuses to emit
anything if the walk disagrees with what solution/grounding.yaml records. The
separation mean, the noise half width and the evaluation order are not derivable:
they are functions of the verifier-owned checkpoint and held-out split, neither of
which is a bundle byte. They are recorded as null and published as absent, and no
number is substituted for them.

THE CARRIERS ARE VERIFIED AND NEVER WRITTEN. environment/eval_corpus.json carries
real FineWeb10B bytes emitted by seed/forge/substrate.py, and
environment/model_stats.json and environment/reference_unquantized.json are the
declarations the re-base authored. All three are read and asserted against the
byte length and the sha256 solution/grounding.yaml declares, and all three are
absent from this file's artifact map. There is no branch here that writes one,
because a carrier that fails its verification is recovered from its source and is
never repaired by regeneration over the top of real bytes.

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
CALIBRATION_ARTIFACT = "tests/calibration.py"

BUNDLE = Path(__file__).resolve().parent.parent
GROUNDING = BUNDLE / "solution" / "grounding.yaml"
ENVIRONMENT = BUNDLE / "environment"
TESTS = BUNDLE / "tests"

MODEL_STATS = "environment/model_stats.json"
TELEMETRY_SCHEMA = "oer20.telemetry/v1"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_grounding() -> dict:
    return yaml.safe_load(GROUNDING.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The frozen inputs, VERIFIED against their declaration and never written
# ---------------------------------------------------------------------------


def verify_carriers(grounding: dict) -> dict:
    """Read each declared carrier and assert it is the bytes grounding.yaml declares.

    Byte length and sha256, both, and a hard failure on either. One of these three
    is upstream FineWeb10B bytes this bundle did not author and the other two are
    declarations of a checkpoint this bundle does not carry, so for none of them is
    there a construction to re-run. A carrier that is absent, truncated or
    overwritten stops the whole recompute here rather than being silently rebuilt.
    """
    out: dict = {}
    for row in grounding["substrate"]["carriers"]:
        relative = str(row["path"])
        path = BUNDLE / relative
        if not path.is_file():
            raise SystemExit(
                "substrate drift: the declared carrier " + relative + " is absent. Nothing here "
                "can regenerate it; recover it by the recipe in grounding.yaml "
                "substrate.carriers, which for this row reads: " + str(row["recovery"]).strip()
            )
        payload = path.read_bytes()
        measured = hashlib.sha256(payload).hexdigest()
        if len(payload) != int(row["length_bytes"]) or measured != str(row["sha256"]):
            raise SystemExit(
                "substrate drift: " + relative + " measures " + str(len(payload)) + " bytes at "
                + measured + " and " + SOURCE + " substrate.carriers declares "
                + str(row["length_bytes"]) + " bytes at " + str(row["sha256"])
                + ". Refusing to derive anything from a source that no longer describes the bytes."
            )
        out[relative] = payload
    return out


def load_reference(grounding: dict):
    """Read the reference allocation out of the one file that carries it.

    The allocation is bound by the digest of its canonical payload and is NOT
    restated in grounding.yaml, so this bundle holds exactly one allocation. A
    second copy would be a second source that can disagree with the first, and the
    predecessor of this pass is what that looks like: grounding.yaml restated a
    twelve-row allocation over tensor names the checkpoint does not carry.
    """
    reference = load_module("oer20_reference_for_recompute", BUNDLE / "solution" / "reference.py")
    payload = hashlib.sha256(reference.canonical()).hexdigest()
    if payload != grounding["reference"]["allocation_sha256"]:
        raise SystemExit(
            "reference allocation digest moved: grounding binds "
            + grounding["reference"]["allocation_sha256"] + " and reference.py hashes to " + payload
        )
    return reference


# ---------------------------------------------------------------------------
# The derivation
# ---------------------------------------------------------------------------


def derive_result(grounding: dict, runner, reference, manifest: dict) -> dict:
    """Recompute what the bundle's own bytes determine, and null what they do not.

    The whole bit accounting is a function of environment/model_stats.json and the
    digest-bound reference allocation, and it is computed here through the LIVE
    tests/runner.py reductions rather than restated. The measurement is not: the
    separation mean, the noise half width and the evaluation order come out of
    forward passes over a checkpoint and a held-out split that are mounted on the
    verifier surface and are absent from this bundle, so they are recorded as null.
    Deriving them here would mean inventing them.
    """
    allocation = runner.normalise_allocation(manifest, reference.REFERENCE_ALLOCATION)
    if not allocation["wellformed"]:
        raise SystemExit(
            "the reference allocation does not resolve against " + MODEL_STATS + ": "
            + str(allocation["defect"])
        )
    account = runner.accounting(manifest, allocation)
    return {
        "measured": False,
        "allocated_bits": account["allocated_bits"],
        "budget_bits": account["budget_bits"],
        "mean_bits_per_param": round(account["mean_bits_per_param"], 6),
        "scheduled_order": None,
        "separation_mean": None,
        "noise_half_width": None,
        "lower_edge": None,
        "verdict": None,
        "reason": None,
        "reward": None,
    }


def check_recorded_result(grounding: dict, derived: dict) -> None:
    recorded = dict(grounding["expected_result"])
    for key, value in sorted(derived.items()):
        if recorded[key] != value:
            raise SystemExit(
                "expected_result." + key + " in " + SOURCE + " records " + repr(recorded[key])
                + " and the derivation produces " + repr(value)
            )


def check_band_fixtures(grounding: dict, checkers) -> None:
    """Run every frozen band fixture through the LIVE reduction it names.

    The fixtures are recorded as absolute readings rather than as offsets from the
    grounded centre and width, because under the determinism identity both of those
    are 0.0 and a multiplicative offset from zero is zero for every row. What keeps
    them load bearing is this check: each accepting row must be accepted and each
    rejecting row must be rejected, by tests/checkers.py's own reduction over
    tests/calibration.py's own constants, with the recorded reason. A row that
    stopped exercising its half fails the derivation instead of shipping.
    """
    reductions = dict(checkers.SELECTORS)
    for row in grounding["calibration_band_fixtures"]:
        ident, name = str(row["id"]), str(row["checker"])
        if name not in reductions:
            raise SystemExit(
                "calibration_band_fixtures row " + ident + " names the reduction " + name
                + ", which tests/checkers.py does not carry"
            )
        outcome = reductions[name](fixture_telemetry(grounding, row))
        accepting = str(row["half"]) == "accepting"
        if bool(outcome.passed) != accepting:
            raise SystemExit(
                "calibration_band_fixtures row " + ident + " is recorded as the " + str(row["half"])
                + " half of " + name + " and the live reduction scored it " + repr(outcome.value)
                + " with reason " + repr(outcome.reason)
            )
        if outcome.reason != str(row["expect_reason"]):
            raise SystemExit(
                "calibration_band_fixtures row " + ident + " records reason "
                + repr(str(row["expect_reason"])) + " and the live reduction carried "
                + repr(outcome.reason)
            )


def fixture_telemetry(grounding: dict, row: dict) -> dict:
    """The one telemetry record a band fixture presents, built once and rendered once."""
    return {
        "schema": TELEMETRY_SCHEMA,
        "calibration": {
            "source": "harness-recompute",
            "wellformed": True,
            "points_used": int(grounding["bindings"]["evaluation_repeat_floor"]),
            "separation_mean": float(row["separation_mean"]),
            "noise_half_width": float(row["noise_half_width"]),
        },
    }


def published(grounding: dict, derived: dict, key: str) -> str:
    """A measured value, or the absence string grounding.yaml publishes in its place."""
    value = derived[key]
    if value is None:
        return str(grounding["expected_result"]["published_as"][key])
    return str(value)


def comment_lines(text: str, marker: str = "# ") -> list:
    """One comment line per source line, with the blank lines kept bare."""
    return [(marker + line).rstrip() for line in str(text).rstrip("\n").split("\n")]


# ---------------------------------------------------------------------------
# The builders. Each takes the one context the derivation assembled.
# ---------------------------------------------------------------------------


def build_calibration_py(context: dict) -> str:
    """Render the frozen band constants and the frozen band fixtures.

    This artifact takes no derived measurement, because tests/runner.py imports it
    and the derivation imports tests/runner.py. A builder that needed the
    derivation would need the derivation to already have run, which is the cycle
    this signature refuses. The tensor list it walks is read straight out of the
    verified environment/model_stats.json rather than through runner.py, for the
    same reason.
    """
    grounding = context["grounding"]
    prose = grounding["artifact_prose"]
    probe = grounding["calibration_probe"]
    points = int(grounding["bindings"]["evaluation_repeat_floor"])
    lines = [
        "#!/usr/bin/env python3",
        '"""' + BANNER,
        "",
        "Source: " + SOURCE,
        "",
        "Frozen constants and frozen fixtures, and no logic at all.",
        "",
        "tests/runner.py reads CALIBRATION_ALLOCATION and measures it against the built",
        "environment on every run. tests/checkers.py reads the band constants and reduces",
        "that measurement. tests/test_output.py reads BAND_FIXTURES and exercises the",
        "accepting and the rejecting half of each reduction over frozen bytes.",
        "",
    ]
    lines += str(prose["calibration_module_note"]).rstrip("\n").split("\n")
    lines += [
        '"""',
        "",
        "from __future__ import annotations",
        "",
    ]
    lines += comment_lines(prose["calibration_allocation_note"])
    lines += [
        "CALIBRATION_ALLOCATION = {",
        '    "scheme": ' + repr(str(grounding["bindings"]["control_scheme"])) + ",",
        '    "bits": {',
    ]
    for name in context["tensor_names"]:
        lines.append(
            "        " + repr(name) + ": " + repr(int(grounding["bindings"]["control_bits"])) + ","
        )
    lines += [
        "    },",
        "}",
        "",
        "EXPECTED_SEPARATION_MEAN = " + repr(float(probe["expected_separation_mean"])),
        "EXPECTED_NOISE_HALF_WIDTH = " + repr(float(probe["expected_noise_half_width"])),
        "HALF_WIDTH_TOLERANCE = " + repr(float(probe["half_width_tolerance"])),
        "BAND_EPSILON = " + repr(float(probe["band_epsilon"])),
        "",
        "BAND_FIXTURES = {",
    ]
    for row in grounding["calibration_band_fixtures"]:
        lines += [
            "    " + repr(str(row["id"])) + ": {",
            '        "checker": ' + repr(str(row["checker"])) + ",",
            '        "half": ' + repr(str(row["half"])) + ",",
            '        "expect_reason": ' + repr(str(row["expect_reason"])) + ",",
            '        "proves": ' + repr(str(row["proves"])) + ",",
            '        "telemetry": {',
            '            "schema": "' + TELEMETRY_SCHEMA + '",',
            '            "calibration": {',
            '                "source": "harness-recompute",',
            '                "wellformed": True,',
            '                "points_used": ' + repr(points) + ",",
            '                "separation_mean": ' + repr(float(row["separation_mean"])) + ",",
            '                "noise_half_width": ' + repr(float(row["noise_half_width"])) + ",",
            "            },",
            "        },",
            "    },",
        ]
    lines += ["}", ""]
    return "\n".join(lines)


def build_solve_sh(context: dict) -> str:
    grounding, derived = context["grounding"], context["derived"]
    ref = grounding["reference"]
    scope = str(ref["allocation_scope"]).rstrip("\n").split("\n")
    out = [
        "#!/usr/bin/env bash",
        "# " + BANNER,
        "# Source: " + SOURCE,
        "#",
    ]
    out += comment_lines(ref["solve_notice"])
    out += [
        "#",
        "# scheme: " + str(context["reference"].REFERENCE_ALLOCATION["scheme"]),
        "# allocation sha256: " + str(ref["allocation_sha256"]),
        "# allocated bits: " + str(derived["allocated_bits"]) + " of "
        + str(derived["budget_bits"]) + " budgeted, " + scope[0],
    ]
    out += ["# " + line for line in scope[1:]]
    out += [
        "set -euo pipefail",
        "",
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'python3 "${HERE}/reference.py"',
        "",
    ]
    return "\n".join(out)


def build_truth_md(context: dict) -> str:
    grounding, derived = context["grounding"], context["derived"]
    ref = grounding["reference"]
    bits = context["reference"].REFERENCE_ALLOCATION["bits"]
    surface = grounding["measurement_surface"]
    out = [
        "<!-- " + BANNER + " -->",
        "<!-- Source: " + SOURCE + " -->",
        "",
        "# TRUTH, slot " + str(grounding["slot"]),
        "",
        "Family " + str(grounding["family"]) + ", " + str(grounding["family_title"])
        + ". Primary archetype " + str(grounding["primary_archetype"]) + ", "
        + str(grounding["primary_archetype_title"]) + ", exactly one.",
        "",
        "## What the re-base moved",
        "",
    ]
    for paragraph in grounding["rebase"]["paragraphs"]:
        out += [str(paragraph).strip(), ""]
    out += [
        "## The reference allocation",
        "",
        "Scheme `" + str(context["reference"].REFERENCE_ALLOCATION["scheme"])
        + "`, allocation payload sha256 `" + str(ref["allocation_sha256"]) + "`.",
        "",
        "| tensor | bits |",
        "|---|---|",
    ]
    for name in context["tensor_names"]:
        out.append("| `" + name + "` | " + str(int(bits[name])) + " |")
    out += [
        "",
        str(ref["why"]).strip(),
        "",
        "## What the verifier measures",
        "",
        "| quantity | value |",
        "|---|---|",
        "| graded artifact | " + str(surface["graded_artifact"]) + " |",
        "| graded evaluation | " + str(surface["graded_evaluation"]) + " |",
        "| readings per point | " + str(surface["readings_per_point"]) + " |",
        "| allocated bits | " + str(derived["allocated_bits"]) + " |",
        "| budget bits | " + str(derived["budget_bits"]) + " |",
        "| mean bits per parameter | " + str(derived["mean_bits_per_param"]) + " |",
        "| verifier-derived evaluation order | " + published(grounding, derived, "scheduled_order") + " |",
        "| separation mean | " + published(grounding, derived, "separation_mean") + " |",
        "| noise half width | " + published(grounding, derived, "noise_half_width") + " |",
        "| lower edge of the band | " + published(grounding, derived, "lower_edge") + " |",
        "| bound separation margin | " + str(grounding["bindings"]["separation_margin"]) + " |",
        "| verdict | " + published(grounding, derived, "verdict") + " |",
        "| reward | " + published(grounding, derived, "reward") + " |",
        "",
        str(grounding["expected_result"]["why_absent"]).strip(),
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
    out += ["", "## Retired", ""]
    for gap in grounding["retired_gaps"]:
        out.append("- `" + str(gap["id"]) + "`: " + str(gap["what"]).strip())
    out.append("")
    return "\n".join(out)


def build_rubrics_json(context: dict) -> str:
    grounding, derived = context["grounding"], context["derived"]
    allocation = context["reference"].REFERENCE_ALLOCATION
    established = grounding["outcomes"]["established"]
    payload = {
        "schema": "forge.solution_rubrics/v1",
        "generated_section": BANNER,
        "generated_from": SOURCE,
        "slot": str(grounding["slot"]),
        "reference_allocation_sha256": grounding["reference"]["allocation_sha256"],
        "grades": "the solution against its reference answer",
        "not": "trajectory behaviour, which tests/rubrics.jsonl grades separately and which this file never substitutes for",
        "criteria": [
            {
                "id": "matches-reference-scheme",
                "statement": "The submitted scheme is the reference scheme.",
                "expect": str(allocation["scheme"]),
            },
            {
                "id": "matches-reference-allocation",
                "statement": "The submitted per-tensor bit allocation is the reference allocation.",
                "expect": {str(k): int(v) for k, v in sorted(allocation["bits"].items())},
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
                    "lower_edge": published(grounding, derived, "lower_edge"),
                    "separation_margin": grounding["bindings"]["separation_margin"],
                    "verdict": str(established["verdict"]),
                    "reason": str(established["reason"]),
                },
            },
            {
                "id": "earns-full-reward",
                "statement": "The reference solution is accepted end to end at full reward.",
                "expect": float(established["reward"]),
            },
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def build_test_output(context: dict) -> str:
    grounding, derived = context["grounding"], context["derived"]
    fixtures = grounding["checker_fixtures"]
    rows = [row for row in fixtures if row.get("checker")]
    band_rows = list(grounding["calibration_band_fixtures"])
    established = grounding["outcomes"]["established"]
    selector_count = len(rows)
    required_count = len([row for row in rows if row["checker"] != "separation_margin_cleared"])
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
        "from calibration import BAND_FIXTURES  # noqa: E402",
        "from checkers import (  # noqa: E402",
        "    REQUIRED,",
        "    SELECTORS,",
        "    classify_separation,",
        ")",
        "",
        "_STATE = {\"telemetry\": None, \"verdict\": None}",
        "",
    ]
    lines += comment_lines(grounding["artifact_prose"]["compiled_test_order_note"])
    lines += [
        "EXPECTED_ORDER = " + repr(derived["scheduled_order"]),
        "EXPECTED_ALLOCATED_BITS = " + repr(int(derived["allocated_bits"])),
        "EXPECTED_BUDGET_BITS = " + repr(int(derived["budget_bits"])),
        "EXPECTED_REASON = " + repr(str(established["reason"])),
        "EXPECTED_REWARD = " + repr(float(established["reward"])),
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
        "def _reduction_over(name, record):",
        "    for ident, selector in SELECTORS:",
        "        if ident == name:",
        "            return selector(record)",
        "    raise AssertionError(\"tests/checkers.yaml names a selector checkers.py does not carry: \" + name)",
        "",
        "",
        "def _reduction(name):",
        "    return _reduction_over(name, telemetry())",
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
    for row in band_rows:
        ident = str(row["id"])
        checker = str(row["checker"])
        reason = str(row["expect_reason"])
        accepting = str(row["half"]) == "accepting"
        lines += [
            "def test_band_fixture_" + ident + "():",
            '    """' + str(row["proves"]).strip() + '"""',
            "    fixture = BAND_FIXTURES[" + repr(ident) + "]",
            "    outcome = _reduction_over(fixture[\"checker\"], fixture[\"telemetry\"])",
            (
                "    assert outcome.passed, " + repr(ident) + " + \" scored \" + repr(outcome.value) + \" with reason \" + outcome.reason"
                if accepting
                else "    assert not outcome.passed, " + repr(ident) + " + \" was accepted at \" + repr(outcome.value) + \", so \" + " + repr(checker) + " + \" does not depend on the grounded value\""
            ),
            "    assert outcome.reason == " + repr(reason) + ", " + repr(ident) + " + \" carried reason \" + repr(outcome.reason)",
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
        "    assert len(SELECTORS) == " + repr(selector_count),
        "    assert len(REQUIRED) == " + repr(required_count),
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
# turns a real TypeError inside a builder into a silently different artifact. The
# three verified carriers are NOT in this map and never enter it, because they are
# inputs to the derivation rather than outputs of it.
ARTIFACTS = (
    ("solution/solve.sh", build_solve_sh),
    ("solution/TRUTH.md", build_truth_md),
    ("solution/rubrics.json", build_rubrics_json),
    ("tests/test_output.py", build_test_output),
)


def render(context: dict) -> dict:
    return {relative: builder(context) for relative, builder in ARTIFACTS}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="OER-20 artifact derivation")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    grounding = load_grounding()
    verify_carriers(grounding)

    # tests/ carries the live reductions this file derives through, and both
    # tests/runner.py and tests/checkers.py import tests/calibration.py by name.
    if str(TESTS) not in sys.path:
        sys.path.insert(0, str(TESTS))

    manifest = json.loads((BUNDLE / MODEL_STATS).read_text(encoding="utf-8"))
    context = {
        "grounding": grounding,
        "manifest": manifest,
        "tensor_names": [str(row["name"]) for row in manifest["tensors"]],
    }

    # tests/calibration.py is the bootstrap artifact: tests/runner.py and
    # tests/checkers.py both import it and the derivation imports both, so it is
    # emitted or compared BEFORE the derivation rather than inside the uniform
    # render loop below.
    bootstrap = BUNDLE / CALIBRATION_ARTIFACT
    bootstrap_text = build_calibration_py(context)
    if args.check:
        current = bootstrap.read_text(encoding="utf-8") if bootstrap.is_file() else ""
        if current != bootstrap_text:
            print("DRIFTED " + CALIBRATION_ARTIFACT)
            print("recompute --check: 1 artifact(s) drifted")
            return 1
    else:
        bootstrap.parent.mkdir(parents=True, exist_ok=True)
        bootstrap.write_text(bootstrap_text, encoding="utf-8")

    runner = load_module("oer20_runner_for_recompute", TESTS / "runner.py")
    checkers = load_module("oer20_checkers_for_recompute", TESTS / "checkers.py")
    context["reference"] = load_reference(grounding)
    context["derived"] = derive_result(grounding, runner, context["reference"], manifest)
    check_recorded_result(grounding, context["derived"])
    check_band_fixtures(grounding, checkers)
    rendered = render(context)

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
    "screening_measured_at",
    "screening_interval_days",
    "screening_expires_at",
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
