# FORGE-CANARY-BEGIN
# 0: a43cbc8e00d4b1ef5d5a005fa19c65fc14b635a4fec09066ca2ea7b99e73978c
# 1: 679638b291eda2a66c3081c64c054c162a3974be50f32748f47d69c1022bedf1
# 2: e96fac6f076dca23d030e6f8521fe5a9f07db3cd1263e9f3ab041128c4fee4a3
# 3: eaf81790ee48713d672e589140d6692e1681d8bcff7fc7fe2da7cd1ea10c327b
# FORGE-CANARY-END
#!/usr/bin/env python3
"""Derive every generated artifact in this bundle from solution/grounding.yaml.

One private grounding file is the single derivation source. This script reads it,
asserts that the frozen surface in tests/lab.py is the same source rather than a
second one, walks the recorded reference plan through that surface, and writes:

    solution/solve.sh
    solution/TRUTH.md
    solution/rubrics.json
    solution/golden_trajectory.json
    tests/fixtures/golden_session.json
    tests/fixtures/reject_<checker>.json
    tests/test_output.py

It invokes no model, no network, no clock, no locale and no random source. It
launches no subprocess and never runs the submission. Every artifact carries the
generated banner naming grounding.yaml as its source, and running this twice over
frozen bytes produces byte-identical output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BUNDLE / "tests"))

import lab  # noqa: E402
import runner  # noqa: E402

import yaml  # noqa: E402

BANNER = "GENERATED SECTION. DO NOT HAND-EDIT."
SOURCE = "solution/grounding.yaml"
GROUNDING = BUNDLE / "solution" / "grounding.yaml"
FIXTURES = BUNDLE / "tests" / "fixtures"


def banner(prefix: str) -> str:
    return prefix + BANNER + " Source: " + SOURCE + "."


def load_grounding() -> dict:
    return yaml.safe_load(GROUNDING.read_text(encoding="utf-8"))


def assert_single_source(ground: dict) -> None:
    """The surface block and tests/lab.py must be one source, not two.

    A drift between them would let the golden trajectory describe a surface the
    verifier does not grade against, which is the exact failure a single
    derivation source exists to prevent.
    """
    surface = ground["surface"]
    pairs = [
        ("surface_id", lab.SURFACE_ID),
        ("base_steps", lab.BASE_STEPS),
        ("axis_order", list(lab.AXIS_ORDER)),
        ("axis_milli_per_unit", lab.AXIS_MILLI_PER_UNIT),
        ("axis_plateau_units", lab.AXIS_PLATEAU_UNITS),
        ("flattening_axis", lab.FLATTENING_AXIS),
        ("unit_scale", lab.UNIT_SCALE),
        ("delta_cap_units", lab.DELTA_CAP_UNITS),
        ("attempt_budget", lab.ATTEMPT_BUDGET),
        ("probe_allowance", lab.PROBE_ALLOWANCE),
        ("seed_ladder", list(lab.SEED_LADDER)),
        ("jitter_span", lab.JITTER_SPAN),
        ("separation_margin", lab.SEPARATION_MARGIN),
        ("seed_artifact_band", lab.SEED_ARTIFACT_BAND),
        ("target_loss", lab.TARGET_LOSS),
        ("eval_every", lab.EVAL_EVERY),
        ("sustain_points", lab.SUSTAIN_POINTS),
        ("curve_max_steps", lab.CURVE_MAX_STEPS),
        ("dip_offset_points", lab.DIP_OFFSET_POINTS),
        ("dip_depth", lab.DIP_DEPTH),
        ("zero_state", lab.ZERO_STATE),
        ("reference_state", lab.REFERENCE_STATE),
    ]
    for key, live in pairs:
        if surface[key] != live:
            raise SystemExit(
                "grounding.yaml surface." + key + " is " + repr(surface[key])
                + " while tests/lab.py holds " + repr(live)
                + "; the derivation source and the graded surface have drifted apart"
            )

    declared = ground["reference_policy"]["expected_final_state"]
    if declared != lab.REFERENCE_FINAL_STATE:
        raise SystemExit(
            "grounding.yaml reference_policy.expected_final_state is " + repr(declared)
            + " while tests/lab.py holds REFERENCE_FINAL_STATE " + repr(lab.REFERENCE_FINAL_STATE)
            + "; the graded reference operating point and the derivation source have drifted apart"
        )
    if lab.gain_milli(declared) != lab.gain_milli(lab.REFERENCE_STATE):
        raise SystemExit(
            "the declared final state holds " + repr(lab.gain_milli(declared))
            + " milli-steps while the canonical reference state holds "
            + repr(lab.gain_milli(lab.REFERENCE_STATE))
            + "; they are not one operating point and split invariance does not hold between them"
        )


def expand_plan(ground: dict) -> list:
    plan = []
    for axis, units, repeat in ground["reference_policy"]["reference_plan"]:
        plan.extend([(str(axis), int(units))] * int(repeat))
    return plan


def golden_session(ground: dict) -> list:
    """Walk the recorded reference plan through the frozen surface, purely.

    The record builders in tests/runner.py are reused rather than restated, so the
    golden telemetry is produced by the same code the live run produces it with.
    No process is launched: only the pure measurement functions are called.
    """
    digest = _sha256(BUNDLE / "solution" / "reference.py")
    controls = lab.control_detail()
    records = [runner._header(digest, controls)]
    attempts = []
    state = lab.zero_state()
    for index, (axis, units) in enumerate(expand_plan(ground)):
        before = dict(state)
        applied = lab.apply_delta(state, axis, units)
        record = runner._measure(index, axis, applied, before, dict(state), {})
        attempts.append(record)
        records.append(record)
    records.append(runner._footer(attempts, controls[0], controls[2], digest))
    return records


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _graded_index(records: list) -> int:
    footer = records[-1]
    return int(footer["best_attempt_index"])


def _clone(records: list) -> list:
    return json.loads(json.dumps(records, sort_keys=True))


def reject_fixtures(records: list) -> dict:
    """One planted-defect telemetry per checker, each moving only its own field."""
    graded = _graded_index(records)
    out = {}

    rows = _clone(records)
    rows[1 + 3]["index"], rows[1 + 4]["index"] = rows[1 + 4]["index"], rows[1 + 3]["index"]
    out["attempt_sequence_contiguous"] = rows

    rows = _clone(records)
    out["session_holds_attempt_budget"] = rows[:1] + rows[1:-1][:-2] + rows[-1:]

    rows = _clone(records)
    rows[1 + 20]["state_before"]["a"] = 0
    out["carried_state_monotone"] = rows

    rows = _clone(records)
    rows[-1]["post_flattening_attempts_on_flattened_axis"] = 39
    rows[-1]["first_reallocation_attempt"] = None
    out["reallocation_follows_flattening"] = rows

    rows = _clone(records)
    rows[1 + graded]["weights_digest"] = "0" * 32
    out["graded_weights_are_run_weights"] = rows

    rows = _clone(records)
    rows[1 + 7]["readout_filter"] = "ema"
    rows[1 + 7]["directives"]["report_ema"] = True
    out["graded_readout_unsmoothed"] = rows

    rows = _clone(records)
    crossing = rows[1 + graded]["graded_crossing_step"]
    rows[1 + graded]["halted_at"] = crossing + lab.EVAL_EVERY
    rows[1 + graded]["directives"]["stop_at"] = crossing + lab.EVAL_EVERY
    out["early_stop_is_not_a_crossing"] = rows

    rows = _clone(records)
    dip = crossing - lab.DIP_OFFSET_POINTS * lab.EVAL_EVERY
    rows[1 + graded]["claimed_crossing_step"] = dip
    rows[1 + graded]["claimed_step_sustained"] = False
    rows[1 + graded]["directives"]["claim_crossing_at"] = dip
    out["crossing_sustained"] = rows

    rows = _clone(records)
    rows[-1]["best_mean_steps"] = rows[0]["baseline_control_mean"]
    out["target_band_reached_by_session"] = rows

    rows = _clone(records)
    ladder = rows[0]["seed_ladder"]
    baseline_seeds = rows[0]["baseline_control_per_seed"]
    carried = dict(rows[-1]["best_per_seed_steps"])
    for position, seed in enumerate(ladder):
        if position:
            carried[seed] = baseline_seeds[seed]
    rows[-1]["best_per_seed_steps"] = carried
    out["multi_seed_separation_established"] = rows

    rows = _clone(records)
    wrong = dict(lab.REFERENCE_FINAL_STATE)
    paying = [axis for axis in lab.AXIS_ORDER if axis != lab.FLATTENING_AXIS]
    wrong[lab.FLATTENING_AXIS] += lab.DELTA_CAP_UNITS
    wrong[paying[-1]] -= lab.DELTA_CAP_UNITS
    wrong_mean, wrong_seeds = lab.mean_steps(dict(wrong))
    rows[0]["reference_final_state"] = wrong
    rows[0]["reference_final_state_gain_milli"] = lab.gain_milli(wrong)
    rows[0]["reference_final_state_control_mean"] = wrong_mean
    rows[0]["reference_final_state_control_per_seed"] = wrong_seeds
    out["reference_operating_point_matches_final_state"] = rows

    return out


def write_jsonl_fixture(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "banner": BANNER,
        "source": SOURCE,
        "records": records,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_solve(ground: dict) -> None:
    text = "\n".join([
        "#!/usr/bin/env bash",
        "# " + banner(""),
        "#",
        "# The reference session policy is a single self-contained file, because the",
        "# harness copies the submission alone into a fresh directory and runs it there.",
        "set -euo pipefail",
        "",
        'SOLUTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'TARGET="${OER06_SUBMISSION:-/app/submission.py}"',
        "",
        'mkdir -p "$(dirname "$TARGET")"',
        'cp "$SOLUTION_DIR/reference.py" "$TARGET"',
        'chmod 0644 "$TARGET"',
        "",
        'echo "installed reference session policy at $TARGET"',
        "",
    ])
    (BUNDLE / "solution" / "solve.sh").write_text(text, encoding="utf-8")
    (BUNDLE / "solution" / "solve.sh").chmod(0o755)


def write_truth(ground: dict, records: list) -> None:
    footer = records[-1]
    header = records[0]
    lines = [
        "# TRUTH.md",
        "",
        "<!-- " + banner("") + " -->",
        "",
        "## " + ground["truth"]["headline"],
        "",
    ]
    lines.extend(paragraph + "\n" for paragraph in ground["truth"]["paragraphs"])
    lines.extend([
        "## What the reference session measures",
        "",
        "| quantity | value |",
        "|---|---|",
        "| attempts recorded | " + str(footer["attempts_recorded"]) + " |",
        "| attempt terminator | " + str(footer["terminator"]) + " |",
        "| flattening onset attempt | " + str(footer["flattening_onset_attempt"]) + " |",
        "| first reallocation attempt | " + str(footer["first_reallocation_attempt"]) + " |",
        "| attempts spent on the flattened direction after onset | "
        + str(footer["post_flattening_attempts_on_flattened_axis"]) + " |",
        "| graded attempt | " + str(footer["best_attempt_index"]) + " |",
        "| best multi-seed mean | " + repr(footer["best_mean_steps"]) + " |",
        "| zero-progress control | " + repr(header["baseline_control_mean"]) + " |",
        "| reference-operating-point control | " + repr(header["target_control_mean"]) + " |",
        "| reward | " + repr(
            lab.margin_taper(
                lab.raw_reward(
                    header["baseline_control_mean"],
                    header["target_control_mean"],
                    footer["best_mean_steps"],
                )
            )
        ) + " |",
        "",
        "## Anchors",
        "",
        "This family's `baseline_metric` and `target_metric` are **absent**, under",
        "`gap-oer-per-family-anchors-unmeasured`. The two control values in the table",
        "above are in-run measurements of the frozen surrogate surface taken by the",
        "verifier, not published anchors, and neither may be transcribed into them.",
        "",
        "## The two budgets",
        "",
        "`max_timeout` is 6.0 hours and terminates the refinement loop **across**",
        "attempts. `budget_hours` is 0.12 hours and bounds **one** attempt. They are",
        "distinct fields with distinct roles and never stand in for each other.",
        "",
    ])
    (BUNDLE / "solution" / "TRUTH.md").write_text("\n".join(lines), encoding="utf-8")


def write_rubrics(ground: dict, records: list) -> None:
    header = records[0]
    footer = records[-1]
    payload = {
        "banner": BANNER,
        "source": SOURCE,
        "judged": "the solution against its reference answer",
        "not_this_file": "tests/rubrics.jsonl judges the trajectory and is never generated from here",
        "reference_sha256": header["submission_sha256"],
        "expected_reward": lab.margin_taper(
            lab.raw_reward(
                header["baseline_control_mean"],
                header["target_control_mean"],
                footer["best_mean_steps"],
            )
        ),
        "expected_reason": ground["reference_policy"]["expected_reason"],
        "criteria": [
            {"id": row["id"], "criterion": row["criterion"]}
            for row in ground["rubric_criteria"]
        ],
    }
    (BUNDLE / "solution" / "rubrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_golden(ground: dict, records: list) -> None:
    payload = {
        "banner": BANNER,
        "source": SOURCE,
        "reference_plan": ground["reference_policy"]["reference_plan"],
        "expected_final_state": ground["reference_policy"]["expected_final_state"],
        "records": records,
    }
    (BUNDLE / "solution" / "golden_trajectory.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


TEST_TEMPLATE = '''#!/usr/bin/env python3
"""{banner}

Compiled per-checker tests. Each one runs BOTH halves of its checker: the
accepting half over the golden session fixture and the rejecting half over a
planted-defect fixture that moves exactly the field that checker grades.

Runs standalone under python3 and needs no test framework.
"""

import json
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

from checkers import Session  # noqa: E402
{imports}

FIXTURES = TESTS / "fixtures"


def load(name):
    payload = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    session = Session()
    for row in payload["records"]:
        kind = row.get("kind")
        if kind == "session_header":
            session.header = row
        elif kind == "attempt":
            session.attempts.append(row)
        elif kind == "session_footer":
            session.footer = row
    return session


def _both(check, reject_name, reason):
    accepted = check(load("golden_session.json"))
    assert accepted.passed, check.__name__ + " rejected the golden session: " + accepted.detail
    rejected = check(load(reject_name))
    assert not rejected.passed, check.__name__ + " accepted its planted defect"
    assert rejected.zero_reason == reason, (
        check.__name__ + " emitted " + repr(rejected.zero_reason) + ", expected " + repr(reason)
    )


{bodies}

CASES = (
{cases}
)


def main():
    failures = []
    for name, case in CASES:
        try:
            case()
        except AssertionError as exc:
            failures.append(name + ": " + str(exc))
    for line in failures:
        print("FAIL " + line)
    if failures:
        return 1
    print("PASS " + str(len(CASES)) + " compiled checker tests, both halves each")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

BODY_TEMPLATE = '''def test_{ident}():
    _both({selector}, "reject_{ident}.json", "{reason}")
'''


def write_test_output(manifest: list) -> None:
    imports = "\n".join(
        "from checkers import " + row["selector"] + "  # noqa: E402" for row in manifest
    )
    bodies = "\n\n".join(
        BODY_TEMPLATE.format(ident=row["id"], selector=row["selector"], reason=row["zero_reason"])
        for row in manifest
    )
    cases = "\n".join(
        '    ("test_{0}", test_{0}),'.format(row["id"]) for row in manifest
    )
    text = TEST_TEMPLATE.format(
        banner=banner(""), imports=imports, bodies=bodies.rstrip("\n"), cases=cases
    )
    (BUNDLE / "tests" / "test_output.py").write_text(text, encoding="utf-8")


def main() -> int:
    ground = load_grounding()
    assert_single_source(ground)
    records = golden_session(ground)

    expected = ground["reference_policy"]["expected_final_state"]
    observed = records[-2]["state_after"]
    if {key: int(value) for key, value in observed.items()} != expected:
        raise SystemExit(
            "the recorded reference plan reaches " + repr(observed)
            + " while grounding.yaml records " + repr(expected)
        )

    manifest = yaml.safe_load(
        (BUNDLE / "tests" / "checkers.yaml").read_text(encoding="utf-8")
    )["checkers"]

    write_jsonl_fixture(FIXTURES / "golden_session.json", records)
    for ident, rows in reject_fixtures(records).items():
        write_jsonl_fixture(FIXTURES / ("reject_" + ident + ".json"), rows)

    write_solve(ground)
    write_truth(ground, records)
    write_rubrics(ground, records)
    write_golden(ground, records)
    write_test_output(manifest)

    print(
        json.dumps(
            {
                "generated": [
                    "solution/solve.sh",
                    "solution/TRUTH.md",
                    "solution/rubrics.json",
                    "solution/golden_trajectory.json",
                    "tests/test_output.py",
                    "tests/fixtures/golden_session.json",
                ]
                + sorted(
                    "tests/fixtures/reject_" + row["id"] + ".json" for row in manifest
                ),
                "source": SOURCE,
            },
            indent=2,
            sort_keys=True,
        )
    )
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
    raise SystemExit(main())
