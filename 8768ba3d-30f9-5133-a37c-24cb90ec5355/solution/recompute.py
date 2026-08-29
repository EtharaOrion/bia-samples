# FORGE-CANARY-BEGIN
# 0: b4838a0f83087adb74f843addf9cf410672fc4439ff89f594122056a1215cf86
# 1: 9a7e2db8ac5e6292a7c3679351f727b7075c39900a860d3cabbe99f27038f9cf
# 2: 577b9896b76539cafe222e3284c9d9a4fdfe580f7317a66e5bf38e766c001873
# 3: ec3f0dde595e2b1504a9319d79f0b0472d04fd53e226087e2518265bfff7d73f
# FORGE-CANARY-END
#!/usr/bin/env python3
"""Derive every generated artifact in this bundle from solution/grounding.yaml alone.

Generated here, and nowhere else:
    solution/fixtures/            checker fixtures and their manifest
    solution/trajectory.golden.json
    solution/solve.sh
    solution/TRUTH.md
    solution/rubrics.json
    tests/test_output.py

This module invokes no model, no network, no clock, no locale and no random
source. It reads one YAML file and writes text. Running it twice over frozen
bytes produces byte-identical output, which `--check` asserts by regenerating
into memory and comparing against what is committed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
GROUNDING = HERE / "grounding.yaml"
BANNER = "GENERATED SECTION. DO NOT HAND-EDIT."
SOURCE = "solution/grounding.yaml"


def load() -> dict:
    with GROUNDING.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _json(payload) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _stamp(payload: dict) -> dict:
    row = {"_generated": BANNER, "_source": SOURCE}
    row.update(payload)
    return row


# --------------------------------------------------------------------------
# plans


def expand_groups(ground: dict, name: str) -> dict:
    draws = []
    for group in ground["plan_groups"][name]:
        for index in group["indices"]:
            draws.append({"doc": group["source"] + ":" + str(index), "repeat": int(group["repeat"])})
    return {"schema": "oer08.plan/v1", "mode": "schedule", "draws": draws}


def constant_plan(ground: dict) -> dict:
    budget = int(ground["substrate"]["budget_tokens"])
    tokens = ground["calibration"]["default_simplex_optimum"]["tokens_per_source"]
    weights = {key: tokens[key] / budget for key in sorted(tokens) if tokens[key] > 0}
    return {"schema": "oer08.plan/v1", "mode": "constant", "weights": weights}


def fixture_rows(ground: dict) -> list:
    rows = []
    for raw in ground["fixtures"]:
        row = {
            "id": raw["id"],
            "kind": raw["kind"],
            "role": raw["role"],
            "substitution": bool(raw.get("substitution", False)),
            "expected_reward": float(raw["expected_reward"]),
            "expected_reason": raw["expected_reason"],
            "fires": raw.get("fires"),
            "submission": raw.get("submission"),
            "plan_file": None,
            "pool_override": raw.get("pool_override"),
            "readout_override": raw.get("readout_override"),
            "telemetry_override": raw.get("telemetry_override"),
        }
        if raw.get("plan_groups") or raw.get("plan_constant"):
            row["plan_file"] = "plan-" + raw["id"] + ".json"
        elif raw.get("telemetry_override"):
            # A planted-telemetry fixture grades the reference run and then moves
            # one handle, so the defect is the planted state and nothing else.
            row["plan_file"] = "plan-reference.json"
        rows.append(row)
    return rows


def plan_files(ground: dict) -> dict:
    out = {}
    for raw in ground["fixtures"]:
        if raw.get("plan_groups"):
            out["plan-" + raw["id"] + ".json"] = expand_groups(ground, raw["plan_groups"])
        elif raw.get("plan_constant"):
            out["plan-" + raw["id"] + ".json"] = constant_plan(ground)
    return out


def override_files(ground: dict) -> dict:
    pool = {}
    for name, row in (ground.get("pool_overrides") or {}).items():
        pool[name] = {key: value for key, value in row.items()}
    readout = {"ema": {"smoothing": "ema", "ema_beta": 0.9}}
    telemetry = {
        "model_counts": {
            "handle": "model",
            "set": {"counts_substituted": True},
            "scale_counts": 0.5,
            "note": "a count table that does not derive from the feed ledger",
        },
        "eval_batch_index": {
            "handle": "eval",
            "set_record_batch_index": 999,
            "note": "an evaluation recorded at a batch the feed never reached",
        },
    }
    return {"pool": pool, "readout": readout, "telemetry": telemetry}


# --------------------------------------------------------------------------
# artifacts


def trajectory(ground: dict) -> str:
    steps = []
    for position, step in enumerate(ground["reference_derivation"]["steps"]):
        steps.append({"order": position, "id": step["id"], "says": step["says"]})
    plan = expand_groups(ground, "reference")
    return _json(
        _stamp(
            {
                "slot": ground["slot"],
                "steps": steps,
                "plan": plan,
                "expected": ground["reference_derivation"]["expected"],
            }
        )
    )


def solve_sh(ground: dict) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "# " + BANNER,
        "# Source: " + SOURCE,
        "#",
        "# The reference solution's entry point. It runs solution/reference.py, which",
        "# derives the plan from the pool and the agent-visible dev probe and writes",
        "# plan.json into the working directory. Nothing else is produced, because",
        "# plan.json is the only artifact the grader reads.",
        "set -euo pipefail",
        "",
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'CORPUS="${OER08_CORPUS:-${HERE}/../environment/corpus_spec.json}"',
        'OUT="${OER08_PLAN_OUT:-plan.json}"',
        "",
        'python3 "${HERE}/reference.py" --corpus "${CORPUS}" --out "${OUT}"',
        "",
        "# Derivation, one line per step, from " + SOURCE + ":",
    ]
    for step in ground["reference_derivation"]["steps"]:
        lines.append("#   " + step["id"] + ": " + step["says"])
    lines.append("")
    return "\n".join(lines)


def truth_md(ground: dict) -> str:
    calibration = ground["calibration"]
    expected = ground["reference_derivation"]["expected"]
    lines = [
        "<!-- " + BANNER + " Source: " + SOURCE + " -->",
        "",
        "# TRUTH: " + ground["slot"] + ", " + ground["title"],
        "",
        "Family " + ground["family"] + ", " + ground["family_title"] + ". Archetype "
        + ground["archetype"] + ", " + ground["archetype_title"] + ".",
        "",
        "## The graded quantity",
        "",
        ground["readout_contract"]["statement"].strip(),
        "",
        "## Why the handed parameterisation is a trap",
        "",
        ground["archetype_construction"]["why_the_default_is_bounded_away"].strip(),
        "",
        ground["archetype_construction"]["the_wider_construction"].strip(),
        "",
        ground["archetype_construction"]["never_hidden"].strip(),
        "",
        "## The reference derivation",
        "",
    ]
    for position, step in enumerate(ground["reference_derivation"]["steps"], start=1):
        lines.append(str(position) + ". **" + step["id"] + "** " + step["says"])
    lines.extend(
        [
            "",
            "## The numbers",
            "",
            "| quantity | value |",
            "|---|---|",
            "| token budget | " + str(ground["substrate"]["budget_tokens"]) + " |",
            "| constant-weight template optimum, whole split | " + repr(calibration["default_simplex_optimum"]["full"]) + " |",
            "| constant-weight template optimum, fold_a | " + repr(calibration["default_simplex_optimum"]["fold_a"]) + " |",
            "| constant-weight template optimum, fold_b | " + repr(calibration["default_simplex_optimum"]["fold_b"]) + " |",
            "| reference optimum, whole split | " + repr(calibration["reference_optimum"]["full"]) + " |",
            "| reference plan, whole split | " + repr(expected["loss_full"]) + " |",
            "| reference plan, fold_a | " + repr(expected["loss_fold_a"]) + " |",
            "| reference plan, fold_b | " + repr(expected["loss_fold_b"]) + " |",
            "| separation | " + repr(calibration["separation_nats"]) + " nats |",
            "| reference reward | " + repr(expected["reward"]) + " |",
            "",
            "## Anchors",
            "",
            "`anchors_state: " + ground["reward"]["anchors_state"] + "`. baseline_metric and "
            "target_metric are absent under `" + ground["reward"]["anchors_gap"] + "`. The two "
            "calibration points above are substrate-local, are recomputed by the verifier from "
            "frozen bytes on every run, and are carried under `" + ground["reward"]["calibration_gap"]
            + "`. They are not the family anchors and are not comparable across slots.",
            "",
            "## Statement ambiguity",
            "",
            "Reading one: " + ground["statement_ambiguity"]["reading_one"].strip(),
            "",
            "Reading two: " + ground["statement_ambiguity"]["reading_two"].strip(),
            "",
            "Reduces to: " + ground["statement_ambiguity"]["reduces_to"].strip(),
            "",
            "## Tier",
            "",
            "Target tier " + ground["tier"]["target_tier"] + ", tier_exemption_granted "
            + ground["tier"]["tier_exemption_granted"] + ", anchorable tier "
            + ground["tier"]["anchorable_tier"] + ".",
            "",
            ground["tier"]["perception_axis_exemption"].strip(),
            "",
            ground["tier"]["no_difficulty_claim"].strip(),
            "",
            "## Gaps",
            "",
        ]
    )
    for gap in ground["gaps"]:
        lines.append("- `" + gap["id"] + "` (" + gap["state"] + "): " + gap["effect"].strip())
    lines.append("")
    return "\n".join(lines)


def rubrics_json(ground: dict) -> str:
    rows = [{"id": row["id"], "rubric": row["rubric"]} for row in ground["rubric_rows"]]
    return _json(
        _stamp(
            {
                "slot": ground["slot"],
                "judged": "the solution against its reference answer",
                "not_judged": "the trajectory; tests/rubrics.jsonl does that and never substitutes for this file",
                "rubrics": rows,
            }
        )
    )


TEST_ROWS = (
    ("bound_evaluation_point_reached", "check_bound_evaluation_point_reached"),
    ("token_budget_respected_as_fed", "check_token_budget_respected_as_fed"),
    ("evaluation_split_untrained", "check_evaluation_split_untrained"),
    ("pool_state_matches_graded_pool", "check_pool_state_matches_graded_pool"),
    ("graded_weights_are_harness_owned", "check_graded_weights_are_harness_owned"),
    ("graded_loss_recomputed_unsmoothed", "check_graded_loss_recomputed_unsmoothed"),
    ("feed_precedes_every_graded_evaluation", "check_feed_precedes_every_graded_evaluation"),
    ("loss_sustained_across_verifier_folds", "check_loss_sustained_across_verifier_folds"),
    ("mixture_beats_default_simplex_optimum", "check_mixture_beats_default_simplex_optimum"),
)


def test_output_py(ground: dict) -> str:
    lines = [
        "#!/usr/bin/env python3",
        '"""' + BANNER,
        "",
        "Source: " + SOURCE,
        "",
        "The compiled surface of tests/checkers.yaml: one test per graded row, each",
        "re-asserting that row against the same live harness handles tests/runner.py",
        "produced. It never enters the reward; tests/grade.py owns that number.",
        '"""',
        "from __future__ import annotations",
        "",
        "import argparse",
        "import sys",
        "from pathlib import Path",
        "",
        "HERE = Path(__file__).resolve().parent",
        "if str(HERE) not in sys.path:",
        "    sys.path.insert(0, str(HERE))",
        "",
        "import checkers  # noqa: E402",
        "import runner  # noqa: E402",
        "",
        "EVIDENCE = Path(__file__).resolve().parent",
        "",
        "",
        "def _handles():",
        "    return runner.handles(EVIDENCE)",
        "",
    ]
    for ident, selector in TEST_ROWS:
        lines.extend(
            [
                "",
                "def test_" + ident + "():",
                '    outcome = checkers.' + selector + "(_handles())",
                "    assert outcome.ok, outcome.reason",
                "",
            ]
        )
    lines.extend(
        [
            "",
            "ROWS = (",
        ]
    )
    for ident, selector in TEST_ROWS:
        lines.append('    ("' + ident + '", test_' + ident + "),")
    lines.extend(
        [
            ")",
            "",
            "",
            "def main() -> int:",
            "    parser = argparse.ArgumentParser()",
            '    parser.add_argument("--evidence", required=True)',
            "    args = parser.parse_args()",
            "    global EVIDENCE",
            "    EVIDENCE = Path(args.evidence)",
            "    failed = []",
            "    for ident, row in ROWS:",
            "        try:",
            "            row()",
            "        except AssertionError as problem:",
            "            failed.append(ident + ': ' + str(problem))",
            "    for line in failed:",
            "        print(line)",
            "    print(str(len(ROWS) - len(failed)) + ' of ' + str(len(ROWS)) + ' compiled rows passed')",
            "    return 1 if failed else 0",
            "",
            "",
            'if __name__ == "__main__":',
            "    raise SystemExit(main())",
            "",
        ]
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------


def artifacts(ground: dict) -> dict:
    out = {
        "solution/trajectory.golden.json": trajectory(ground),
        "solution/solve.sh": solve_sh(ground),
        "solution/TRUTH.md": truth_md(ground),
        "solution/rubrics.json": rubrics_json(ground),
        "tests/test_output.py": test_output_py(ground),
        "solution/fixtures/manifest.json": _json(
            _stamp({"slot": ground["slot"], "fixtures": fixture_rows(ground)})
        ),
        "solution/fixtures/overrides.json": _json(_stamp(override_files(ground))),
    }
    for name, payload in plan_files(ground).items():
        out["solution/fixtures/" + name] = _json(_stamp(payload))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    ground = load()
    produced = artifacts(ground)
    drifted = []
    for relative, text in sorted(produced.items()):
        target = BUNDLE / relative
        if args.check:
            current = target.read_text(encoding="utf-8") if target.is_file() else None
            if current != text:
                drifted.append(relative)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        if relative.endswith(".sh"):
            target.chmod(0o755)
    if args.check:
        for relative in drifted:
            print("DRIFTED: " + relative)
        print(str(len(produced) - len(drifted)) + " of " + str(len(produced)) + " generated artifacts are byte-identical")
        return 1 if drifted else 0
    print("regenerated " + str(len(produced)) + " artifacts from " + SOURCE)
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
