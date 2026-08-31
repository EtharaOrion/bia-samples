# FORGE-CANARY-BEGIN
# 0: d04d795d02651642af923ace32af4fcacd4e7c20ea05e47ae41c16df3a06b938
# 1: ae4f5df69d663d250eefaedd6b74fc2ddf866a06cb7d1fc41ca53697c73430fd
# 2: 7c8f8b31d2adda127a77e84e0f579daf02301d6a452e1210efa0ba623d7e236c
# 3: 8d919a97fec24f4fa6616f39a57511aec78553c24cfc7531eb9b4404327d4e25
# FORGE-CANARY-END
#!/usr/bin/env python3
"""Derive every generated artifact in the OER-09 bundle from solution/grounding.yaml alone.

No model, no network, no clock, no locale and no random source is consulted. Every mapping is
serialised with sorted keys and every list order comes from the source file, so running this
twice over frozen bytes produces byte-identical output.

Generates:
    solution/solve.sh
    solution/TRUTH.md
    solution/rubrics.json
    solution/golden_trajectory.json
    solution/fixtures.json
    tests/test_output.py
"""
from __future__ import annotations

import copy
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
    return yaml.safe_load(GROUNDING.read_text(encoding="utf-8"))


def hash_banner(prefix: str) -> str:
    return prefix + " " + BANNER + " Source: " + SOURCE


def apply_patch(state: dict, patch) -> dict:
    """Apply a list of {path, value} onto a deep copy. Integer segments index lists."""
    out = copy.deepcopy(state)
    for entry in patch or []:
        node = out
        parts = str(entry["path"]).split(".")
        for part in parts[:-1]:
            node = node[int(part)] if isinstance(node, list) else node[part]
        last = parts[-1]
        if isinstance(node, list):
            node[int(last)] = entry["value"]
        else:
            node[last] = entry["value"]
    return out


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def gen_fixtures(source: dict) -> str:
    base = source["fixture_base"]
    document = {
        "banner": BANNER,
        "source": SOURCE,
        "reference_binding": source["reference_binding"],
        "base": base,
        "accepting": {
            "id": source["accepting"]["id"],
            "state": base,
            "expected_reward": source["accepting"]["expected_reward"],
            "expected_reason": source["accepting"]["expected_reason"],
        },
        "continuity": {
            "id": source["continuity"]["id"],
            "state": apply_patch(base, source["continuity"]["patch"]),
            "expected_reward": source["continuity"]["expected_reward"],
            "expected_reason": source["continuity"]["expected_reason"],
        },
        "negative_controls": [
            {
                "id": row["id"],
                "description": row["description"],
                "checker": row["checker"],
                "expected_reason": row["expected_reason"],
                "expected_reward": 0.0,
                "state": apply_patch(base, row["patch"]),
            }
            for row in source["negative_controls"]
        ],
    }
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def gen_golden(source: dict) -> str:
    document = {
        "banner": BANNER,
        "source": SOURCE,
        "slot": source["slot"],
        "reference_identity": source["reference_binding"]["identity"],
        "steps": source["golden_trajectory"],
        "terminal_state": source["fixture_base"],
        "terminal_reward": source["accepting"]["expected_reward"],
    }
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def gen_rubrics(source: dict) -> str:
    document = {
        "banner": BANNER,
        "source": SOURCE,
        "judged": "the solution against its reference answer",
        "not_judged": "the trajectory; tests/rubrics.jsonl carries the trajectory-judged rubrics",
        "reference": source["reference_binding"]["identity"],
        "criteria": source["solution_rubrics"],
        "total_weight": sum(int(row["weight"]) for row in source["solution_rubrics"]),
    }
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def gen_solve(source: dict) -> str:
    chain = json.dumps({"stages": source["reference_chain"]}, indent=2, sort_keys=True)
    lines = [
        "#!/usr/bin/env bash",
        "# " + BANNER + " Source: " + SOURCE,
        "# OER-09 reference entry point.",
        "set -euo pipefail",
        "",
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        "# tests/runner.py copies this entry point ALONE into a fresh sandbox, so its own",
        "# location does not resolve the bundle. The runner hands the bundle root over on its",
        "# environment allowlist as OER09_BUNDLE; the fallback is for running this in place.",
        'BUNDLE="${OER09_BUNDLE:-$(dirname "${HERE}")}"',
        "",
        "mkdir -p work",
        "",
        "# The filter chain names the field the REGISTER carries, not the one REGISTER.md documents.",
        "cat > work/filters.json <<'FILTERS'",
        chain,
        "FILTERS",
        "",
        'python3 "${BUNDLE}/environment/curate.py" \\',
        '  --filters work/filters.json \\',
        '  --out work/curated_pool.jsonl \\',
        '  --report work/curation_report.json \\',
        "  --explain",
        "",
        "# Exit status 0 established that the chain did not raise. It established nothing about",
        "# the pool. The harness observation is the thing that does, and it is read before a",
        "# single training token is spent.",
        'python3 "${BUNDLE}/environment/pool_probe.py"',
        'python3 "${BUNDLE}/solution/reference.py"',
        "",
        'echo "OER-09 reference: curated pool verified against the harness observation, then trained"',
    ]
    return "\n".join(lines) + "\n"


def gen_truth(source: dict) -> str:
    truth = source["truth"]
    pool = source["pool"]
    rows = "\n".join(
        "| " + str(step["step"]) + " | " + step["action"] + " | " + step["establishes"] + " |"
        for step in source["golden_trajectory"]
    )
    stages = ", ".join(str(n) for n in pool["stage_counts"])
    lines = [
        "<!-- " + BANNER + " Source: " + SOURCE + " -->",
        "",
        "# TRUTH, OER-09",
        "",
        "Family " + source["family"] + ", " + source["family_title"] + ". Archetype "
        + source["primary_archetype"] + ", " + source["primary_archetype_title"] + ".",
        "",
        "## The graded quantity",
        "",
        truth["graded_quantity"],
        "",
        "## The trap",
        "",
        truth["the_trap"],
        "",
        "## The tell",
        "",
        truth["the_tell"],
        "",
        "## The skill under test",
        "",
        truth["the_skill"],
        "",
        "## The register drift",
        "",
        "| | documented | observed |",
        "|---|---|---|",
        "| schema | " + pool["documented_schema"] + " | " + pool["observed_schema"] + " |",
        "| quality field | `" + pool["documented_quality_field"] + "` | `"
        + pool["observed_quality_field"] + "` |",
        "",
        "## The reference chain",
        "",
        "Stage counts: " + stages + ".",
        "",
        "Source pool " + str(pool["source_documents"]) + " documents at `" + pool["source_digest"] + "`.",
        "",
        "Curated pool " + str(pool["curated_documents"]) + " documents at `" + pool["curated_digest"]
        + "`, " + str(pool["curated_tokens"]) + " tokens.",
        "",
        "## The golden trajectory",
        "",
        "| step | action | establishes |",
        "|---|---|---|",
        rows,
        "",
        "## Anchors",
        "",
        "`anchors_state: " + source["anchors"]["anchors_state"] + "` under `"
        + source["anchors"]["gap"] + "`. " + source["anchors"]["authority"],
        "",
        "Bound formula: `" + source["anchors"]["formula_bound"] + "`.",
        "",
        "Operated normalisation: `" + source["anchors"]["formula_operated"] + "`.",
        "",
        source["anchors"]["substitute_note"].strip(),
        "",
        "## Solver egress",
        "",
        "Bound here: `" + source["solver_egress_divergence"]["bound_here"] + "`. Contract field `"
        + source["solver_egress_divergence"]["contract_field"] + "` reads `"
        + source["solver_egress_divergence"]["contract_value"] + "` under `"
        + source["solver_egress_divergence"]["gap"] + "`.",
        "",
        source["solver_egress_divergence"]["deny_list_is_not_containment"].strip(),
        "",
        "## Envelope",
        "",
        "`reference_hours: " + str(source["envelope"]["reference_hours"]) + "` against `budget_hours: "
        + str(source["envelope"]["budget_hours"]) + "`.",
        "",
        "## Tier",
        "",
        "Target tier " + source["tier"]["target_tier"] + ", `tier_exemption_granted: "
        + source["tier"]["tier_exemption_granted"] + "`. Anchorable tier "
        + source["tier"]["anchorable_tier"] + ".",
        "",
        source["tier"]["perception_axis_exemption"].strip(),
    ]
    return "\n".join(lines) + "\n"


def gen_test_output(source: dict) -> str:
    controls = source["negative_controls"]
    by_checker: dict = {}
    for row in controls:
        by_checker.setdefault(row["checker"], []).append(row)

    order = [
        "filter_claim_matches_observed_pool",
        "curated_pool_consumed_by_trainer",
        "curation_precedes_first_feed",
        "token_budget_respected_as_fed",
        "eval_split_never_trained_on",
        "graded_weights_harness_owned",
        "graded_loss_unsmoothed",
        "improvement_sustained_across_points",
        "early_stop_does_not_establish_loss",
    ]

    head = [
        "#!/usr/bin/env python3",
        '"""' + BANNER + " Source: " + SOURCE + "",
        "",
        "Both halves of every OER-09 checker, compiled from the fixtures grounding.yaml declares.",
        "The accepting half runs the clean fixture through the real checker. The rejecting half",
        "runs the planted fixture and asserts the exact zero_reason that checker owns.",
        '"""',
        "from __future__ import annotations",
        "",
        "import json",
        "import sys",
        "from pathlib import Path",
        "",
        "HERE = Path(__file__).resolve().parent",
        "BUNDLE = HERE.parent",
        "sys.path.insert(0, str(HERE))",
        "",
        "import checkers  # noqa: E402",
        "import grade  # noqa: E402",
        "",
        "FIXTURES = json.loads((BUNDLE / 'solution' / 'fixtures.json').read_text(encoding='utf-8'))",
        "BASE = FIXTURES['base']",
        "CONTROLS = {row['id']: row for row in FIXTURES['negative_controls']}",
        "",
        "",
        "def _verdict(state, ident):",
        "    for row in checkers.run_all(state):",
        "        if row.ident == ident:",
            "            return row",
        "    raise AssertionError('no verdict for ' + ident)",
        "",
        "",
        "def test_reference_accepted_end_to_end():",
        "    document = grade.evaluate(FIXTURES['accepting']['state'])",
        "    assert document['reward'] == FIXTURES['accepting']['expected_reward'], document",
        "    assert document['reason'] == FIXTURES['accepting']['expected_reason'], document",
        "",
        "",
        "def test_reward_is_not_binary():",
        "    document = grade.evaluate(FIXTURES['continuity']['state'])",
        "    assert 0.0 < document['reward'] < 1.0, document",
        "    assert document['reward'] == FIXTURES['continuity']['expected_reward'], document",
        "",
        "",
    ]

    body = []
    for ident in order:
        rows = by_checker.get(ident, [])
        body.append("def test_" + ident + "():")
        body.append("    accepted = _verdict(BASE, '" + ident + "')")
        body.append("    assert accepted.ok, accepted.detail")
        for row in rows:
            body.append("    planted = _verdict(CONTROLS['" + row["id"] + "']['state'], '" + ident + "')")
            body.append("    assert not planted.ok, '" + row["id"] + " did not fire " + ident + "'")
            body.append(
                "    assert planted.reason == '" + row["expected_reason"] + "', planted.reason"
            )
            body.append(
                "    document = grade.evaluate(CONTROLS['" + row["id"] + "']['state'])"
            )
            body.append("    assert document['reward'] == 0.0, document")
            body.append(
                "    assert document['reason'] == '" + row["expected_reason"] + "', document"
            )
        body.append("")
        body.append("")

    tail = [
        "TESTS = [",
        "    'test_reference_accepted_end_to_end',",
        "    'test_reward_is_not_binary',",
    ]
    tail.extend("    'test_" + ident + "'," for ident in order)
    tail.extend(
        [
            "]",
            "",
            "",
            "def main() -> int:",
            "    passed = 0",
            "    for name in TESTS:",
            "        globals()[name]()",
            "        passed += 1",
            "    # A positive count, never a zero one. An empty parse is ambiguous between a suite",
            "    # that ran clean and a suite that never ran, so the count is asserted positive.",
            "    assert passed > 0, 'no compiled test ran'",
            "    print('compiled tests passed: ' + str(passed))",
            "    return 0",
            "",
            "",
            "if __name__ == '__main__':",
            "    sys.exit(main())",
        ]
    )
    return "\n".join(head + body + tail) + "\n"


def verify_measured(source: dict) -> list:
    """Recompute the fixture_base evaluation against tests/harness.py and report disagreement.

    `fixture_base` carries the graded state the whole fixture set is built on. Its evaluation
    numbers are MEASURED quantities, so leaving them as literals nobody re-derives is exactly
    the authored ground truth invariant 22 forbids. This re-runs the verifier's own measurement
    over the reference curation and refuses to regenerate anything if the two disagree, so a
    change to the substrate cannot silently leave a stale golden trajectory behind.
    """
    import sys as _sys

    tests = BUNDLE / "tests"
    if str(tests) not in _sys.path:
        _sys.path.insert(0, str(tests))
    import harness

    config = harness.load_config(BUNDLE / "environment" / "frozen_config.yaml")
    register = harness.load_register(BUNDLE / "environment" / "pool" / "source_register.jsonl")
    curated = harness.reference_pool(register, harness.REFERENCE_CHAIN, config["substrate"])

    bound = int(config["evaluation"]["bound_point"])
    substrate = config["substrate"]
    measured = {
        "control_loss": harness.loss_at(
            harness.effective_tokens(register, harness.consumed_at(bound, config), substrate), substrate
        ),
        "reference_loss": harness.loss_at(
            harness.effective_tokens(curated, harness.consumed_at(bound, config), substrate), substrate
        ),
    }
    points = {}
    for point in [bound] + [int(value) for value in config["evaluation"]["sustain_points"]]:
        effective = harness.effective_tokens(curated, harness.consumed_at(point, config), substrate)
        points[point] = {
            "loss": harness.loss_at(effective, substrate),
            "weights_digest": harness.checkpoint_digest(point, effective, substrate),
        }

    evaluation = source["fixture_base"]["evaluation"]
    ledger = source["fixture_base"]["weights"]["ledger"]
    drift = []
    for name, value in measured.items():
        if evaluation.get(name) != value:
            drift.append("fixture_base.evaluation." + name + ": " + repr(evaluation.get(name))
                         + " but the harness measures " + repr(value))
    if evaluation.get("graded_loss") != measured["reference_loss"]:
        drift.append("fixture_base.evaluation.graded_loss: " + repr(evaluation.get("graded_loss"))
                     + " but the reference curation measures " + repr(measured["reference_loss"]))
    for row in evaluation.get("points") or []:
        expected = points.get(int(row["point"]))
        if expected is None:
            drift.append("fixture_base names point " + repr(row["point"]) + ", which is not scheduled")
            continue
        if row.get("loss") != expected["loss"]:
            drift.append("point " + str(row["point"]) + " loss: " + repr(row.get("loss"))
                         + " but the harness measures " + repr(expected["loss"]))
        if row.get("weights_digest") != expected["weights_digest"]:
            drift.append("point " + str(row["point"]) + " weights_digest is not the harness checkpoint")
        if ledger.get(str(row["point"])) != expected["weights_digest"]:
            drift.append("weights.ledger[" + repr(str(row["point"])) + "] is not the harness checkpoint")
    return drift


def main() -> int:
    source = load()
    drift = verify_measured(source)
    if drift:
        print("REFUSING to regenerate: fixture_base disagrees with the measured harness state")
        for line in drift:
            print("  " + line)
        return 3
    artifacts = {
        BUNDLE / "solution" / "fixtures.json": gen_fixtures(source),
        BUNDLE / "solution" / "golden_trajectory.json": gen_golden(source),
        BUNDLE / "solution" / "rubrics.json": gen_rubrics(source),
        BUNDLE / "solution" / "solve.sh": gen_solve(source),
        BUNDLE / "solution" / "TRUTH.md": gen_truth(source),
        BUNDLE / "tests" / "test_output.py": gen_test_output(source),
    }
    for path, text in artifacts.items():
        write(path, text)
    (BUNDLE / "solution" / "solve.sh").chmod(0o755)
    for path in sorted(artifacts):
        print("wrote " + path.relative_to(BUNDLE).as_posix())
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
