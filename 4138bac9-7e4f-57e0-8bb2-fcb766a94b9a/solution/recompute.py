# FORGE-CANARY-BEGIN
# 0: dd99e7c53f40e180f4343172906761d15ff23bb4e32bcb08f80b969f010db4d7
# 1: 6a07c5f28814591a10824f98da79dfc94b45ceca675f4dec37b0b9ced24504ee
# 2: a3e0d6f0540bf931849b8877d5f1c90bd959cb182b019fbd6743d61b8d5eae20
# 3: 16e809c567ea6222f40b6d15369541613b0e00541f8331fa847034694cb01310
# FORGE-CANARY-END
#!/usr/bin/env python3
"""Derive every generated artifact of this slot from solution/grounding.yaml alone.

Generated here, and nowhere else:

    solution/solve.sh          the entry point that installs the reference generator
    solution/TRUTH.md          what is graded, what is not, and what is known to be open
    solution/rubrics.json      the solution graded against its reference answer
    solution/trajectory.json   the golden trajectory
    tests/fixtures/records.json  the golden record, one single-defect fixture per checker,
                                 and one stale control per silent mutation
    tests/test_output.py       the compiled tests, one per checker, both halves

No model, no network, no clock, no locale and no random source is reachable from this
module: it reads one YAML file, does arithmetic over strings and dictionaries, and
writes bytes. Two runs over frozen bytes therefore produce byte-identical output, and
`--check` regenerates in memory and exits non-zero if any committed artifact drifted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
GROUNDING = HERE / "grounding.yaml"
BANNER = "GENERATED SECTION. DO NOT HAND-EDIT."
SOURCE = "solution/grounding.yaml"
SCHEMA = "forge.oer17.run/v1"


def banner_line(comment: str) -> str:
    return comment + " " + BANNER + " Source: " + SOURCE


def load_grounding() -> dict:
    with GROUNDING.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def digest_vector(vector: dict) -> str:
    rows = [[str(key), format(float(vector[key]), ".12g")] for key in sorted(vector)]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def seeded_digest(seed: str) -> str:
    return hashlib.sha256(str(seed).encode("utf-8")).hexdigest()


# --- the golden record and its single-defect fixtures --------------------------


def golden_record(ground: dict) -> dict:
    block = ground["golden_record"]
    bound = ground["bound_values"]
    weights = dict(block["weights"])
    digest = digest_vector(weights)
    tree = seeded_digest(block["grading_tree_seed"])
    events = [{"index": i, "name": name, "detail": ""} for i, name in enumerate(block["events"])]
    opened = next(row["index"] for row in events if row["name"] == "benchmark-loaded")
    steps = list(block["eval_points"])
    scores = list(block["raw_scores"])
    training = dict(block["training"])
    return {
        "schema": SCHEMA,
        "events": events,
        "generation": {
            "exit_code": 0,
            "timed_out": False,
            "workspace_after": list(block["workspace_after"]),
            "reported_score": block["reported_score"],
            "reported_smoothing": None,
        },
        "samples": [dict(row) for row in block["samples"]],
        "benchmark": {
            "items": [dict(row) for row in block["benchmark_items"]],
            "item_count": len(block["benchmark_items"]),
            "revision": bound["benchmark_revision"],
            "opened_at_event_index": opened,
        },
        "training": {
            "optimizer": bound["optimizer_id"],
            "learning_rate": bound["learning_rate"],
            "steps_fed": training["steps_fed"],
            "tokens_fed": training["tokens_fed"],
            "corpus_samples": training["corpus_samples"],
            "usable_samples": training["usable_samples"],
            "halted_at_step": training["halted_at_step"],
            "completed": training["completed"],
            "checkpoint_selected_by": training["checkpoint_selected_by"],
            "weights_by_step": {str(step): digest for step in steps},
        },
        "eval_points": [
            {"step": step, "raw_score": score, "smoothed_score": None, "weights_sha256": digest}
            for step, score in zip(steps, scores)
        ],
        "measurement": {
            "score": block["measurement_score"],
            "weights": weights,
            "computed_by": "tests/harness.py::evaluate",
        },
        "graded": {
            "eval_point_index": bound["graded_eval_point_index"],
            "step": steps[bound["graded_eval_point_index"]],
            "score": block["graded_score"],
            "smoothing": "none",
            "source": "verifier-measurement",
            "weights_sha256": digest,
        },
        "reported": {
            "score": block["reported_score"],
            "smoothing": None,
            "note": "recorded for divergence only; never graded",
        },
        "grading_tree": {"digest_before": tree, "digest_after": tree, "file_count": 9},
    }


def with_raw_scores(record: dict, scores: list) -> dict:
    out = json.loads(json.dumps(record))
    for point, score in zip(out["eval_points"], scores):
        point["raw_score"] = score
    return out


def apply_patch(record: dict, patch: dict, ground: dict) -> dict:
    """Apply one single-defect patch. Every key handled here is named in grounding.yaml."""
    out = json.loads(json.dumps(record))
    for key in sorted(patch):
        value = patch[key]
        if key == "training.steps_fed":
            out["training"]["steps_fed"] = value
        elif key == "training.completed":
            out["training"]["completed"] = value
        elif key == "training.halted_at_step":
            out["training"]["halted_at_step"] = value
        elif key == "graded.smoothing":
            out["graded"]["smoothing"] = value
        elif key == "graded.source":
            out["graded"]["source"] = value
        elif key == "graded.weights_sha256":
            out["graded"]["weights_sha256"] = seeded_digest(ground["golden_record"]["foreign_weights_seed"])
        elif key == "grading_tree.digest_after":
            out["grading_tree"]["digest_after"] = seeded_digest(ground["golden_record"]["mutated_tree_seed"])
        elif key == "raw_scores":
            out = with_raw_scores(out, list(value))
        elif key == "events.swap":
            names = [row["name"] for row in out["events"]]
            first, second = names.index(value[0]), names.index(value[1])
            out["events"][first]["name"], out["events"][second]["name"] = value[1], value[0]
            for index, row in enumerate(out["events"]):
                row["index"] = index
            out["benchmark"]["opened_at_event_index"] = next(
                row["index"] for row in out["events"] if row["name"] == "benchmark-loaded"
            )
        elif key == "samples.append":
            out["samples"].append(dict(value))
        else:
            raise KeyError("unhandled fixture patch key: " + key)
    return out


def build_records(ground: dict) -> dict:
    golden = golden_record(ground)
    fixtures = {}
    for row in ground["fixtures"]:
        fixtures[row["id"]] = {
            "fires": row["fires"],
            "reason": row["reason"],
            "record": apply_patch(golden, row["patch"], ground),
        }
    controls = {}
    for row in ground["stale_controls"]:
        record = json.loads(json.dumps(golden))
        if "sample" in row:
            record["samples"].append(dict(row["sample"]))
        if "raw_scores" in row:
            record = with_raw_scores(record, list(row["raw_scores"]))
        if "steps_fed" in row:
            record["training"]["steps_fed"] = row["steps_fed"]
        controls[row["id"]] = {
            "mutation": row["mutation"],
            "checker": row["checker"],
            "stale": dict(row["stale"]),
            "live": dict(row["live"]),
            "record": record,
        }
    return {
        "_generated": BANNER,
        "_source": SOURCE,
        "schema": "forge.oer17.fixtures/v1",
        "bound_values": dict(ground["bound_values"]),
        "golden": golden,
        "fixtures": fixtures,
        "stale_controls": controls,
    }


# --- the generated text artifacts ---------------------------------------------


def build_solve_sh(ground: dict) -> str:
    reference = ground["reference"]
    lines = [
        "#!/usr/bin/env bash",
        banner_line("#"),
        "# The reference solution's entry point. It installs the reference generator at the",
        "# submission path and does nothing else: the harness runs the generator itself, in",
        "# isolation, and measures the held-out score in the verifier's own process.",
        "set -euo pipefail",
        "",
        'SOLUTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'SUBMISSION="${OER17_SUBMISSION:-' + reference["submission_path"] + '}"',
        "",
        'mkdir -p "$(dirname "${SUBMISSION}")"',
        'cp "${SOLUTION_DIR}/reference.py" "${SUBMISSION}"',
        "",
        "# The reference bytes are sha256-bound, so the accepting half of every checker is a",
        "# statement about THIS reference and not about some file that happened to be here.",
        'EXPECTED="' + reference["sha256"] + '"',
        'ACTUAL="$(sha256sum "${SUBMISSION}" | cut -d" " -f1)"',
        'if [ "${EXPECTED}" != "${ACTUAL}" ]; then',
        '  echo "reference digest mismatch: expected ${EXPECTED}, got ${ACTUAL}" >&2',
        "  exit 1",
        "fi",
        "",
        'echo "installed reference generator at ${SUBMISSION}"',
        "",
    ]
    return "\n".join(lines)


def build_truth(ground: dict) -> str:
    truth = ground["truth"]
    observed = ground["observed"]
    out = [
        "<!-- " + BANNER + " Source: " + SOURCE + " -->",
        "",
        "# TRUTH.md, slot " + ground["slot"],
        "",
        "## What is graded",
        "",
        truth["what_is_graded"].strip(),
        "",
        "## What is not graded",
        "",
        truth["what_is_not_graded"].strip(),
        "",
        "## Why the screen is the load-bearing control",
        "",
        truth["why_the_screen_is_load_bearing"].strip(),
        "",
        "## The reference, as measured on this host",
        "",
        "| quantity | observed |",
        "|---|---|",
    ]
    for key in sorted(observed):
        if key == "note":
            continue
        out.append("| " + key + " | " + json.dumps(observed[key]) + " |")
    out += [
        "",
        observed["note"].strip(),
        "",
        "## Anchors",
        "",
        "anchors_state: " + str(ground["anchors"]["anchors_state"]) + ", gap " + str(ground["anchors"]["gap"]) + ".",
        ground["anchors"]["never_invented"].strip(),
        "",
        "## Known limits",
        "",
    ]
    for row in truth["known_limits"]:
        out.append("- " + str(row).strip())
    out.append("")
    return "\n".join(out)


def build_rubrics(ground: dict) -> str:
    payload = {
        "_generated": BANNER,
        "_source": SOURCE,
        "slot": ground["slot"],
        "graded_against": "solution/reference.py",
        "rubrics": [
            {
                "id": row["id"],
                "criterion": row["criterion"].strip(),
                "reference_answer": row["reference_answer"].strip(),
            }
            for row in ground["rubrics_solution"]
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def build_trajectory(ground: dict) -> str:
    payload = {
        "_generated": BANNER,
        "_source": SOURCE,
        "slot": ground["slot"],
        "steps": [
            {"step": row["step"], "action": row["action"].strip(), "observation": row["observation"].strip()}
            for row in ground["trajectory"]
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def build_test_output(ground: dict) -> str:
    fixtures = {row["fires"]: row for row in ground["fixtures"]}
    order = [
        "heldout_contamination_absent",
        "benchmark_unread_on_generation_path",
        "grading_tree_immutable",
        "training_budget_respected",
        "graded_score_unsmoothed",
        "score_sustained_across_eval_points",
        "early_stop_not_a_score",
        "graded_score_from_harness_state",
        "evaluated_weights_harness_owned",
    ]
    lines = [
        "#!/usr/bin/env python3",
        '"""' + BANNER + " Source: " + SOURCE + "",
        "",
        "The compiled tests. One per declared checker, each exercising BOTH halves: the",
        "golden record must be accepted, and a single-defect fixture must be rejected with",
        "exactly that checker's zero reason. The stale controls prove drift causality: the",
        "same fixture reads clean under a pre-mutation bound value and dirty under the live",
        "one, so a verdict carried across a silent mutation is caught.",
        "",
        "Runs standalone (`python3 tests/test_output.py`) and under pytest.",
        '"""',
        "from __future__ import annotations",
        "",
        "import json",
        "import sys",
        "from pathlib import Path",
        "",
        "HERE = Path(__file__).resolve().parent",
        "sys.path.insert(0, str(HERE))",
        "",
        "import checkers  # noqa: E402",
        "",
        'FIXTURES = json.loads((HERE / "fixtures" / "records.json").read_text(encoding="utf-8"))',
        'BOUND = FIXTURES["bound_values"]',
        'GOLDEN = FIXTURES["golden"]',
        "",
        "",
        "def _bound(**overrides):",
        "    merged = dict(BOUND)",
        "    merged.update(overrides)",
        "    return merged",
        "",
        "",
    ]
    for ident in order:
        row = fixtures[ident]
        lines += [
            "def test_" + ident + "():",
            "    accepted = checkers.selector(" + json.dumps(ident) + ")(GOLDEN, BOUND)",
            '    assert accepted.ok, "golden record rejected: " + accepted.reason + " " + accepted.detail',
            '    fixture = FIXTURES["fixtures"][' + json.dumps(row["id"]) + "]",
            "    rejected = checkers.selector(" + json.dumps(ident) + ')(fixture["record"], BOUND)',
            '    assert not rejected.ok, "planted defect ' + row["id"] + ' was accepted"',
            "    assert rejected.reason == " + json.dumps(row["reason"]) + ", rejected.reason",
            "",
            "",
        ]
    for row in ground["stale_controls"]:
        ident = row["id"].replace("-", "_")
        lines += [
            "def test_" + ident + "():",
            '    control = FIXTURES["stale_controls"][' + json.dumps(row["id"]) + "]",
            '    checker = checkers.selector(control["checker"])',
            '    stale = checker(control["record"], _bound(**control["stale"]))',
            '    live = checker(control["record"], _bound(**control["live"]))',
            '    assert stale.ok, "the pre-mutation verdict should read clean: " + stale.reason',
            '    assert not live.ok, "the post-mutation verdict should read dirty"',
            "",
            "",
        ]
    lines += [
        "def main() -> int:",
        "    tests = sorted(name for name in globals() if name.startswith(\"test_\"))",
        "    collected = len(tests)",
        "    failed = []",
        "    for name in tests:",
        "        try:",
        "            globals()[name]()",
        "        except AssertionError as error:",
        "            failed.append(name + \": \" + str(error))",
        "    passed = collected - len(failed)",
        "    print(json.dumps({\"collected\": collected, \"passed\": passed, \"failed\": len(failed)}, sort_keys=True))",
        "    for row in failed:",
        "        print(row, file=sys.stderr)",
        "    if collected == 0 or failed:",
        "        return 1",
        "    return 0",
        "",
        "",
        'if __name__ == "__main__":',
        "    raise SystemExit(main())",
        "",
    ]
    return "\n".join(lines)


ARTIFACTS = (
    ("solution/solve.sh", build_solve_sh, 0o755),
    ("solution/TRUTH.md", build_truth, 0o644),
    ("solution/rubrics.json", build_rubrics, 0o644),
    ("solution/trajectory.json", build_trajectory, 0o644),
    ("tests/test_output.py", build_test_output, 0o755),
)


def render(ground: dict) -> dict:
    out = {relative: builder(ground) for relative, builder, _ in ARTIFACTS}
    out["tests/fixtures/records.json"] = json.dumps(build_records(ground), indent=2, sort_keys=True) + "\n"
    return out


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description="derive every generated artifact from grounding.yaml")
    parser.add_argument("--check", action="store_true", help="regenerate in memory and report drift")
    args = parser.parse_args(argv[1:])
    ground = load_grounding()
    rendered = render(ground)
    modes = {relative: mode for relative, _, mode in ARTIFACTS}

    drift = []
    for relative in sorted(rendered):
        path = BUNDLE / relative
        text = rendered[relative]
        if args.check:
            current = path.read_text(encoding="utf-8") if path.is_file() else None
            if current != text:
                drift.append(relative)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        if relative in modes:
            path.chmod(modes[relative])

    if args.check:
        print(json.dumps({"checked": len(rendered), "drifted": sorted(drift)}, sort_keys=True))
        return 1 if drift else 0
    print(json.dumps({"generated": sorted(rendered)}, sort_keys=True))
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
    raise SystemExit(main(sys.argv))
