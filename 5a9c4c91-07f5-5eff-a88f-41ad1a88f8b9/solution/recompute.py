# FORGE-CANARY-BEGIN
# 0: 8f6edf50f0b0a79f3b5cccbb48cbcda8533285b656bc7d2b940812c052c73b1a
# 1: 0cfc283f4972a61aa4e72f2264f5d887de63b5d7b2a74004eba4728fdb20956b
# 2: 193e977a81b54ab343415a8d134cca4bc12a55a877a75279436ef77af8884291
# 3: 41d173135e14e5dc5fbfa167a0ba109ccb5e635b29fb4ec69ebb002aab317eac
# FORGE-CANARY-END
#!/usr/bin/env python3
"""Derive every generated artifact from solution/grounding.yaml alone.

Generates:
    solution/solve.sh
    solution/TRUTH.md
    solution/rubrics.json
    solution/golden_trajectory.json
    tests/test_output.py
    tests/fixtures/*.json

INVOKES NO MODEL, NO NETWORK, NO CLOCK, NO LOCALE AND NO RANDOM SOURCE. The only
bytes it reads are solution/grounding.yaml and solution/reference.py, both frozen
files in this bundle; reference.py is read to bind its sha256 into the accepting
fixture, so the accepting half proves the checkers accept THIS reference and not
merely some file. Running this twice over frozen bytes produces byte-identical
output.

    python3 solution/recompute.py            # write
    python3 solution/recompute.py --check     # fail if committed bytes drifted
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import yaml

BUNDLE = pathlib.Path(__file__).resolve().parent.parent
GROUNDING = BUNDLE / "solution" / "grounding.yaml"
REFERENCE = BUNDLE / "solution" / "reference.py"
FIXTURES = BUNDLE / "tests" / "fixtures"

BANNER = "GENERATED SECTION. DO NOT HAND-EDIT."
SOURCE_LINE = "Derived from solution/grounding.yaml by solution/recompute.py."


def digest(payload) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def summary_digest(prior) -> str:
    return digest([[int(row["index"]), str(row["recipe_fingerprint"]),
                    row.get("sustained_crossing_step"), row.get("attempt_reward")]
                   for row in prior])


def fingerprint(recipe: dict) -> str:
    blob = json.dumps(recipe, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def curve(crossing, target: float, stride: int, max_steps: int) -> list:
    """A raw loss series whose only sustained crossing is exactly `crossing`.

    Below the crossing the series sits strictly above the target and decreases
    monotonically; at and after it the series sits strictly below. A `None`
    crossing gives a series that never reaches the target at all.
    """
    steps = list(range(stride, max_steps + 1, stride))
    out = []
    for step in steps:
        if crossing is None:
            value = target + 0.20 + 0.60 * (1.0 - step / float(max_steps))
        elif step >= int(crossing):
            value = target - 0.05 - 0.20 * (step - int(crossing)) / float(max_steps)
        else:
            value = target + 0.02 + 1.40 * (int(crossing) - step) / float(int(crossing))
        out.append(round(value, 6))
    return out


def build_session(spec: dict, attempts_spec: list) -> dict:
    stride = int(spec["stride"])
    window = int(spec["sustain_window"])
    target = float(spec["target_loss"])
    max_steps = int(spec["max_steps"])
    baseline = int(spec["baseline_metric"])
    goal = int(spec["target_metric"])
    span = float(baseline - goal)
    bases = spec["recipe_bases"]

    rows: list = []
    for item in attempts_spec:
        recipe = dict(bases[item["family"]])
        recipe.update(item.get("variant") or {})
        recipe["max_steps"] = max_steps
        mark = fingerprint(recipe)
        run = digest({"recipe": recipe, "index": item["index"]})
        losses = curve(item["crossing"], target, stride, max_steps)
        evals = []
        for position, value in enumerate(losses):
            step = stride * (position + 1)
            evals.append({
                "step": step,
                "loss": value,
                "smoothing": "none",
                "source": "verifier-recomputed",
                "weights_provenance": "harness-step-state",
                "weights_run_digest": run,
                "weights_digest": digest({"run": run, "step": step}),
            })
        crossing = item["crossing"]
        contribution = 0.0 if crossing is None else min(max((baseline - float(crossing)) / span, 0.0), 1.0)
        prior = list(rows)
        handed = summary_digest(prior)
        by_index = {int(row["index"]): row for row in prior}
        inherited = {}
        for claim in item.get("inherits_from") or []:
            source = by_index.get(int(claim))
            if source is None:
                continue
            shared = sorted(key for key, value in recipe.items()
                            if key in source["recipe"] and source["recipe"][key] == value)
            if shared:
                inherited[str(int(claim))] = shared
        rows.append({
            "index": int(item["index"]),
            "run_digest": run,
            "recipe": recipe,
            "recipe_fingerprint": mark,
            "frozen_axis_writes": [],
            "unknown_keys": [],
            "summary_digest_expected": handed,
            "summary_digest_echoed": handed,
            "carried_attempt_count": len(prior),
            "inherits_from": [int(claim) for claim in item.get("inherits_from") or []],
            "inherited_components": inherited,
            "attempt_seconds": float(spec["attempt_seconds"]),
            "halted_at_step": max_steps,
            "forward_passes_per_step": 1,
            "backward_passes_per_step": 1,
            "verifier_evals": evals,
            "sustained_crossing_step": crossing,
            "attempt_reward": round(contribution, 6),
            "reported": {"crossing_step": None, "claimed_val_loss": None, "readout": "raw"},
        })
    return {
        "schema": "bia.session/v1",
        "slot": "OER-02",
        "bindings": {"max_timeout_hours": 6.0, "budget_hours": 0.12, "max_attempts": 50,
                     "final_selection": "best", "final_selection_k": 50},
        "anchors": {"baseline_metric": baseline, "target_metric": goal,
                    "target_loss": target, "pass_threshold": 0.65,
                    "anchors_state": "fixture-calibrated"},
        "consolidation": dict(spec["consolidation_rules"]),
        "eval_schedule": {"owner": "verifier", "stride": stride,
                          "sustain_window": window, "smoothing": "none"},
        "session_seconds": round(float(spec["attempt_seconds"]) * len(rows), 6),
        "attempts": rows,
        "failed_launches": [],
    }


def apply_patch(spec: dict, attempts_spec: list, patch: dict):
    """Spec-level patches rebuild the session; post-build patches edit the record."""
    operation = patch.get("op", "none")
    if operation == "keep_one_crossing":
        keep = int(patch["index"])
        return [dict(row, crossing=(row["crossing"] if int(row["index"]) == keep else None))
                for row in attempts_spec], None
    if operation == "flat_sweep":
        return [dict(row, inherits_from=[], crossing=150) for row in attempts_spec], None
    return attempts_spec, operation


def edit_session(session: dict, patch: dict, operation) -> dict:
    if operation in (None, "none"):
        return session
    rows = session["attempts"]
    if operation == "set_report":
        rows[int(patch["index"]) - 1]["reported"]["crossing_step"] = int(patch["crossing_step"])
    elif operation == "set_eval_smoothing":
        for item in rows[int(patch["index"]) - 1]["verifier_evals"]:
            item["smoothing"] = str(patch["value"])
    elif operation == "break_window":
        row = rows[int(patch["index"]) - 1]
        where = [position for position, item in enumerate(row["verifier_evals"])
                 if int(item["step"]) == int(row["sustained_crossing_step"])][0]
        item = row["verifier_evals"][where + int(patch["offset"])]
        item["loss"] = round(float(item["loss"]) + float(patch["delta"]), 6)
    elif operation == "early_stop_claim":
        row = rows[int(patch["index"]) - 1]
        halted = int(patch["halted_at_step"])
        row["halted_at_step"] = halted
        row["verifier_evals"] = [item for item in row["verifier_evals"] if int(item["step"]) <= halted]
        row["reported"]["crossing_step"] = int(patch["claim"])
    elif operation == "foreign_weights":
        row = rows[int(patch["index"]) - 1]
        row["verifier_evals"][int(patch["offset"])]["weights_run_digest"] = digest({"foreign": True})
    elif operation == "set_attempt_seconds":
        rows[int(patch["index"]) - 1]["attempt_seconds"] = float(patch["seconds"])
    elif operation == "frozen_write":
        rows[int(patch["index"]) - 1]["frozen_axis_writes"] = [str(patch["key"])]
    elif operation == "stale_summary":
        position = int(patch["index"]) - 1
        stale = rows[position - 1]["summary_digest_expected"]
        rows[position]["summary_digest_expected"] = stale
        rows[position]["summary_digest_echoed"] = stale
    else:
        raise ValueError("patch operation outside the closed set: " + str(operation))
    return session


def write(path: pathlib.Path, text: str, check: bool, drifted: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if check:
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            drifted.append(path.relative_to(BUNDLE).as_posix())
        return
    path.write_text(text, encoding="utf-8")


def compile_solve_sh(ground: dict) -> str:
    return "\n".join([
        "#!/usr/bin/env bash",
        "# " + BANNER,
        "# " + SOURCE_LINE,
        "#",
        "# The reference solution is a refinement POLICY, not a configuration. The",
        "# harness invokes it once per attempt with state.json beside it, and it",
        "# writes proposal.json. This script only puts it where the harness looks.",
        "set -euo pipefail",
        'HERE="$(cd "$(dirname "$0")" && pwd)"',
        'TARGET="${BIA_SUBMISSION:-/workspace/refine.py}"',
        'mkdir -p "$(dirname "${TARGET}")"',
        'cp "${HERE}/reference.py" "${TARGET}"',
        'chmod +x "${TARGET}"',
        'echo "reference policy installed at ${TARGET}"',
        "",
    ])


def compile_truth(ground: dict, session: dict, reference_sha: str) -> str:
    expected = ground["golden_trajectory"]["expected"]
    lines = [
        "<!-- " + BANNER + " -->",
        "<!-- " + SOURCE_LINE + " -->",
        "",
        "# TRUTH, slot OER-02",
        "",
        "## What is graded",
        "",
        "The **consolidated sustained crossing step** of the session, lower is better.",
        "",
        "1. The verifier trains and evaluates. Every graded number is the raw loss the",
        "   verifier's own loop measured on a held-out draw whose seed exists only in the",
        "   verifier environment. Nothing on the graded path is smoothed, averaged or",
        "   filtered.",
        "2. A crossing at step `p` requires the target to hold at `p` and at the next",
        "   " + str(ground["golden_trajectory"]["sustain_window"]) + " evaluation points the verifier schedules.",
        "3. The graded step is the " + str(ground["consolidation_rules"]["multiplicity"]) +
        "rd smallest sustained crossing over attempts with pairwise-distinct recipe fingerprints.",
        "",
        "## Why one lucky attempt cannot reach the band",
        "",
        ground["consolidation_rules"]["why"],
        "",
        "## Anchors",
        "",
        "| field | value | source |",
        "|---|---|---|",
        "| baseline_metric | " + str(ground["anchors"]["baseline_metric"]) + " | " + ground["anchors"]["baseline_identity"] + " |",
        "| target_metric | " + str(ground["anchors"]["target_metric"]) + " | " + ground["anchors"]["target_identity"] + " |",
        "| authority | " + ground["anchors"]["authority"] + " | |",
        "",
        "Reward: `" + ground["anchors"]["reward_formula"] + "`",
        "",
        ground["anchors"]["reward_full_rule"] + ".",
        "",
        "## The reference",
        "",
        "`solution/reference.py`, sha256 `" + reference_sha + "`.",
        "",
        "It bootstraps three optimizer families, then hill-climbs a fixed ladder on top",
        "of the running best, so every later proposal is built from an earlier one. It",
        "reports no crossing, because claiming a number the verifier did not measure is",
        "what the DIVERGENCE checker exists to make expensive.",
        "",
        "## The golden trajectory",
        "",
        "Synthetic, compiled from `solution/grounding.yaml`. It carries no measured",
        "number; it fixes the SHAPE the checker fixtures are built from.",
        "",
        "| attempt | family | crossing | inherits from |",
        "|---|---|---|---|",
    ]
    for spec, row in zip(ground["golden_trajectory"]["attempts"], session["attempts"]):
        lines.append("| " + str(row["index"]) + " | " + str(spec["family"]) + " | " +
                     str(row["sustained_crossing_step"]) + " | " +
                     (", ".join(str(item) for item in row["inherits_from"]) or "none") + " |")
    lines += [
        "",
        "- running best: `" + json.dumps(expected["frontier"]) + "`",
        "- strict improvements at attempts: `" + json.dumps(expected["frontier_improvements"]) + "`",
        "- consolidation indices: `" + json.dumps(expected["consolidation_indices"]) + "`",
        "- graded step: `" + str(expected["consolidated_metric"]) + "`",
        "- reward: `" + str(expected["reward"]) + "`, reason `" + expected["reason"] + "`",
        "",
        "## What is unmeasured",
        "",
        ground["operating_point"]["consequence"],
        "",
        "Gap: `" + ground["operating_point"]["gap"] + "`.",
        "",
        "## Divergences carried",
        "",
        "- solver egress: " + ground["solver_egress_divergence"]["why_this_lane_binds_open"],
        "- reward path: " + ground["reward_path_divergence"]["resolution"],
        "- free axis: " + ground["free_axis_narrowing"]["cost"],
        "",
    ]
    return "\n".join(lines)


def compile_rubrics(ground: dict, reference_sha: str) -> str:
    rows = []
    for item in ground["rubric_targets"]:
        rows.append({
            "id": item["id"],
            "criterion": item["criterion"],
            "reference": "solution/reference.py",
            "reference_sha256": reference_sha,
            "graded_against": "the reference answer, not the trajectory",
        })
    payload = {
        "banner": BANNER,
        "source": "solution/grounding.yaml",
        "slot": "OER-02",
        "note": "This file grades a SOLUTION against its reference answer. tests/rubrics.jsonl is a different file with a different job: it judges the TRAJECTORY and may lower an outcome, never raise one.",
        "rubrics": rows,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def compile_tests(ground: dict, manifest: dict) -> str:
    order = [row["id"] for row in ground["checker_order"]]
    lines = [
        '"""' + BANNER,
        "",
        SOURCE_LINE,
        "",
        "Both halves of every checker, over the compiled fixtures in tests/fixtures/.",
        "The accepting half is the clean fixture, on which every checker passes and the",
        "reward is exactly 1.0. The rejecting half is that checker's planted fixture, on",
        "which that checker and only that checker fails, with its declared reason.",
        '"""',
        "from __future__ import annotations",
        "",
        "import json",
        "import pathlib",
        "import sys",
        "",
        "HERE = pathlib.Path(__file__).resolve().parent",
        "sys.path.insert(0, str(HERE))",
        "",
        "import checkers  # noqa: E402",
        "import reward  # noqa: E402",
        "",
        "FIXTURES = HERE / \"fixtures\"",
        "",
        "",
        "def load(name):",
        "    return json.loads((FIXTURES / (name + \".json\")).read_text(encoding=\"utf-8\"))",
        "",
        "",
        "def first_failure(session):",
        "    for name, verdict in checkers.run_chain(session):",
        "        if not verdict.ok:",
        "            return name, verdict.reason",
        "    return None, \"\"",
        "",
        "",
        "def test_accepting_half_scores_full_reward():",
        "    session = load(\"accept_reference_session\")",
        "    assert first_failure(session) == (None, \"\")",
        "    metric, _ = checkers.consolidation(session)",
        "    anchors = session[\"anchors\"]",
        "    value = reward.anchored(metric, anchors[\"baseline_metric\"], anchors[\"target_metric\"])",
        "    assert value == 1.0",
        "",
        "",
        "def test_accepting_half_survives_a_non_empty_report():",
        "    session = load(\"accept_reported_within_tolerance\")",
        "    claims = [row for row in session[\"attempts\"]",
        "              if (row.get(\"reported\") or {}).get(\"crossing_step\") is not None]",
        "    assert claims",
        "    assert first_failure(session) == (None, \"\")",
        "",
    ]
    for ident in order:
        row = manifest["by_checker"][ident]
        lines += [
            "",
            "def test_" + ident + "():",
            "    clean = load(\"accept_reference_session\")",
            "    assert checkers." + row["selector"] + "(clean).ok",
            "    planted = load(\"" + row["fixture"] + "\")",
            "    verdict = checkers." + row["selector"] + "(planted)",
            "    assert not verdict.ok",
            "    assert verdict.reason == \"" + row["reason"] + "\"",
            "    assert first_failure(planted) == (\"" + ident + "\", \"" + row["reason"] + "\")",
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    check = "--check" in sys.argv[1:]
    ground = yaml.safe_load(GROUNDING.read_text(encoding="utf-8"))
    reference_sha = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()

    spec = dict(ground["golden_trajectory"])
    spec["consolidation_rules"] = ground["consolidation_rules"]
    attempts_spec = ground["golden_trajectory"]["attempts"]

    drifted: list = []
    manifest = {"banner": BANNER, "source": "solution/grounding.yaml",
                "reference_sha256": reference_sha, "fixtures": {}, "by_checker": {}}

    for name, item in ground["fixtures"].items():
        patched_spec, operation = apply_patch(spec, attempts_spec, item["patch"])
        session = build_session(spec, patched_spec)
        session = edit_session(session, item["patch"], operation)
        session["fixture"] = {"name": name, "kind": item["kind"],
                              "reference_sha256": reference_sha,
                              "expect_reason": item["expect_reason"]}
        write(FIXTURES / (name + ".json"),
              json.dumps(session, indent=2, sort_keys=True) + "\n", check, drifted)
        manifest["fixtures"][name] = {"kind": item["kind"], "reason": item["expect_reason"],
                                      "checker": item.get("checker", "")}
        if item["kind"] == "reject":
            manifest["by_checker"][item["checker"]] = {
                "fixture": name, "reason": item["expect_reason"],
                "selector": "check_" + item["checker"],
            }

    ground["checker_order"] = [{"id": ident} for ident in
                              [row["checker"] for row in ground["fixtures"].values()
                               if row["kind"] == "reject"]]

    golden = build_session(spec, attempts_spec)
    write(BUNDLE / "solution" / "golden_trajectory.json",
          json.dumps({"banner": BANNER, "source": "solution/grounding.yaml",
                      "session": golden}, indent=2, sort_keys=True) + "\n", check, drifted)
    write(FIXTURES / "manifest.json",
          json.dumps(manifest, indent=2, sort_keys=True) + "\n", check, drifted)
    write(BUNDLE / "solution" / "solve.sh", compile_solve_sh(ground), check, drifted)
    write(BUNDLE / "solution" / "TRUTH.md", compile_truth(ground, golden, reference_sha), check, drifted)
    write(BUNDLE / "solution" / "rubrics.json", compile_rubrics(ground, reference_sha), check, drifted)
    write(BUNDLE / "tests" / "test_output.py", compile_tests(ground, manifest), check, drifted)

    if not check:
        (BUNDLE / "solution" / "solve.sh").chmod(0o755)
    if drifted:
        print("DRIFTED: " + ", ".join(sorted(drifted)))
        return 1
    print("recomputed " + str(len(ground["fixtures"])) + " fixtures and 5 generated artifacts")
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
