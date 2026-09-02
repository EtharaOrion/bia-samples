# FORGE-CANARY-BEGIN
# 0: 0cfa8ecc39b13741a4ee1c895e21c2578b8cabf2868f7525724ce69c3ed5027e
# 1: 403dfc64c0fcd55b294c2e758da1e71c9c8c5cce441a27b12013d2a2fdf6d1ef
# 2: a9db2ef93ce7fef28fd06d099c9f9a1ef9587a267b048060c060c50c73fffb64
# 3: b2843ef1ff82466b6202a4f35cf12b576813c0c376d5521d99c6595b9e11beab
# FORGE-CANARY-END
#!/usr/bin/env python3
"""Derive every generated artifact for OER-07 from solution/grounding.yaml alone.

Generates:
    solution/golden_trajectory.json
    solution/fixtures/<id>.json        the checker fixtures adequacy.py materialises
    solution/solve.sh                  the solution entry point
    solution/TRUTH.md
    solution/rubrics.json
    tests/test_output.py               the compiled tests checkers.yaml names
    tests/schedule_expected.json       the verifier's frozen copy of its own schedule

No model, no network, no clock, no locale, no random source. The only bytes read
are solution/grounding.yaml and the one file grounding.yaml names under
reference_binding, whose sha256 is measured rather than authored. Running this
twice over frozen bytes produces byte-identical output.

    python3 solution/recompute.py            write
    python3 solution/recompute.py --check    report drift without writing
"""

import hashlib
import json
import pathlib
import sys

import yaml

BANNER = "GENERATED SECTION. DO NOT HAND-EDIT."
SOURCE = "solution/grounding.yaml"
HERE = pathlib.Path(__file__).resolve().parent
BUNDLE = HERE.parent
FOLD_ID = "digest"


def load():
    with (HERE / "grounding.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def digest(payload):
    return hashlib.sha256(canonical(payload)).hexdigest()


def token_digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def reference_digest(ground):
    """sha256 of the file grounding.yaml names. Measured, never authored here."""
    named = pathlib.Path(str(ground["reference_binding"]["file"]))
    return hashlib.sha256((BUNDLE / named).read_bytes()).hexdigest()


def weights_digest(seed, step):
    return token_digest("OER-07|weights|seed" + str(seed) + "|step" + str(step))


def evaluation_points(ground, seed, losses=None):
    steps = list(ground["schedule"]["points"])
    values = list(losses if losses is not None else ground["clean_arm"]["losses"])
    return [
        {"step": int(step), "loss": float(value), "weights_digest": weights_digest(seed, step)}
        for step, value in zip(steps, values)
    ]


def clean_files(ground):
    """The accepting fixture: the telemetry a reference run leaves behind."""
    files = {}
    schedule = ground["schedule"]
    files["eval/schedule.json"] = {
        "points": [int(item) for item in schedule["points"]],
        "sustain_points_required": int(schedule["sustain_points_required"]),
        "target_loss": float(schedule["target_loss"]),
        "min_seeds": int(schedule["min_seeds"]),
        "scheduled_by": str(schedule["scheduled_by"]),
    }
    for seed in ground["seeds"]:
        files["eval/seed-" + str(seed) + "/evaluations.json"] = {
            "seed": int(seed),
            "smoothing": str(ground["clean_arm"]["smoothing"]),
            "source": str(ground["clean_arm"]["source"]),
            "halted_at_step": ground["clean_arm"]["halted_at_step"],
            "graded_step": ground["clean_arm"]["graded_step"],
            "points": evaluation_points(ground, seed),
        }
        files["eval/seed-" + str(seed) + "/checkpoint_manifest.json"] = {
            "seed": int(seed),
            "owner": "harness",
            "checkpoints": [
                {"step": int(step), "weights_digest": weights_digest(seed, step)}
                for step in schedule["points"]
            ],
        }
    files["eval/reference_arm.json"] = {
        "seeds": {str(key): float(value) for key, value in ground["reference_arm"]["seeds"].items()},
        "produced_by": "verifier",
    }
    files["anchors.json"] = {
        "resolved": True,
        "baseline_metric": int(ground["fixture_anchors"]["baseline"]),
        "target_metric": int(ground["fixture_anchors"]["target"]),
        "source": "fixture-resolved synthetic constants, NOT anchors",
        "anchors_state_in_bundle": "absent",
        "gap": "gap-oer-per-family-anchors-unmeasured",
    }
    files["frozen_axes.json"] = {
        "source": str(ground["frozen_axes"]["source"]),
        "declared": dict(ground["frozen_axes"]["declared"]),
        "observed": dict(ground["frozen_axes"]["observed"]),
    }
    files.update(loop_files(ground))
    return files


def summary_for(ground, index):
    window = int(ground["loop"]["summary_window"])
    priors = [row for row in ground["loop"]["approaches"] if int(row["iteration"]) < int(index)]
    entries = [
        {
            "approach_id": str(row["approach_id"]),
            "description": str(row["description"]),
            "reward": float(row["reward"]),
            "established": [str(item) for item in row["established"]],
        }
        for row in priors
    ]
    compacted = len(entries) > window
    if compacted:
        folded = len(entries) - window
        entries = [
            {
                "approach_id": FOLD_ID,
                "description": str(folded) + " earlier approaches, merged",
                "reward": None,
                "established": [],
            }
        ] + entries[folded:]
    asserts = ground["loop"]["asserts"].get(str(index), [])
    return {
        "index": int(index),
        "compacted": bool(compacted),
        "entries": entries,
        "asserts": [dict(row) for row in asserts],
    }


def ledger_lines(ground):
    """The durable record, in append order, with the reconstruction records."""
    recoveries = {int(row["iteration"]): list(row["recovers"]) for row in ground["loop"]["reconstructions"]}
    lines = []
    lengths = {}
    for index in ground["loop"]["iterations"]:
        if index in recoveries:
            lines.append(
                {
                    "approach_id": "recon-" + str(index),
                    "status": "tried",
                    "established": [],
                    "iteration": int(index),
                    "reconstructed_from_ledger": True,
                    "recovers": [str(item) for item in recoveries[index]],
                }
            )
        for row in ground["loop"]["approaches"]:
            if int(row["iteration"]) != int(index):
                continue
            lines.append(
                {
                    "approach_id": str(row["approach_id"]),
                    "status": str(row["status"]),
                    "established": [str(item) for item in row["established"]],
                    "iteration": int(index),
                    "reward": float(row["reward"]),
                }
            )
        lengths[int(index)] = len(lines)
    return lines, lengths


def loop_files(ground):
    lines, lengths = ledger_lines(ground)
    files = {
        "loop/ledger.jsonl": "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in lines
        )
    }
    rows = []
    final = max(int(item) for item in ground["loop"]["iterations"])
    for index in ground["loop"]["iterations"]:
        summary = summary_for(ground, index)
        files["loop/summaries/iter-" + str(index) + ".json"] = summary
        submission = (
            reference_digest(ground)
            if int(index) == final
            else token_digest("OER-07|attempt|" + str(index))
        )
        rows.append(
            {
                "index": int(index),
                "summary_digest": digest(summary),
                "submission_digest": submission,
                "ledger_len_after": int(lengths[int(index)]),
                "reconciled_against_ledger": bool(ground["loop"]["reconciled"][str(index)]),
                "halted": False,
            }
        )
    files["loop/iterations.json"] = {
        "compaction_at": int(ground["loop"]["compaction_at"]),
        "summary_window": int(ground["loop"]["summary_window"]),
        "iterations": rows,
    }
    return files


# --------------------------------------------------------------------------
# The op vocabulary. Every defect fixture and every negative control is one
# clean tree plus an ordered list of these, so a fixture is a diff against the
# accepting case rather than a second hand-written tree that could drift from it.
# --------------------------------------------------------------------------


def _seed_key(seed):
    return "eval/seed-" + str(seed) + "/evaluations.json"


def apply_op(ground, files, op):
    kind = str(op["op"])
    if kind == "seed_field":
        files[_seed_key(op["seed"])][str(op["field"])] = op["value"]
    elif kind == "all_seed_field":
        for seed in ground["seeds"]:
            files[_seed_key(seed)][str(op["field"])] = op["value"]
    elif kind == "seed_losses":
        files[_seed_key(op["seed"])]["points"] = evaluation_points(ground, op["seed"], op["losses"])
    elif kind == "all_seed_losses":
        for seed in ground["seeds"]:
            files[_seed_key(seed)]["points"] = evaluation_points(ground, seed, op["losses"])
    elif kind == "drop_point":
        record = files[_seed_key(op["seed"])]
        record["points"] = [row for row in record["points"] if int(row["step"]) != int(op["step"])]
    elif kind == "point_digest":
        for row in files[_seed_key(op["seed"])]["points"]:
            if int(row["step"]) == int(op["step"]):
                row["weights_digest"] = str(op["value"])
    elif kind == "reference_arm":
        files["eval/reference_arm.json"]["seeds"] = {
            str(key): float(value) for key, value in op["seeds"].items()
        }
    elif kind == "swap_iterations":
        rows = files["loop/iterations.json"]["iterations"]
        first = next(i for i, row in enumerate(rows) if row["index"] == int(op["first"]))
        second = next(i for i, row in enumerate(rows) if row["index"] == int(op["second"]))
        rows[first], rows[second] = rows[second], rows[first]
    elif kind == "set_asserts":
        files["loop/summaries/iter-" + str(op["iteration"]) + ".json"]["asserts"] = [
            dict(row) for row in op["asserts"]
        ]
    elif kind == "drop_reconstructions":
        kept = [
            row
            for row in files["loop/ledger.jsonl"].splitlines()
            if '"reconstructed_from_ledger":true' not in row.replace(" ", "")
        ]
        files["loop/ledger.jsonl"] = "".join(row + "\n" for row in kept)
    elif kind == "schedule_points":
        files["eval/schedule.json"]["points"] = [int(item) for item in op["points"]]
    elif kind == "observed_axis":
        files["frozen_axes.json"]["observed"][str(op["key"])] = op["value"]
    else:
        raise ValueError("op outside the closed vocabulary: " + kind)
    return files


def reseal(files):
    """Re-derive each iteration's summary digest from the summary bytes it names.

    A fixture that edits a summary must not incidentally fire the ordering
    checker through a stale digest, because then the fixture would prove a
    different checker than the one it was written for. Resealing keeps every
    fixture pointed at exactly the checker it targets.
    """
    for row in files["loop/iterations.json"]["iterations"]:
        key = "loop/summaries/iter-" + str(row["index"]) + ".json"
        if key in files:
            row["summary_digest"] = digest(files[key])
    return files


def fixture(ground, spec):
    files = clean_files(ground)
    for op in spec.get("ops") or []:
        files = apply_op(ground, files, op)
    files = reseal(files)
    return {
        "generated": BANNER + " Source: " + SOURCE,
        "id": str(spec["id"]),
        "fires": str(spec.get("fires", "")),
        "expect_reason": str(spec["expect_reason"]),
        "statement": str(spec["statement"]),
        "files": files,
    }


def accepting_fixture(ground):
    return {
        "generated": BANNER + " Source: " + SOURCE,
        "id": "fx-accepting",
        "fires": "",
        "expect_reason": "graded",
        "statement": "The reference solution's telemetry. Every checker passes and the reward is exactly 1.0.",
        "expect_reward": 1.0,
        "reference_sha256": reference_digest(ground),
        "files": clean_files(ground),
    }


def golden_trajectory(ground):
    lines, lengths = ledger_lines(ground)
    steps = []
    for index in ground["loop"]["iterations"]:
        summary = summary_for(ground, index)
        approach = next(
            (row for row in ground["loop"]["approaches"] if int(row["iteration"]) == int(index)),
            None,
        )
        steps.append(
            {
                "iteration": int(index),
                "summary_compacted": bool(summary["compacted"]),
                "summary_visible_ids": [
                    entry["approach_id"]
                    for entry in summary["entries"]
                    if entry["approach_id"] != FOLD_ID
                ],
                "recovered_from_ledger": [
                    str(item)
                    for row in ground["loop"]["reconstructions"]
                    if int(row["iteration"]) == int(index)
                    for item in row["recovers"]
                ],
                "proposed": None if approach is None else str(approach["approach_id"]),
                "established": [] if approach is None else [str(x) for x in approach["established"]],
                "reward": None if approach is None else float(approach["reward"]),
                "ledger_len_after": int(lengths[int(index)]),
            }
        )
    return {
        "generated": BANNER + " Source: " + SOURCE,
        "slot": "OER-07",
        "compaction_at": int(ground["loop"]["compaction_at"]),
        "summary_window": int(ground["loop"]["summary_window"]),
        "reference_sha256": reference_digest(ground),
        "steps": steps,
        "ledger": lines,
        "final_reward": 1.0,
    }


def rubrics_json(ground):
    return {
        "generated": BANNER + " Source: " + SOURCE,
        "slot": "OER-07",
        "judged_against": "the solution and its reference answer, not the trajectory",
        "not_a_substitute_for": "tests/rubrics.jsonl, which is judged against the trajectory",
        "criteria": [
            {
                "id": str(row["id"]),
                "weight": int(row["weight"]),
                "expected": str(row["expected"]),
            }
            for row in ground["rubrics_json"]["criteria"]
        ],
    }


def solve_sh(ground):
    lines = [
        "#!/usr/bin/env bash",
        "# " + BANNER,
        "# Source: " + SOURCE + ". Regenerate with: python3 solution/recompute.py",
        "#",
        "# The solution entry point. Installs the reference optimizer as the",
        "# submission and drives the bound refinement loop under the recovery",
        "# discipline: reconcile the durable ledger against the handed summary",
        "# BEFORE proposing, at every iteration, and record the reconciliation.",
        "set -euo pipefail",
        "",
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'BUNDLE="$(dirname "${HERE}")"',
        'WORKSPACE="${OER_WORKSPACE:-/workspace}"',
        'LEDGER_DIR="${OER_LEDGER_DIR:-/workspace/ledger}"',
        "",
        'mkdir -p "${LEDGER_DIR}"',
        'cp "${HERE}/reference.py" "${WORKSPACE}/submission.py"',
        "",
        "# The compaction lands at iteration "
        + str(ground["loop"]["compaction_at"])
        + " and the summary window is "
        + str(ground["loop"]["summary_window"])
        + ".",
        "# Nothing announces it. Detection is the difference between the durable",
        "# ledger and the summary handed in, which is what reference.recover()",
        "# computes; it appends a reconstruction record naming every dropped id.",
        "for ITERATION in " + " ".join(str(item) for item in ground["loop"]["iterations"]) + "; do",
        '  OER_ITERATION="${ITERATION}" OER_LEDGER_DIR="${LEDGER_DIR}" \\',
        '    python3 "${HERE}/reconcile.py"',
        "done",
        "",
        'echo "OER-07 reference installed at ${WORKSPACE}/submission.py"',
        "",
    ]
    return "\n".join(lines)


def truth_md(ground):
    truth = ground["truth"]
    lines = [
        "<!-- " + BANNER + " Source: " + SOURCE + " -->",
        "",
        "# OER-07 — TRUTH",
        "",
        "## Headline",
        "",
        str(truth["headline"]).strip(),
        "",
        "## The move",
        "",
        str(truth["the_move"]).strip(),
        "",
        "## Why the summary is not enough",
        "",
        str(truth["why_the_summary_is_not_enough"]).strip(),
        "",
        "## What a shortcut would look like",
        "",
        str(truth["what_a_shortcut_would_look_like"]).strip(),
        "",
        "## The graded quantity",
        "",
        "- target loss: " + str(ground["schedule"]["target_loss"]),
        "- verifier-scheduled evaluation points: "
        + ", ".join(str(item) for item in ground["schedule"]["points"]),
        "- sustain window: " + str(ground["schedule"]["sustain_points_required"]) + " further points",
        "- graded seeds: " + ", ".join(str(item) for item in ground["seeds"]),
        "- minimum seeds for a mean: " + str(ground["schedule"]["min_seeds"]),
        "- separation margin: 0.05, entering as a continuous ramp and never as a cliff",
        "",
        "## Anchors",
        "",
        "anchors_state is `absent` under `gap-oer-per-family-anchors-unmeasured`. No",
        "baseline_metric and no target_metric is authored anywhere in this bundle. The",
        "numbers under `fixture_anchors` in " + SOURCE + " are synthetic unit-test",
        "constants that exercise the arithmetic and are not a claim about this family.",
        "",
        "## Reference binding",
        "",
        "- file: " + str(ground["reference_binding"]["file"]),
        "- sha256: " + reference_digest(ground),
        "",
        "The digest above is measured from the bytes, not authored. It is bound into",
        "the accepting fixture as the final iteration's submission_digest, so the",
        "accepting half proves the checkers accept this reference and not merely some",
        "file.",
        "",
    ]
    return "\n".join(lines)


def schedule_expectation(ground):
    """The verifier's own frozen copy of the evaluation frame it schedules.

    Every checker on the graded path reads its frame — the target loss, the
    sustain window, the minimum seed count and the point series — out of the
    run's own eval/schedule.json. Trusting that file makes the frame whatever the
    run says it is, and a shortened point series shrinks the window a crossing
    has to survive. So the frame is graded against this copy, which is derived
    from grounding.yaml `schedule` and from nothing else.

    It lands in tests/ because tests/ is verifier-only: it is not on the agent
    surface, is not in instruction.md, and is not in environment/. The point
    series is therefore obtainable only by reading the live schedule the harness
    wrote, which is exactly what makes it a discovery value.
    """
    schedule = ground["schedule"]
    return {
        "generated": BANNER + " Source: " + SOURCE,
        "read_by": "tests/checkers.py::check_evaluation_schedule_as_scheduled",
        "points": [int(item) for item in schedule["points"]],
        "target_loss": float(schedule["target_loss"]),
        "sustain_points_required": int(schedule["sustain_points_required"]),
        "min_seeds": int(schedule["min_seeds"]),
        "scheduled_by": str(schedule["scheduled_by"]),
    }


def test_output_py(ground):
    header = [
        "# " + BANNER,
        "# Source: " + SOURCE + ". Regenerate with: python3 solution/recompute.py",
        '"""Compiled tests, one per checker declared in tests/checkers.yaml.',
        "",
        "Each test drives the live checker against the bound runtime telemetry root.",
        "There is no clock read, no random source, no network and no submission import",
        "here; the checker does the reading and this file only asserts its verdict.",
        '"""',
        "",
        "import pathlib",
        "",
        "import checkers",
        "",
        'RUN_ROOT = pathlib.Path("/logs/verifier")',
        "",
        "",
    ]
    body = []
    for row in ground["compiled_tests"]:
        ident = str(row["id"])
        selector = str(row["selector"])
        reason = str(row["zero_reason"])
        body.extend(
            [
                "def test_" + ident + "():",
                '    result = checkers.' + selector + "(RUN_ROOT)",
                "    assert result.ok, " + repr(reason) + ' + ": " + result.detail',
                "",
                "",
            ]
        )
    return "\n".join(header + body).rstrip("\n") + "\n"


def artifacts(ground):
    out = {
        "solution/golden_trajectory.json": json.dumps(
            golden_trajectory(ground), sort_keys=True, indent=1
        )
        + "\n",
        "solution/rubrics.json": json.dumps(rubrics_json(ground), sort_keys=True, indent=1) + "\n",
        "solution/solve.sh": solve_sh(ground),
        "solution/TRUTH.md": truth_md(ground),
        "tests/test_output.py": test_output_py(ground),
        "tests/schedule_expected.json": json.dumps(
            schedule_expectation(ground), sort_keys=True, indent=1
        )
        + "\n",
        "solution/fixtures/fx-accepting.json": json.dumps(
            accepting_fixture(ground), sort_keys=True, indent=1
        )
        + "\n",
    }
    for spec in list(ground["defect_fixtures"]) + list(ground["negative_controls"]):
        out["solution/fixtures/" + str(spec["id"]) + ".json"] = (
            json.dumps(fixture(ground, spec), sort_keys=True, indent=1) + "\n"
        )
    return out


def main(argv):
    ground = load()
    generated = artifacts(ground)
    check_only = "--check" in argv
    drifted = []
    for relative, text in sorted(generated.items()):
        path = BUNDLE / relative
        if path.is_file() and path.read_text(encoding="utf-8") == text:
            continue
        drifted.append(relative)
        if not check_only:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    if not check_only:
        (BUNDLE / "solution" / "solve.sh").chmod(0o755)
    sys.stdout.write(
        json.dumps(
            {"generated": len(generated), "drifted": sorted(drifted), "check_only": check_only},
            sort_keys=True,
        )
        + "\n"
    )
    return 1 if (check_only and drifted) else 0


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
