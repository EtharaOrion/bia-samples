# FORGE-CANARY-BEGIN
# 0: 3d3cf051b7f0bf8ce4100c7e72a60384cf0ce9a48fce01d8f0099aea6cf2f85a
# 1: 0fd794da38cb5d7c756b90eb8d47bbc420957877688dc010e1d7d7a720afa0b6
# 2: b8d663336a7d8f97ca45cd2ea93d6f55d3d3b8c3ca2bb298fb704a300518049c
# 3: b6aa4c95c9194078802116c8b76c4a14a9ec5c12e87856f6b10177506f7cf4fa
# FORGE-CANARY-END
#!/usr/bin/env python3
"""Derive every generated artifact of OER-12 from solution/grounding.yaml alone.

Generated here, and nowhere else: the golden trajectory, every checker fixture,
solution/solve.sh, solution/TRUTH.md, solution/rubrics.json and tests/test_output.py.

This module invokes NO model, NO network, NO clock, NO locale and NO random source.
Every value it writes is a pure function of solution/grounding.yaml plus the frozen
bundle bytes that file names under derivation_inputs. Running it twice over frozen bytes
produces byte-identical output, which solution/recompute.py --check asserts directly.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import yaml

BUNDLE = pathlib.Path(__file__).resolve().parent.parent
SOLUTION = BUNDLE / "solution"
TESTS = BUNDLE / "tests"
ENVIRONMENT = BUNDLE / "environment"
FIXTURES = SOLUTION / "fixtures"
GROUNDING = SOLUTION / "grounding.yaml"

BANNER = "GENERATED SECTION. DO NOT HAND-EDIT."
BANNER_SOURCE = "solution/grounding.yaml"
BANNER_LINE = BANNER + " Source: " + BANNER_SOURCE

sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(SOLUTION))

import reference  # noqa: E402
import runner  # noqa: E402

HARNESS_DOCUMENTS = (
    "run",
    "eval_ledger",
    "weight_ledger",
    "corpus_ledger",
    "feed_ledger",
    "split_ledger",
    "event_log",
    "freeze_ledger",
    "policy_state",
    "anchors",
)


def load_grounding() -> dict:
    return yaml.safe_load(GROUNDING.read_text(encoding="utf-8"))


def stable_digest(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def dump_json(payload) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------------
# Fixture construction.
# --------------------------------------------------------------------------------


def base_documents(ground: dict, curve_id: str) -> dict:
    run_shape = ground["harness_run"]
    state = dict(ground["bound_state"])
    version = int(state["parse_schema_version"])
    budget = int(state["token_budget_tokens"])

    schema = json.loads((ENVIRONMENT / "schema.json").read_text(encoding="utf-8"))
    split = json.loads((ENVIRONMENT / "split.json").read_text(encoding="utf-8"))
    corpus = (ENVIRONMENT / "corpus" / "corpus.rec").read_text(encoding="utf-8")
    eval_ordinals = list(split["eval_record_ordinals"])

    built = reference.build(corpus, schema, split, version, eval_ordinals, budget)

    curve = {int(step): float(loss) for step, loss in ground["loss_curves"][curve_id]}
    offsets = [float(value) for value in run_shape["raw_batch_offsets"]]

    eval_rows = []
    weight_rows = []
    for step in [run_shape["bound_eval_step"]] + list(run_shape["sustain_points"]):
        loss = curve[int(step)]
        digest = stable_digest("oer12-weights", str(step))
        eval_rows.append(
            {
                "step": int(step),
                "loss": loss,
                "raw_batch_losses": [loss + offset for offset in offsets],
                "weights_source": run_shape["weights_source"],
                "weights_digest": digest,
                "reported_by_submission": False,
            }
        )
        weight_rows.append({"step": int(step), "digest": digest})

    anchors = ground["fixture_anchors"]
    return {
        "run": {
            "slot": ground["slot"],
            "bound_eval_step": int(run_shape["bound_eval_step"]),
            "sustain_points": [int(step) for step in run_shape["sustain_points"]],
            "sustain_band": float(run_shape["sustain_band"]),
            "halt_kind": run_shape["halt_kind"],
            "halted_at_step": int(run_shape["halted_at_step"]),
        },
        "eval_ledger": {
            "origin": run_shape["eval_origin"],
            "evaluated_by": run_shape["evaluated_by"],
            "smoothing": run_shape["smoothing_on_graded_path"],
            "split_id": state["eval_split_id"],
            "records": eval_rows,
        },
        "weight_ledger": {"records": weight_rows},
        "corpus_ledger": runner.corpus_ledger(corpus, version),
        "feed_ledger": runner.feed_ledger(
            corpus, version, built["pipeline"]["ambiguity_policy"], eval_ordinals, budget
        ),
        "split_ledger": {
            "split_id": state["eval_split_id"],
            "eval_ordinals": eval_ordinals,
            "eval_ids": list(split["eval_record_ids"]),
        },
        "event_log": {"events": built["events"]},
        "freeze_ledger": {
            "axes": [
                {
                    "axis": axis,
                    "open_digest": stable_digest("oer12-frozen", axis),
                    "close_digest": stable_digest("oer12-frozen", axis),
                }
                for axis in ground["frozen_axes_digested"]
            ]
        },
        "policy_state": {"state": state},
        "anchors": {
            "anchors_state": anchors["anchors_state"],
            "baseline_metric": float(anchors["baseline_metric"]),
            "target_metric": float(anchors["target_metric"]),
            "authority": anchors["authority"],
        },
        "_submission_parse_ledger": built["parse_ledger"],
        "_corpus": corpus,
        "_schema": schema,
        "_split": split,
        "_budget": budget,
        "_eval_ordinals": eval_ordinals,
    }


def _row_index(rows: list, ordinal: str) -> int:
    for index, row in enumerate(rows):
        if row.get("ordinal") == ordinal:
            return index
    raise KeyError("no ledger row for ordinal " + str(ordinal))


def apply_defect(documents: dict, defect: dict, ground: dict) -> dict:
    kind = defect["kind"]

    if kind == "drop_documents":
        for name in defect["documents"]:
            documents.pop(name, None)
        if not defect.get("submission_ledger", True):
            documents["_submission_parse_ledger"] = None
        return documents

    if kind == "eval_origin":
        documents["eval_ledger"]["origin"] = defect["origin"]
        documents["eval_ledger"]["records"][0]["reported_by_submission"] = bool(
            defect["reported_by_submission"]
        )
        return documents

    if kind == "smooth_readout":
        documents["eval_ledger"]["smoothing"] = defect["smoothing"]
        documents["eval_ledger"]["records"][0]["loss"] = float(defect["graded_loss_override"])
        return documents

    if kind == "early_stop":
        documents["run"]["halt_kind"] = defect["halt_kind"]
        documents["run"]["halted_at_step"] = int(defect["halted_at_step"])
        return documents

    if kind == "selected_checkpoint":
        for row in documents["eval_ledger"]["records"]:
            if row["step"] == int(defect["step"]):
                row["weights_source"] = defect["weights_source"]
        return documents

    if kind == "ledger_rows":
        rows = documents["_submission_parse_ledger"]["records"]
        for patch in defect["rows"]:
            index = _row_index(rows, patch["ordinal"])
            rows[index] = dict(rows[index])
            rows[index].update(
                {
                    "verdict": patch["verdict"],
                    "reason": patch["reason"],
                    "policy_applied": patch["policy_applied"],
                }
            )
        return documents

    if kind == "ledger_policy":
        documents["_submission_parse_ledger"]["ambiguity_policy"] = defect["ambiguity_policy"]
        return documents

    if kind == "stale_schema":
        stale = int(defect["schema_version"])
        documents["corpus_ledger"] = runner.corpus_ledger(documents["_corpus"], stale)
        stale_build = reference.build(
            documents["_corpus"],
            documents["_schema"],
            documents["_split"],
            stale,
            documents["_eval_ordinals"],
            documents["_budget"],
        )
        documents["_submission_parse_ledger"] = stale_build["parse_ledger"]
        return documents

    if kind == "reorder_events":
        documents["event_log"]["events"] = [
            {"seq": int(defect["classified_seq"]), "event": "first-document-classified", "doc": "r01"},
            {"seq": int(defect["declared_seq"]), "event": "ambiguity-policy-declared",
             "policy": "quarantine-partial"},
        ]
        return documents

    if kind == "overspend":
        documents["feed_ledger"]["fed_tokens"] = int(defect["fed_tokens"])
        return documents

    if kind == "feed_accounting":
        documents["feed_ledger"]["accounting"] = defect["accounting"]
        return documents

    if kind == "leak_split":
        documents["feed_ledger"]["fed_ordinals"] = sorted(
            set(documents["feed_ledger"]["fed_ordinals"]) | set(defect["ordinals"])
        )
        return documents

    if kind == "stale_split":
        documents["split_ledger"]["split_id"] = defect["split_id"]
        documents["feed_ledger"]["fed_ordinals"] = sorted(
            set(documents["feed_ledger"]["fed_ordinals"]) | set(defect["ordinals"])
        )
        return documents

    if kind == "move_axis":
        for row in documents["freeze_ledger"]["axes"]:
            if row["axis"] == defect["axis"]:
                row["close_digest"] = stable_digest("oer12-frozen-moved", defect["axis"])
        return documents

    if kind == "forge_writer":
        documents[defect["document"]]["_writer_override"] = defect["writer"]
        return documents

    raise KeyError("unknown fixture defect kind: " + str(kind))


def emit_fixture(ground: dict, scenario: dict) -> dict:
    documents = base_documents(ground, scenario.get("curve", "accepted"))
    if scenario.get("anchors") == "absent":
        documents["anchors"] = {
            "anchors_state": "absent",
            "baseline_metric": None,
            "target_metric": None,
            "authority": ground["anchors"]["authority_for_absence"],
        }
    if scenario.get("defect"):
        documents = apply_defect(documents, scenario["defect"], ground)

    root = FIXTURES / scenario["id"]
    harness = root / "harness"
    view = root / "submission_view"
    for name in HARNESS_DOCUMENTS:
        payload = documents.get(name)
        if payload is None:
            continue
        body = dict(payload)
        body["writer"] = body.pop("_writer_override", runner.HARNESS_WRITER)
        body["_generated"] = BANNER_LINE
        write(harness / (name + ".json"), dump_json(body))

    ledger = documents.get("_submission_parse_ledger")
    if ledger is not None:
        body = dict(ledger)
        body["_generated"] = BANNER_LINE
        write(view / "parse_ledger.json", dump_json(body))

    return {
        "id": scenario["id"],
        "role": scenario["role"],
        "targets": scenario.get("targets"),
        "expect_reward": scenario.get("expect_reward"),
        "expect_reason": scenario.get("expect_reason"),
        "path": root.relative_to(BUNDLE).as_posix(),
    }


# --------------------------------------------------------------------------------
# Generated artifacts.
# --------------------------------------------------------------------------------


def emit_golden_trajectory(ground: dict, fixtures: list) -> None:
    payload = {
        "_generated": BANNER_LINE,
        "slot": ground["slot"],
        "reference_sha256": hashlib.sha256(
            (SOLUTION / "reference.py").read_bytes()
        ).hexdigest(),
        "reference_binding_rule": ground["reference_binding_rule"],
        "steps": ground["golden_trajectory"]["steps"],
        "what_the_agent_must_not_do": ground["golden_trajectory"]["what_the_agent_must_not_do"],
        "fixtures": fixtures,
        "loss_curves_are_fixtures_not_measurements": ground[
            "loss_curves_are_fixtures_not_measurements"
        ],
    }
    write(SOLUTION / "golden_trajectory.json", dump_json(payload))


def emit_solve(ground: dict) -> None:
    block = ground["solve"]
    lines = [
        block["shell"],
        "# " + BANNER_LINE,
        "# " + block["description"],
        "set -euo pipefail",
        "",
    ]
    lines.extend(block["steps"])
    lines.append("")
    path = SOLUTION / "solve.sh"
    write(path, "\n".join(lines))
    path.chmod(0o755)


def emit_truth(ground: dict) -> None:
    truth = ground["truth"]
    lines = ["<!-- " + BANNER_LINE + " -->", "", "# TRUTH.md — OER-12", "",
             truth["headline"], ""]
    for section in truth["sections"]:
        lines.append("## " + section["heading"])
        lines.append("")
        lines.append(section["body"].strip())
        lines.append("")
    lines.append("## Declared gaps")
    lines.append("")
    for gap in ground["gaps"]:
        lines.append("- **" + gap["id"] + "** — " + gap["statement"].strip()
                     + " Closes by: " + gap["closes_by"].strip() + ".")
    lines.append("")
    write(SOLUTION / "TRUTH.md", "\n".join(lines))


def emit_solution_rubrics(ground: dict) -> None:
    payload = {
        "_generated": BANNER_LINE,
        "schema": "forge.solution_rubrics/v1",
        "slot": ground["slot"],
        "judged_against": "the solution's reference answer, not the trajectory",
        "distinct_from": "tests/rubrics.jsonl, which is judged against the trajectory",
        "rubrics": ground["solution_rubrics"],
    }
    write(SOLUTION / "rubrics.json", dump_json(payload))


TEST_OUTPUT_HEAD = '''#!/usr/bin/env python3
"""{banner}

The compiled mirror of the graded checker set. One test per checker, in the same order
tests/checkers.py registers them, because that order decides which machine-readable
reason a defect is attributed to.
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import checkers  # noqa: E402

DEFAULT_RUN_DIR = "/tmp/oer12-run"


def run_dir() -> pathlib.Path:
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        return pathlib.Path(sys.argv[1])
    return pathlib.Path(DEFAULT_RUN_DIR)


def handles() -> checkers.Handles:
    root = run_dir()
    return checkers.Handles(harness=root / "harness", submission_view=root / "submission_view")


def _assert(ident: str, zero_reason: str, verdict) -> None:
    assert verdict.passed, ident + " scored zero with reason " + (verdict.reason or zero_reason) \\
        + ": " + verdict.detail


'''

TEST_OUTPUT_CASE = '''def test_{ident}() -> None:
    _assert("{ident}", "{zero_reason}", checkers.{selector}(handles()))


'''

TEST_OUTPUT_TAIL = '''CASES = (
{cases}
)


def main() -> int:
    failures = []
    for name, case in CASES:
        try:
            case()
        except AssertionError as exc:
            failures.append(name + ": " + str(exc))
    for line in failures:
        print("FAIL " + line)
    print("checkers collected " + str(len(CASES)) + ", failed " + str(len(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def emit_test_output(ground: dict) -> None:
    body = TEST_OUTPUT_HEAD.format(banner=BANNER_LINE)
    for row in ground["graded_checkers"]:
        body += TEST_OUTPUT_CASE.format(
            ident=row["id"], zero_reason=row["zero_reason"], selector=row["selector"]
        )
    cases = "\n".join(
        '    ("{id}", test_{id}),'.format(id=row["id"]) for row in ground["graded_checkers"]
    )
    body += TEST_OUTPUT_TAIL.format(cases=cases)
    write(TESTS / "test_output.py", body)


def generate() -> dict:
    ground = load_grounding()
    fixtures = [emit_fixture(ground, scenario) for scenario in ground["scenarios"]]
    emit_golden_trajectory(ground, fixtures)
    emit_solve(ground)
    emit_truth(ground)
    emit_solution_rubrics(ground)
    emit_test_output(ground)
    return {"fixtures": len(fixtures), "checkers": len(ground["graded_checkers"])}


def snapshot() -> str:
    rows = []
    targets = [
        SOLUTION / "golden_trajectory.json",
        SOLUTION / "solve.sh",
        SOLUTION / "TRUTH.md",
        SOLUTION / "rubrics.json",
        TESTS / "test_output.py",
    ]
    targets.extend(sorted(FIXTURES.rglob("*.json")))
    for path in targets:
        rows.append([path.relative_to(BUNDLE).as_posix(),
                     hashlib.sha256(path.read_bytes()).hexdigest()])
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode("utf-8")).hexdigest()


def main(argv) -> int:
    summary = generate()
    first = snapshot()
    if "--check" in argv:
        generate()
        second = snapshot()
        if first != second:
            print("NOT BYTE-IDENTICAL across two runs: " + first + " then " + second,
                  file=sys.stderr)
            return 1
        summary["byte_identical_across_two_runs"] = True
        summary["snapshot"] = first
    print(json.dumps(summary, sort_keys=True))
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
