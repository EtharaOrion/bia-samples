# FORGE-CANARY-BEGIN
# 0: c199358b38f4647b48b6f360e792c40ef36627cf3f88024aeb1abe53748c5cc4
# 1: 2afa68b911e2a0cc9eae108c5fbac826ad175b9dd1dccad33d272bc56c3aaa23
# 2: 75e6589d14e0d3f7135c34277c33453c2ccbc3ce4d25026ededdd976c1a552e7
# 3: 4c19b8c6d15ae67ec7744c97fb511c34b3f68743ce819d9828557072c7197683
# FORGE-CANARY-END
"""Derive every generated artifact for OER-11 from solution/grounding.yaml alone.

Invokes NO model, NO network, NO clock, NO locale and NO random source. Running it twice
over frozen bytes produces byte-identical output, which `seed/tasks/OER-11/adequacy.py`
checks by running it twice and comparing bytes.

Derives:

    solution/solve.sh
    solution/TRUTH.md
    solution/rubrics.json
    solution/golden_trajectory.json
    solution/fixtures/clean_handles.json
    solution/fixtures/planted.json
    tests/test_output.py

Nothing below is hand-authored downstream of the grounding source. Every digest is
recomputed here rather than transcribed, and every loss value is the exact unsmoothed
mean of its own per-batch records, so no number in a generated file can drift away from
the source that produced it.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
GROUNDING = HERE / "grounding.yaml"
FIXTURES = HERE / "fixtures"

BANNER = "GENERATED SECTION. DO NOT HAND-EDIT."
SOURCE = "solution/grounding.yaml"
BANNER_LINE = BANNER + " Source: " + SOURCE


def load_grounding() -> dict:
    with GROUNDING.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_ids(ids) -> str:
    payload = json.dumps(sorted(str(item) for item in ids), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def unsmoothed_mean(values) -> float:
    rows = [float(item) for item in values]
    return math.fsum(rows) / len(rows)


def document_ids(shard_count: int, docs_per_shard: int) -> list:
    return ["d-%04d" % index for index in range(shard_count * docs_per_shard)]


def split_documents(ids, modulus: int, residue: int):
    held, trained = [], []
    for name in ids:
        index = int(name.split("-")[-1])
        (held if index % modulus == residue else trained).append(name)
    return held, trained


def distribute(total: int, buckets: int) -> list:
    base, extra = divmod(total, buckets)
    return [base + (1 if index < extra else 0) for index in range(buckets)]


def evaluation_point(step: int, per_batch, prefix: str) -> dict:
    return {
        "step": step,
        "loss": unsmoothed_mean(per_batch),
        "per_batch_losses": [float(item) for item in per_batch],
        "filter": "none",
        "origin": "verifier-recompute",
        "weights_source": "harness-run-produced",
        "weights_digest": sha256_text(prefix + str(step)),
    }


def clean_handles(grounding: dict) -> dict:
    corpus = grounding["corpus"]
    post = corpus["post_move"]
    evaluation = grounding["evaluation"]

    entries = []
    for index, row in enumerate(grounding["snapshot_events"]):
        phase = corpus[row["phase"]]
        entries.append(
            {"seq": index, "event": row["event"], "snapshot_version": int(phase["snapshot_version"])}
        )
    graded_version = int(post["snapshot_version"])
    budget = int(post["token_budget_tokens"])

    ids = document_ids(int(post["shard_count"]), int(post["docs_per_shard"]))
    held, trained = split_documents(
        ids, int(corpus["eval_split_modulus"]), int(corpus["eval_split_residue"])
    )

    prefix = str(evaluation["weights_seed_prefix"])
    graded_step = int(evaluation["bound_evaluation_step"])
    scheduled = [int(step) for step in evaluation["scheduled_sustain_steps"]]
    graded_point = evaluation_point(graded_step, evaluation["graded_per_batch_losses"], prefix)
    sustain_points = [
        evaluation_point(step, batches, prefix)
        for step, batches in zip(scheduled, evaluation["sustain_per_batch_losses"])
    ]
    split_digest = digest_ids(held)
    stream_digest = sha256_text(str(evaluation["token_stream_seed"]))

    return {
        "_generated": BANNER_LINE,
        "snapshot_ledger": {
            "schema": "forge.snapshot_ledger/v1",
            "owner": "harness",
            "entries": entries,
            "graded_eval_seq": entries[-1]["seq"],
            "ordering_note": "seq is a monotone integer the harness owns; no wall clock is recorded or read",
        },
        "parse_manifest": {
            "schema": "forge.parse_manifest/v1",
            "owner": "harness",
            "built_against_snapshot": graded_version,
            "recorded_at_seq": entries[-3]["seq"],
            "documents": len(trained),
        },
        "pipeline_effect": {
            "schema": "forge.pipeline_effect/v1",
            "owner": "harness",
            "produced": True,
            "cache_hit": False,
            "produced_from_snapshot": graded_version,
            "token_stream_digest": stream_digest,
        },
        "token_budget": {
            "schema": "forge.token_budget/v1",
            "measured_by": "harness",
            "budget_tokens": budget,
            "budget_at_graded_snapshot": budget,
            "tokens_fed": budget,
            "feed_passes": 1,
            "per_shard_tokens": distribute(budget, int(post["shard_count"])),
            "token_stream_digest": stream_digest,
        },
        "split_manifest": {
            "schema": "forge.split_manifest/v1",
            "owner": "harness",
            "split_id": str(post["eval_split_id"]),
            "split_digest": split_digest,
            "eval_batches": len(evaluation["graded_per_batch_losses"]),
            "eval_doc_ids": held,
            "train_doc_ids": trained,
        },
        "weights_ledger": {
            "schema": "forge.weights_ledger/v1",
            "owner": "harness",
            "records": [
                {"step": point["step"], "digest": point["weights_digest"]}
                for point in [graded_point] + sustain_points
            ],
        },
        "eval_ledger": {
            "schema": "forge.eval_ledger/v1",
            "producer": "verifier",
            "submission_reported_used": False,
            "snapshot_version": graded_version,
            "split_id": str(post["eval_split_id"]),
            "split_digest": split_digest,
            "bound_evaluation_step": graded_step,
            "scheduled_sustain_steps": scheduled,
            "sustain_points_required": int(evaluation["sustain_points_required"]),
            "sustain_slack": float(evaluation["sustain_slack"]),
            "halted_early": False,
            "terminated_by": "schedule-exhausted",
            "final_step": scheduled[-1],
            "graded_point": graded_point,
            "sustain_points": sustain_points,
        },
        "anchor_record": {
            "schema": "forge.anchor_record/v1",
            "family": "F13",
            "anchors_state": "absent",
            "baseline_metric": None,
            "target_metric": None,
            "gap": "gap-oer-per-family-anchors-unmeasured",
            "note": "This lane authors no baseline or target for F13. An absent record resolves to reward 0.0 with reason anchors-unresolved.",
        },
    }


def planted(grounding: dict) -> dict:
    rows = []
    for row in grounding["planted_defects"]:
        edits = []
        if "path" in row:
            edits.append({"path": list(row["path"]), "op": "set", "value": row["value"]})
        for extra in row.get("also_set") or []:
            edits.append({"path": list(extra["path"]), "op": "set", "value": extra["value"]})
        for extra in row.get("append") or []:
            edits.append({"path": list(extra["path"]), "op": "append", "value": extra["value"]})
        rows.append(
            {
                "id": row["id"],
                "checker": row["checker"],
                "zero_reason": row["zero_reason"],
                "narrative": " ".join(str(row["narrative"]).split()),
                "edits": edits,
            }
        )
    return {"_generated": BANNER_LINE, "defects": rows}


def apply_edits(handles: dict, edits) -> dict:
    payload = json.loads(json.dumps(handles))
    for edit in edits:
        cursor = payload
        for key in edit["path"][:-1]:
            cursor = cursor[key]
        last = edit["path"][-1]
        if edit["op"] == "append":
            cursor[last] = list(cursor[last]) + [edit["value"]]
        else:
            cursor[last] = edit["value"]
    return payload


def golden_trajectory(grounding: dict) -> dict:
    return {
        "_generated": BANNER_LINE,
        "slot": grounding["slot"],
        "archetype": grounding["archetype"],
        "pivot": "iteration 4, where the agent re-asks the snapshot version instead of reusing the iteration-3 parse verdict",
        "iterations": [
            {
                "iteration": int(row["iteration"]),
                "snapshot_asked": int(row["snapshot_asked"]),
                "action": " ".join(str(row["action"]).split()),
                "holds": " ".join(str(row["holds"]).split()),
                "pivot": bool(row.get("this_is_the_pivot", False)),
            }
            for row in grounding["golden_trajectory"]
        ],
    }


def rubrics_json(grounding: dict) -> dict:
    return {
        "_generated": BANNER_LINE,
        "slot": grounding["slot"],
        "judged": "the solution against its reference answer",
        "not_the_same_file_as": "tests/rubrics.jsonl, which is judged against the trajectory in bucket N and may lower an outcome but never raise one",
        "criteria": [
            {
                "id": row["id"],
                "criterion": " ".join(str(row["criterion"]).split()),
                "reference_answer": " ".join(str(row["reference_answer"]).split()),
                "weight": int(row["weight"]),
            }
            for row in grounding["solution_rubrics"]
        ],
    }


def solve_sh(grounding: dict) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "# " + BANNER_LINE,
        "#",
        "# Reference solve path for " + str(grounding["slot"]) + ".",
        "#",
        "# The shape of this script IS the lesson: the snapshot version is asked, work is",
        "# done, and then the snapshot version is asked AGAIN before anything derived from",
        "# the first answer is committed to. solution/reference.py re-resolves the corpus",
        "# whenever the handle it was given has gone stale.",
        "#",
        "# THIS ORACLE INSTALLS NO SUBMISSION, AND THAT IS THE SLOT'S CONTRACT RATHER THAN AN",
        "# OMISSION. Everything tests/grade.py grades is a harness-owned record under",
        "# /logs/harness, listed in tests/checkers.py HANDLE_FILES and described in",
        "# environment/README.md as written by the harness process and not writable from here.",
        "# There is therefore no destination for this script to name and nothing for it to",
        "# copy, install or redirect: it reads the corpus, builds the token stream, and reports",
        "# on stdout. A reader looking for the graded artifact should look at /logs/harness.",
        "set -euo pipefail",
        "",
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'export PYTHONPATH="${HERE}/../environment:${PYTHONPATH:-}"',
        "",
        "# NOTHING THIS SCRIPT RUNS MAY LEAVE A FILE IN THE SHARED WORKSPACE. Without this,",
        "# the two python3 -c steps below import corpus_api and pipeline from /task and the",
        "# interpreter writes /task/__pycache__/*.pyc, which were the only paths the oracle",
        "# phase created. A harness that infers an install destination from what the phase",
        "# wrote then has exactly one parent directory to choose and picks that bytecode",
        "# directory as the submission. An oracle that installs nothing must also WRITE",
        "# nothing, otherwise its incidental output is read as its submission.",
        "export PYTHONDONTWRITEBYTECODE=1",
        'cd "${HERE}"',
        "",
    ]
    for step in grounding["solve_steps"]:
        lines.append(str(step))
    lines.append("")
    return "\n".join(lines)


def truth_md(grounding: dict) -> str:
    truth = grounding["truth"]
    temporal = grounding["temporal_design"]
    anchors = grounding["anchors"]
    out = [
        "<!-- " + BANNER_LINE + " -->",
        "",
        "# TRUTH: " + str(grounding["slot"]) + " -- " + str(grounding["title"]),
        "",
        "Family " + str(grounding["family"]) + ", " + str(grounding["family_title"])
        + ". Archetype " + str(grounding["archetype"]) + ", " + str(grounding["archetype_title"]) + ".",
        "",
        "## Headline",
        "",
        " ".join(str(truth["headline"]).split()),
        "",
        "## What the agent must do",
        "",
    ]
    for row in truth["what_the_agent_must_do"]:
        out.append("- " + " ".join(str(row).split()))
    out += ["", "## What defeats a submission", ""]
    for row in truth["what_defeats_a_submission"]:
        out.append("- " + " ".join(str(row).split()))
    out += [
        "",
        "## Why the handle exists",
        "",
        " ".join(str(truth["why_the_handle_exists"]).split()),
        "",
        "## Ordering basis",
        "",
        " ".join(str(temporal["ordering_basis"]).split()),
        "",
        "## Anchors",
        "",
        "anchors_state: " + str(anchors["anchors_state"]),
        "",
        "baseline_metric: absent. target_metric: absent. gap: " + str(anchors["gap"]) + ".",
        "",
        " ".join(str(anchors["schema_bound"]).split()),
        "",
        " ".join(str(anchors["resolution"]).split()),
        "",
        "## Golden trajectory",
        "",
        "| iteration | snapshot asked | action |",
        "|---|---|---|",
    ]
    for row in grounding["golden_trajectory"]:
        marker = " **(pivot)**" if row.get("this_is_the_pivot") else ""
        out.append(
            "| " + str(row["iteration"]) + " | " + str(row["snapshot_asked"]) + " | "
            + " ".join(str(row["action"]).split()) + marker + " |"
        )
    out += ["", "## Declared gaps", ""]
    for row in grounding["gaps_declared"]:
        out.append("- `" + str(row["id"]) + "`: " + " ".join(str(row["why"]).split()))
    out.append("")
    return "\n".join(out)


def test_output_py(grounding: dict, handles: dict, defects: dict) -> str:
    order = [row["id"] for row in grounding["planted_defects"]]
    by_checker = {}
    for row in defects["defects"]:
        by_checker.setdefault(row["checker"], []).append(row["id"])
    checker_ids = []
    for row in defects["defects"]:
        if row["checker"] not in checker_ids:
            checker_ids.append(row["checker"])

    head = '''"""{banner}

The compiled per-checker suite for {slot}. Every function below is the BOTH-HALVES
witness for one checker: it drives that checker over the clean fixture and asserts it
accepts, then over a planted-defect fixture and asserts it rejects with EXACTLY that
checker's zero_reason and that no other checker rejects.

This file is generated from {source}. Edit the source, not this file.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import checkers

CLEAN = json.loads(r\'\'\'{clean}\'\'\')
PLANTED = json.loads(r\'\'\'{planted}\'\'\')

DEFECTS = {{row["id"]: row for row in PLANTED["defects"]}}


def apply_edits(handles, edits):
    payload = json.loads(json.dumps(handles))
    for edit in edits:
        cursor = payload
        for key in edit["path"][:-1]:
            cursor = cursor[key]
        last = edit["path"][-1]
        if edit["op"] == "append":
            cursor[last] = list(cursor[last]) + [edit["value"]]
        else:
            cursor[last] = edit["value"]
    return payload


def accepting_half(name):
    verdict = checkers.SELECTORS[name](CLEAN)
    assert verdict.passed, name + " rejected the clean reference fixture: " + verdict.observed
    return verdict


def rejecting_half(name, defect_id):
    row = DEFECTS[defect_id]
    handles = apply_edits(CLEAN, row["edits"])
    verdict = checkers.SELECTORS[name](handles)
    assert not verdict.passed, name + " accepted planted defect " + defect_id
    assert verdict.zero_reason == row["zero_reason"], (
        name + " emitted " + verdict.zero_reason + " for " + defect_id
        + " where " + row["zero_reason"] + " was declared"
    )
    others = [
        other.checker
        for other in checkers.run_all(handles)
        if not other.passed and other.checker != name
    ]
    assert not others, defect_id + " also fired " + ", ".join(others)
    return verdict

'''.format(
        banner=BANNER_LINE,
        slot=grounding["slot"],
        source=SOURCE,
        clean=json.dumps(handles, sort_keys=True, indent=2),
        planted=json.dumps(defects, sort_keys=True, indent=2),
    )

    body = []
    for checker in checker_ids:
        ids = [name for name in order if name in by_checker[checker]]
        body.append("def test_" + checker + "():")
        body.append("    accepting_half(" + repr(checker) + ")")
        for defect_id in ids:
            body.append("    rejecting_half(" + repr(checker) + ", " + repr(defect_id) + ")")
        body.append("")
        body.append("")

    tail = '''TESTS = [value for name, value in sorted(globals().items()) if name.startswith("test_")]


def main():
    for test in TESTS:
        test()
    print("OER-11 compiled suite: " + str(len(TESTS)) + " checker(s), both halves each")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    return head + "\n" + "\n".join(body) + tail


def write(path: Path, text: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def build() -> dict:
    grounding = load_grounding()
    handles = clean_handles(grounding)
    defects = planted(grounding)
    return {
        "grounding": grounding,
        "handles": handles,
        "defects": defects,
        "artifacts": {
            "solution/fixtures/clean_handles.json": json.dumps(handles, sort_keys=True, indent=2) + "\n",
            "solution/fixtures/planted.json": json.dumps(defects, sort_keys=True, indent=2) + "\n",
            "solution/golden_trajectory.json": json.dumps(golden_trajectory(grounding), sort_keys=True, indent=2) + "\n",
            "solution/rubrics.json": json.dumps(rubrics_json(grounding), sort_keys=True, indent=2) + "\n",
            "solution/solve.sh": solve_sh(grounding),
            "solution/TRUTH.md": truth_md(grounding),
            "tests/test_output.py": test_output_py(grounding, handles, defects),
        },
    }


def main(argv: list) -> int:
    result = build()
    changed = []
    for relative, text in sorted(result["artifacts"].items()):
        target = BUNDLE / relative
        if write(target, text):
            changed.append(relative)
    (BUNDLE / "solution" / "solve.sh").chmod(0o755)
    print(
        json.dumps(
            {
                "source": SOURCE,
                "artifacts": sorted(result["artifacts"]),
                "changed": changed,
                "deterministic": "no model, no network, no clock, no locale, no random source",
            },
            sort_keys=True,
            indent=2,
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
    raise SystemExit(main(sys.argv[1:]))
