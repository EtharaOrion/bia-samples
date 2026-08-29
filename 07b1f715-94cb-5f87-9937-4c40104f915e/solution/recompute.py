#!/usr/bin/env python3
"""Derive every generated canonical artifact of this bundle from solution/grounding.yaml.

GENERATOR. Reads solution/grounding.yaml and nothing else. It never invokes a
model, a network, a clock, a locale or a random source, so two runs on two hosts
produce byte-identical output.

Generated artifacts:
    tests/anchors.py          the anchor and frozen-field constants the grader imports
    tests/checkers.yaml       the checker reduction table
    tests/rubrics.jsonl       the trajectory rubrics, one JSON object per line
    tests/test_output.py      the compiled deterministic rubric tests
    solution/rubrics.json     the reference-based rubric
    solution/TRUTH.md         the human-readable golden trajectory

Also:
    --hash                    print the canonical content hash and the derived uuid
    --check                   regenerate into memory and report any drift from disk
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUNDING = os.path.join(ROOT, "solution", "grounding.yaml")
FORGE_TASK_NAMESPACE = uuid.UUID("c53e8f3b-526f-52c0-a04e-89e2269b237d")
HASH_EXCLUDED_DIRS = ("trajectories",)
HASH_EXCLUDED_FILES = ("solution/provenance.yaml",)
BANNER = "GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml."


def load():
    with open(GROUNDING) as f:
        return yaml.safe_load(f)


def one_line(text: str) -> str:
    return " ".join(str(text).split())


def _literal(obj, indent=0) -> str:
    """Deterministic Python literal for a plain nested mapping."""
    pad = " " * indent
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        rows = ["%s    %s: %s," % (pad, json.dumps(k), _literal(obj[k], indent + 4))
                for k in sorted(obj)]
        return "{\n" + "\n".join(rows) + "\n" + pad + "}"
    if isinstance(obj, bool):
        return "True" if obj else "False"
    if obj is None:
        return "None"
    if isinstance(obj, (int, float)):
        return json.dumps(obj)
    return json.dumps(obj)


def gen_anchors(g) -> str:
    profiles = g["profiles"]
    anchors = g["anchors"]
    arch = {}
    batch = {}
    train = {}
    val = {}
    prof_anchor = {}
    for name in sorted(profiles):
        p = profiles[name]
        arch[name] = {
            "n_layer": p["n_layer"],
            "n_head": p["n_head"],
            "d_model": p["d_model"],
            "seq_len": p["seq_len"],
            "vocab_size": 50304,
            "norm": "rmsnorm",
            "mlp": "gelu_4x",
            "pos": "learned",
            "head_tied": False,
        }
        batch[name] = p["batch_sequences"]
        train[name] = p["train_shard_digest"]
        val[name] = p["val_shard_digest"]
        a = anchors[name]
        prof_anchor[name] = {
            "target_loss": a["target_loss"],
            "baseline_steps": a["baseline_steps"],
            "target_steps": a["target_steps"],
            "min_seeds": a["min_seeds"],
            "sig_margin": a["sig_margin"],
            "status": a["status"],
        }
    lines = [
        '"""%s' % BANNER,
        "",
        "Anchor and frozen-field constants the S02 verifier grades against. The full",
        "profile anchors carry status %s, and the runs they were" % anchors["full"]["status"],
        "bound from are recorded in solution/grounding.yaml under",
        "calibration_measurement.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "PROFILE_ANCHORS = %s" % _literal(prof_anchor),
        "",
        "ARCHITECTURE = %s" % _literal(arch),
        "",
        "BATCH_SEQUENCES = %s" % _literal(batch),
        "",
        "TRAIN_SHARD = %s" % _literal(train),
        "",
        "VAL_SHARD = %s" % _literal(val),
        "",
    ]
    return "\n".join(lines)


def gen_checkers_yaml(g) -> str:
    contract = g["verifier_contract"]
    out = [
        "# %s" % BANNER,
        "schema: forge.checkers/v1",
        "reduction_kinds: [VALUE, EFFECT, ABSENCE, INVARIANT, ORDERING, DIVERGENCE]",
        "note: >-",
        "  Every checker below reduces to exactly one reduction kind, names the live-state read",
        "  it traces to, names the reachable file that carries it and the selector inside that",
        "  file, and names the machine-readable reason tests/grade.py emits alongside the zero",
        "  this checker can cause. The outcome key is the entry the verifier writes into",
        "  /logs/verifier/outcomes.json, and tests/test_output.py asserts it.",
        "reward_path: %s" % contract["reward_path"],
        "aggregation:",
        "  mode: %s" % contract["aggregation_mode"],
        "  note: >-",
        "    %s" % one_line(contract["note"]),
        "checkers:",
    ]
    for c in g["checkers"]:
        out.append("  - id: %s" % c["id"])
        out.append("    name: %s" % c["name"])
        out.append("    reduction: %s" % c["kind"])
        out.append("    outcome_key: %s" % c["name"])
        out.append("    reached_by: %s" % c["reached_by"])
        out.append("    selector: %s" % c["selector"])
        out.append("    zero_reason: %s" % c["zero_reason"])
        out.append("    live_state_read: >-")
        out.append("      %s" % one_line(c["live_state_read"]))
        out.append("    asserts: >-")
        out.append("      %s" % one_line(c["asserts"]))
        out.append("    gate: hard_pass")
        out.append("    required: true")
    out.append("")
    return "\n".join(out)


def gen_rubrics_jsonl(g) -> str:
    lines = []
    for item in g["trajectory_rubrics"]:
        lines.append(json.dumps({"id": item["id"], "rubric": one_line(item["rubric"])},
                                sort_keys=True))
    return "\n".join(lines) + "\n"


def gen_reference_rubric(g) -> str:
    block = g["reference_rubric"]
    items = []
    for item in block["items"]:
        record = {
            "id": item["id"],
            "dimension": item["dimension"],
            "weight": item["weight"],
            "evaluation_target": item["evaluation_target"],
            "criterion": one_line(item["criterion"]),
            "judgment": one_line(item["judgment"]),
            "evidence": [item["evidence"]],
            "mode": item["mode"],
        }
        if item["mode"] == "compiled":
            record["outcome_kind"] = item["outcome_kind"]
        else:
            record["residue"] = one_line(item["residue"])
        items.append(record)
    total = sum(i["weight"] for i in items)
    compiled = sum(i["weight"] for i in items if i["mode"] == "compiled")
    payload = {
        "generated_from": "solution/grounding.yaml",
        "banner": BANNER,
        "schema": "bia.s02.rubric/v1",
        "compilation_floor": block["compilation_floor"],
        "compiled_weight_share": round(compiled / float(total), 6),
        "evaluation_targets": list(block["evaluation_targets"]),
        "items": items,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def gen_test_output(g) -> str:
    compiled = [i for i in g["reference_rubric"]["items"] if i["mode"] == "compiled"]
    by_outcome = {c["name"]: c for c in g["checkers"]}
    lines = [
        '"""%s' % BANNER,
        "",
        "One test per compiled rubric item. Each asserts the outcome entry the verifier",
        "wrote after reading live state, so a test cannot pass on an unrun checker.",
        '"""',
        "",
        "import os",
        "import sys",
        "",
        "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))",
        "",
        "from checkers import outcomes",
        "",
        "",
        "def _o():",
        "    return outcomes.load()",
        "",
    ]
    for item in compiled:
        key = item["judgment"].split(" is true")[0].strip()
        checker = by_outcome[key]
        lines.append("")
        lines.append("def test_%s_%s():" % (item["id"].lower(), key))
        lines.append('    """%s | %s | %s"""' % (item["id"], checker["kind"], one_line(item["criterion"])))
        lines.append("    assert _o()['%s'] is True" % key)
        lines.append("")
    return "\n".join(lines)


def gen_truth(g) -> str:
    task = g["task"]
    op = g["operating_point"]
    anchors_full = g["anchors"]["full"]
    parts = []
    parts.append("# TRUTH: %s" % task["name"])
    parts.append("")
    parts.append(BANNER)
    parts.append("")
    parts.append("## What this task is")
    parts.append("")
    parts.append(one_line(
        "Slot %s, %s. The learning-rate schedule is frozen, published at "
        "environment/frozen_schedule.py and unmodifiable. The update rule is the only free axis. "
        "The reward is the earliest sustained multi-seed crossing of the target validation loss, "
        "mapped onto the closed interval from zero to one by the spec scoring form."
        % (task["slot"], task["slot_name"])))
    parts.append("")
    parts.append("## Frozen surface")
    parts.append("")
    for row in g["frozen_surface"]:
        parts.append("- %s" % one_line(row))
    parts.append("")
    parts.append("## Free surface")
    parts.append("")
    for row in g["free_surface"]:
        parts.append("- %s" % one_line(row))
    parts.append("")
    parts.append("## Defeat mechanism, stated as design intent and marked UNMEASURED")
    parts.append("")
    parts.append(one_line(g["defeat_mechanism"]["design_intent"]))
    parts.append("")
    parts.append(one_line(
        "Status: %s. Measured only by %s. No pass probability, no self-solve claim and no "
        "difficulty tier is asserted anywhere in this bundle."
        % (g["defeat_mechanism"]["status"], g["defeat_mechanism"]["measured_by"])))
    parts.append("")
    parts.append("## Operating point")
    parts.append("")
    parts.append(one_line(op["binding"]))
    parts.append("")
    parts.append("| Field | Value |")
    parts.append("|---|---|")
    parts.append("| budget_hours | %s |" % op["budget_hours"])
    parts.append("| max_timeout hours | %s |" % op["max_timeout_hours"])
    parts.append("| max_attempts | %s |" % op["max_attempts"])
    parts.append("| envelope | %s |" % op["envelope"])
    parts.append("| wall clock | %s |" % op["wallclock_status"])
    parts.append("")
    parts.append(one_line(op["enforcement"]))
    parts.append("")
    parts.append(one_line(op["wallclock_note"]))
    parts.append("")
    parts.append("## Calibration sweep the anchors are bound from")
    parts.append("")
    cal = g["calibration_measurement"]
    parts.append(one_line(
        "Measured %s on %s. %s" % (cal["date"], one_line(cal["host"]), one_line(cal["grader_rule"]))))
    parts.append("")
    parts.append("| Id | Rule | Seeds | Steps run | Observed |")
    parts.append("|---|---|---|---|---|")
    for run in cal["runs"]:
        parts.append("| %s | %s | %s | %s | %s |" % (
            run["id"], one_line(run["rule"]), ", ".join(str(s) for s in run["seeds"]),
            run["steps_run"], one_line(run["observed"])))
    parts.append("")
    parts.append(one_line(cal["conclusion"]))
    parts.append("")
    parts.append("## Measured reference at the graded operating point")
    parts.append("")
    mr = g["measured_reference"]
    parts.append("| Field | Value |")
    parts.append("|---|---|")
    for key in ("command", "profile", "date", "device", "steps_run", "stop_reason",
                "graded_step", "score", "checkers_true", "repeats", "score_across_repeats",
                "wallclock_two_seed_total_s", "wallclock_slowest_invocation_per_repeat_s",
                "wallclock_best_invocation_s"):
        parts.append("| %s | %s |" % (key, one_line(mr[key])))
    parts.append("")
    parts.append(one_line(mr["reading"]))
    parts.append("")
    parts.append(one_line(mr["budget_caveat"]))
    parts.append("")
    parts.append(one_line(mr["smoke_parity"]))
    parts.append("")
    parts.append("## Golden trajectory")
    parts.append("")
    parts.append("| Step | Action | Established state | Checkers |")
    parts.append("|---|---|---|---|")
    for step in g["truth_steps"]:
        parts.append("| %s | %s | %s | %s |" % (
            step["id"], one_line(step["action"]), one_line(step["established_state"]),
            ", ".join(step["checkers"])))
    parts.append("")
    parts.append("## Rejected routes and the control that rejects each")
    parts.append("")
    parts.append("| Route | Negative control | Rejected by |")
    parts.append("|---|---|---|")
    for row in g["rejected_routes"]:
        parts.append("| %s | %s | %s |" % (one_line(row["route"]), row["control"], row["rejected_by"]))
    parts.append("")
    parts.append("## Checkers and their reduction")
    parts.append("")
    parts.append("| Id | Name | Kind | Live-state read |")
    parts.append("|---|---|---|---|")
    for c in g["checkers"]:
        parts.append("| %s | %s | %s | %s |" % (c["id"], c["name"], c["kind"], one_line(c["live_state_read"])))
    parts.append("")
    parts.append("## Anchors, and what is not established about them")
    parts.append("")
    parts.append(one_line(anchors_full["note"]))
    parts.append("")
    parts.append("| Anchor | Value | Status |")
    parts.append("|---|---|---|")
    for key in ("target_loss", "baseline_steps", "target_steps", "min_seeds", "sig_margin"):
        parts.append("| %s | %s | %s |" % (key, anchors_full[key], anchors_full["status"]))
    parts.append("")
    parts.append("## Negative controls, both halves proven")
    parts.append("")
    parts.append(one_line(g["negative_controls"]["note"]))
    parts.append("")
    parts.append("| Control | Planted defect | Exercises | Observed |")
    parts.append("|---|---|---|---|")
    for row in g["negative_controls"]["controls"]:
        parts.append("| %s | %s | %s | %s |" % (
            row["id"], one_line(row["plants"]), row["exercises"], one_line(row["observed"])))
    parts.append("")
    parts.append("## Coverage gaps carried at sign-off")
    parts.append("")
    parts.append("| Id | Reason | Cap |")
    parts.append("|---|---|---|")
    for gap in g["coverage_gaps"]:
        parts.append("| %s | %s | %s |" % (gap["id"], one_line(gap["reason"]), gap["cap"]))
    parts.append("")
    parts.append("## Disposition")
    parts.append("")
    parts.append(one_line(
        "This bundle ships at %s, which is the correct terminal state for a locally verified and "
        "externally unproven task. Only an external signed pilot over these frozen bytes can move "
        "it." % g["disposition"]))
    parts.append("")
    return "\n".join(parts)


ARTIFACTS = {
    "tests/anchors.py": gen_anchors,
    "tests/checkers.yaml": gen_checkers_yaml,
    "tests/rubrics.jsonl": gen_rubrics_jsonl,
    "tests/test_output.py": gen_test_output,
    "solution/rubrics.json": gen_reference_rubric,
    "solution/TRUTH.md": gen_truth,
}


def iter_hash_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in HASH_EXCLUDED_DIRS and d != "__pycache__")
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if rel in HASH_EXCLUDED_FILES:
                continue
            if rel.startswith(tuple(d + "/" for d in HASH_EXCLUDED_DIRS)):
                continue
            if rel.endswith(".pyc"):
                continue
            yield rel, full


def canonical_content_hash(root):
    entries = []
    for rel, full in iter_hash_files(root):
        with open(full, "rb") as f:
            entries.append((rel, hashlib.sha256(f.read()).hexdigest()))
    entries.sort(key=lambda e: e[0].encode("utf-8"))
    manifest = "".join("%s  %s\n" % (digest, rel) for rel, digest in entries)
    return hashlib.sha256(manifest.encode("utf-8")).hexdigest(), entries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--manifest", action="store_true")
    args = ap.parse_args()

    if args.hash or args.manifest:
        digest, entries = canonical_content_hash(ROOT)
        if args.manifest:
            for rel, d in entries:
                print("%s  %s" % (d, rel))
        print("canonical_content_hash: %s" % digest)
        print("uuid: %s" % uuid.uuid5(FORGE_TASK_NAMESPACE, digest))
        return 0

    g = load()
    drift = []
    for rel, fn in sorted(ARTIFACTS.items()):
        body = fn(g)
        path = os.path.join(ROOT, rel)
        if args.check:
            current = open(path).read() if os.path.exists(path) else None
            if current != body:
                drift.append(rel)
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(body)
        print("wrote %s" % rel)
    if args.check:
        if drift:
            print("DRIFT: %s" % ", ".join(drift))
            return 1
        print("no drift across %d generated artifacts" % len(ARTIFACTS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
