#!/usr/bin/env python3
"""Derives every generated bundle artifact from solution/grounding.yaml.

Run with no argument to write. Run with --check to verify that the committed
bytes are exactly what this generator produces, which is the property that
keeps the oracle, the human readable truth, the rubric, the compiled tests and
the checker fixtures from ever disagreeing with each other.

The generator reads a frozen literal source and computes digests from the frozen
substrate. It invokes no model, opens no network socket, reads no clock, reads
no locale and draws no random number.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BUNDLE, "environment"))

BANNER = "GENERATED SECTION. DO NOT HAND-EDIT."
SOURCE = "solution/grounding.yaml via solution/recompute.py"

GENERATED = [
    "environment/frozen_manifest.json",
    "tests/corpus/MANIFEST.json",
    "tests/checkers.yaml",
    "tests/rubrics.jsonl",
    "tests/test_output.py",
    "solution/rubrics.json",
    "solution/TRUTH.md",
    "solution/solve.sh",
]


def load_grounding() -> dict:
    with open(os.path.join(HERE, "grounding.yaml")) as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------
# generators
# --------------------------------------------------------------------------


def gen_frozen_manifest(g: dict) -> str:
    import bia_core as core

    cache = os.environ.get("BIA_CACHE", "/tmp/bia_cache")
    scales = {}
    for name in ("full", "smoke"):
        cfg = core.resolve_scale(name)
        corpus = core.build_corpus(cfg, cache)
        model = core.build_model(cfg)
        core.frozen_init(model, 0)
        scales[name] = {
            "arch_signature": core.arch_signature(model),
            "corpus_digest": corpus.digest,
            "val_conditional_loss": round(corpus.cond_loss, 6),
            "val_marginal_loss": round(corpus.uni_loss, 6),
            "target_loss": round(corpus.target_loss, 6),
            "target_closure": cfg["target_closure"],
            "baseline_steps": cfg["baseline_steps"],
            "target_steps": cfg["target_steps"],
            "total_steps": cfg["total_steps"],
            "tokens_per_step": cfg["tokens_per_step"],
            "min_seeds": cfg["min_seeds"],
            "sig_margin": cfg["sig_margin"],
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "anchor_status": g["anchors"][name]["status"],
        }
        # the anchors in bia_core are the single source; grounding must agree
        assert cfg["baseline_steps"] == g["anchors"][name]["baseline_steps"], name
        assert cfg["target_steps"] == g["anchors"][name]["target_steps"], name
    payload = {
        "banner": BANNER,
        "source": SOURCE,
        "schema": "bia.s01/frozen_manifest/v1",
        "baseline_optimizer_sha256": sha256_file(
            os.path.join(BUNDLE, "environment", "baseline_optimizer.py")),
        "core_sha256": sha256_file(os.path.join(BUNDLE, "environment", "bia_core.py")),
        "scales": scales,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def gen_corpus_manifest(g: dict) -> str:
    d = os.path.join(BUNDLE, "tests", "corpus")
    entries = []
    for name in sorted(os.listdir(d)):
        if not name.endswith(".py"):
            continue
        entries.append({"file": name, "sha256": sha256_file(os.path.join(d, name))})
    payload = {
        "banner": BANNER,
        "source": SOURCE,
        "schema": "bia.s01/exclusion_corpus/v1",
        "purpose": (
            "the pinned exclusion set the no_verbatim_record_copy checker compares "
            "a submission against, standing for the published record lineage this "
            "family names as its leakage surface"),
        "similarity_threshold": 0.90,
        "normalization": "python abstract syntax tree dump with docstrings removed",
        "entries": entries,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def split_implemented_by(implemented_by: str) -> tuple:
    carrier_path, carrier_selector = implemented_by.split()
    return carrier_path, carrier_selector


def gen_checkers_yaml(g: dict) -> str:
    r = g["reward"]
    lines = [
        f"# {BANNER}",
        f"# Source: {SOURCE}",
        "schema: forge.checkers/v1",
        "reduction_kinds: [VALUE, EFFECT, ABSENCE, INVARIANT, ORDERING, DIVERGENCE]",
        f"reward_path: {r['path']}",
        f"score_document_path: {r['score_document_path']}",
        "aggregation:",
        f"  mode: {r['aggregation_mode']}",
        f"  rationale: {json.dumps(r['aggregation_rationale'].strip())}",
        "checkers:",
    ]
    for c in g["checkers"]:
        carrier_path, carrier_selector = split_implemented_by(c["implemented_by"])
        lines.append(f"  - id: {c['id']}")
        lines.append(f"    statement: {json.dumps(c['asserts'].strip())}")
        lines.append(f"    reduction: {c['kind']}")
        lines.append(f"    reached_by: {carrier_path}")
        lines.append(f"    selector: {carrier_selector}")
        lines.append(f"    zero_reason: {c['zero_reason']}")
        lines.append(f"    required: {json.dumps(bool(c['required']))}")
        lines.append(f"    weight: {g['compiled_item_weight']}")
        lines.append(f"    live_state_read: {json.dumps(c['live_state_read'].strip())}")
        lines.append(f"    compiled_test: {json.dumps('tests/test_output.py::' + c['compiled_test'])}")
        if c.get("bounded_claim"):
            lines.append(f"    bounded_claim: {json.dumps(c['bounded_claim'].strip())}")
    return "\n".join(lines) + "\n"


def gen_rubrics_jsonl(g: dict) -> str:
    out = []
    for r in g["trajectory_rubrics"]:
        out.append(json.dumps({"id": r["id"], "rubric": r["rubric"].strip()}, sort_keys=True))
    return "\n".join(out) + "\n"


def gen_rubrics_json(g: dict) -> str:
    items = []
    w = g["compiled_item_weight"]
    for c in g["checkers"]:
        items.append({
            "id": c["id"],
            "dimension": "outcome",
            "weight": w,
            "evaluation_target": "verifier_outcomes",
            "criterion": c["asserts"].strip(),
            "judgment": (
                f"The {c['id']} entry of the verifier outcome vector is true, which the "
                f"verifier decides as a {c['kind']} assertion over {c['live_state_read'].strip()}"),
            "evidence": ["verifier_outcomes", "telemetry_record"],
            "mode": "compiled",
            "reduction_kind": c["kind"],
        })
    for r in g["trajectory_rubrics"]:
        items.append({
            "id": r["id"],
            "dimension": "process",
            "weight": r["weight"],
            "evaluation_target": "agent_transcript",
            "criterion": r["rubric"].strip(),
            "judgment": r["rubric"].strip(),
            "evidence": ["agent_transcript"],
            "mode": "judged",
            "residue": r["residue"].strip(),
        })
    total = sum(i["weight"] for i in items)
    compiled = sum(i["weight"] for i in items if i["mode"] == "compiled")
    payload = {
        "banner": BANNER,
        "source": SOURCE,
        "schema": "bia.s01/rubrics/v1",
        "evaluation_target_vocabulary": {
            "telemetry_record": "/telemetry/run_record.jsonl",
            "submission_optimizer": "submission/optimizer.py",
            "submission_logs": "submission/logs/full_seed*.log",
            "verifier_outcomes": "/logs/verifier/outcomes.json",
            "verifier_score": "/logs/verifier/score.json",
            "agent_transcript": "trajectories/<model>/<run>/agent/trajectory.json",
        },
        "compilation_floor": g["compilation_floor"],
        "compiled_weight_share": round(compiled / total, 4),
        "items": items,
    }
    assert compiled / total >= g["compilation_floor"], "compilation floor breached"
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def gen_test_output(g: dict) -> str:
    head = [
        f'"""{BANNER}',
        "",
        f"Source: {SOURCE}.",
        "",
        "One test per compiled rubric item. Each reads the outcome vector the",
        "verifier wrote for the run being graded and nothing else, so a compiled",
        "test can never disagree with the score for the same run.",
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
    body = []
    for c in g["checkers"]:
        body += [
            "",
            f"def {c['compiled_test']}():",
            f"    # reduction kind {c['kind']}",
            f"    assert _o()[{c['id']!r}] is True",
        ]
    return "\n".join(head + body) + "\n"


def gen_truth(g: dict) -> str:
    idn = g["identity"]
    L = []
    L.append(f"# TRUTH: {idn['task_name']}")
    L.append("")
    L.append(BANNER)
    L.append("")
    L.append(f"Source: {SOURCE}.")
    L.append("")
    L.append("## Headline")
    L.append("")
    L.append(g["truth"]["headline"].strip())
    L.append("")
    L.append("## Frozen surface")
    L.append("")
    for f in g["frozen_surface"]:
        L.append(f"- {f}")
    L.append("")
    L.append("## Free surface")
    L.append("")
    for f in g["free_surface"]:
        L.append(f"- {f}")
    L.append("")
    L.append("## Ordered path")
    L.append("")
    for i, s in enumerate(g["truth"]["steps"], 1):
        L.append(f"### Step {i}")
        L.append("")
        L.append(s["action"].strip())
        L.append("")
        L.append(f"Established state: {s['established_state'].strip()}")
        L.append("")
        L.append(f"Survives: {s['survives'].strip()}")
        L.append("")
        L.append(f"Satisfies: {', '.join(s['checkers'])}")
        L.append("")
    L.append("## Rejected routes")
    L.append("")
    L.append("| Route | Rejected by | Known-wrong control |")
    L.append("|---|---|---|")
    for r in g["truth"]["rejected_routes"]:
        L.append(f"| {r['route']} | {r['rejected_by']} | {r['control']} |")
    L.append("")
    L.append("## Reward")
    L.append("")
    L.append(f"{g['reward']['expression']}")
    L.append("")
    L.append(f"Reward is a single float on {g['reward']['range']}, higher is better, and {g['reward']['zero_attribution']}.")
    L.append("")
    L.append("## Anchors")
    L.append("")
    L.append("| Scale | baseline_steps | target_steps | Status |")
    L.append("|---|---|---|---|")
    for name in ("full", "smoke"):
        a = g["anchors"][name]
        L.append(f"| {name} | {a['baseline_steps']} | {a['target_steps']} | {a['status']} |")
    L.append("")
    L.append(g["anchors"]["binding_rule"].strip())
    L.append("")
    L.append("## Measurement")
    L.append("")
    m = g["measurement"]
    L.append(f"Taken on {m['taken_on']}, on {m['machine'].strip()}. Reading convention: {m['sweeps_seeds'].strip()}.")
    L.append("")
    L.append(m["diagnosis"].strip())
    L.append("")
    L.append("| Sweep | Varied | Decided |")
    L.append("|---|---|---|")
    for s in m["sweeps"]:
        L.append(f"| {s['name']} | {s['varied']} | {s['decided']} |")
    L.append("")
    for s in m["sweeps"]:
        L.append(f"Observed in the {s['name']} sweep: {s['observed'].strip()}")
        L.append("")
    r = m["reference_result"]
    L.append(
        f"The frozen baseline recipe crosses at step {m['anchors_bound_here']['baseline_graded_step_full']} "
        f"and the reference crosses at step {r['graded_step_full']}, so solve.sh at the graded full scale "
        f"scores {r['score_full']} in {' and '.join(str(x) for x in r['elapsed_sec_full'])} seconds "
        f"across {r['runs']} runs, against a bound of {r['elapsed_bound_sec']} seconds, with all eleven "
        f"verifier outcomes true. Reproducibility: {r['reproducibility'].strip()}.")
    L.append("")
    L.append(r["note"].strip())
    L.append("")
    sm = m["smoke_result"]
    L.append(
        f"The same call graph on the smoke scale runs on {sm['device']} in {sm['elapsed_sec']} seconds, "
        f"where the baseline crosses at step {sm['baseline_graded_step']}, the reference crosses at step "
        f"{sm['reference_graded_step']}, and the score is {sm['score']}.")
    L.append("")
    L.append("## Defeat mechanism")
    L.append("")
    L.append(g["defeat_mechanism"]["statement"].strip())
    L.append("")
    L.append(f"Status: {g['defeat_mechanism']['status']}. Evidence owed: {g['defeat_mechanism']['evidence_owed']}.")
    L.append("")
    L.append("## Coverage gaps")
    L.append("")
    for gap in g["coverage_gaps"]:
        L.append(f"### {gap['id']}")
        L.append("")
        L.append(gap["statement"].strip())
        L.append("")
        L.append(f"Cap: {gap['cap']}. Closes when: {gap['closes_when'].strip()}")
        L.append("")
    L.append("## Checker set")
    L.append("")
    L.append("| Checker | Kind | Compiled test |")
    L.append("|---|---|---|")
    for c in g["checkers"]:
        L.append(f"| {c['id']} | {c['kind']} | tests/test_output.py::{c['compiled_test']} |")
    L.append("")
    return "\n".join(L)


def gen_solve(g: dict) -> str:
    return g["solve_script"]


GENERATORS = {
    "environment/frozen_manifest.json": gen_frozen_manifest,
    "tests/corpus/MANIFEST.json": gen_corpus_manifest,
    "tests/checkers.yaml": gen_checkers_yaml,
    "tests/rubrics.jsonl": gen_rubrics_jsonl,
    "tests/test_output.py": gen_test_output,
    "solution/rubrics.json": gen_rubrics_json,
    "solution/TRUTH.md": gen_truth,
    "solution/solve.sh": gen_solve,
}


# --------------------------------------------------------------------------
# canonical content hash
# --------------------------------------------------------------------------

HASH_EXCLUDED_DIRS = ("trajectories",)
HASH_EXCLUDED_FILES = ("solution/provenance.sig",)
BINDING_PLACEHOLDER = b"binding: PENDING_BINDING_BLOCK\n"


def normalize_for_hash(rel: str, data: bytes) -> bytes:
    """Replaces the later-applied binding block with a fixed placeholder.

    The binding block is written after the hash exists, so it must sit outside
    the hash preimage or the binding would be cyclic. Every other byte of the
    carrier, and every byte of every other file, stays inside the domain.
    """
    if rel != "solution/provenance.yaml":
        return data
    marker = b"\nbinding:\n"
    idx = data.find(marker)
    if idx == -1:
        if data.endswith(BINDING_PLACEHOLDER):
            return data
        raise SystemExit("provenance.yaml carries neither a binding block nor its placeholder")
    return data[:idx + 1] + BINDING_PLACEHOLDER


def hash_domain() -> list:
    out = []
    for root, dirs, files in os.walk(BUNDLE):
        dirs[:] = sorted(d for d in dirs
                         if os.path.relpath(os.path.join(root, d), BUNDLE) not in HASH_EXCLUDED_DIRS
                         and d not in ("__pycache__",))
        for name in sorted(files):
            rel = os.path.relpath(os.path.join(root, name), BUNDLE).replace(os.sep, "/")
            if rel in HASH_EXCLUDED_FILES or rel.split("/")[0] in HASH_EXCLUDED_DIRS:
                continue
            if rel.endswith(".pyc"):
                continue
            out.append(rel)
    return sorted(out)


def canonical_content_hash() -> str:
    h = hashlib.sha256()
    h.update(b"bia.s01/canonical/v1\n")
    for rel in hash_domain():
        with open(os.path.join(BUNDLE, rel), "rb") as fh:
            data = normalize_for_hash(rel, fh.read())
        h.update(len(rel).to_bytes(4, "big"))
        h.update(rel.encode("utf-8"))
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    return h.hexdigest()


def bundle_uuid(content_hash: str) -> str:
    import uuid
    return str(uuid.uuid5(uuid.UUID("c53e8f3b-526f-52c0-a04e-89e2269b237d"), content_hash))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--hash", action="store_true")
    args = ap.parse_args()
    if args.hash:
        ch = canonical_content_hash()
        print(json.dumps({
            "canonical_content_hash": ch,
            "uuid": bundle_uuid(ch),
            "files_hashed": len(hash_domain()),
        }, indent=2, sort_keys=True))
        return 0
    g = load_grounding()
    bad = []
    for rel in GENERATED:
        text = GENERATORS[rel](g)
        path = os.path.join(BUNDLE, rel)
        if args.check:
            current = open(path).read() if os.path.exists(path) else None
            if current != text:
                bad.append(rel)
                print(f"DRIFT {rel}")
            else:
                print(f"ok    {rel}")
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as fh:
                fh.write(text)
            if rel.endswith(".sh"):
                os.chmod(path, 0o755)
            print(f"wrote {rel}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
