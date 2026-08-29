"""Private recomputation script for bia slot S06.

Every generated canonical artifact in this bundle descends from solution/grounding.yaml
through this script and through nothing else. It derives solution/TRUTH.md, the
reference based rubric at solution/rubrics.json, and the compiled deterministic rubric
tests at tests/test_output.py. It reads no clock, no locale, no network and no random
source, and it invokes no model, so two hosts produce byte identical output.

Usage:
  python3 solution/recompute.py --generate        rewrite the three generated artifacts
  python3 solution/recompute.py --check           regenerate into memory and diff
  python3 solution/recompute.py --hash            print canonical_content_hash and uuid
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)
GROUNDING = os.path.join(HERE, "grounding.yaml")
FORGE_TASK_NAMESPACE = uuid.UUID("c53e8f3b-526f-52c0-a04e-89e2269b237d")
BANNER = "GENERATED SECTION. DO NOT HAND-EDIT."
SOURCE_LINE = "Source of truth: solution/grounding.yaml. Generator: solution/recompute.py."

KINDS = ["VALUE", "EFFECT", "ABSENCE", "INVARIANT", "ORDERING", "DIVERGENCE"]

# Frozen mapping from compiled rubric item id to its deterministic assertion body.
# The bodies read committed bundle bytes only. No criterion prose and no reference
# text is carried into the generated test module.
COMPILED_BODIES = {
    "checker_roster_reconciles": """
    declared = {c["id"] for c in _checkers()}
    src = _read("tests/grade.py")
    block = src.split("CHECKER_IDS = [", 1)[1].split("]", 1)[0]
    executed = set(re.findall(r'"([a-z0-9_]+)"', block))
    assert declared == executed, (declared ^ executed)
""",
    "checker_kinds_are_singleton": """
    for c in _checkers():
        assert len(c["kinds"]) == 1, (c["id"], c["kinds"])
        assert c["kinds"][0] in KINDS, (c["id"], c["kinds"])
""",
    "checker_kinds_cover_six": """
    seen = {c["kinds"][0] for c in _checkers()}
    assert seen == set(KINDS), sorted(set(KINDS) - seen)
""",
    "each_checker_names_live_state_read": """
    for c in _checkers():
        assert c["live_state_read"].strip(), c["id"]
""",
    "reward_clamped_to_unit_interval": """
    assert "min(max(raw, 0), 1)" in _read("tests/checkers.yaml")
    grade = _read("tests/grade.py")
    assert "min(max(raw, 0.0), 1.0)" in grade
""",
    "rubrics_jsonl_shape": """
    lines = [ln for ln in _read("tests/rubrics.jsonl").splitlines() if ln.strip()]
    assert lines
    for ln in lines:
        obj = json.loads(ln)
        assert isinstance(obj, dict)
        assert set(obj) == {"id", "rubric"}, set(obj)
""",
    "optimization_bindings_present": """
    import tomllib
    with open(os.path.join(BUNDLE, "task.toml"), "rb") as fh:
        doc = tomllib.load(fh)
    opt = doc["optimization"]
    assert float(opt["max_timeout"]) == 6.0
    assert float(opt["budget_hours"]) == 0.12
    assert int(opt["max_attempts"]) == 50
    assert opt["final_selection"] == "best"
    assert opt["solver_egress"] == "setup-only"
    assert opt["compute_envelope"] == "one-h100"
    rw = opt["reward"]
    assert rw["type"] == "float"
    assert list(rw["range"]) == [0.0, 1.0]
    assert rw["clamp"] == "min(max(raw, 0), 1)"
    assert float(opt["budget_hours"]) < float(opt["max_timeout"])
""",
    "frozen_fixture_digests_bound": """
    bound = _bound()
    assert bound["noise_fixture_sha256"] == _sha("environment/fixtures/noise_process.py")
    assert bound["corpus_sha256"] == _sha("environment/fixtures/corpus.bin")
""",
    "zero_score_has_machine_reason": """
    grade = _read("tests/grade.py")
    calls = re.findall(r"emit\\(\\s*0\\.0\\s*,\\s*([^,]+),", grade)
    assert calls
    for arg in calls:
        arg = arg.strip()
        assert arg and arg not in ('""', "''", "None")
    sh = _read("tests/test.sh")
    assert '"reason"' in sh
""",
    "smoke_path_is_cpu_only": """
    solve = _read("solution/solve.sh")
    assert "CUDA_VISIBLE_DEVICES" in solve
    assert "BIA_SMOKE" in solve
    grade = _read("tests/grade.py")
    assert 'env["CUDA_VISIBLE_DEVICES"] = ""' in grade
    train = _read("environment/harness/train.py")
    assert "modelmod.SMOKE" in train
    assert 'torch.device("cpu")' in train
""",
    "no_em_dash_in_markdown": """
    bad = []
    for root, _dirs, files in os.walk(BUNDLE):
        for name in files:
            if not name.endswith(".md"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as fh:
                if "\\u2014" in fh.read():
                    bad.append(os.path.relpath(path, BUNDLE))
    assert not bad, bad
""",
    "estimator_contract_documented": """
    text = _read("instruction.md")
    assert "build_estimator" in text
    assert "estimate" in text
    assert "submission/estimator.py" in text
""",
}


def load_grounding():
    with open(GROUNDING, encoding="utf-8") as fh:
        return json.load(fh)


def render_truth(g) -> str:
    lines = []
    lines.append("# TRUTH: bia slot S06, gradient estimator under injected noise")
    lines.append("")
    lines.append(BANNER)
    lines.append("")
    lines.append(SOURCE_LINE)
    lines.append("")
    lines.append("## What is frozen and what is free")
    lines.append("")
    lines.append("Frozen surface, in full, because a solver that edits any of it is not solving this task.")
    lines.append("")
    for item in g["frozen_surface"]:
        lines.append("- " + item)
    lines.append("")
    lines.append("Free surface, in full, because a task that leaves nothing free measures nothing.")
    lines.append("")
    for item in g["free_surface"]:
        lines.append("- " + item)
    lines.append("")
    lines.append("## The operating point")
    lines.append("")
    op = g["operating_point"]
    lines.append(op["why_scaled"] + ". The declared point is " + op["name"] + " with budget_hours "
                 + str(op["budget_hours"]) + " per attempt, max_timeout " + str(op["max_timeout_hours"])
                 + " hours across the refinement loop, and max_attempts " + str(op["max_attempts"])
                 + ". One graded attempt runs " + str(op["graded_arms_per_attempt"])
                 + " training arms: " + op["shape"] + ".")
    lines.append("")
    lines.append("Reference score at this operating point is " + op["scaled_measurement_state"]
                 + ". This bundle states no reference score it has not measured. A value offered"
                 + " in place of that measurement would be the exact defect this project names"
                 + " unverifiable-recorded-value.")
    lines.append("")
    lines.append("## The reward")
    lines.append("")
    r = g["reward"]
    lines.append("Raw reward is " + r["raw"] + ", and the emitted reward is " + r["clamp"]
                 + ". The target loss is " + r["target_loss"] + ". Baseline steps is " + r["baseline_steps"]
                 + ". Target steps is " + r["target_steps"] + ". " + r["no_unmeasured_anchor"] + ".")
    lines.append("")
    lines.append("## Ordered golden path")
    lines.append("")
    lines.append("Each step names the action, the state it establishes, the pressure it must survive, and the checker identifier it satisfies.")
    lines.append("")
    for i, s in enumerate(g["truth_steps"], start=1):
        lines.append("### Step " + str(i) + ", checker " + s["checker"])
        lines.append("")
        lines.append(s["action"])
        lines.append("")
        lines.append("State established: " + s["state"] + ". Survives: " + s["survives"] + ".")
        lines.append("")
    lines.append("## Checker reconciliation")
    lines.append("")
    lines.append("The checker identifiers named by the golden path equal the committed checker set exactly, in both directions.")
    lines.append("")
    for c in g["checkers"]:
        lines.append("- " + c["id"] + " reduces to " + c["kind"])
    lines.append("")
    lines.append("## Rejected routes")
    lines.append("")
    lines.append("Each rejected route names the known wrong control that stands for it. The controls are the negative half of the feasibility bundle, and a route with no control would be an author's assertion rather than a measured rejection.")
    lines.append("")
    for rr in g["rejected_routes"]:
        lines.append("- " + rr["route"] + " Control " + rr["control"] + ". " + rr["why"] + ".")
    lines.append("")
    lines.append("## The injected process and why it is frozen")
    lines.append("")
    nf = g["noise_fixture"]
    lines.append("The fixture at " + nf["path"] + " carries seed " + str(nf["seed"]) + " and sha256 "
                 + nf["sha256"] + ". " + nf["why_it_is_frozen"] + ".")
    lines.append("")
    for comp in nf["components"]:
        lines.append("- " + comp)
    lines.append("")
    lines.append("## Design intent, and what was measured")
    lines.append("")
    for sentence in g["design_intent"]:
        lines.append(sentence)
        lines.append("")
    return "\n".join(lines) + "\n"


def render_rubrics(g) -> str:
    items = []
    total = 0.0
    compiled = 0.0
    for it in g["rubric_items"]:
        entry = {
            "id": it["id"],
            "dimension": it["dimension"],
            "weight": it["weight"],
            "evaluation_target": it["evaluation_target"],
            "criterion": it["criterion"],
            "judgment": it["judgment"],
            "evidence": list(it["evidence"]),
            "mode": it["mode"],
        }
        if it["mode"] == "judged":
            entry["residue"] = it["residue"]
        else:
            entry["outcome_kind"] = "VALUE"
        items.append(entry)
        total += float(it["weight"])
        if it["mode"] == "compiled":
            compiled += float(it["weight"])
    doc = {
        "_generated_by": "solution/recompute.py from solution/grounding.yaml",
        "_banner": BANNER,
        "slot": g["slot"],
        "compilation_floor": g["compilation_floor"],
        "compiled_weight_share": round(compiled / total, 6),
        "evaluation_target_vocabulary": ["bundle_bytes", "agent_trajectory"],
        "items": items,
    }
    return json.dumps(doc, indent=1, sort_keys=True) + "\n"


def render_test_output(g) -> str:
    out = []
    out.append('"""Compiled deterministic rubric tests for bia slot S06.')
    out.append("")
    out.append(BANNER)
    out.append("")
    out.append(SOURCE_LINE)
    out.append("")
    out.append("One test per compiled rubric item. Each reads committed bundle bytes only and")
    out.append("carries no criterion prose and no reference text.")
    out.append('"""')
    out.append("")
    out.append("import hashlib")
    out.append("import json")
    out.append("import os")
    out.append("import re")
    out.append("")
    out.append("TESTS_DIR = os.path.dirname(os.path.abspath(__file__))")
    out.append('BUNDLE = os.environ.get("BIA_BUNDLE_SOURCE") or os.path.dirname(TESTS_DIR)')
    out.append("KINDS = " + repr(KINDS))
    out.append("")
    out.append("")
    out.append("def _read(rel):")
    out.append('    with open(os.path.join(BUNDLE, rel), encoding="utf-8") as fh:')
    out.append("        return fh.read()")
    out.append("")
    out.append("")
    out.append("def _sha(rel):")
    out.append('    with open(os.path.join(BUNDLE, rel), "rb") as fh:')
    out.append("        return hashlib.sha256(fh.read()).hexdigest()")
    out.append("")
    out.append("")
    out.append("def _checkers():")
    out.append('    text = _read("tests/checkers.yaml")')
    out.append('    body = text.split("checkers:", 1)[1].split("\\nreward:", 1)[0]')
    out.append("    blocks = []")
    out.append('    for chunk in body.split("  - id:"):')
    out.append("        chunk = chunk.strip()")
    out.append("        if not chunk:")
    out.append("            continue")
    out.append("        lines = chunk.splitlines()")
    out.append("        entry = {")
    out.append('            "id": lines[0].strip(),')
    out.append('            "kinds": [ln.split(":", 1)[1].strip() for ln in lines if ln.strip().startswith("reduction:")],')
    out.append('            "live_state_read": "".join(')
    out.append('                ln.split(":", 1)[1] for ln in lines if ln.strip().startswith("live_state_read:")),')
    out.append("        }")
    out.append("        blocks.append(entry)")
    out.append("    return blocks")
    out.append("")
    out.append("")
    out.append("def _bound():")
    out.append('    text = _read("tests/checkers.yaml")')
    out.append("    out = {}")
    out.append("    for line in text.splitlines():")
    out.append('        if ":" not in line:')
    out.append("            continue")
    out.append('        key, _, val = line.strip().partition(":")')
    out.append('        if key.strip() in ("noise_fixture_sha256", "corpus_sha256", "identity_estimator_digest"):')
    out.append('            out[key.strip()] = val.strip().strip(\'"\')')
    out.append("    return out")
    for it in g["rubric_items"]:
        if it["mode"] != "compiled":
            continue
        body = COMPILED_BODIES[it["id"]]
        out.append("")
        out.append("")
        out.append("def test_" + it["id"] + "():")
        out.append(body.strip("\n"))
    out.append("")
    return "\n".join(out) + "\n"


TARGETS = [
    ("solution/TRUTH.md", render_truth),
    ("solution/rubrics.json", render_rubrics),
    ("tests/test_output.py", render_test_output),
]


def canonical_content_hash(root=BUNDLE):
    lines = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if rel.split("/")[0] == "trajectories":
                continue
            if rel == "solution/provenance.yaml":
                continue
            with open(full, "rb") as fh:
                lines.append("%s  %s\n" % (rel, hashlib.sha256(fh.read()).hexdigest()))
    lines.sort()
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--hash", action="store_true")
    ap.add_argument("--root", default=BUNDLE)
    args = ap.parse_args(argv)

    if args.hash:
        h = canonical_content_hash(args.root)
        print(json.dumps({
            "canonical_content_hash": h,
            "uuid": str(uuid.uuid5(FORGE_TASK_NAMESPACE, h)),
        }, sort_keys=True))
        return 0

    g = load_grounding()
    rc = 0
    for rel, fn in TARGETS:
        text = fn(g)
        path = os.path.join(args.root, rel)
        if args.generate:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
            print("wrote " + rel)
        else:
            current = ""
            if os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    current = fh.read()
            if current != text:
                print("DRIFT " + rel)
                rc = 1
            else:
                print("ok " + rel)
    return rc


if __name__ == "__main__":
    sys.exit(main())
