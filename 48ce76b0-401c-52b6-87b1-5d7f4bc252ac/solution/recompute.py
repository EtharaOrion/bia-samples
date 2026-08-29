"""Regenerate every generated artifact of this bundle from solution/grounding.yaml.

This script is the single derivation source. It invokes no model, no network, no clock, no
locale and no random source, so running it twice under different host identities produces
byte-identical output. Running it is how an auditor proves the ground truth, the rubric and
the compiled tests cannot disagree with each other or with the frozen fixture.

Usage: python3 solution/recompute.py [--check]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)
RUNNER = os.path.join(BUNDLE, "environment", "runner")
if RUNNER not in sys.path:
    sys.path.insert(0, RUNNER)

KINDS = ("VALUE", "EFFECT", "ABSENCE", "INVARIANT", "ORDERING", "DIVERGENCE")

TEST_BODIES = {
    "checker_registry_covers_six_kinds": '''
def test_checker_registry_covers_six_kinds():
    import re
    from checkers import kinds as kinds_mod
    carrier = open(os.path.join(BUNDLE, "tests", "checkers", "kinds.py"), encoding="utf-8").read()
    reg = load_checker_registry()
    assert reg, "tests/checkers.yaml declared no checkers"
    ids = [c["id"] for c in reg]
    assert len(set(ids)) == len(ids), "duplicate checker identifier"
    for c in reg:
        assert c["reduction"] in KINDS, (c["id"], c["reduction"])
        assert c["live_state_read"].strip(), c["id"]
        assert c["reached_by"] == "tests/checkers/kinds.py", c["id"]
        assert ("def " + c["selector"] + "(") in carrier, c["id"]
        assert re.match(r"^[a-z0-9]+(?:-[a-z0-9]+)*$", c["zero_reason"]), c["id"]
        assert kinds_mod.ZERO_REASON[c["id"]] == c["zero_reason"], c["id"]
''',
    "reward_is_bounded_float": '''
def test_reward_is_bounded_float(tmp_path):
    grade = load_grader(tmp_path)
    for raw, want in ((0.42, 0.42), (1.7, 1.0), (-0.3, 0.0)):
        grade.emit(raw, "probe")
        got = json.loads(open(grade.SCORE_PATH).read())
        assert isinstance(got["score"], float)
        assert 0.0 <= got["score"] <= 1.0
        assert abs(got["score"] - want) < 1e-12
''',
    "zero_score_is_attributed": '''
def test_zero_score_is_attributed(tmp_path):
    grade = load_grader(tmp_path)
    grade.emit(0.0, "")
    got = json.loads(open(grade.SCORE_PATH).read())
    assert got["score"] == 0.0
    assert got["reason"].strip(), "an unattributed zero reached the reward path"
    grade.emit(0.0, "improvement_clears_noise_floor:below_floor")
    got = json.loads(open(grade.SCORE_PATH).read())
    assert got["reason"] == "improvement_clears_noise_floor:below_floor"
''',
    "frozen_bindings_present_in_task_manifest": '''
def test_frozen_bindings_present_in_task_manifest():
    import tomllib
    with open(os.path.join(BUNDLE, "task.toml"), "rb") as f:
        man = tomllib.load(f)
    opt = man["optimization"]
    assert man["metadata"]["max_timeout_hours"] == 6.0
    assert opt["budget_hours"] == 0.12
    assert opt["budget_hours"] != 6
    assert opt["max_attempts"] == 50 and opt["max_attempts"] <= 100
    assert opt["final_selection"] == "best"
    assert opt["solver_egress"] == "setup-only"
    assert man["environment"]["gpus"] == 1
    assert man["environment"]["gpu_types"] == ["H100"]
    assert opt["reward_kind"] == "single_float"
    assert opt["reward_range"] == [0.0, 1.0]
    assert opt["reward_clip"] == "min(max(raw,0),1)"
    assert "optimization" in man["task"]["keywords"]
    assert "bounded-continuous" in man["task"]["keywords"]
''',
    "trajectory_rubrics_shape": '''
def test_trajectory_rubrics_shape():
    path = os.path.join(BUNDLE, "tests", "rubrics.jsonl")
    raw = open(path, encoding="utf-8").read()
    assert not raw.lstrip().startswith("["), "rubrics.jsonl is wrapped in an array"
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    assert lines, "rubrics.jsonl is empty"
    seen = set()
    for ln in lines:
        obj = json.loads(ln)
        assert isinstance(obj, dict)
        assert set(obj) == {"id", "rubric"}, sorted(obj)
        assert obj["id"] not in seen
        seen.add(obj["id"])
''',
    "order_contract_rejects_non_permutation": '''
def test_order_contract_rejects_non_permutation():
    import order as order_mod
    import profiles as profiles_mod
    cfg = profiles_mod.get_profile("smoke")
    b, s, n = cfg["batch_sequences"], cfg["n_steps"], cfg["n_train_sequences"]
    good = [[i * b + j for j in range(b)] for i in range(s)]
    ok, reason = order_mod.validate_order(good, cfg)
    assert ok, reason
    dup = [list(row) for row in good]
    dup[1][0] = dup[0][0]
    assert not order_mod.validate_order(dup, cfg)[0]
    short = [list(row) for row in good][:-1]
    assert not order_mod.validate_order(short, cfg)[0]
    wide = [list(row) for row in good]
    wide[0] = wide[0] + [wide[0][0]]
    assert not order_mod.validate_order(wide, cfg)[0]
    out_of_range = [list(row) for row in good]
    out_of_range[0][0] = n
    assert not order_mod.validate_order(out_of_range, cfg)[0]
''',
    "ordering_policy_is_deterministic": '''
def test_ordering_policy_is_deterministic():
    import order as order_mod
    import profiles as profiles_mod
    path = resolve_order_module()
    cfg = profiles_mod.get_profile("smoke")
    meta = smoke_meta(cfg)
    first = order_mod.call_build_order(path, meta)
    os.environ["BIA_TEST_IDENTITY"] = "beta"
    second = order_mod.call_build_order(path, meta)
    assert order_mod.validate_order(first, cfg)[0]
    assert order_mod.order_digest(first) == order_mod.order_digest(second)
''',
    "corpus_manifest_digests_regenerate": '''
def test_corpus_manifest_digests_regenerate():
    import corpus as corpus_mod
    import profiles as profiles_mod
    manifest = json.loads(open(os.path.join(BUNDLE, "environment", "corpus_manifest.json")).read())
    cfg = profiles_mod.get_profile("smoke")
    train, domains, val = corpus_mod.generate_corpus(cfg)
    got = corpus_mod.corpus_digests(train, domains, val)
    assert got == manifest["profiles"]["smoke"]["corpus"], "smoke corpus does not regenerate"
    assert manifest["profiles"]["smoke"]["substrate_digest"] == profiles_mod.substrate_digest(cfg)
    graded = profiles_mod.get_profile("graded")
    assert manifest["profiles"]["graded"]["substrate_digest"] == profiles_mod.substrate_digest(graded)
''',
    "checkers_read_live_state_only": '''
def test_checkers_read_live_state_only():
    import ast
    src = open(os.path.join(BUNDLE, "tests", "checkers", "kinds.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    forbidden = {"random", "time", "datetime", "locale", "secrets"}
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    assert not (found & forbidden), sorted(found & forbidden)
''',
}

PROLOGUE = '''"""GENERATED SECTION. DO NOT HAND-EDIT.

Source of truth: solution/grounding.yaml, compiled by solution/recompute.py.
One test per rubric item whose mode is compiled. Judged items carry a named semantic residue
in solution/rubrics.json and are deliberately absent here.
"""

from __future__ import annotations

import json
import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.environ.get("BIA_BUNDLE", os.path.dirname(TESTS_DIR))
for _p in (TESTS_DIR, os.path.join(BUNDLE, "environment", "runner")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

KINDS = ("VALUE", "EFFECT", "ABSENCE", "INVARIANT", "ORDERING", "DIVERGENCE")


def load_checker_registry():
    path = os.path.join(BUNDLE, "tests", "checkers.yaml")
    text = open(path, encoding="utf-8").read()
    try:
        import yaml

        return yaml.safe_load(text)["checkers"]
    except ImportError:
        pass
    out, cur = [], None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- id:"):
            if cur:
                out.append(cur)
            cur = {"id": stripped.split(":", 1)[1].strip(), "reduction": "",
                   "live_state_read": "", "reached_by": "", "selector": "", "zero_reason": ""}
        elif cur is not None and any(
            stripped.startswith(k + ":") for k in ("reduction", "reached_by", "selector", "zero_reason")
        ):
            key, val = stripped.split(":", 1)
            cur[key] = val.strip()
        elif cur is not None and stripped.startswith("live_state_read:"):
            cur["live_state_read"] = stripped.split(":", 1)[1].strip().strip('"')
        elif cur is not None and stripped.startswith("reward:"):
            break
    if cur:
        out.append(cur)
    return out


def load_grader(tmp_path):
    # Every path the grader writes is redirected into tmp_path. A probe emit that
    # reached the real reward file would overwrite a scored run with its own value.
    os.environ["BIA_SCORE_PATH"] = str(tmp_path / "score.json")
    os.environ["BIA_OUTCOMES_PATH"] = str(tmp_path / "outcomes.json")
    os.environ["BIA_REWARD_PATH"] = str(tmp_path / "reward.txt")
    for name in ("grade",):
        sys.modules.pop(name, None)
    import grade

    return grade


def resolve_order_module():
    for cand in (
        os.environ.get("BIA_REFERENCE_ORDER", ""),
        os.path.join(BUNDLE, "solution", "reference_order.py"),
        os.environ.get("BIA_ORDER_MODULE", ""),
        "/workspace/submission/order.py",
    ):
        if cand and os.path.exists(cand):
            return cand
    raise AssertionError("no ordering policy module is resolvable")


def smoke_meta(cfg):
    import corpus as corpus_mod
    import order as order_mod

    train, domains, _ = corpus_mod.generate_corpus(cfg)
    feats = corpus_mod.sequence_features(train, domains, cfg)
    return order_mod.build_meta(cfg, feats)
'''


def load_grounding() -> dict:
    import yaml

    with open(os.path.join(HERE, "grounding.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def write(path: str, text: str, check: bool) -> tuple[str, bool]:
    rel = os.path.relpath(path, BUNDLE)
    if check:
        if not os.path.exists(path):
            return rel, False
        return rel, open(path, encoding="utf-8").read() == text
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return rel, True


def build_corpus_manifest() -> str:
    import corpus as corpus_mod
    import profiles as profiles_mod

    out = {"schema": "bia.corpus_manifest/v1", "profiles": {}}
    for name in ("graded", "smoke"):
        cfg = profiles_mod.get_profile(name)
        train, domains, val = corpus_mod.generate_corpus(cfg)
        out["profiles"][name] = {
            "substrate_digest": profiles_mod.substrate_digest(cfg),
            "frozen": profiles_mod.frozen_substrate_fields(cfg),
            "corpus": corpus_mod.corpus_digests(train, domains, val),
        }
    return json.dumps(out, indent=1, sort_keys=True) + "\n"


def build_rubrics_json(g: dict) -> str:
    rub = g["rubric"]
    items = []
    for it in rub["items"]:
        row = {
            "id": it["id"],
            "dimension": it["dimension"],
            "weight": it["weight"],
            "evaluation_target": it["evaluation_target"],
            "criterion": it["criterion"],
            "judgment": it["judgment"],
            "evidence": [it.get("evidence", it["evaluation_target"])],
            "mode": it["mode"],
        }
        if it["mode"] == "compiled":
            row["outcome_class"] = it["outcome_class"]
            row["compiled_test"] = f"tests/test_output.py::test_{it['id']}"
        else:
            row["judged_residue"] = it["judged_residue"]
        items.append(row)
    total = sum(i["weight"] for i in items)
    compiled = sum(i["weight"] for i in items if i["mode"] == "compiled")
    doc = {
        "schema": "bia.rubrics/v1",
        "source_of_truth": "solution/grounding.yaml",
        "generated_by": "solution/recompute.py",
        "banner": "GENERATED SECTION. DO NOT HAND-EDIT.",
        "evaluation_targets": rub["evaluation_targets"],
        "compilation_floor": rub["compilation_floor"],
        "compiled_weight_share": round(compiled / total, 6),
        "items": items,
    }
    return json.dumps(doc, indent=1, sort_keys=True) + "\n"


def build_test_output(g: dict) -> str:
    parts = [PROLOGUE]
    for it in g["rubric"]["items"]:
        if it["mode"] != "compiled":
            continue
        body = TEST_BODIES.get(it["id"])
        if body is None:
            raise SystemExit(f"compiled_item_without_test_body_{it['id']}")
        parts.append(f"\n# rubric item: {it['id']} ({it['outcome_class']}), weight {it['weight']}")
        parts.append(body.rstrip("\n"))
        parts.append("")
    return "\n".join(parts) + "\n"


def _para(text: str) -> str:
    return " ".join(str(text).split())


def build_truth_md(g: dict) -> str:
    L = []
    L.append("# Ground truth for S03 data-order optimization")
    L.append("")
    L.append("GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml, regenerated by solution/recompute.py.")
    L.append("")
    L.append("## What this bundle grades")
    L.append("")
    L.append(_para(
        "The model, the optimizer and the total token budget are frozen. The only free axis is the order in which the frozen training sequences are consumed, expressed as a permutation of those sequences into batches. Reward is the clipped fraction of a declared target improvement in final validation loss, measured against a uniform random shuffle run on the same graded seed inside the same attempt."
    ))
    L.append("")
    L.append("## The scaled operating point")
    L.append("")
    L.append(_para(g["scaled_operating_point"]["rationale"]))
    L.append("")
    sop = g["scaled_operating_point"]
    L.append("| Quantity | Value |")
    L.append("|---|---|")
    for key in ("n_layer", "d_model", "n_head", "seq_len", "vocab_size", "batch_sequences", "n_steps",
                "n_train_sequences", "n_val_sequences", "n_domains", "total_train_tokens_per_run",
                "graded_seeds", "comparator_draws_per_seed", "runs_per_graded_attempt",
                "total_tokens_per_graded_attempt", "precision", "wall_clock_status",
                "wall_clock_design_target"):
        L.append(f"| {key} | {_para(sop[key])} |")
    L.append("")
    L.append("## Measured reference outcome")
    L.append("")
    mr = g["measured_reference"]
    L.append(_para(mr["statement"]))
    L.append("")
    L.append("| Quantity | Value |")
    L.append("|---|---|")
    for key in ("status", "profile", "command", "exit_code", "score", "reason", "wall_clock_seconds",
                "budget_seconds", "wall_clock_note", "baseline_ensemble_mean", "submitted_mean",
                "mean_delta", "delta_sd", "t_statistic", "t_threshold", "n_observations",
                "compiled_tests_passed", "reproducibility"):
        L.append(f"| {key} | {_para(mr[key])} |")
    L.append("")
    L.append("## Measured experiments behind this reference")
    L.append("")
    for ex in g["measured_experiments"]:
        L.append(f"### {ex['id']}")
        L.append("")
        L.append(_para("Question: " + ex["question"]))
        L.append("")
        L.append(_para("Method: " + ex["method"]))
        L.append("")
        L.append(_para("Result: " + ex["result"]))
        L.append("")
        L.append(_para("Consequence: " + ex["consequence"]))
        L.append("")
    L.append("## Ordered path through instruction.md")
    L.append("")
    for step in g["truth_steps"]:
        L.append(f"### {step['id']}")
        L.append("")
        L.append(_para(step["action"]))
        L.append("")
        L.append(_para("Established state: " + step["established_state"]))
        L.append("")
        L.append(_para("Survives: " + step["survives"]))
        L.append("")
        L.append(f"Satisfied checker: `{step['checker']}`")
        L.append("")
    L.append("## Checker set this path closes over")
    L.append("")
    L.append("| Checker | Kind |")
    L.append("|---|---|")
    for c in g["checkers"]:
        L.append(f"| `{c['id']}` | {c['kind']} |")
    L.append("")
    L.append("## Rejected routes and their known-wrong controls")
    L.append("")
    for r in g["rejected_routes"]:
        L.append(f"- `{r['control']}`: " + _para(r["route"] + " " + r["why_rejected"]))
    L.append("")
    L.append("## Defeat mechanism")
    L.append("")
    L.append(_para(g["defeat_mechanism"]["statement"]))
    L.append("")
    L.append(_para(
        "Status: " + g["defeat_mechanism"]["status"] + ". This is stated as design intent and is not a difficulty claim. No pass probability is authored, no self-solve is claimed, and the terminal disposition of this bundle is "
        + g["defeat_mechanism"]["terminal_disposition"] + ". The cohort this slot is designed against is "
        + ", ".join(g["defeat_mechanism"]["cohort"]) + ", valid to " + str(g["defeat_mechanism"]["cohort_validity_end"]) + "."
    ))
    L.append("")
    L.append("## Coverage gaps carried at sign-off")
    L.append("")
    L.append("| Gap | Reason | Cap |")
    L.append("|---|---|---|")
    for gap in g["coverage_gaps"]:
        L.append(f"| `{gap['id']}` | {_para(gap['reason'])} | {gap['cap']} |")
    L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="regenerate the generated artifacts of this bundle")
    ap.add_argument("--check", action="store_true", help="verify committed bytes without writing")
    args = ap.parse_args(argv)

    g = load_grounding()
    outputs = [
        (os.path.join(BUNDLE, "environment", "corpus_manifest.json"), build_corpus_manifest()),
        (os.path.join(BUNDLE, "solution", "rubrics.json"), build_rubrics_json(g)),
        (os.path.join(BUNDLE, "tests", "test_output.py"), build_test_output(g)),
        (os.path.join(BUNDLE, "solution", "TRUTH.md"), build_truth_md(g)),
    ]
    bad = []
    for path, text in outputs:
        rel, ok = write(path, text, args.check)
        print(("ok   " if ok else "DRIFT") + " " + rel)
        if not ok:
            bad.append(rel)
    if bad:
        print(json.dumps({"drift": bad}, sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
