"""GENERATED SECTION. DO NOT HAND-EDIT.

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


# rubric item: checker_registry_covers_six_kinds (VALUE), weight 3

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


# rubric item: reward_is_bounded_float (VALUE), weight 3

def test_reward_is_bounded_float(tmp_path):
    grade = load_grader(tmp_path)
    for raw, want in ((0.42, 0.42), (1.7, 1.0), (-0.3, 0.0)):
        grade.emit(raw, "probe")
        got = json.loads(open(grade.SCORE_PATH).read())
        assert isinstance(got["score"], float)
        assert 0.0 <= got["score"] <= 1.0
        assert abs(got["score"] - want) < 1e-12


# rubric item: zero_score_is_attributed (ABSENCE), weight 2

def test_zero_score_is_attributed(tmp_path):
    grade = load_grader(tmp_path)
    grade.emit(0.0, "")
    got = json.loads(open(grade.SCORE_PATH).read())
    assert got["score"] == 0.0
    assert got["reason"].strip(), "an unattributed zero reached the reward path"
    grade.emit(0.0, "improvement_clears_noise_floor:below_floor")
    got = json.loads(open(grade.SCORE_PATH).read())
    assert got["reason"] == "improvement_clears_noise_floor:below_floor"


# rubric item: frozen_bindings_present_in_task_manifest (VALUE), weight 3

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


# rubric item: trajectory_rubrics_shape (INVARIANT), weight 2

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


# rubric item: order_contract_rejects_non_permutation (VALUE), weight 3

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


# rubric item: ordering_policy_is_deterministic (DIVERGENCE), weight 3

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


# rubric item: corpus_manifest_digests_regenerate (DIVERGENCE), weight 3

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


# rubric item: checkers_read_live_state_only (ABSENCE), weight 2

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

