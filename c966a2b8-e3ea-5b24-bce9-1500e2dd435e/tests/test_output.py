"""Deterministic tests compiled from solution/rubrics.json.

Advisory to the reward: the reward comes from tests/grade.py and from
nothing here. These tests are the compiled half of the reference rubric and
they assert structure over frozen bundle bytes, never criterion prose and
never a reference answer.
"""

# GENERATED FILE. DO NOT HAND-EDIT. Regenerate with: python3 solution/recompute.py

import functools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)


@functools.lru_cache(maxsize=1)
def _checkers_yaml():
    import yaml
    with open(os.path.join(BUNDLE, "tests", "checkers.yaml")) as f:
        return yaml.safe_load(f)


@functools.lru_cache(maxsize=1)
def _task_toml():
    import tomllib
    with open(os.path.join(BUNDLE, "task.toml"), "rb") as f:
        return tomllib.load(f)


@functools.lru_cache(maxsize=1)
def _checkers_module():
    sys.path.insert(0, os.path.join(BUNDLE, "tests"))
    import checkers
    return checkers


@functools.lru_cache(maxsize=1)
def _controls_module():
    sys.path.insert(0, os.path.join(BUNDLE, "tests"))
    import controls
    return controls


@functools.lru_cache(maxsize=1)
def _substrate_config():
    sys.path.insert(0, os.path.join(BUNDLE, "environment"))
    from substrate import config
    return config


def test_checkers_roster_matches_the_registry():
    declared = [c["id"] for c in _checkers_yaml()["checkers"]]
    registry = [n for n, _, _ in _checkers_module().REGISTRY]
    assert declared == registry, (declared, registry)
    assert len(declared) == len(registry) and declared


def test_each_checker_reduces_to_exactly_one_kind():
    import re
    closed = {"VALUE", "EFFECT", "ABSENCE", "INVARIANT", "ORDERING", "DIVERGENCE"}
    for c in _checkers_yaml()["checkers"]:
        assert isinstance(c["kind"], str), c["id"]
        assert c["kind"] in closed, (c["id"], c["kind"])
    mod = _checkers_module()
    reg = {n: k for n, k, _ in mod.REGISTRY}
    kebab = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    seen = set()
    for c in _checkers_yaml()["checkers"]:
        assert reg[c["id"]] == c["kind"], c["id"]
        assert c["reduction"] == c["kind"], c["id"]
        assert c["zero_reason"] == mod.ZERO_REASON[c["id"]], c["id"]
        assert kebab.match(c["zero_reason"]), (c["id"], c["zero_reason"])
        assert c["zero_reason"] not in seen, c["id"]
        seen.add(c["zero_reason"])
        carrier = os.path.join(BUNDLE, c["reached_by"])
        assert os.path.isfile(carrier), (c["id"], c["reached_by"])
        assert "def %s(" % c["selector"] in open(carrier).read(), (c["id"], c["selector"])


def test_every_checker_names_a_live_state_read():
    for c in _checkers_yaml()["checkers"]:
        assert c.get("live_state_read"), c["id"]
        assert len(c["live_state_read"].strip()) > 40, c["id"]


def test_kind_census_matches_declared_kinds():
    y = _checkers_yaml()
    counted = {}
    for c in y["checkers"]:
        counted[c["kind"]] = counted.get(c["kind"], 0) + 1
    declared = dict(y["kind_census"])
    total = declared.pop("total")
    assert counted == declared, (counted, declared)
    assert total == sum(counted.values()) == len(y["checkers"])
    assert set(counted) == {"VALUE", "EFFECT", "ABSENCE", "INVARIANT", "ORDERING", "DIVERGENCE"}


def test_trajectory_rubrics_shape():
    import json
    raw = open(os.path.join(BUNDLE, "tests", "rubrics.jsonl")).read()
    assert not raw.lstrip().startswith("["), "rubrics.jsonl must not be a wrapping array"
    ids = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        assert set(obj.keys()) == {"id", "rubric"}, obj.keys()
        assert isinstance(obj["id"], str) and isinstance(obj["rubric"], str)
        ids.append(obj["id"])
    assert len(ids) == len(set(ids))
    assert len(ids) >= 1


def test_seven_optimization_bindings_present():
    t = _task_toml()
    o = t["optimization"]
    assert o["max_timeout"] == 6.0
    assert o["budget_hours"] == 0.12
    assert o["max_attempts"] == 50
    assert o["final_selection"] == "best"
    assert o["solver_egress"] == "setup-only"
    assert o["compute_envelope"] == "one-H100"
    assert o["reward_range"] == [0.0, 1.0]
    assert o["reward_clamp"] == "min(max(raw,0),1)"
    assert t["environment"]["gpus"] == 1
    assert t["environment"]["gpu_types"] == ["H100"]


def test_budget_hours_bound_by_role():
    o = _task_toml()["optimization"]
    assert o["budget_hours"] != 6
    assert o["budget_hours"] < o["max_timeout"]
    assert abs(o["budget_hours"] * o["max_attempts"] - o["max_timeout"]) < 1e-9


def test_reward_clamp_is_the_bound_formula():
    C = _checkers_module()
    for raw in (-2.0, -0.001, 0.0, 0.37, 1.0, 4.2):
        good = {"raw": raw, "score": min(max(raw, 0.0), 1.0), "anchors": {}, "agent_metric": None}
        ok, reason, _ = C.check_reward_is_clamped_unit_float(good)
        assert ok, (raw, reason)
    bad = {"raw": 0.25, "score": 0.9, "anchors": {}, "agent_metric": None}
    ok, reason, _ = C.check_reward_is_clamped_unit_float(bad)
    assert not ok and reason == "score_not_clamped_raw"
    out = {"raw": 0.25, "score": 1.7, "anchors": {}, "agent_metric": None}
    ok, reason, _ = C.check_reward_is_clamped_unit_float(out)
    assert not ok and reason == "score_outside_unit_interval"


def test_substrate_manifest_covers_substrate():
    import hashlib, json
    d = os.path.join(BUNDLE, "environment", "substrate")
    pinned = json.load(open(os.path.join(d, "SUBSTRATE_MANIFEST.json")))["files"]
    on_disk = sorted(n for n in os.listdir(d) if n.endswith(".py"))
    assert sorted(pinned) == on_disk, (sorted(pinned), on_disk)
    for name, digest in pinned.items():
        got = hashlib.sha256(open(os.path.join(d, name), "rb").read()).hexdigest()
        assert got == digest, name


def test_smoke_fixture_digest_matches():
    import hashlib, json
    f = os.path.join(BUNDLE, "environment", "fixtures")
    meta = json.load(open(os.path.join(f, "FIXTURES.json")))
    blob = open(os.path.join(f, meta["fixture"]), "rb").read()
    assert len(blob) == meta["bytes"]
    assert hashlib.sha256(blob).hexdigest() == meta["sha256"]


def test_ablation_scope_is_closed_and_shared():
    C = _substrate_config()
    assert len(C.ABLATED_MODULE_CLASSES) >= 6
    assert len(C.ABLATED_FUNCTIONALS) >= 5
    assert "RMSNorm" in C.ABLATED_MODULE_CLASSES
    assert "layer_norm" in C.ABLATED_FUNCTIONALS
    assert set(C.FORBIDDEN_PHASES) == {"forward", "backward", "eval"}
    src = open(os.path.join(BUNDLE, "environment", "substrate", "model.py")).read()
    assert 'norm_mode == "rmsnorm"' in src
    assert 'norm_mode == "none"' in src


def test_every_checker_has_a_rejecting_control():
    controls = _controls_module()
    targets = {t for t, _, _, _ in controls.DEFECTS}
    registry = {n for n, _, _ in _checkers_module().REGISTRY}
    assert registry <= targets, registry - targets


def test_emitted_markdown_prose_rule():
    offenders = []
    for root, dirs, files in os.walk(BUNDLE):
        dirs[:] = [d for d in dirs if d not in ("trajectories", "__pycache__")]
        for name in files:
            if not name.endswith(".md"):
                continue
            p = os.path.join(root, name)
            text = open(p, encoding="utf-8").read()
            if "\u2014" in text:
                offenders.append((p, "em-dash"))
            in_fence = False
            prev_kind = None
            for line in text.split("\n"):
                if line.startswith("```"):
                    in_fence = not in_fence
                    prev_kind = None
                    continue
                if in_fence:
                    continue
                kind = "blank" if not line.strip() else "prose"
                if kind == "prose" and prev_kind == "prose":
                    if not (line.lstrip().startswith(("|", "-", "*", "#", ">", "<!--"))
                            or line.lstrip()[:2].rstrip(".").isdigit()):
                        offenders.append((p, "intra-paragraph newline: " + line[:60]))
                prev_kind = kind
    assert not offenders, offenders
