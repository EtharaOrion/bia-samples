"""Compiled deterministic rubric tests for bia slot S06.

GENERATED SECTION. DO NOT HAND-EDIT.

Source of truth: solution/grounding.yaml. Generator: solution/recompute.py.

One test per compiled rubric item. Each reads committed bundle bytes only and
carries no criterion prose and no reference text.
"""

import hashlib
import json
import os
import re

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.environ.get("BIA_BUNDLE_SOURCE") or os.path.dirname(TESTS_DIR)
KINDS = ['VALUE', 'EFFECT', 'ABSENCE', 'INVARIANT', 'ORDERING', 'DIVERGENCE']


def _read(rel):
    with open(os.path.join(BUNDLE, rel), encoding="utf-8") as fh:
        return fh.read()


def _sha(rel):
    with open(os.path.join(BUNDLE, rel), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _checkers():
    text = _read("tests/checkers.yaml")
    body = text.split("checkers:", 1)[1].split("\nreward:", 1)[0]
    blocks = []
    for chunk in body.split("  - id:"):
        chunk = chunk.strip()
        if not chunk:
            continue
        lines = chunk.splitlines()
        entry = {
            "id": lines[0].strip(),
            "kinds": [ln.split(":", 1)[1].strip() for ln in lines if ln.strip().startswith("reduction:")],
            "live_state_read": "".join(
                ln.split(":", 1)[1] for ln in lines if ln.strip().startswith("live_state_read:")),
        }
        blocks.append(entry)
    return blocks


def _bound():
    text = _read("tests/checkers.yaml")
    out = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.strip().partition(":")
        if key.strip() in ("noise_fixture_sha256", "corpus_sha256", "identity_estimator_digest"):
            out[key.strip()] = val.strip().strip('"')
    return out


def test_checker_roster_reconciles():
    declared = {c["id"] for c in _checkers()}
    src = _read("tests/grade.py")
    block = src.split("CHECKER_IDS = [", 1)[1].split("]", 1)[0]
    executed = set(re.findall(r'"([a-z0-9_]+)"', block))
    assert declared == executed, (declared ^ executed)


def test_checker_kinds_are_singleton():
    for c in _checkers():
        assert len(c["kinds"]) == 1, (c["id"], c["kinds"])
        assert c["kinds"][0] in KINDS, (c["id"], c["kinds"])


def test_checker_kinds_cover_six():
    seen = {c["kinds"][0] for c in _checkers()}
    assert seen == set(KINDS), sorted(set(KINDS) - seen)


def test_each_checker_names_live_state_read():
    for c in _checkers():
        assert c["live_state_read"].strip(), c["id"]


def test_reward_clamped_to_unit_interval():
    assert "min(max(raw, 0), 1)" in _read("tests/checkers.yaml")
    grade = _read("tests/grade.py")
    assert "min(max(raw, 0.0), 1.0)" in grade


def test_rubrics_jsonl_shape():
    lines = [ln for ln in _read("tests/rubrics.jsonl").splitlines() if ln.strip()]
    assert lines
    for ln in lines:
        obj = json.loads(ln)
        assert isinstance(obj, dict)
        assert set(obj) == {"id", "rubric"}, set(obj)


def test_optimization_bindings_present():
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


def test_frozen_fixture_digests_bound():
    bound = _bound()
    assert bound["noise_fixture_sha256"] == _sha("environment/fixtures/noise_process.py")
    assert bound["corpus_sha256"] == _sha("environment/fixtures/corpus.bin")


def test_zero_score_has_machine_reason():
    grade = _read("tests/grade.py")
    calls = re.findall(r"emit\(\s*0\.0\s*,\s*([^,]+),", grade)
    assert calls
    for arg in calls:
        arg = arg.strip()
        assert arg and arg not in ('""', "''", "None")
    sh = _read("tests/test.sh")
    assert '"reason"' in sh


def test_smoke_path_is_cpu_only():
    solve = _read("solution/solve.sh")
    assert "CUDA_VISIBLE_DEVICES" in solve
    assert "BIA_SMOKE" in solve
    grade = _read("tests/grade.py")
    assert 'env["CUDA_VISIBLE_DEVICES"] = ""' in grade
    train = _read("environment/harness/train.py")
    assert "modelmod.SMOKE" in train
    assert 'torch.device("cpu")' in train


def test_no_em_dash_in_markdown():
    bad = []
    for root, _dirs, files in os.walk(BUNDLE):
        for name in files:
            if not name.endswith(".md"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as fh:
                if "\u2014" in fh.read():
                    bad.append(os.path.relpath(path, BUNDLE))
    assert not bad, bad


def test_estimator_contract_documented():
    text = _read("instruction.md")
    assert "build_estimator" in text
    assert "estimate" in text
    assert "submission/estimator.py" in text

