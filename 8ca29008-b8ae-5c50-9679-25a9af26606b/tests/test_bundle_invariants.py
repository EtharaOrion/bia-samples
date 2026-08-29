"""Structural assertions over the bundle's own grading surface.

These are separate from tests/test_output.py on purpose. That file carries exactly one
test per compiled rubric item and nothing else, because its test-name set is the proof
of compiled-item completeness. This file carries everything else: that the checker
declaration and the registry the verifier executes agree, that the compiled and judged
lanes are what the rubric says they are, and that a graded run's outputs have the shape
the reward contract promises.

Nothing here asserts difficulty, a pass probability, or a tier.
"""

from __future__ import annotations

import json
import os
import re
import sys

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_DIR = os.path.dirname(TESTS_DIR)
sys.path.insert(0, TESTS_DIR)

from checkers.reduce import CHECKERS, KINDS, reduce_all  # noqa: E402

LOG_DIR = os.environ.get("LOG_DIR", "/logs/verifier")
SCORE_PATH = os.environ.get("SCORE_PATH", os.path.join(LOG_DIR, "score.json"))
OUTCOMES_PATH = os.environ.get("S08_OUTCOMES", os.path.join(LOG_DIR, "outcomes.json"))

TEST_NAME = re.compile(r"^def test_([a-z0-9_]+)\(\):$", re.MULTILINE)
REASON_CODE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _load_checkers_yaml():
    try:
        import yaml
    except ImportError:  # pragma: no cover
        pytest.skip("pyyaml unavailable in this image")
    with open(os.path.join(TESTS_DIR, "checkers.yaml")) as f:
        return yaml.safe_load(f)


def _rubric_lines():
    with open(os.path.join(TESTS_DIR, "rubrics.jsonl")) as f:
        return [line for line in f.read().split("\n") if line.strip()]


def _rubrics_json():
    path = os.path.join(BUNDLE_DIR, "solution", "rubrics.json")
    if not os.path.exists(path):
        pytest.skip("solution/ is not mounted on the verifier surface")
    with open(path) as f:
        return json.load(f)


def _compiled_test_names():
    with open(os.path.join(TESTS_DIR, "test_output.py")) as f:
        return set(TEST_NAME.findall(f.read()))


def test_every_checker_reduces_to_exactly_one_kind():
    for name, spec in CHECKERS.items():
        matches = [k for k in KINDS if k == spec["kind"]]
        assert len(matches) == 1, f"{name} reduces to {len(matches)} kinds, expected exactly 1"


def test_every_checker_names_a_live_state_read():
    for name, spec in CHECKERS.items():
        assert spec.get("live_state_read"), f"{name} names no live-state read"


def test_checkers_yaml_agrees_with_the_registry():
    doc = _load_checkers_yaml()
    declared = {c["id"]: c for c in doc["checkers"]}
    assert set(declared) == set(CHECKERS), "checkers.yaml and the registry declare different checker sets"
    for name, spec in CHECKERS.items():
        assert declared[name]["kind"] == spec["kind"], f"{name} kind disagrees between yaml and registry"
        assert declared[name]["live_state_read"] == spec["live_state_read"], (
            f"{name} live-state read disagrees between yaml and registry"
        )
        assert declared[name]["kind"] in doc["kinds"]


def test_checkers_yaml_reduction_field_agrees_with_the_registry_kind():
    doc = _load_checkers_yaml()
    for c in doc["checkers"]:
        assert c["reduction"] == CHECKERS[c["id"]]["kind"]
        assert c["reduction"] == c["kind"]


def test_every_checker_declares_the_zero_reason_its_carrier_emits():
    doc = _load_checkers_yaml()
    declared = {c["id"]: c for c in doc["checkers"]}
    reasons = set()
    for name, spec in CHECKERS.items():
        reason = spec["zero_reason"]
        assert REASON_CODE.match(reason), f"{name} zero_reason {reason!r} is not a kebab reason code"
        assert declared[name]["zero_reason"] == reason, f"{name} zero_reason disagrees with the registry"
        assert reason not in reasons, f"{name} reuses a zero_reason, so a zero would be ambiguous"
        reasons.add(reason)


def test_every_checker_carrier_defines_the_selector_it_names():
    for name, spec in CHECKERS.items():
        path = os.path.join(BUNDLE_DIR, spec["reached_by"])
        assert os.path.isfile(path), f"{name} names carrier {spec['reached_by']}, which is not a file"
        with open(path) as f:
            assert f"def {spec['selector']}(" in f.read(), f"{name} selector is undefined in its carrier"


def test_entry_point_writes_the_bound_reward_path_on_every_exit():
    doc = _load_checkers_yaml()
    with open(os.path.join(TESTS_DIR, "test.sh")) as f:
        entry = f.read()
    assert doc["reward_path"] in entry, "test.sh never writes the reward path the manifest binds"
    assert "trap write_reward EXIT" in entry, "the reward write is not guarded by an EXIT trap"


def test_checkers_yaml_uses_only_the_six_kinds():
    doc = _load_checkers_yaml()
    assert list(doc["kinds"]) == list(KINDS)
    for c in doc["checkers"]:
        assert c["kind"] in KINDS


def test_every_kind_carries_at_least_one_checker():
    covered = {spec["kind"] for spec in CHECKERS.values()}
    assert covered == set(KINDS), f"kinds with no checker: {sorted(set(KINDS) - covered)}"


def test_rubrics_jsonl_is_one_object_per_line_with_exactly_two_keys():
    ids = []
    for i, line in enumerate(_rubric_lines()):
        obj = json.loads(line)
        assert isinstance(obj, dict), f"rubrics.jsonl line {i + 1} is not an object"
        assert set(obj) == {"id", "rubric"}, f"rubrics.jsonl line {i + 1} keys are {sorted(obj)}"
        assert isinstance(obj["id"], str) and obj["id"]
        assert isinstance(obj["rubric"], str) and obj["rubric"]
        ids.append(obj["id"])
    assert len(ids) == len(set(ids)), "duplicate rubric id"
    assert ids, "rubrics.jsonl is empty"


def test_reduce_all_refuses_an_unregistered_checker():
    ok, why = reduce_all({name: (True, None) for name in CHECKERS})
    assert ok, why
    bad = {name: (True, None) for name in CHECKERS}
    bad["invented_checker"] = (True, None)
    ok, why = reduce_all(bad)
    assert not ok and "invented_checker" in why


def test_reduce_all_refuses_a_missing_checker():
    partial = {name: (True, None) for name in list(CHECKERS)[:-1]}
    ok, why = reduce_all(partial)
    assert not ok and "did_not_run" in why


# --------------------------------------------------------------------------
# The rubric-compilation obligations, asserted over frozen bytes rather than
# recorded in prose: completeness by identifier set equality, the floor by
# arithmetic, and honesty by requiring every judged item to name its residue.
# --------------------------------------------------------------------------


def test_compiled_rubric_items_and_compiled_tests_are_the_same_identifier_set():
    compiled = {i["id"] for i in _rubrics_json()["items"] if i["mode"] == "compiled"}
    assert compiled == _compiled_test_names(), (
        "compiled rubric items and tests/test_output.py tests are not the same set; "
        f"only in rubric: {sorted(compiled - _compiled_test_names())}; "
        f"only in tests: {sorted(_compiled_test_names() - compiled)}"
    )


def test_compiled_rubric_items_are_exactly_the_registered_checkers():
    compiled = {i["id"] for i in _rubrics_json()["items"] if i["mode"] == "compiled"}
    assert compiled == set(CHECKERS)


def test_compiled_weight_share_meets_the_compilation_floor():
    doc = _rubrics_json()
    total = sum(i["weight"] for i in doc["items"])
    compiled = sum(i["weight"] for i in doc["items"] if i["mode"] == "compiled")
    share = compiled / total
    assert share == pytest.approx(doc["compiled_weight_share"], abs=1e-6)
    assert share >= doc["compilation_floor"], (
        f"compiled weight share {share} is below the floor {doc['compilation_floor']}"
    )


def test_every_judged_item_names_its_residue():
    for item in _rubrics_json()["items"]:
        if item["mode"] != "judged":
            continue
        residue = item.get("residue", "")
        assert isinstance(residue, str) and residue.strip(), f"{item['id']} names no residue"


def test_every_rubric_item_carries_one_mode_and_one_evaluation_target():
    doc = _rubrics_json()
    for item in doc["items"]:
        assert item["mode"] in ("compiled", "judged")
        assert item["evaluation_target"] in doc["evaluation_targets"]
        for entry in item["evidence"]:
            assert entry in doc["evaluation_targets"]


def test_judged_rubric_ids_match_the_trajectory_rubric_file():
    judged = {i["id"] for i in _rubrics_json()["items"] if i["mode"] == "judged"}
    assert judged == {json.loads(line)["id"] for line in _rubric_lines()}


# --------------------------------------------------------------------------
# Graded-run assertions. Skipped when no run is present, so the bundle byte
# assertions above still run in a bare inspection.
# --------------------------------------------------------------------------


def _outcomes():
    if not os.path.exists(OUTCOMES_PATH):
        pytest.skip("no graded run present")
    with open(OUTCOMES_PATH) as f:
        return json.load(f)


def _score():
    if not os.path.exists(SCORE_PATH):
        pytest.skip("no graded run present")
    with open(SCORE_PATH) as f:
        return json.load(f)


def test_outcomes_cover_every_registered_checker():
    out = _outcomes()
    assert set(out) == set(CHECKERS), "outcomes.json does not cover the registered checker set"


def test_score_is_a_single_float_on_the_unit_interval():
    s = _score()
    assert "score" in s
    v = s["score"]
    assert isinstance(v, (int, float)) and not isinstance(v, bool)
    assert 0.0 <= float(v) <= 1.0


def test_a_failing_checker_forces_a_zero_score():
    out = _outcomes()
    s = _score()
    if not all(out.values()):
        assert float(s["score"]) == 0.0, "a checker failed and the score was not zero"
