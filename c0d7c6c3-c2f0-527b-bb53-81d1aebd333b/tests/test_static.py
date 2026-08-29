"""Hardware-free unit tests of the graded logic.

Every required instrument must demonstrate BOTH halves of its decision under
frozen bytes, per invariant 17: it must accept a known-good input and reject a
known-wrong one. A checker that only ever accepts, or only ever rejects, is
inert and is a blocking defect rather than a coverage gap.

Three of the checkers this file exercises did not exist as decisions at all in
the previous grader for this slot. `frozen_axes_unchanged` and `uniform_stopping`
were the literal `True`, and `no_val_shard_touch` compared the string 'absent'
with itself whenever the shard was missing. Each is now a comparison over
measured state, and each has a rejecting half below.
"""
from __future__ import annotations

import json
import math
import pathlib

import pytest

import sandbox
from crossing import (clears_bar, common_grid, find_crossing, identical_cadence,
                      seed_interface_live, seed_means)
from fingerprint import fingerprint, names_a_forbidden_constant, numeric_constants
from reward import Anchors, AnchorsUnmeasured, claim_reconciles, compute, gap_closed

TARGET = 1.84
MARGIN = 0.01
MIN_SEEDS = 2

MEASURED = dict(baseline_steps=200, oracle_steps=120, target_loss=TARGET,
                margin=MARGIN, min_seeds=MIN_SEEDS, pass_threshold=0.35,
                claim_tolerance_steps=20, measured=True,
                provenance="unit-test fixture; not a real measurement")


def anchors(**over):
    cfg = dict(MEASURED)
    cfg.update(over)
    return Anchors(**cfg)


# --------------------------------------------------------------------------
# common_grid and find_crossing.
# --------------------------------------------------------------------------

def test_common_grid_accepts_a_fully_shared_grid():
    assert common_grid({0: {10: 3.5, 20: 3.3}, 1: {10: 3.5, 20: 3.3}}) == [10, 20]


def test_common_grid_rejects_a_point_missing_on_one_seed():
    per_seed = {0: {10: 3.5, 15: 3.35, 20: 3.3}, 1: {10: 3.5, 20: 3.3}}
    assert common_grid(per_seed) == [10, 20]


def test_common_grid_rejects_a_non_finite_loss():
    assert common_grid({0: {10: 3.5, 20: float("nan")}, 1: {10: 3.5, 20: 3.3}}) == [10]


def test_clears_bar_accepts_a_clearing_mean():
    assert clears_bar(TARGET - 0.05, 4, TARGET, MARGIN) is True


def test_clears_bar_rejects_a_non_clearing_mean():
    assert clears_bar(TARGET - 0.0001, 4, TARGET, MARGIN) is False


def test_seed_means_averages_across_seeds():
    assert seed_means({0: {10: 3.0}, 1: {10: 4.0}}, [10]) == {10: 3.5}


def test_find_crossing_finds_the_earliest_clearing_step():
    per_seed = {0: {50: 2.0, 100: 1.80, 150: 1.7}, 1: {50: 2.0, 100: 1.80, 150: 1.7}}
    c = find_crossing(per_seed, TARGET, MARGIN, MIN_SEEDS)
    assert c.reason == "crossing-found" and c.step == 100


def test_find_crossing_fails_closed_on_no_seeds():
    assert find_crossing({}, TARGET, MARGIN, MIN_SEEDS).step is None


def test_find_crossing_fails_closed_on_insufficient_seeds():
    c = find_crossing({0: {10: 1.0}}, TARGET, MARGIN, MIN_SEEDS)
    assert c.step is None and c.reason == "insufficient-seeds"


def test_find_crossing_fails_closed_when_the_target_is_never_cleared():
    per_seed = {0: {10: 3.9, 20: 3.8}, 1: {10: 3.9, 20: 3.8}}
    c = find_crossing(per_seed, TARGET, MARGIN, MIN_SEEDS)
    assert c.step is None and c.reason == "target-never-cleared"


# --------------------------------------------------------------------------
# uniform_stopping. Previously the literal True.
# --------------------------------------------------------------------------

def test_identical_cadence_accepts_the_same_milestones_on_every_seed():
    assert identical_cadence({0: (20, 40, 60), 1: (20, 40, 60)}) is True


def test_identical_cadence_rejects_a_per_seed_stopping_point():
    """A submission that stops against its own run's loss stops at different steps."""
    assert identical_cadence({0: (20, 40, 60), 1: (20, 40)}) is False


def test_identical_cadence_rejects_an_empty_cadence():
    assert identical_cadence({0: (), 1: ()}) is False


def test_identical_cadence_rejects_no_seeds():
    assert identical_cadence({}) is False


# --------------------------------------------------------------------------
# seed_interface_present. Previously a substring search for "BIA_SEED".
# --------------------------------------------------------------------------

def test_seed_interface_accepts_two_seeds_that_move_the_measured_loss():
    assert seed_interface_live({0: {20: 3.0, 40: 2.0}, 1: {20: 3.1, 40: 2.0}}) is True


def test_seed_interface_rejects_identical_curves():
    """A comment reading BIA_SEED cannot move a loss, so this cannot be faked in text."""
    assert seed_interface_live({0: {20: 3.0, 40: 2.0}, 1: {20: 3.0, 40: 2.0}}) is False


def test_seed_interface_rejects_a_single_seed():
    assert seed_interface_live({0: {20: 3.0}}) is False


# --------------------------------------------------------------------------
# claim_reconciles. Previously opt-out via `if claimed else True`.
# --------------------------------------------------------------------------

def test_claim_accepts_a_declaration_matching_the_measurement():
    assert claim_reconciles(120, 120, 20) is True


def test_claim_accepts_within_the_grid_tolerance():
    assert claim_reconciles(140, 120, 20) is True


def test_claim_rejects_a_declaration_outside_the_tolerance():
    assert claim_reconciles(2690, 120, 20) is False


def test_claim_rejects_absence_rather_than_waiving_the_check():
    """The measured defect: omitting the field made the check vacuously true."""
    assert claim_reconciles(None, 120, 20) is False


def test_claim_rejects_an_unparseable_declaration():
    assert claim_reconciles("soon", 120, 20) is False


def test_reward_magnitude_does_not_depend_on_the_claim():
    """The reward is a function of the measured crossing, never of the claim."""
    per_seed = {0: {100: 1.70}, 1: {100: 1.70}}
    c = find_crossing(per_seed, TARGET, MARGIN, MIN_SEEDS)
    honest = compute(c, anchors(), {}, 100)
    generous = compute(c, anchors(), {}, 120)
    assert honest.value == generous.value


# --------------------------------------------------------------------------
# Anchors.
# --------------------------------------------------------------------------

def test_validate_accepts_well_formed_anchors():
    anchors().validate()


def test_validate_rejects_inverted_anchors():
    with pytest.raises(ValueError, match="invert"):
        anchors(baseline_steps=100, oracle_steps=200).validate()


def test_validate_rejects_a_zero_pass_threshold():
    with pytest.raises(ValueError, match="pass_threshold"):
        anchors(pass_threshold=0.0).validate()


def test_validate_rejects_a_single_seed():
    with pytest.raises(ValueError, match="min_seeds"):
        anchors(min_seeds=1).validate()


def test_require_measured_accepts_measured_anchors_with_provenance():
    anchors().require_measured()


def test_require_measured_rejects_unmeasured_anchors():
    with pytest.raises(AnchorsUnmeasured):
        anchors(measured=False).require_measured()


def test_require_measured_rejects_a_measured_claim_without_provenance():
    with pytest.raises(AnchorsUnmeasured, match="provenance"):
        anchors(provenance="  ").require_measured()


def test_unmeasured_anchors_score_zero_even_on_a_perfect_run():
    c = find_crossing({0: {20: 0.0}, 1: {20: 0.0}}, TARGET, MARGIN, MIN_SEEDS)
    r = compute(c, anchors(measured=False), {}, 20)
    assert (r.value, r.passed, r.reason) == (0.0, False, "anchors-unmeasured")


def test_gap_closed_is_one_at_the_reference_anchor():
    assert gap_closed(120, anchors()) == 1.0


def test_gap_closed_is_zero_at_the_baseline_anchor():
    assert gap_closed(200, anchors()) == 0.0


def test_gap_closed_is_monotone_in_steps():
    assert gap_closed(140, anchors()) > gap_closed(180, anchors())


def test_compute_zeroes_on_any_validity_failure():
    c = find_crossing({0: {100: 1.7}, 1: {100: 1.7}}, TARGET, MARGIN, MIN_SEEDS)
    r = compute(c, anchors(), {"uniform_stopping": False}, 100)
    assert (r.value, r.passed) == (0.0, False)
    assert r.reason == "validity-failed:uniform_stopping"


def test_every_zero_carries_a_machine_readable_reason():
    for r in (compute(find_crossing({}, TARGET, MARGIN, MIN_SEEDS), anchors(), {}, 1),
              compute(find_crossing({0: {1: 9.0}, 1: {1: 9.0}}, TARGET, MARGIN, MIN_SEEDS),
                      anchors(), {}, 1),
              compute(find_crossing({0: {100: 1.7}, 1: {100: 1.7}}, TARGET, MARGIN, MIN_SEEDS),
                      anchors(), {}, None)):
        assert r.value == 0.0 and r.passed is False
        assert isinstance(r.reason, str) and r.reason


# --------------------------------------------------------------------------
# The structural fingerprint. Both halves, including the measured defeat.
# --------------------------------------------------------------------------

RECIPE_A = """
import torch
def update(p, g, lr):
    return p - lr * g
for i in range(10):
    x = update(1.0, 2.0, 0.5)
"""


def test_fingerprint_is_stable_under_comments_and_blank_lines():
    b = "# header\n" + RECIPE_A.replace("\n", "\n\n") + "\n# trailing\n"
    assert fingerprint(RECIPE_A) == fingerprint(b)


def test_fingerprint_is_stable_under_renaming():
    b = RECIPE_A.replace("update", "apply_step").replace("(p, g, lr)", "(w, grad, eta)") \
                .replace("p - lr * g", "w - eta * grad").replace("x =", "y =")
    assert fingerprint(RECIPE_A) == fingerprint(b)


def test_fingerprint_is_stable_under_hyperparameter_retuning():
    assert fingerprint(RECIPE_A) == fingerprint(RECIPE_A.replace("0.5", "0.0271"))


def test_fingerprint_is_stable_under_an_appended_no_op():
    """THE MEASURED DEFEAT. Eleven characters of dead code moved the old digest."""
    assert fingerprint(RECIPE_A) == fingerprint(RECIPE_A + "\n_forge_noop = 1\n")


def test_fingerprint_is_stable_under_several_dead_stores():
    assert fingerprint(RECIPE_A) == fingerprint(RECIPE_A + "\n_a=1\n_b=2\n_c=[1,2,3]\n")


def test_fingerprint_moves_when_the_computation_changes():
    """The accepting half: a real change to the update rule is a different recipe."""
    b = RECIPE_A.replace("p - lr * g", "p - lr * g / (g * g + 1e-8) ** 0.5")
    assert fingerprint(RECIPE_A) != fingerprint(b)


def test_fingerprint_moves_when_a_live_assignment_is_added():
    b = RECIPE_A + "\nz = update(3.0, 4.0, 0.5)\nprint(z)\n"
    assert fingerprint(RECIPE_A) != fingerprint(b)


def test_fingerprint_raises_on_source_that_does_not_parse():
    with pytest.raises(SyntaxError):
        fingerprint("def broken(:\n  pass\n")


def test_the_pinned_set_contains_the_published_corpus_and_not_this_bundle():
    """Both halves of no_stale_algorithm over the bytes actually shipped."""
    here = pathlib.Path(__file__).resolve().parent
    pinned = set(json.loads((here / "fingerprints.json").read_text())["fingerprints"])
    assert len(pinned) > 100
    reference = (here.parent / "solution" / "recipe.py").read_text()
    baseline = (here.parent / "environment" / "train_baseline.py").read_text()
    assert fingerprint(reference) not in pinned
    assert fingerprint(baseline) not in pinned
    assert fingerprint(reference) != fingerprint(baseline)


# --------------------------------------------------------------------------
# Constant folding for no_hardcoded_target.
# --------------------------------------------------------------------------

def test_numeric_constants_folds_arithmetic():
    assert 2690.0 in numeric_constants("S = 2000 + 690\n")


def test_forbidden_constant_caught_in_plain_form():
    assert names_a_forbidden_constant("L = 3.28\n", {"target_loss": 3.28}) == ["target_loss"]


def test_forbidden_constant_caught_in_trailing_zero_form():
    """A surface regex missed this: 3.280 destroys the word boundary of 3.28."""
    assert names_a_forbidden_constant("L = 3.280\n", {"target_loss": 3.28}) == ["target_loss"]


def test_forbidden_constant_caught_in_decomposed_form():
    assert names_a_forbidden_constant("S = 2000 + 690\n", {"oracle_steps": 2690.0}) \
        == ["oracle_steps"]


def test_clean_source_names_no_forbidden_constant():
    assert names_a_forbidden_constant("x = 1.0\ny = 42\n",
                                      {"target_loss": 3.28, "oracle_steps": 2690.0}) == []


# --------------------------------------------------------------------------
# Isolation, and the red line that compared 'absent' with itself.
# --------------------------------------------------------------------------

def _world(tmp_path):
    private, data = tmp_path / "private", tmp_path / "mount"
    private.mkdir()
    data.mkdir()
    (data / "train_0.bin").write_bytes(b"\x01\x02")
    (private / "val.bin").write_bytes(b"\x03\x04")
    (private / "shape.json").write_text(json.dumps({"seq_len": 8}))
    (private / "step_grid.json").write_text("[8]")
    (tmp_path / "recipe.py").write_text("pass\n")
    staged = sandbox.stage_run(tmp_path / "sandbox", submission=tmp_path / "recipe.py",
                               shape_path=private / "shape.json",
                               step_grid_path=private / "step_grid.json",
                               train_shard_glob=str(data / "train_*.bin"))
    env = sandbox.child_env(seed=3, shape_path=staged["shape_path"],
                            step_grid_path=staged["step_grid_path"],
                            train_shards=staged["train_shards"],
                            checkpoint_dir=staged["checkpoint_dir"], device="cpu",
                            module_path=str(tmp_path))
    return private, data, staged, env


def test_child_env_grants_exactly_the_task_variables(tmp_path):
    _, _, _, env = _world(tmp_path)
    for granted in sandbox.GRANTED:
        assert granted in env


def test_child_env_withholds_every_grading_artifact(tmp_path):
    _, _, _, env = _world(tmp_path)
    for withheld in ("BIA_VAL_SHARD", "BIA_ANCHORS", "BIA_REWARD_PATH",
                     "BIA_BUNDLE_HASH", "BIA_CLAIM", "BIA_GRADING_TREE",
                     "BIA_FINGERPRINTS"):
        assert withheld not in env


def test_isolation_accepts_a_cleanly_staged_world(tmp_path):
    private, _, staged, env = _world(tmp_path)
    assert sandbox.isolation_violations(env, staged["sandbox_root"],
                                        {"val_shard": private / "val.bin"},
                                        staged["matched_sources"]) == []


def test_isolation_rejects_a_shard_staged_by_the_training_glob(tmp_path):
    private, data, _, _ = _world(tmp_path)
    mounted = data / "train_val.bin"
    mounted.write_bytes(b"\x03\x04")
    staged = sandbox.stage_run(tmp_path / "sandbox2", submission=tmp_path / "recipe.py",
                               shape_path=private / "shape.json",
                               step_grid_path=private / "step_grid.json",
                               train_shard_glob=str(data / "train_*.bin"))
    assert sandbox.isolation_violations({}, staged["sandbox_root"],
                                        {"val_shard": mounted},
                                        staged["matched_sources"]) == [
        "val_shard-named-by-submission-environment"]


def test_isolation_rejects_a_shard_named_directly(tmp_path):
    private, _, staged, _ = _world(tmp_path)
    assert sandbox.isolation_violations({"BIA_VAL_SHARD": str(private / "val.bin")},
                                        staged["sandbox_root"],
                                        {"val_shard": private / "val.bin"}) == [
        "val_shard-named-by-submission-environment"]


def test_withheld_during_removes_the_split_and_restores_it(tmp_path):
    shard = tmp_path / "val.bin"
    shard.write_bytes(b"payload")
    with sandbox.withheld_during(shard) as enforced:
        assert enforced is True and not shard.exists()
    assert shard.read_bytes() == b"payload"


def test_file_digest_returns_none_for_an_absent_file_rather_than_a_marker(tmp_path):
    """THE MEASURED DEFECT. The old runner returned the string 'absent' here and
    the grader then compared it with itself, so a missing validation shard passed
    the red line."""
    assert sandbox.file_digest(tmp_path / "missing.bin") is None


def test_file_digest_distinguishes_a_changed_file(tmp_path):
    p = tmp_path / "val.bin"
    p.write_bytes(b"a")
    before = sandbox.file_digest(p)
    p.write_bytes(b"b")
    assert sandbox.file_digest(p) != before


def test_tree_digest_moves_when_the_grading_tree_is_edited(tmp_path):
    d = tmp_path / "tree"
    d.mkdir()
    (d / "grade.py").write_text("x = 1\n")
    before = sandbox.tree_digest(d)
    assert sandbox.tree_digest(d) == before
    (d / "grade.py").write_text("x = 2\n")
    assert sandbox.tree_digest(d) != before
