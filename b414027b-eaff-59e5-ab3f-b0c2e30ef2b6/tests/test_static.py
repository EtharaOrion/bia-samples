"""Hardware-free unit tests of the graded logic.

Every required instrument must demonstrate BOTH halves of its decision under
frozen bytes, per invariant 17: it must accept a known-good input and reject a
known-wrong one. A checker that only ever accepts, or only ever rejects, is inert
and is a blocking defect rather than a coverage gap.

Nothing here needs a GPU. The prior attempt designed the graded logic as pure
functions over measurements precisely so this suite could run anywhere, and that
property is preserved in the port.
"""
from __future__ import annotations

import json
import math
import pathlib

import pytest

import sandbox
from crossing import Crossing, clears_bar, common_grid, find_crossing, seed_means
from grade import digest_or_absent, unmodified
from reward import Anchors, AnchorsUnmeasured, compute, gap_closed

TARGET = 3.40
MARGIN = 0.004
MIN_SEEDS = 2

MEASURED = dict(
    baseline_tokens=150_000_000,
    oracle_tokens=100_000_000,
    target_loss=TARGET,
    margin=MARGIN,
    min_seeds=MIN_SEEDS,
    pass_threshold=0.10,
    measured=True,
    provenance="unit-test fixture; not a real measurement",
)


def anchors(**over):
    cfg = dict(MEASURED)
    cfg.update(over)
    return Anchors(**cfg)


# --------------------------------------------------------------------------
# common_grid. Both halves: includes shared points, excludes unshared ones.
# --------------------------------------------------------------------------

def test_common_grid_accepts_fully_shared_grid():
    per_seed = {0: {10: 3.5, 20: 3.3}, 1: {10: 3.5, 20: 3.3}}
    assert common_grid(per_seed) == [10, 20]


def test_common_grid_rejects_point_missing_on_one_seed():
    """C2, finer evaluation cadence. An extra point on one seed must not count."""
    per_seed = {0: {10: 3.5, 15: 3.35, 20: 3.3}, 1: {10: 3.5, 20: 3.3}}
    assert common_grid(per_seed) == [10, 20]
    assert 15 not in common_grid(per_seed)


def test_common_grid_rejects_non_finite_loss():
    per_seed = {0: {10: 3.5, 20: float("nan")}, 1: {10: 3.5, 20: 3.3}}
    assert common_grid(per_seed) == [10]


def test_common_grid_empty_on_no_seeds():
    assert common_grid({}) == []


def test_common_grid_empty_on_disjoint_grids():
    assert common_grid({0: {10: 3.5}, 1: {20: 3.5}}) == []


# --------------------------------------------------------------------------
# clears_bar. Both halves.
# --------------------------------------------------------------------------

def test_clears_bar_accepts_clearing_mean():
    assert clears_bar(TARGET - 0.01, 4, TARGET, MARGIN) is True


def test_clears_bar_rejects_non_clearing_mean():
    assert clears_bar(TARGET - 0.0001, 4, TARGET, MARGIN) is False


def test_clears_bar_rejects_mean_above_target():
    assert clears_bar(TARGET + 0.01, 8, TARGET, MARGIN) is False


def test_clears_bar_uses_sqrt_n_so_seed_count_matters():
    mean = TARGET - 0.002
    assert clears_bar(mean, 2, TARGET, MARGIN) is False
    assert clears_bar(mean, 8, TARGET, MARGIN) is True


def test_seed_means_averages_across_seeds():
    per_seed = {0: {10: 3.0}, 1: {10: 4.0}}
    assert seed_means(per_seed, [10]) == {10: 3.5}


# --------------------------------------------------------------------------
# find_crossing. Accepting half, then every fail-closed branch.
# --------------------------------------------------------------------------

def test_find_crossing_finds_earliest_clearing_point():
    per_seed = {
        0: {50: 3.60, 100: 3.38, 150: 3.30},
        1: {50: 3.60, 100: 3.38, 150: 3.30},
    }
    c = find_crossing(per_seed, TARGET, MARGIN, MIN_SEEDS)
    assert c.reason == "crossing-found"
    assert c.training_tokens == 100


def test_find_crossing_returns_earliest_not_best():
    per_seed = {0: {50: 3.30, 100: 3.20}, 1: {50: 3.30, 100: 3.20}}
    c = find_crossing(per_seed, TARGET, MARGIN, MIN_SEEDS)
    assert c.training_tokens == 50


def test_find_crossing_fails_closed_on_no_seeds():
    c = find_crossing({}, TARGET, MARGIN, MIN_SEEDS)
    assert c.training_tokens is None and c.reason == "no-seeds-measured"


def test_find_crossing_fails_closed_on_insufficient_seeds():
    c = find_crossing({0: {10: 3.0}}, TARGET, MARGIN, MIN_SEEDS)
    assert c.training_tokens is None and c.reason == "insufficient-seeds"


def test_find_crossing_fails_closed_on_no_common_grid():
    c = find_crossing({0: {10: 3.0}, 1: {20: 3.0}}, TARGET, MARGIN, MIN_SEEDS)
    assert c.training_tokens is None and c.reason == "no-common-evaluation-grid"


def test_find_crossing_fails_closed_when_target_never_cleared():
    per_seed = {0: {10: 3.9, 20: 3.8}, 1: {10: 3.9, 20: 3.8}}
    c = find_crossing(per_seed, TARGET, MARGIN, MIN_SEEDS)
    assert c.training_tokens is None and c.reason == "target-never-cleared"


def test_find_crossing_unshared_early_point_cannot_create_earlier_crossing():
    """C2 end to end: a seed-local extra point must not move the crossing earlier."""
    shared = {50: 3.60, 100: 3.30}
    per_seed = {0: {**shared, 75: 3.31}, 1: dict(shared)}
    c = find_crossing(per_seed, TARGET, MARGIN, MIN_SEEDS)
    assert c.training_tokens == 100


# --------------------------------------------------------------------------
# Anchors.validate. Both halves on every rule.
# --------------------------------------------------------------------------

def test_validate_accepts_well_formed_anchors():
    anchors().validate()


def test_validate_rejects_inverted_anchors():
    with pytest.raises(ValueError, match="invert"):
        anchors(baseline_tokens=100, oracle_tokens=200).validate()


def test_validate_rejects_equal_anchors_zero_span():
    with pytest.raises(ValueError, match="invert"):
        anchors(baseline_tokens=100, oracle_tokens=100).validate()


def test_validate_rejects_non_positive_margin():
    with pytest.raises(ValueError, match="margin"):
        anchors(margin=0.0).validate()


def test_validate_rejects_zero_min_seeds():
    with pytest.raises(ValueError, match="min_seeds"):
        anchors(min_seeds=0).validate()


def test_validate_rejects_zero_pass_threshold():
    with pytest.raises(ValueError, match="pass_threshold"):
        anchors(pass_threshold=0.0).validate()


def test_validate_rejects_pass_threshold_above_one():
    with pytest.raises(ValueError, match="pass_threshold"):
        anchors(pass_threshold=1.5).validate()


# --------------------------------------------------------------------------
# require_measured. THE GUARD THE PRIOR ATTEMPT LACKED.
# --------------------------------------------------------------------------

def test_require_measured_accepts_measured_anchors_with_provenance():
    anchors().require_measured()


def test_require_measured_rejects_unmeasured_anchors():
    with pytest.raises(AnchorsUnmeasured):
        anchors(measured=False).require_measured()


def test_require_measured_rejects_measured_claim_without_provenance():
    with pytest.raises(AnchorsUnmeasured, match="provenance"):
        anchors(measured=True, provenance="   ").require_measured()


def test_unmeasured_anchors_score_zero_and_never_pass():
    """BINDING. This is the defect that voided the prior attempt."""
    per_seed = {0: {50: 3.0}, 1: {50: 3.0}}
    c = find_crossing(per_seed, TARGET, MARGIN, MIN_SEEDS)
    r = compute(c, anchors(measured=False), {}, None)
    assert r.value == 0.0
    assert r.passed is False
    assert r.reason == "anchors-unmeasured"
    assert r.checkers["anchors_measured"] is False


def test_unmeasured_anchors_zero_even_on_a_perfect_run():
    per_seed = {0: {1: 0.0}, 1: {1: 0.0}}
    c = find_crossing(per_seed, TARGET, MARGIN, MIN_SEEDS)
    r = compute(c, anchors(measured=False), {}, None)
    assert (r.value, r.passed) == (0.0, False)


# --------------------------------------------------------------------------
# gap_closed and compute.
# --------------------------------------------------------------------------

def test_gap_closed_is_one_at_the_oracle_anchor():
    assert gap_closed(100_000_000, anchors()) == 1.0


def test_gap_closed_is_zero_at_the_baseline_anchor():
    assert gap_closed(150_000_000, anchors()) == 0.0


def test_gap_closed_clamps_beyond_oracle():
    assert gap_closed(10_000_000, anchors()) == 1.0


def test_gap_closed_clamps_worse_than_baseline():
    assert gap_closed(900_000_000, anchors()) == 0.0


def test_gap_closed_is_monotone_in_tokens():
    a = anchors()
    assert gap_closed(120_000_000, a) > gap_closed(140_000_000, a)


def test_compute_scores_and_passes_a_good_run():
    per_seed = {0: {110_000_000: 3.30}, 1: {110_000_000: 3.30}}
    c = find_crossing(per_seed, TARGET, MARGIN, MIN_SEEDS)
    r = compute(c, anchors(), {"run_completes": True}, None)
    assert r.passed is True and 0.0 < r.value <= 1.0
    assert r.reason == "scored"


def test_compute_scores_without_passing_below_threshold():
    per_seed = {0: {149_000_000: 3.30}, 1: {149_000_000: 3.30}}
    c = find_crossing(per_seed, TARGET, MARGIN, MIN_SEEDS)
    r = compute(c, anchors(), {"run_completes": True}, None)
    assert r.passed is False
    assert r.reason == "scored-below-pass-threshold"
    assert r.value > 0.0


def test_compute_zeroes_on_validity_failure():
    c = find_crossing({0: {50: 3.0}, 1: {50: 3.0}}, TARGET, MARGIN, MIN_SEEDS)
    r = compute(c, anchors(), {"run_completes": False}, None)
    assert (r.value, r.passed) == (0.0, False)
    assert r.reason.startswith("validity-failed:")


def test_compute_zeroes_on_rubric_failure():
    c = find_crossing({0: {50: 3.0}, 1: {50: 3.0}}, TARGET, MARGIN, MIN_SEEDS)
    r = compute(c, anchors(), {}, {"honest_result_claims": False})
    assert (r.value, r.passed) == (0.0, False)
    assert r.reason == "rubric-failed:honest_result_claims"


def test_compute_passes_when_all_rubrics_pass():
    per_seed = {0: {110_000_000: 3.30}, 1: {110_000_000: 3.30}}
    c = find_crossing(per_seed, TARGET, MARGIN, MIN_SEEDS)
    r = compute(c, anchors(), {}, {"honest_result_claims": True})
    assert r.passed is True


def test_every_zero_carries_a_machine_readable_reason():
    cases = [
        compute(find_crossing({}, TARGET, MARGIN, MIN_SEEDS), anchors(), {}, None),
        compute(find_crossing({0: {1: 9.0}, 1: {1: 9.0}}, TARGET, MARGIN, MIN_SEEDS), anchors(), {}, None),
        compute(find_crossing({0: {1: 3.0}, 1: {1: 3.0}}, TARGET, MARGIN, MIN_SEEDS), anchors(measured=False), {}, None),
        compute(find_crossing({0: {1: 3.0}, 1: {1: 3.0}}, TARGET, MARGIN, MIN_SEEDS), anchors(), {"x": False}, None),
    ]
    for r in cases:
        assert r.value == 0.0 and r.passed is False
        assert isinstance(r.reason, str) and r.reason and " " not in r.reason.split(":")[0]
        assert r.as_payload()["reason"] == r.reason


# --------------------------------------------------------------------------
# The legitimate strategy. This MUST NOT be blocked, because it is the task.
# --------------------------------------------------------------------------

def test_batch_size_reduction_that_saves_tokens_scores_higher():
    """L1. Fewer training tokens to the same loss must score strictly better.

    A submission halving its batch size and taking somewhat more steps, but
    reaching the target on fewer total tokens, is doing exactly what the task
    asks. Nothing may penalise it.
    """
    big_batch = {0: {140_000_000: 3.30}, 1: {140_000_000: 3.30}}
    small_batch = {0: {115_000_000: 3.30}, 1: {115_000_000: 3.30}}
    a = anchors()
    r_big = compute(find_crossing(big_batch, TARGET, MARGIN, MIN_SEEDS), a, {}, None)
    r_small = compute(find_crossing(small_batch, TARGET, MARGIN, MIN_SEEDS), a, {}, None)
    assert r_small.value > r_big.value


def test_token_axis_ignores_step_count_entirely():
    """Two submissions at identical tokens score identically regardless of steps.

    The graded record carries no step count at all, so a step-efficient but
    token-hungry submission cannot benefit from its step count.
    """
    per_seed = {0: {120_000_000: 3.30}, 1: {120_000_000: 3.30}}
    r = compute(find_crossing(per_seed, TARGET, MARGIN, MIN_SEEDS), anchors(), {}, None)
    assert "crossing_training_tokens" in r.measurement
    assert not any("step" in k for k in r.measurement)


# --------------------------------------------------------------------------
# Submission isolation. Both halves of every mechanism in tests/sandbox.py.
#
# These are the instruments that replace `env = dict(os.environ)`. Under
# invariant 17 each must be shown to accept a clean world AND to reject a
# leaking one, because a predicate that has only ever returned the empty list
# is indistinguishable from `return []`.
# --------------------------------------------------------------------------

def _world(tmp_path: pathlib.Path):
    """A staged run whose withheld artifacts sit outside the sandbox."""
    private = tmp_path / "private"
    data = tmp_path / "mount"
    private.mkdir()
    data.mkdir()
    (data / "train_0.bin").write_bytes(b"\x01\x02")
    (private / "val.bin").write_bytes(b"\x03\x04")
    (private / "anchors.json").write_text("{}")
    (private / "shape.json").write_text(json.dumps({"seq_len": 8}))
    (private / "token_grid.json").write_text("[8]")
    staged = sandbox.stage_run(
        tmp_path / "sandbox",
        submission=_write(tmp_path / "train.py", "pass\n"),
        shape_path=private / "shape.json",
        token_grid_path=private / "token_grid.json",
        train_shard_glob=str(data / "train_*.bin"),
    )
    env = sandbox.child_env(
        seed=7,
        shape_path=staged["shape_path"],
        token_grid_path=staged["token_grid_path"],
        train_shards=staged["train_shards"],
        checkpoint_dir=staged["checkpoint_dir"],
        device="cpu",
        module_path=str(tmp_path),
    )
    return private, data, staged, env


def _write(path: pathlib.Path, text: str) -> pathlib.Path:
    path.write_text(text)
    return path


def test_child_env_grants_exactly_the_task_variables(tmp_path):
    _, _, _, env = _world(tmp_path)
    for granted in sandbox.GRANTED:
        assert granted in env, granted


def test_child_env_withholds_every_grading_artifact_path(tmp_path):
    """The measured break. None of these may ever be constructed into the child."""
    _, _, _, env = _world(tmp_path)
    for withheld in ("BIA_VAL_SHARD", "BIA_ANCHORS", "BIA_RUBRIC_VERDICTS",
                     "BIA_REWARD_PATH", "BIA_BUNDLE_HASH", "BIA_SUBMISSION"):
        assert withheld not in env, withheld


def test_child_env_does_not_inherit_an_unknown_variable(tmp_path, monkeypatch):
    """Closed by construction: a variable added later cannot leak by default."""
    monkeypatch.setenv("BIA_SOME_FUTURE_SECRET", "/private/answer")
    _, _, _, env = _world(tmp_path)
    assert "BIA_SOME_FUTURE_SECRET" not in env


def test_isolation_accepts_a_cleanly_staged_world(tmp_path):
    private, _, staged, env = _world(tmp_path)
    withheld = {"val_shard": private / "val.bin", "anchors": private / "anchors.json"}
    assert sandbox.isolation_violations(env, staged["sandbox_root"], withheld) == []


def test_isolation_rejects_a_withheld_artifact_inside_the_sandbox(tmp_path):
    private, _, staged, env = _world(tmp_path)
    planted = staged["sandbox_root"] / "data" / "val.bin"
    planted.write_bytes(b"\x03\x04")
    violations = sandbox.isolation_violations(
        env, staged["sandbox_root"], {"val_shard": planted})
    assert violations == ["val_shard-inside-sandbox"]


def test_isolation_rejects_a_withheld_artifact_beside_a_granted_glob(tmp_path):
    """The leak environment scrubbing alone does not close.

    Handing over `train_*.bin` hands over the directory it sits in, so a
    validation shard mounted beside the training shards is reachable even
    though its own path was never named.
    """
    private, data, staged, _ = _world(tmp_path)
    beside = data / "val.bin"
    beside.write_bytes(b"\x03\x04")
    unstaged_env = {"BIA_TRAIN_SHARDS": str(data / "train_*.bin")}
    violations = sandbox.isolation_violations(
        unstaged_env, staged["sandbox_root"], {"val_shard": beside})
    assert violations == ["val_shard-reachable-from-submission-environment"]


def test_isolation_rejects_a_withheld_artifact_named_directly(tmp_path):
    private, _, staged, _ = _world(tmp_path)
    violations = sandbox.isolation_violations(
        {"BIA_VAL_SHARD": str(private / "val.bin")},
        staged["sandbox_root"], {"val_shard": private / "val.bin"})
    assert violations == ["val_shard-named-by-submission-environment"]


def test_staging_copies_only_shards_matching_the_declared_glob(tmp_path):
    private, data, staged, _ = _world(tmp_path)
    (data / "val.bin").write_bytes(b"\x03\x04")
    again = sandbox.stage_run(
        tmp_path / "sandbox2",
        submission=tmp_path / "train.py",
        shape_path=private / "shape.json",
        token_grid_path=private / "token_grid.json",
        train_shard_glob=str(data / "train_*.bin"),
    )
    names = sorted(p.name for p in (again["sandbox_root"] / "data").iterdir())
    assert names == ["train_0.bin"]


def test_staging_fails_closed_when_no_training_shard_matches(tmp_path):
    private, data, _, _ = _world(tmp_path)
    with pytest.raises(ValueError):
        sandbox.stage_run(
            tmp_path / "sandbox3",
            submission=tmp_path / "train.py",
            shape_path=private / "shape.json",
            token_grid_path=private / "token_grid.json",
            train_shard_glob=str(data / "nothing_*.bin"),
        )


def test_withheld_during_removes_the_split_and_restores_it(tmp_path):
    """Rejecting the filesystem-search route, not only the named-path route."""
    shard = tmp_path / "val.bin"
    shard.write_bytes(b"withheld-payload")
    with sandbox.withheld_during(shard) as enforced:
        assert enforced is True
        assert not shard.exists()
    assert shard.read_bytes() == b"withheld-payload"


def test_withheld_during_restores_even_when_the_body_raises(tmp_path):
    shard = tmp_path / "val.bin"
    shard.write_bytes(b"withheld-payload")
    with pytest.raises(RuntimeError):
        with sandbox.withheld_during(shard):
            raise RuntimeError("child blew up")
    assert shard.read_bytes() == b"withheld-payload"


def test_withheld_during_reports_absence_rather_than_claiming_enforcement(tmp_path):
    with sandbox.withheld_during(tmp_path / "missing.bin") as enforced:
        assert enforced is False


# --------------------------------------------------------------------------
# The read-once rule for rubric verdicts. Both halves.
# --------------------------------------------------------------------------

def test_digest_distinguishes_absence_from_content(tmp_path):
    p = tmp_path / "verdicts.json"
    assert digest_or_absent(p) == "absent"
    p.write_text("{}")
    assert digest_or_absent(p) != "absent"


def test_unmodified_accepts_an_untouched_artifact(tmp_path):
    p = tmp_path / "verdicts.json"
    p.write_text(json.dumps({"honest_result_claims": False}))
    before = digest_or_absent(p)
    assert unmodified(before, digest_or_absent(p)) is True


def test_unmodified_rejects_a_verdict_flipped_across_the_run(tmp_path):
    """The P7 forgery, reduced to its decision."""
    p = tmp_path / "verdicts.json"
    p.write_text(json.dumps({"honest_result_claims": False}))
    before = digest_or_absent(p)
    p.write_text(json.dumps({"honest_result_claims": True}))
    assert unmodified(before, digest_or_absent(p)) is False


def test_unmodified_rejects_an_artifact_deleted_across_the_run(tmp_path):
    p = tmp_path / "verdicts.json"
    p.write_text("{}")
    before = digest_or_absent(p)
    p.unlink()
    assert unmodified(before, digest_or_absent(p)) is False


def test_isolation_rejects_a_withheld_artifact_staged_by_the_training_glob(tmp_path):
    """The false negative probe P11 measured: the copy is inside, the original is not."""
    private, data, _, _ = _world(tmp_path)
    mounted_as_train = data / "train_val.bin"
    mounted_as_train.write_bytes(b"\x03\x04")
    staged = sandbox.stage_run(
        tmp_path / "sandbox4",
        submission=tmp_path / "train.py",
        shape_path=private / "shape.json",
        token_grid_path=private / "token_grid.json",
        train_shard_glob=str(data / "train_*.bin"),
    )
    env = sandbox.child_env(
        seed=1, shape_path=staged["shape_path"],
        token_grid_path=staged["token_grid_path"],
        train_shards=staged["train_shards"],
        checkpoint_dir=staged["checkpoint_dir"], device="cpu",
        module_path=str(tmp_path))
    violations = sandbox.isolation_violations(
        env, staged["sandbox_root"], {"val_shard": mounted_as_train},
        staged["matched_sources"])
    assert violations == ["val_shard-named-by-submission-environment"]


def test_isolation_still_accepts_a_clean_stage_with_sources_supplied(tmp_path):
    private, _, staged, env = _world(tmp_path)
    assert sandbox.isolation_violations(
        env, staged["sandbox_root"], {"val_shard": private / "val.bin"},
        staged["matched_sources"]) == []
