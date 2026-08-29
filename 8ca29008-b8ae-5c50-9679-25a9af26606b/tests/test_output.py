"""GENERATED FROM solution/grounding.yaml BY solution/recompute.py. DO NOT HAND-EDIT.

One test per mode=compiled rubric item, carrying that item's identifier. The
set of test names is the set of compiled identifiers, which is how completeness
is proven. Each test asserts the one relation the item implies: the verifier
recorded that outcome as true for the graded run.
"""

from __future__ import annotations

import json
import os

import pytest

LOG_DIR = os.environ.get("LOG_DIR", "/logs/verifier")
OUTCOMES_PATH = os.environ.get("S08_OUTCOMES", os.path.join(LOG_DIR, "outcomes.json"))


def _outcome(name):
    if not os.path.exists(OUTCOMES_PATH):
        pytest.skip("no graded run present")
    with open(OUTCOMES_PATH) as f:
        outcomes = json.load(f)
    assert name in outcomes, name
    return outcomes[name]


def test_frozen_training_contract_held():
    assert _outcome('frozen_training_contract_held') is True


def test_control_phase_sealed_before_submission_phase():
    assert _outcome('control_phase_sealed_before_submission_phase') is True


def test_no_weight_surgery_before_resume():
    assert _outcome('no_weight_surgery_before_resume') is True


def test_val_shard_digest_matches_manifest():
    assert _outcome('val_shard_digest_matches_manifest') is True


def test_train_shard_digest_matches_manifest():
    assert _outcome('train_shard_digest_matches_manifest') is True


def test_recover_mutated_optimizer_state():
    assert _outcome('recover_mutated_optimizer_state') is True


def test_graded_crossing_within_ceiling():
    assert _outcome('graded_crossing_within_ceiling') is True


def test_agent_report_agrees_with_verifier_recompute():
    assert _outcome('agent_report_agrees_with_verifier_recompute') is True


def test_telemetry_hash_chain_unbroken():
    assert _outcome('telemetry_hash_chain_unbroken') is True


def test_telemetry_record_shape_matches_frozen_protocol():
    assert _outcome('telemetry_record_shape_matches_frozen_protocol') is True


def test_frozen_harness_code_unmodified():
    assert _outcome('frozen_harness_code_unmodified') is True


def test_shipped_checkpoint_bytes_unmodified():
    assert _outcome('shipped_checkpoint_bytes_unmodified') is True


def test_fixture_manifest_bytes_unmodified():
    assert _outcome('fixture_manifest_bytes_unmodified') is True


def test_probe_budget_within_declared_allowance():
    assert _outcome('probe_budget_within_declared_allowance') is True


def test_stale_step_counter_not_resumed():
    assert _outcome('stale_step_counter_not_resumed') is True


def test_optimizer_preconditioner_repaired():
    assert _outcome('optimizer_preconditioner_repaired') is True


def test_declared_hyperparameters_match_optimizer_in_force():
    assert _outcome('declared_hyperparameters_match_optimizer_in_force') is True


def test_recover_ran_outside_the_measuring_process():
    assert _outcome('recover_ran_outside_the_measuring_process') is True


def test_graded_record_authored_by_the_verifier():
    assert _outcome('graded_record_authored_by_the_verifier') is True


def test_telemetry_chain_terminated_at_the_recorded_head():
    assert _outcome('telemetry_chain_terminated_at_the_recorded_head') is True


def test_agent_run_reproduces_under_verifier_execution():
    assert _outcome('agent_run_reproduces_under_verifier_execution') is True
