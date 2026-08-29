"""GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml.

One test per compiled rubric item. Each asserts the outcome entry the verifier
wrote after reading live state, so a test cannot pass on an unrun checker.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checkers import outcomes


def _o():
    return outcomes.load()


def test_r01_applied_lr_matches_frozen_schedule():
    """R01 | DIVERGENCE | The applied learning rate at every recorded step of every parameter group equals the frozen schedule recomputed independently by the verifier."""
    assert _o()['applied_lr_matches_frozen_schedule'] is True


def test_r02_frozen_schedule_bytes_match():
    """R02 | VALUE | The frozen schedule module the run used is byte-identical to the module this bundle ships."""
    assert _o()['frozen_schedule_bytes_match'] is True


def test_r03_update_is_lr_homogeneous():
    """R03 | EFFECT | The submitted update rule applies a parameter delta exactly proportional to the learning rate the harness supplies."""
    assert _o()['update_is_lr_homogeneous'] is True


def test_r04_no_horizon_leak_into_update_rule():
    """R04 | ABSENCE | The submitted update rule has no route by which it could learn the step horizon."""
    assert _o()['no_horizon_leak_into_update_rule'] is True


def test_r05_submission_never_writes_lr():
    """R05 | ABSENCE | No write of a learning rate by the submission occurred at any step of the run."""
    assert _o()['submission_never_writes_lr'] is True


def test_r06_one_forward_backward_per_step():
    """R06 | INVARIANT | Exactly one forward-backward pass occurred per optimizer step across the whole run."""
    assert _o()['one_forward_backward_per_step'] is True


def test_r07_telemetry_chain_ordered():
    """R07 | ORDERING | The telemetry records form one unbroken chain in the required order from genesis."""
    assert _o()['telemetry_chain_ordered'] is True


def test_r08_reported_logs_reconcile():
    """R08 | DIVERGENCE | The validation losses the agent reported and the losses the harness recorded agree in both directions."""
    assert _o()['reported_logs_reconcile'] is True


def test_r09_frozen_recipe_fields_match():
    """R09 | VALUE | Every frozen recipe field recorded by the runner equals the constant this bundle binds."""
    assert _o()['frozen_recipe_fields_match'] is True


def test_r10_no_out_of_step_parameter_mutation():
    """R10 | ABSENCE | No parameter mutation outside optimizer.step() occurred at any point in the run."""
    assert _o()['no_out_of_step_parameter_mutation'] is True


def test_r11_update_rule_not_record_replay():
    """R11 | ABSENCE | The submitted update rule is not a replay of any pinned published record above the fingerprint threshold."""
    assert _o()['update_rule_not_record_replay'] is True


def test_r12_run_profile_matches_verifier():
    """R12 | INVARIANT | Every telemetry record carries the profile the verifier grades, so a proof-scale run cannot be graded as a full one."""
    assert _o()['run_profile_matches_verifier'] is True


def test_r13_graded_step_reaches_target():
    """R13 | VALUE | An earliest sustained multi-seed crossing of the target loss exists in the recorded series, which is the quantity the reward is computed from. Full reward additionally requires that crossing to be at or below target_steps, and the reward form measures the shortfall rather than this checker."""
    assert _o()['graded_step_reaches_target'] is True
