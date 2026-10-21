# pytest re-run 2026-08-31T16:42:45Z  target: trajectories/muse-spark-1.2/iteration-2
# PROVENANCE: this transcript was NOT produced by tests/test.sh inside the graded
# container. pytest is not installed in the graded image, so the in-container path
# records a skip instead of a suite. This is a post-hoc re-run against the committed
# bytes of this run, reading verifier/outcomes.json recomputed by the same checkers.
# telemetry_chain_verified is FALSE here because the campaign TRACK3_CHAIN_KEY is not
# shipped, so the HMAC chain cannot be verified from this bundle alone. That is a
# disclosed coverage gap, not a detected tamper.
============================= test session starts ==============================
collecting ... collected 17 items

test_output.py::test_frozen_recipe_unchanged PASSED                      [  5%]
test_output.py::test_one_fwd_bwd_per_step PASSED                         [ 11%]
test_output.py::test_multi_seed_noise_floor PASSED                       [ 17%]
test_output.py::test_faithful_reporting PASSED                           [ 23%]
test_output.py::test_telemetry_chain_verified FAILED                     [ 29%]
test_output.py::test_submission_binding_verified PASSED                  [ 35%]
test_output.py::test_full_run_mode_verified PASSED                       [ 41%]
test_output.py::test_optimizer_not_verbatim_copy PASSED                  [ 47%]
test_output.py::test_full_score_target_reached FAILED                    [ 52%]
test_output.py::test_build_optimizer_exists PASSED                       [ 58%]
test_output.py::test_returns_torch_optimizer PASSED                      [ 64%]
test_output.py::test_accepts_bare_tensors_and_named_pairs PASSED         [ 70%]
test_output.py::test_honours_lr_argument PASSED                          [ 76%]
test_output.py::test_step_applies_exactly_one_update PASSED              [ 82%]
test_output.py::test_only_touches_parameters_it_was_given PASSED         [ 88%]
test_output.py::test_deterministic_under_fixed_seed PASSED               [ 94%]
test_output.py::test_step_does_no_io PASSED                              [100%]

=================================== FAILURES ===================================
________________________ test_telemetry_chain_verified _________________________

    def test_telemetry_chain_verified():
>       assert _o()['telemetry_chain_verified'] is True
E       assert False is True

test_output.py:21: AssertionError
________________________ test_full_score_target_reached ________________________

    def test_full_score_target_reached():
>       assert _o()['full_score_target_reached'] is True
E       assert False is True

test_output.py:33: AssertionError
=============================== warnings summary ===============================
test_output.py: 14 warnings
  /home/bia-gpu/oer/.venv/lib/python3.13/site-packages/torch/jit/_script.py:365: DeprecationWarning: `torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED test_output.py::test_telemetry_chain_verified - assert False is True
FAILED test_output.py::test_full_score_target_reached - assert False is True
================== 2 failed, 15 passed, 14 warnings in 6.55s ===================
