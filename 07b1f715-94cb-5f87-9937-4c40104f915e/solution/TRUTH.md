# TRUTH: bia/s02-optimizer-search-frozen-schedule

GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml.

## What this task is

Slot S02, optimizer-search-frozen-schedule. The learning-rate schedule is frozen, published at environment/frozen_schedule.py and unmodifiable. The update rule is the only free axis. The reward is the earliest sustained multi-seed crossing of the target validation loss, mapped onto the closed interval from zero to one by the spec scoring form.

## Frozen surface

- the learning-rate schedule, published at environment/frozen_schedule.py and unmodifiable
- the per-group learning-rate multiplier table and the base learning rate
- the dataset, its token order, the batch shape and the sequence length
- the architecture, its parameter-group partition and its initialization
- one forward-backward pass per optimizer step
- the validation path and the weights validation reads

## Free surface

- the update rule only, meaning the map from gradient and optimizer state to a parameter delta
- the optimizer's own internal state, its initialization and its hyperparameters

## Defeat mechanism, stated as design intent and marked UNMEASURED

Every published record co-designs its update rule with its schedule, so the reported gain is a property of the pair and not of either half. This slot freezes a schedule that no published record was tuned against: a four-tenths-of-one-percent warmup, a discontinuous restart drop at forty percent of the run, and a tail that floors at a quarter of peak with no terminal anneal. A ported record therefore arrives with its own missing half. The rule is additionally never told the step horizon, so the cooldown a record relies on cannot be reimplemented inside the update, and the verifier enforces that the applied delta is exactly linear in the harness learning rate. What is left is the question the slot asks, which is which part of the record actually carried the gain.

Status: UNMEASURED. Measured only by external signed pilot over frozen bytes against the pinned cohort. No pass probability, no self-solve claim and no difficulty tier is asserted anywhere in this bundle.

## Operating point

6 hours per session divided by 50 attempts equals 7.2 minutes per graded attempt

| Field | Value |
|---|---|
| budget_hours | 0.12 |
| max_timeout hours | 6.0 |
| max_attempts | 50 |
| envelope | one H100, single accelerator, no multi-GPU scaling |
| wall clock | MEASURED |

The runner carries a wall-clock guard of 420 seconds per graded attempt, so the per-attempt budget is an enforced byte rather than a claim. A run that trips the guard stops, records stop_reason wallclock_guard, and is graded on the telemetry it produced.

Recorded under measured_reference below. The measurement was taken on an H100 shared with concurrent sibling lanes, so it is an upper bound on the same run on an idle card and it is not a difficulty measurement.

## Calibration sweep the anchors are bound from

Measured 2026-08-21 on one H100 80GB shared with four concurrent sibling lanes, so every wall clock recorded here is an upper bound on the same run on an idle card and none of them is a difficulty measurement. Crossing steps below are computed with the same seed-mean, noise-floor, every-seed and sustained rule that tests/grade.py applies, run offline over the runner's own logs.

| Id | Rule | Seeds | Steps run | Observed |
|---|---|---|---|---|
| M01 | environment/starter/update_rule.py | 0, 1 | 1400 | val_loss plateaus by step 500 and ends at 7.166 for seed 0 and 7.192 for seed 1, a two-seed mean of 7.179. The lowest target it crosses inside the horizon is 7.25, at step 1125. It never reaches the 5.x region, so it cannot bind baseline_steps at any target the reference is graded against. |
| M02 | solution/reference_update_rule.py | 0, 1 | 1400 | Two-seed mean 5.3939 at step 975, 5.3030 at 1075, 5.3073 at 1100, 5.2748 at 1125 and 5.0563 at 1400. First sustained two-seed crossing of 5.30 is step 1125, stable under every sig_margin from 0.010 to 0.030. The 1075 dip is not sustained, because the mean rises back above 5.30 at step 1100. Mean absolute seed-to-seed difference over the recorded steps from 200 onward is 0.0321, which is a seed-to-seed standard deviation of 0.0227, and sig_margin is bound to 0.023 from it. |
| M03 | the reference with its orthogonalized hidden-matrix answer removed, one ablation | 0 | 1400 | Ends at 6.851, which is 1.79 above the reference at the same step. The orthogonalized hidden-matrix branch is what carries the gain, and the ablation does not reach the target inside the horizon. |
| M04 | reference variant, orthogonalized on every matrix, momentum warm-started from the first gradient | 0 | 1400 | Ends at 5.054 against the reference's 5.068 at seed 0, and is behind the reference for most of the run. Its crossing of 5.30 is step 1150, later than the reference's. |
| M05 | the M04 variant with the agreement contraction floor raised to one | 0 | 1400 | Ends at 5.015 against the reference's 5.068 at seed 0, the best terminal loss in the sweep, but its crossing of 5.30 is step 1125, the same step as the reference. The contraction floor is nearly inert because the measured agreement stays high, so raising it buys terminal loss and no step count. |

Across five measured runs the earliest sustained crossing of 5.30 is step 1075, from seed 1 of the reference alone, and the earliest two-seed crossing is 1125. No measured rule reaches the target before the frozen schedule's tail begins at step 981, which is why target_steps is bound to 980 and why the reference scores strictly inside the reward interval rather than at either end of it.

## Measured reference at the graded operating point

| Field | Value |
|---|---|
| command | bash solution/solve.sh |
| profile | full |
| date | 2026-08-21 |
| device | one H100 80GB shared with four concurrent sibling workloads |
| steps_run | 1400 |
| stop_reason | completed |
| graded_step | 1125 |
| score | 0.6547619047619048 |
| checkers_true | 13 of 13 |
| repeats | 4 |
| score_across_repeats | identical at 0.6547619047619048 with graded_step 1125 in all four |
| wallclock_two_seed_total_s | [481, 343, 265, 433] |
| wallclock_slowest_invocation_per_repeat_s | [239, 190, 165, 230] |
| wallclock_best_invocation_s | 89 |

The reference lands at an interior point of the reward map, so the reward can still rank a submission above the reference and a reference failure is still distinguishable from a reference success. This is a reference measurement and it is not a difficulty measurement, and no pass probability, self-solve claim or difficulty tier follows from it.

The graded attempt was run four times against different levels of contention on the same card and returned the same score every time, at two-seed totals of 481, 343, 265 and 433 seconds. The fastest single invocation of 1400 steps finished in 89 seconds. Every invocation finished inside the runner's 420 second guard with stop_reason completed. Two of the four totals sit inside the 432 second per-attempt budget and two, taken while sibling workloads shared the card, sit above it. The spread is a property of the measuring host rather than of the bundle, and the figure on an idle card is carried as the named gap idle-card-wallclock-unmeasured rather than estimated away.

BIA_SMOKE=1 bash solution/solve.sh over the same reference bytes and the same runner returns graded_step 28 and score 0.5 with all thirteen checkers true, unchanged by this calibration.

## Golden trajectory

| Step | Action | Established state | Checkers |
|---|---|---|---|
| T01 | Read environment/frozen_schedule.py and establish the shape of the frozen schedule, including the short warmup, the restart drop at forty percent and the tail floor at a quarter of peak. | the frozen surface is known and the free surface is bounded to the update rule | C01, C02 |
| T02 | Copy environment/starter/update_rule.py to submission/update_rule.py and replace the update rule, keeping the build_update_rule entry point and the frozen group names. | a submission exists at the graded path exposing the bound entry point | C03, C05 |
| T03 | Answer the cold start inside the rule by initializing second moments from the first observed gradient rather than from zero. | the rule is scaled from its first step under a four-tenths-of-one-percent warmup | C03 |
| T04 | Answer the restart bump by updating hidden matrices along an orthogonalized momentum direction whose spectral scale is fixed by construction, so a change in applied learning rate rescales the step without changing its geometry. | the search direction is invariant to the schedule's discontinuity | C03, C11 |
| T05 | Answer the missing terminal anneal with a horizon-free contraction driven by the agreement between the current gradient and the accumulated momentum. | the effective step contracts as the run becomes noise-dominated, without knowing the horizon | C04, C05 |
| T06 | Run the runner at the graded profile for seed 0 and seed 1, letting the runner write both the telemetry chain and the logs. | two seeds of harness-owned telemetry and matching logs exist | C06, C07, C09, C10, C12 |
| T07 | Report the losses the runner logged, without editing them, and derive the graded step by applying the seed-mean and noise-floor rule to that series. | the reported result reconciles with telemetry in both directions | C08, C13 |

## Rejected routes and the control that rejects each

| Route | Negative control | Rejected by |
|---|---|---|
| edit environment/frozen_schedule.py to restore a terminal anneal | NC01 | C01 and C02 |
| write a learning rate into a param group inside step() | NC02 | C04 |
| reimplement the schedule inside the update rule by scaling the applied lr | NC03 | C03 |
| submit a verbatim published record from the pinned corpus | NC04 | C11 |
| hand-write logs that telemetry does not contain | NC05 | C08 |
| run the tiny proof profile and present it as a graded result | NC06 | C12 |

## Checkers and their reduction

| Id | Name | Kind | Live-state read |
|---|---|---|---|
| C01 | frozen_schedule_bytes_match | VALUE | sha256 of the file at /workspace/environment/frozen_schedule.py, read from the container filesystem at verification time |
| C02 | applied_lr_matches_frozen_schedule | DIVERGENCE | the lr_applied map of every record the verifier received from the training process it spawned, compared against the verifier-private reimplementation in tests/checkers/schedule_ref.py |
| C03 | update_is_lr_homogeneous | EFFECT | parameter tensor values before and after a real optimizer.step() call on the submitted rule, executed twice from identical state at two harness-supplied learning rates, inside a process the verifier spawns so no submitted byte is executed in the grader itself |
| C04 | submission_never_writes_lr | ABSENCE | the lr_write_violations counter in every record the verifier received from its own training process, which that process increments when a post-step read of a param group learning rate differs from the value it wrote pre-step |
| C05 | no_horizon_leak_into_update_rule | ABSENCE | two surfaces, both required: the AST of the submitted bytes parsed at verification time under an import allowlist, a ban on constructing names at runtime, and rejection of any constant-foldable literal equal to the denied horizon or a simple multiple of it; and the measured update magnitude of the rule driven for three horizons of steps under a constant gradient and a constant learning rate inside a process the verifier spawns |
| C06 | one_forward_backward_per_step | INVARIANT | the forward_calls_this_step and backward_calls_this_step counters the harness measures from hooks it attached to the model it owns, together with the forward_backward_cumulative and steps_checked counters, in every record of every seed |
| C07 | telemetry_chain_ordered | ORDERING | the record sequence the verifier itself assembled from the stream its training process sent, indexed and chained by the verifier under a key it mints per grading pass and never shares |
| C08 | reported_logs_reconcile | DIVERGENCE | validation losses parsed from /workspace/submission/logs against the loss series the verifier measured for itself during its own re-execution, compared in both directions |
| C09 | frozen_recipe_fields_match | VALUE | the architecture, batch_sequences, frozen, model_digest, train_shard_digest and val_shard_digest fields of every record of the verifier's own run, together with the parameter shapes its measurement process read off every streamed checkpoint |
| C10 | no_out_of_step_parameter_mutation | ABSENCE | the out_of_step_mutations counter in every record of the verifier's own run, which its training process increments when the parameter probe taken at the end of a step differs from the probe taken at the start of the next one |
| C11 | update_rule_not_record_replay | ABSENCE | the AST-normalized form of the submitted bytes compared against every pinned record in tests/corpus/ |
| C12 | run_profile_matches_verifier | INVARIANT | the profile field of every record, and the batch shape and sequence length the training process observed off the real tensors, compared against the profile the verifier was invoked to grade |
| C13 | graded_step_reaches_target | VALUE | the per-seed validation loss series the verifier's measurement process computed itself from the parameters each evaluation point streamed to it |

## Anchors, and what is not established about them

Bound from the calibration sweep recorded under calibration_measurement below and not from ambition. target_loss is the level the reference rule's two-seed mean is measured to cross at step 1125 of 1400 while finishing 0.24 below it, so the crossing is sustained rather than a horizon artifact. baseline_steps is the graded horizon, which is where the measurement puts the baseline crossing: the shipped starter rule was run for the full 1400 steps at two seeds and ends at 7.179, and the reference stripped of its orthogonalized hidden-matrix answer ends at 6.851, so no measured rule weaker than the reference reaches this target inside the horizon at all. target_steps is the last step before the published frozen schedule enters its tail branch, which is RESTART_END_FRAC times total_steps, so full reward means reaching the target without spending any of the schedule's final thirty percent. The earliest crossing measured anywhere in the sweep is step 1075, which is above target_steps, so full reward is unattained by every rule measured here and the reward keeps headroom above the reference rather than saturating on it. Whether any rule attains full reward is unmeasured and is not claimed.

| Anchor | Value | Status |
|---|---|---|
| target_loss | 5.3 | CALIBRATED_FROM_MEASURED_RUN |
| baseline_steps | 1400 | CALIBRATED_FROM_MEASURED_RUN |
| target_steps | 980 | CALIBRATED_FROM_MEASURED_RUN |
| min_seeds | 2 | CALIBRATED_FROM_MEASURED_RUN |
| sig_margin | 0.023 | CALIBRATED_FROM_MEASURED_RUN |

## Negative controls, both halves proven

Each control was executed during authoring against the CPU proof profile. The clean fixture passes every checker and each planted defect is rejected with the named machine-readable reason, so no checker in this bundle is inert and none is hardcoded to reject everything.

| Control | Planted defect | Exercises | Observed |
|---|---|---|---|
| NC00 | nothing, the clean reference run | every checker | graded_step=28 score 0.5 |
| NC01 | an edited copy of environment/frozen_schedule.py | C01 | frozen_schedule_edited_digest_66a65c0a2d616480 |
| NC02 | an update rule that writes lr into its own param group | C04 | submission_wrote_lr_4_times_by_step_1 |
| NC03 | an update rule that reads the horizon from the environment | C05 | disallowed_import_os |
| NC04 | a verbatim copy of a pinned corpus record, and a renamed retuned copy of it | C11 | record_replay_of_rec_soap_lite_similarity_1.000 and similarity_0.999 |
| NC05 | one edited telemetry record left unsigned | C07 | chain_break_at_index_6 |
| NC06 | proof-profile telemetry presented to the full-profile verifier | C12 | profile_mismatch_smoke_expected_full |
| NC07 | an update rule that mutates parameters from zero_grad, outside step() | C10 | out_of_step_mutation_3_by_step_4 |
| NC08 | a forged learning rate in telemetry, re-signed with a valid chain | C02 | lr_divergence_group_hidden_matrix_step_16 |
| NC09 | a hand-edited validation loss in the agent log | C08 | reported_loss_diverges_seed_0_step_20 |
| NC10 | a single-seed run presented as a graded result | C13 | need_at_least_2_seeds_got_1 |
| NC11 | an update whose magnitude follows the square root of the supplied lr | C03 | update_not_linear_in_lr_residual_2.929e-01 |
| NC12 | a re-signed telemetry record claiming two forward-backward passes in one step | C06 | forward_backward_not_one_at_step_24 |
| NC13 | a re-signed telemetry record claiming a wider model | C09 | architecture_violation_d_model_step_1 |

## Coverage gaps carried at sign-off

| Id | Reason | Cap |
|---|---|---|
| full-reward-attainability-unmeasured | target_steps is bound to the last step before the frozen schedule's tail branch, and no rule in the calibration sweep was measured to reach the target that early. That the anchor keeps headroom above the reference is measured; that any rule attains full reward is not, and is not claimed. | HOLD:PILOT_REQUIRED |
| idle-card-wallclock-unmeasured | Every wall clock in this bundle was taken on an H100 shared with four concurrent sibling workloads, so each is an upper bound rather than the per-attempt figure the budget is written against. Four repeats of the graded attempt returned two-seed totals of 481, 343, 265 and 433 seconds, every invocation finished inside the 420 second guard, and the two most contended of the four exceeded the 432 second per-attempt budget. The same run on an idle card is owed and is not estimated here. | HOLD:PILOT_REQUIRED |
| difficulty-unmeasured | No pass probability, self-solve claim or difficulty tier is asserted. The defeat mechanism is recorded as design intent and is unmeasured until an external signed pilot over frozen bytes against the pinned cohort exists. | HOLD:PILOT_REQUIRED |
| image-digest-unpinned | task.toml pins the public base image by digest, and the digest of the image environment/Dockerfile builds on top of it does not exist until a campaign builds it. The field names the base and the gap is declared rather than filled with an invented digest. | HOLD:PILOT_REQUIRED |
| screening-unverified | The ENGRAM-owned contamination screening roots carry no pinned identity or digest in this project, so the exclusion list, the freeze-date table and the near-duplicate index cannot be resolved. The bundle ships its own fingerprint corpus, which is a family control and not a substitute for those roots. | HOLD:PILOT_REQUIRED |

## Disposition

This bundle ships at HOLD:PILOT_REQUIRED, which is the correct terminal state for a locally verified and externally unproven task. Only an external signed pilot over these frozen bytes can move it.
