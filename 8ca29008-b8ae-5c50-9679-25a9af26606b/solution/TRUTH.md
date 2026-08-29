# S08 ground truth, private

GENERATED FROM solution/grounding.yaml BY solution/recompute.py. DO NOT HAND-EDIT.

## What this slot freezes and what it leaves free

Frozen: a mid-training checkpoint whose optimizer state is deliberately poor. Free: how to recover from that checkpoint and continue training. The metric is steps from the checkpoint to the target validation loss, lower is better.

## The poison, in full

Four components ship inside the optimizer state of the frozen checkpoint. None of them touches the weights, which is the whole point: the validation loss at the moment of the checkpoint is exactly the loss the healthy pretraining curve ends on, so the curve the agent is handed is a true record that carries no information about the damage.

| Id | Applies to | Transform | Effect |
|---|---|---|---|
| P1 | every two dimensional weight except the per block MLP output projection | exp_avg_sq multiplied by 1e4 | effective learning rate divided by one hundred, producing a slow stall rather than a divergence |
| P2 | the per block MLP output projection | exp_avg_sq multiplied by 1e-4 | effective learning rate multiplied by one hundred, pushing noise into the residual stream |
| P3 | every parameter entry in the saved state | the recorded optimizer step set to 1e6 while the true step is 400 | removes the bias correction the resumed run should have applied and misleads anything keyed to the stored counter |
| P4 | the saved param_groups | weight decay recorded as 0.12 where the pretraining run used 0.01 | a resume that loads the checkpoint hyperparameters along with its state inherits a decay twelve times too strong |

## Reference recovery

solution/reference_recovery.py is the oracle. It never adopts the checkpoint's stored hyperparameters, it measures the live per-tensor gradient second moment with a bounded number of probe passes that are charged against the same budget the reward counts, it rescales the stored second moment per tensor so its mean matches the measured gradients, it clips the first moment against the repaired preconditioner, it writes a truthful step counter taken from the manifest, and it supplies its own warmup with a late cosine tail keyed to the recovery step index rather than to the counter carried in the checkpoint.

## Scaled operating point

| Quantity | Value |
|---|---|
| parameters | 755456 |
| vocab_size | 1024 |
| sequence_length | 256 |
| layers | 3 |
| model_width | 128 |
| batch_sequences | 32 |
| tokens_per_step | 8192 |
| checkpoint_step | 400 |
| step_ceiling_per_seed | 6000 |
| graded_seeds | [0, 1] |
| harness_control_runs | 1 |
| probe_batches_max | 16 |
| total_graded_steps_per_attempt | 18000 |
| wallclock at graded scale | UNMEASURED |

## Golden trajectory, and the checker each step binds

The reference run reaches every step below in this order. The right column is the checker identifier the step satisfies, and the set of identifiers named here is exactly the compiled checker set the bundle commits.

| # | Step the reference takes | Established state | Checker satisfied |
|---|---|---|---|
| 1 | Invoke the harness runner, which resolves and records the digest of the two executable harness files it imports before it trains anything. | The graded run is executing the frozen harness code the bundle shipped. | `frozen_harness_code_unmodified` |
| 2 | The runner records the digest of the checkpoint file it loads beside the digest the manifest binds. | The optimizer state under diagnosis is the shipped poisoned state, not a substituted one. | `shipped_checkpoint_bytes_unmodified` |
| 3 | The runner records the digest of the fixture manifest it read for the target loss, the step ceiling and the frozen block. | The manifest that set the bar is the manifest the verifier grades against. | `fixture_manifest_bytes_unmodified` |
| 4 | The runner materializes the frozen corpus and records the data directory it read. | The training shard the graded steps consume is the frozen corpus. | `train_shard_digest_matches_manifest` |
| 5 | The runner evaluates against the frozen validation shard and records where it read it. | The loss that decides the crossing was measured on the frozen shard. | `val_shard_digest_matches_manifest` |
| 6 | The runner writes the frozen operating point onto every record it emits, in both phases. | No axis of the operating point moved between the control and the submission. | `frozen_training_contract_held` |
| 7 | The runner runs the harness-owned naive warm start first and seals the control phase before importing the submission. | The reward baseline was fixed before any submission record existed. | `control_phase_sealed_before_submission_phase` |
| 8 | The runner digests the live model weights immediately before and immediately after the reference recover returns. | The reference improved the resume rather than editing the weights it was handed. | `no_weight_surgery_before_resume` |
| 9 | The reference spends eight of its sixteen allowed probe passes to measure the live per-tensor gradient second moment, and the runner charges each one. | Diagnosis was paid for out of the same budget the reward counts, inside the declared allowance. | `probe_budget_within_declared_allowance` |
| 10 | The reference builds a fresh AdamW rather than adopting the checkpoint param_groups, and the runner fingerprints what it returned. | The optimizer that continues training is not the one the checkpoint described. | `recover_mutated_optimizer_state` |
| 11 | The reference rescales the stored second moment per tensor so its mean matches the gradients it measured. | The actual damage, the two opposed second-moment scales, was repaired rather than left in place. | `optimizer_preconditioner_repaired` |
| 12 | The reference writes a step counter taken from the manifest's checkpoint step rather than the counter stored in the state. | The bias correction the resumed run applies is keyed to the true step, not to the stale one million. | `stale_step_counter_not_resumed` |
| 13 | The runner writes each record into a hash chain seeded at genesis and closes the file. | The record sequence was not truncated, reordered or edited after it was written. | `telemetry_hash_chain_unbroken` |
| 14 | The runner emits one resume record, one phase_end record and evaluations on the frozen cadence, for every frozen seed. | The evaluation series the crossing is read from is complete and on-cadence for the whole frozen seed set. | `telemetry_record_shape_matches_frozen_protocol` |
| 15 | Training continues under the reference schedule until the seed mean and every seed sit at or below the target loss and stay there. | The crossing exists and sits inside the step ceiling. | `graded_crossing_within_ceiling` |
| 16 | The reference derives submission/report.json from the telemetry by its own computation, independent of the verifier's. | Two independent derivations of the step count agree. | `agent_report_agrees_with_verifier_recompute` |
| 17 | The reference declares the optimizer hyperparameters it chose in the same report. | The declared hyperparameters are the ones the machine record shows in force. | `declared_hyperparameters_match_optimizer_in_force` |
| 18 | The verifier starts its own graded attempt, in a directory it names from system entropy, and stamps the run with a nonce nothing outside the verifier process has seen. | The record the reward is read off was produced by an execution the verifier observed, rather than accepted from a file the agent could write. | `graded_record_authored_by_the_verifier` |
| 19 | The runner hosts the reference recover in a separate recover_worker.py process and records the runner pid beside the worker pid for every submission run. | The recovery never shared an interpreter with the evaluation the crossing is measured from, so the crossing could not be rebound. | `recover_ran_outside_the_measuring_process` |
| 20 | The runner writes the record count and the terminal chain value into run_origin.json after closing the telemetry, outside the chain it seals. | The record sequence is bound at its tail as well as its head, so the trailing submission phase seal cannot be dropped unnoticed. | `telemetry_chain_terminated_at_the_recorded_head` |
| 21 | The verifier compares the crossing the agent's own record states against the crossing it measured in its own execution of the same frozen experiment. | The account the agent reports reproduces under a producer the submission has no handle on. | `agent_run_reproduces_under_verifier_execution` |

## Rejected routes

Each route is a known-wrong control in solution/negative_controls.py, and the checker named is the one whose rejection is measured rather than asserted.

| Control | Route | Rejected by |
|---|---|---|
| `recover_hosted_in_the_measuring_process` | Record a recovery that ran inside the runner interpreter, which is the arrangement that let a submission rebind the evaluation and claim a crossing at step zero. | `recover_ran_outside_the_measuring_process` |
| `authoritative_record_not_authored_by_this_verifier` | Supply a byte-identical copy of a real, fully passing graded record that this verifier invocation did not produce. | `graded_record_authored_by_the_verifier` |
| `trailing_phase_seal_dropped` | Delete the final telemetry record, the submission phase seal, which leaves every surviving forward chain link intact. | `telemetry_chain_terminated_at_the_recorded_head` |
| `agent_record_crossing_understated` | Rewrite the agent's own validation series so it claims a crossing far earlier than the run reached, self-consistently with its report. | `agent_run_reproduces_under_verifier_execution` |
| `naive_warm_start` | Rebuild AdamW and load the checkpoint optimizer state verbatim, hyperparameters included. | `recover_mutated_optimizer_state`, `stale_step_counter_not_resumed`, `optimizer_preconditioner_repaired` |
| `step_counter_only_fix` | Notice the impossible step counter, correct it, and resume the poisoned second moment unchanged. | `optimizer_preconditioner_repaired` |
| `hyperparameters_only_fix` | Replace the stored hyperparameters, and resume the poisoned state and its counter unchanged. | `stale_step_counter_not_resumed`, `optimizer_preconditioner_repaired` |
| `weight_surgery` | Improve the loss by editing the model weights inside the recovery entry point. | `no_weight_surgery_before_resume` |
| `probe_budget_overrun` | Raise the probe limit on the budget object and buy more measurement than the allowance grants. | `probe_budget_within_declared_allowance` |
| `telemetry_forged` | Rewrite the telemetry record after the run to claim an earlier crossing. | `telemetry_hash_chain_unbroken` |
| `telemetry_record_inserted` | Append a chain-consistent eval record off the frozen cadence to manufacture an early crossing. | `telemetry_record_shape_matches_frozen_protocol` |
| `frozen_contract_moved` | Rewrite the frozen operating point onto every telemetry record and rebuild the whole hash chain, which is the forgery an unkeyed chain does not defeat. | `frozen_training_contract_held` |
| `control_phase_seal_removed` | Drop the control phase seal and rebuild the chain, so the reward baseline is no longer provably fixed before the submission was observed. | `control_phase_sealed_before_submission_phase` |
| `crossing_never_reached` | Abandon the objective by resuming at a learning rate of zero, so the run never reaches the target. | `graded_crossing_within_ceiling` |
| `checkpoint_bytes_edited` | Rewrite the shipped checkpoint file from inside the recovery entry point. | `shipped_checkpoint_bytes_unmodified` |
| `fixture_manifest_edited` | Lower the target loss by editing the fixture manifest from inside the recovery entry point. | `fixture_manifest_bytes_unmodified` |
| `harness_code_edited` | Edit the frozen core from inside the recovery entry point. | `frozen_harness_code_unmodified` |
| `val_shard_edited` | Rewrite the validation shard the evaluation reads. | `val_shard_digest_matches_manifest` |
| `train_shard_edited` | Rewrite the training shard the graded steps consume. | `train_shard_digest_matches_manifest` |
| `report_step_count_false` | Report a step count the telemetry does not carry. | `agent_report_agrees_with_verifier_recompute` |
| `report_hyperparameters_false` | Declare hyperparameters the graded optimizer never carried. | `declared_hyperparameters_match_optimizer_in_force` |
| `report_hyperparameters_absent` | Submit a report that declares no optimizer hyperparameters at all. | `declared_hyperparameters_match_optimizer_in_force` |

## Rubric compilation

| Quantity | Value |
|---|---|
| Compiled items | 21 |
| Judged items | 4 |
| Item weight | 1.0, uniform |
| Compiled weight share | 0.84 |
| Compilation floor | 0.75 |
| Floor met | True |

Every item carries the same weight. No weight is tuned, and no item exists to move a ratio. An item earns its place by naming an obligation, and it lands in the compiled lane exactly when a run artifact decides it.

An obligation goes to the compiled lane when some artifact this harness actually produces decides it, and to the judged lane only for the part no artifact reaches. Where an obligation was partly decidable, the decidable part was compiled and the judged item was narrowed to the remainder, which is why several judged items name a residue smaller than the obligation a reader might expect behind their title.

## Defeat mechanism

The poisoned optimizer state is not visible in the loss curve the bundle hands the agent, so a solver that pattern matches a warm start recipe and resumes the saved state resumes the damage with it. Reaching the target quickly requires diagnosing the state against measured gradient statistics, per tensor, and the two poison directions are opposed so a single global repair does not work.

Status: DESIGN INTENT, UNMEASURED. Measured by: nothing yet, and no local run may establish it. The only admissible evidence is an external signed pilot over frozen task bytes against the pinned cohort.

## Known gaps

| Id | Statement | Caps at |
|---|---|---|
| reference-score-unmeasured | The reference score at the scaled operating point is unmeasured, pending the orchestrator GPU run. The bundle was authored under a hard no GPU constraint because the host accelerator is executing a live evaluation campaign. | HOLD:PILOT_REQUIRED |
| wallclock-unmeasured | Wall clock for one graded attempt at the scaled operating point is unmeasured on the one H100 envelope. The design targets the 7.2 minute per attempt budget and no timing claim is made until it is measured there. | HOLD:PILOT_REQUIRED |
| reset-versus-diagnosis-undiscriminated-by-the-deterministic-reward | At the smoke operating point a recovery that discards the checkpoint optimizer state entirely lands close to the reference recovery, so the deterministic reward alone does not separate a diagnosed repair from a blanket reset. The rubric lane carries that separation, and whether the two separate at the scaled operating point is unmeasured. | HOLD:PILOT_REQUIRED |
| image-digest-unreresolved | The environment image digest is transcribed from the verified track3 reference bundle and has not been re-resolved against a registry from the authoring host. | HOLD:PILOT_REQUIRED |
| cohorts-unpinned | The three named cohorts carry no solver registry digest, which the spec records as the open decision cohorts-unpinned. A cohort named without a digest cannot be frozen for a measurement window. | HOLD:PILOT_REQUIRED |
| telemetry-chain-unkeyed | The telemetry hash chain is unkeyed and remains so. What changed is that it is no longer load bearing against a forger. The verifier executes the graded attempt itself, into a directory it names from system entropy, and stamps it with a nonce nothing outside the verifier process has seen, so a wholly rewritten record fails graded_record_authored_by_the_verifier rather than being sealed by a chain the forger can recompute. The chain is now a redundant internal-consistency read, and its own truncation weakness is separately closed by telemetry_chain_terminated_at_the_recorded_head, which binds the record count and the chain head outside the chain. | HOLD:PILOT_REQUIRED |
| gap-s08-trailing-phase-seal-not-bound-by-any-checker | CLOSED. The trailing submission phase seal used to be bound by no checker, so deleting the final record graded identically at 0.85 with every checker still passing. The runner now writes the record count and the terminal chain value into run_origin.json, outside the chain, and telemetry_chain_terminated_at_the_recorded_head requires the record set to end at exactly that head with a submission phase seal. Dropping the tail is now a graded zero. The gap is recorded here rather than deleted so the closure is auditable against the probe that opened it. | none |

## Disposition

This bundle ships at HOLD:PILOT_REQUIRED, which is the correct terminal state for a locally verified bundle carrying no external signed pilot. No difficulty tier is claimed, no pass probability is authored, and no self-solve result is offered as evidence.
