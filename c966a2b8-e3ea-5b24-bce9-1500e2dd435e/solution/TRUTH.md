# TRUTH: slot S10, reproduction-under-ablation

<!-- GENERATED SECTION. DO NOT HAND-EDIT. Regenerate with: python3 solution/recompute.py -->

This file is private ground truth. It is generated from solution/grounding.yaml by solution/recompute.py and it never crosses the agent-visible boundary.

## What is frozen and what is free

Frozen: a published result and one ablated component the model must compensate for. Free: everything else in the optimizer. The reward is on recovering the published metric without the ablated component.

## The published result this slot freezes

Pre-normalization, an RMSNorm on the input of every residual branch plus a final norm before the output head, is what makes a GPT-style transformer trainable at a single global learning rate without warmup, and removing it collapses training under the same recipe.

Lineage:

- Xiong et al. 2020, On Layer Normalization in the Transformer Architecture, arXiv:2002.04745. Establishes that pre-normalization placement is what removes the warmup requirement and stabilizes large learning rates.
- Zhang and Sennrich 2019, Root Mean Square Layer Normalization, arXiv:1910.07467. The RMSNorm form the frozen un-ablated model uses.
- The modded-nanogpt track 3 record lineage, which is the corpus the consolidated spec names as F1's primary leakage source, and every record in which is co-designed with a pre-normalized architecture.

The harness reproduces the published result itself, at the scaled operating point, as the anchor_target arm: the un-ablated model driven by the frozen reference recipe. The published number is therefore measured on the grading device at grade time and is never a number an author typed.

## The ablated component

Every activation normalization operation in the model's forward pass. It is the component the frozen published result attributes its gain to, so ablating it is ablating the carrier rather than ablating a bystander.

Precise scope: The module classes LayerNorm, RMSNorm, GroupNorm, BatchNorm1d, BatchNorm2d, BatchNorm3d, SyncBatchNorm, InstanceNorm1d, InstanceNorm2d, InstanceNorm3d, LocalResponseNorm and CrossMapLRN2d, and the torch.nn.functional entry points layer_norm, rms_norm, group_norm, batch_norm, instance_norm, local_response_norm and normalize.

Explicitly out of scope: The softmax inside scaled_dot_product_attention is part of the frozen attention operator and is not the ablated component. Normalizing an update tensor inside the submitted rule is an optimizer-side operation, is explicitly permitted, and is not activation normalization.

## How the harness enforces the ablation

The enforcement is harness, not prose. The instrument is environment/substrate/probe.py, class NormProbe. Two independent interception points are installed so a single bypass does not blind it.

- Every torch.nn.functional normalization entry point is wrapped in the live process, so a direct functional call from anywhere, including from inside the submitted update rule, writes an event.
- A forward pre-hook is registered on every module of the live model, so a normalization module smuggled into the module tree at runtime writes an event and the distinct set of executed module classes is recorded per phase.

Every event carries the phase that executed it. Events in forward, backward and eval are forbidden. Events in update are recorded and permitted, because the compensation the task asks for lives there.

Live-state read: The arm_probe_trace and arm_complete records in /logs/verifier/run_record.jsonl, specifically forbidden_norm_events, norm_parameter_names and module_classes_by_phase.

Liveness witness: The un-ablated anchor_target arm runs through the same probe and must record a strictly positive count of the same forbidden-phase events. If it records zero the probe is blind and the ABSENCE checker refuses to certify anything, with reason probe_inert_no_witness.

The submission surface is one file exporting one factory that returns an object with a step method. The model, the data, the batch order and the loop are constructed by harness-owned code whose bytes are hash pinned and rehashed at the start and the end of the run. A submission that monkeypatches its way into the forward pass still executes the operation, and executing it is what the probe records.

## Is the target reachable without the ablated component

Status: MEASURED, PARTIAL. The reference closes about half of the measured gap and does not close all of it, and nothing in this bundle establishes that the whole gap is closable..

The argument that used to sit here was a prediction from published precedent, that layerwise trust ratios, adaptive gradient clipping and decoupled weight decay would restore from the update what the ablation removed from the forward pass. It was measured on the grading device and it lost, so it has been replaced by what the measurement supports. All losses below are validation loss in nats at the frozen 1200 step operating point on one H100, and the two anchors they are read against were themselves measured on that device as 1.789 for anchor_target and 2.013 for anchor_baseline. First, what the ablation removes, from a 200 step instrumented run of the frozen reference recipe on both builds: the root mean square of the residual stream is flat with depth on the un-ablated build, 3.44 at block 0 and 4.09 at block 9, and exponential with depth on the ablated build, 0.055 at block 0 and 0.828 at block 9, a factor of fifteen across ten blocks, because the squared-ReLU MLP makes each branch output grow faster than linearly in the scale of the stream it reads and nothing rescales that stream. Second, what it costs the optimizer: the published learning rate of 1.0e-3 sits on the ablated build's stability cliff, with three runs of the published configuration returning 2.013, 2.085 and a non-finite loss, and 1.5e-3 and 2.0e-3 both diverging under the published schedule, so there is no headroom to spend and turning the learning rate up is not available. Third, three families of scale-restoring update rule were measured and all three lost to the naive port: a layerwise trust ratio in the LARS and LAMB line of You et al. 2020, arXiv:1904.00962, measures 2.587 at a 3.0e-3 relative step, 2.054 at 1.0e-2 and diverges at 2.0e-2; spectrally normalized momentum in the Muon line measures 2.585 at 1.0e-3, 2.206 at 3.0e-3 and diverges at 2.0e-2; rescaling every block matrix back to its initial norm after each update measures 2.226 at 1.0e-3 and diverges at 2.0e-3. Fourth, what does move: lengthening the warmup from 5 percent to 15 percent of the run and halving the global gradient-norm clip from 1.0 to 0.5 raises the ceiling enough to run at 1.5e-3, where the loss measures 1.840, 1.848, 1.855 and 1.865, but two runs in six still diverge, and neither half of that change works alone, since the schedule change at the published rate measures 2.127 and the rate rise under the published schedule diverges. Fifth, what the divergence is, instrumented to the step: weight norms do not run away, the largest parameter norm moving only from 11.39 to 11.92 over the 25 steps before a death, while the largest activation at the last block goes from about 1.5 to 1.3e3 on one batch, recovers, drifts, then reaches 1.2e9 on a single batch with a global gradient norm of 1.8e10 against a running scale near 0.6, after which the next forward pass is non-finite; this is an input-driven activation cascade with nothing downstream to absorb it, which is exactly the job the ablated normalization was doing, and it is why the three scale-restoring families could not fix it. Sixth, the compensation the reference ships, which follows from the fifth point: the optimizer cannot stop the cascade in the forward pass, but it can refuse to write the cascade into the weights, so the rule detects a global gradient norm above 500 times its own running scale, restores every parameter from a rollback point 25 steps back, zeroes the first moment, applies a 50 step linear learning-rate cooldown and continues, measuring finite losses in every run, against four other settings of the same guard that each produced either a broken run or a loss worse than the baseline. Seventh, where inside the now-stable band the reference should sit, which with the guard in place is a choice rather than a constraint: at a 1.5e-3 peak rate the guarded rule measured 1.810, 1.813, 1.827, 1.855, 1.862, 1.902 and 1.928, which is 0.38 to 0.91 of the gap and leaves as little as 0.02 nats before the reward clamps at one and stops being able to rank a better submission above the reference, while at 1.4e-3 it measured 1.896, 1.902, 1.908 and 1.962, which is 0.23 to 0.52 of the gap with about 0.05 nats of room below and 0.11 above, so the reference ships at 1.4e-3 and deliberately declines the higher score.

Half the gap is not the whole gap, and nothing here shows the rest of it is reachable from the update alone. The measurement is also single seed and the substrate is metastable, so the same configuration returns a spread of about 0.1 nats across repeats and occasionally a broken run, which means one score is a draw from a distribution rather than a constant.

How a grader confirms it on an idle GPU: Run solution/solve.sh with no BIA_SMOKE set on an idle H100 and read /logs/verifier/score.json. A score strictly above zero establishes that the target is partially reachable without the ablated component, because the reward is exactly the fraction of the measured baseline to target gap that the submission closed and the baseline is the naive port. A score at or near one establishes full reachability. A score of exactly zero with reason scored, rather than with a checker reason, establishes that this particular reference does not clear the baseline and is a finding against the reference rather than against the task, and the correct response is to strengthen the reference before piloting, not to lower the target. That finding is exactly what happened to the previous reference, which scored zero with reason scored at agent metric 2.587, and strengthening the reference rather than lowering the target is what produced the rule that ships here.

The reward is continuous, so reachability is not all or nothing. A submission that closes a quarter of the measured gap scores a quarter.

## The scaled operating point

Binding source: requirements/bia-environment-spec.md lines 122 to 135. Six hours of single H100 session wallclock divided by fifty attempts is 7.2 minutes of single-H100 time per graded attempt, which is budget_hours 0.12. F1 at the upstream operating point costs 84 minutes per graded run, 11.7 times over budget, so a scaled operating point is mandatory and this slot is authored at the scaled point rather than at upstream nanogpt scale.

| Quantity | Value |
|---|---|
| corpus | enwik8, byte level, vocabulary 256 |
| d_model | 256 |
| n_layer | 10 |
| n_head | 4 |
| seq_len | 512 |
| batch_size | 24 |
| tokens_per_step | 12288 |
| steps | 1200 |
| tokens_per_arm | 14745600 |
| parameters_measured | 8131840 on the un-ablated build and 8126464 on the ablated build, read from the n_params field of the arm_complete records rather than estimated |
| forward_backward_per_step | 1 |
| arms_per_attempt | three on the first attempt of a session, one thereafter, because the two anchors are cached under a key over the substrate manifest, the frozen config, the corpus digest and the device |

Wallclock status: MEASURED on one NVIDIA H100 80GB HBM3 shared with other work. Per arm: anchor_target 32.8 seconds, anchor_baseline 29.4 seconds, agent 46.2 seconds. A first attempt that pays for all three arms therefore costs about 110 seconds of arm time, and the graded reference attempt against a warm anchor cache completed end to end, checkers and negative-control table included, in 50 seconds of wall clock. Both are inside the 432 second per-attempt bound that budget_hours 0.12 sets, with the three-arm case using about a quarter of it. The bundle also carries a per-arm wallclock guard of 900 seconds that aborts an arm and emits a machine-readable reason rather than silently overrunning.

First orchestrator action: Wallclock is now measured and inside budget, so the remaining orchestrator action is the external signed pilot over frozen bytes. The step count does not need lowering.

## The anchors and the reward

Policy: measured, never authored. Target metric: validation loss of the un-ablated model under the frozen reference recipe, measured by the harness as arm anchor_target. Baseline metric: validation loss of the ablated model under the identical frozen reference recipe, measured by the harness as arm anchor_baseline.

Reward: `raw = (baseline_metric - agent_metric) / (baseline_metric - target_metric); score = min(max(raw, 0), 1)`

If the naive port produces a non-finite loss, the baseline is set to the untrained ceiling ln(vocab_size), which is exact for this architecture because the output head is zero initialized, and the substitution is recorded as baseline_diverged in score.json.

baseline_metric minus target_metric must be at least 0.05 nats or the run scores zero with reason anchor_gap_below_margin, because a task whose ablation costs nothing is not this task.

## Optimization bindings

| Binding | Value |
|---|---|
| max_timeout | 6.0 |
| budget_hours | 0.12 |
| max_attempts | 50 |
| final_selection | best |
| solver_egress | setup-only |
| compute_envelope | one H100 |
| reward | single float on [0,1] via min(max(raw,0),1) |

The 6 in the spec bounds one fifty-attempt session, which is max_timeout's role. Writing 6 into budget_hours would encode a 300 hour session.

## Cohort

Pinned by name: gpt5.6-sol, anthropic/claude-opus-5, muse-spark-1.2, valid to 2026-11-17. Registry digest status: ABSENT. The consolidated spec records cohorts-unpinned as an open decision: identity is supplied, the frozen solver registry digest is not.

## Contamination

Primary leakage surface: The modded-nanogpt published record lineage, whose logfiles carry complete reproduction scripts, and the general optimizer literature.

Every published record in that lineage is co-designed with a pre-normalized architecture. Replaying one onto the ablated substrate is precisely the baseline anchor the harness measures, so a replayed recipe lands at raw zero by construction.

A published method, as distinct from a published record, is prior art the agent is entitled to use. The compensation route named under reachability is exactly such prior art, and the task is not made harder by pretending otherwise.

Fingerprint check status: NOT IMPLEMENTED IN THIS BUNDLE. Named as a coverage gap. The scoring geometry above makes replay worthless rather than rejected, which is weaker than an exclusion-set fingerprint and is recorded as weaker.

## Rejected routes, and what rejects them

| Route a submission might take | What rejects it |
|---|---|
| Reintroduce activation normalization into the forward pass by any means | The ABSENCE checker ablation_absent_from_graded_run, reading the live NormProbe trace for the agent arm |
| Edit the substrate to remove the probe or change the model | The INVARIANT checker substrate_bytes_unmodified, which rehashes every substrate file at run start and run end against the manifest baked into the image |
| Submit a rule that leaves the weights alone so a stale evaluation is graded | The EFFECT checker submitted_rule_moved_the_weights, reading the measured parameter delta at every sampled step |
| Report a validation loss the checkpoint does not reproduce | The DIVERGENCE checker validation_loss_recompute_agrees, comparing the streaming value against a fresh recompute from the saved checkpoint |
| Degrade the target anchor instead of improving the agent arm | The ORDERING checker anchors_measured_before_agent_arm, which requires both anchors to be ready before the agent arm starts |
| Port a published record recipe unchanged | Nothing rejects it, and nothing needs to. That recipe is the measured baseline anchor, so it lands at raw zero by construction |
| Emit a score outside the unit interval or one that is not the clamp of raw | The VALUE checker reward_is_clamped_unit_float |

## Defeat mechanism, stated as design intent

The ablation removes the component that actually carried the published gain, so recall alone cannot reach the target. The naive move, porting the published optimizer configuration onto the ablated substrate, is not merely suboptimal here: it is the measured baseline anchor, so it scores exactly zero by construction rather than by assertion.

Status: MEASURED IN PART, AND WEAKER THAN DESIGNED IN ONE RESPECT. The construction half holds: the naive port is the baseline anchor and lands at raw zero. The measured surprise is that the three families of scale-restoring update rule that the literature points at first all land worse than the naive port on this substrate, at every stable setting tried, while a compensation derived from an instrumented reading of how the ablated build actually dies closes about half the measured gap. Recall of a published mechanism is therefore not enough, but neither is the gap closed only by novelty: what the reference needed was a measurement of the failure mode.. No pass probability is authored here, no self-solve is claimed, and no difficulty tier is assigned. Difficulty is measured by an external signed pilot over frozen bytes and by nothing else.

## Declared gaps

| Gap | Statement |
|---|---|
| reference-score-measured-as-a-draw | The reference solution's harness score at the shipped 1.4e-3 operating point is measured through the ordinary graded pipeline on one NVIDIA H100 80GB HBM3 at 0.4683473899549122, reason scored, agent metric 1.9083885649840038, against a measured baseline of 2.0133245289325714 and a measured target of 1.7892686973015468, with all seven checkers passing and all 18 control halves passing. Four further runs of the identical configuration measured 1.896, 1.902, 1.908 and 1.962, which is 0.23 to 0.52 of the measured gap. The earlier 1.5e-3 operating point of the same rule was graded three times at 0.4975, 0.6747 and 0.9093 and measured 1.810, 1.813, 1.827, 1.855, 1.862, 1.902 and 1.928 across seven runs. Every draw of both points landed strictly inside the open unit interval and none clamped, but the 1.5e-3 point came within 0.02 nats of the upper clamp, which is why the shipped point is the lower one. The honest statement is a reference that lands somewhere in a band rather than one that lands at a number. |
| substrate-is-metastable | The ablated build is metastable at every learning rate that beats the naive port, and the naive port is metastable at its own published rate: three runs of the frozen reference recipe on the ablated build returned 2.013, 2.085 and a non-finite loss. The cached baseline anchor is therefore one draw from that distribution and not a constant of the substrate. The shipped reference carries a rollback guard that converts the divergence mode into lost progress rather than a non-finite loss, which is why every measured run of it stayed finite, but the guard reduces the variance rather than removing it. This is the largest single threat to the bundle's gradability and it is recorded rather than smoothed over. |
| wallclock-measured-under-contention | Per-arm wallclock is measured at 32.8, 29.4 and 46.2 seconds and the graded attempt at 50 seconds against a warm cache, all inside the 432 second per-attempt bound. The measurement was taken on an H100 shared with other work, so it is an upper bound on idle-device cost rather than an idle-device figure. |
| seed-ladder-single-seed | One graded seed per attempt, frozen at zero and owned by the harness. The multi-seed confirm rung the consolidated spec binds for stochastic families is not run inside an attempt, because eight seeds do not fit 7.2 minutes. Run-to-run variation at fixed seed is now partly quantified rather than unquantified: repeats of one stable configuration landed within 0.015 nats of each other, while repeats of configurations nearer the stability cliff spread by more than 0.2 nats or diverged outright, so the variation is not a single number and depends on where in the stability landscape the submission sits. |
| anchor-gap-margin-uncalibrated | MIN_ANCHOR_GAP is 0.05 nats and DIVERGENCE_TOL is 0.002 nats. The measured gap is 0.2241 nats, which clears the 0.05 floor by a factor of four and a half, and the DIVERGENCE checker passed with the streaming and post hoc values agreeing to better than 1e-9 nats. Neither constant is calibrated against a seed-to-seed variance study, because the substrate's variance is dominated by stability draws rather than by seed. |
| cohort-registry-digest-absent | The three cohorts are named but not pinned by solver registry digest, per the open decision cohorts-unpinned. |
| contamination-fingerprint-absent | No deterministic fingerprint check against a pinned exclusion set ships in this bundle. |
| trajectories-empty | No rollouts and no inspector page. The optimization lane asks for one rollout per model for at least three models, and producing them requires the GPU and the cohort. |

## Disposition

HOLD:PILOT_REQUIRED. Local design is complete and locally verified, the reference score is now measured on the grading device, and per-arm wallclock is now measured and inside budget. Difficulty remains unmeasured, the substrate's metastability is a recorded open threat to gradability, and no external signed pilot exists. This is the correct terminal state until that pilot runs.
