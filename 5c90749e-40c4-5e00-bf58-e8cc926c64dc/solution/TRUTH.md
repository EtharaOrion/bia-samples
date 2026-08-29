# TRUTH: bia/s01-optimizer-search-frozen-init

GENERATED SECTION. DO NOT HAND-EDIT.

Source: solution/grounding.yaml via solution/recompute.py.

## Headline

The reference path derives a blended-sign decoupled adaptive rule with a trapezoid schedule against the reduced interface, runs it at the graded scale on two seeds through the harness runner, and submits what the runner produced.

## Frozen surface

- dataset, the order-two sparse categorical source and the token arrays it produces
- batch, the sequences per forward pass and the sequence length and therefore the tokens per step
- architecture, the module tree and every parameter shape
- weight initialization, owned by environment/bia_core.py frozen_init and seeded by the run seed
- one forward and backward pass per optimizer step
- evaluation, performed by the harness on the live parameters with no blending and no averaging

## Free surface

- the optimizer update rule, supplied as build_optimizer
- the learning rate schedule of that rule, supplied as build_schedule
- the internal state and hyperparameters of the submitted rule

## Ordered path

### Step 1

Read instruction.md and establish that the free surface is exactly build_optimizer and build_schedule and that the initialization and the evaluation path are harness owned.

Established state: the submission contract and the frozen surface are known

Survives: the silent absence of any interface for initialization or for eval-time weight blending

Satisfies: frozen_init_digest, frozen_architecture_signature

### Step 2

Write submission/optimizer.py exporting build_optimizer and build_schedule, deriving the update rule against the reduced surface rather than transcribing a published record.

Established state: a submitted rule exists and is not a member of the exclusion set

Survives: the temptation to port a record whose gain lives in components this interface does not expose

Satisfies: no_verbatim_record_copy, optimizer_step_moves_parameters

### Step 3

Invoke environment/runner/run_bia.py with mode probe to size the rule and its schedule inside the per attempt budget.

Established state: probe telemetry exists and is excluded from grading by its mode

Survives: the absence of any announcement that a probe cannot be graded

Satisfies: telemetry_bound_to_submission

### Step 4

Invoke environment/runner/run_bia.py with mode full on seeds zero and one with the submission that is to be graded.

Established state: full-mode telemetry exists for at least the minimum seeds and is bound to the submitted file

Survives: the silent invalidation of full-mode telemetry produced by an earlier submission

Satisfies: frozen_batch_contract, frozen_corpus_digest, no_submission_writes

### Step 5

Submit the per-seed logs the runner wrote, unedited, beside the optimizer file.

Established state: the reported series and the recorded series agree in both directions

Survives: the fact that a hand-written log parses cleanly and still fails reconciliation

Satisfies: report_matches_telemetry

### Step 6

Let the verifier derive the graded step from the recorded series and write the reward.

Established state: a graded step exists, is sustained, and is at or below the baseline anchor

Survives: a transient dip that a per-run stopping choice would have banked

Satisfies: crossing_sustained, steps_to_target_within_baseline

## Rejected routes

| Route | Rejected by | Known-wrong control |
|---|---|---|
| submit the frozen baseline recipe unchanged | no_verbatim_record_copy | environment/baseline_optimizer.py used as a known-wrong submission |
| submit a member of the excluded published recipe corpus | no_verbatim_record_copy | tests/corpus/rec_lion.py wrapped to the submission interface, measured at similarity 0.901 |
| return an optimizer whose step leaves the parameters unchanged | optimizer_step_moves_parameters | a null optimizer whose step is a no-op |
| report a loss series the harness never recorded | report_matches_telemetry | a hand-edited per-seed log |
| present a single-seed run as a graded result | crossing_sustained | a full-mode run at one seed only |

## Reward

score = min(max((baseline_steps - graded_step) / (baseline_steps - target_steps), 0.0), 1.0)

Reward is a single float on [0.0, 1.0], higher is better, and every zero written by tests/grade.py carries a machine-readable reason string.

## Anchors

| Scale | baseline_steps | target_steps | Status |
|---|---|---|---|
| full | 310 | 248 | measured on the reference machine, one H100, two seeds, eighty percent rule |
| smoke | 560 | 448 | measured on the authoring host, CPU, two seeds, eighty percent rule |

baseline_steps is the graded step of the frozen baseline recipe at environment/baseline_optimizer.py, measured through the grading rule in tests/grade.py at seeds zero and one on the machine that scale is graded on. target_steps is the step count a submission must reach for full reward, and it is eighty percent of the measured baseline_steps, which is close to the twenty point eight percent gap between the F1 baseline and the F1 target recorded at requirements/bia-environment-spec.md:313. Both anchors at both scales are now measured, so no provisional bracket is carried and no anchor is nominal. Neither anchor is set from what the reference solution achieves, and the reference is not exempted from either of them.

## Measurement

Taken on 2026-08-21, on one H100 with eighty gigabytes, torch 2.13.0 with cuda 13.2, shared with three sibling lanes for the whole measurement window, so every elapsed second recorded here is an upper bound and an idle card is faster. Reading convention: every sweep below was read on seed zero through the same runner call graph the graded run uses.

The graded operating point was mis-sized and has been resized. The full scale used to set vocab to five hundred twelve at order two, which is a context table of two hundred sixty two thousand one hundred forty four rows, and writing that table is the whole of what the model has to learn from a source with no other structure in it. At that row count no update rule reached the target inside any step count that fits the attempt budget. The measured evidence is that the reference finished fourteen hundred steps at validation loss six point two one against a marginal loss of six point two three six eight and a target of three point eight zero four zero seven six, that a peak rate sweep across three point five e minus three, one e minus two and three e minus two moved the four hundred step value only between six point two three four seven and six point two three eight zero, and that the same rule at the same rate on a sixty five thousand five hundred thirty six row table crosses its target near step two hundred eighty. The failure was therefore the row count and not the calibration of the rate, and the fix is to resize the row count rather than to move a bar.

| Sweep | Varied | Decided |
|---|---|---|
| table size | vocab at order two, which is the context table row count | vocab two hundred fifty six, which puts the crossing inside the attempt budget with headroom |
| peak rate | the peak learning rate of the reference rule | one point eight e minus three, the measured interior optimum |
| warmup length | the warmup fraction of the trapezoid schedule | five percent of the total step count |
| per parameter group rate multipliers | separate rate multipliers on the embeddings, the attention matrices, the feed forward matrices and the head | uniform, no per group multiplier in the shipped rule |
| blend share | the share of the sign of bias corrected momentum in the blended direction | one tenth, unchanged from the authored default and now measured at the graded scale |

Observed in the table size sweep: vocab five hundred twelve, two hundred sixty two thousand one hundred forty four rows, no descent at all inside fourteen hundred steps. vocab two hundred fifty six, sixty five thousand five hundred thirty six rows, crossing near step two hundred eighty. vocab one hundred twenty eight, sixteen thousand three hundred eighty four rows, crossing between step one hundred and step two hundred. vocab sixty four, four thousand ninety six rows, crossing before step two hundred.

Observed in the peak rate sweep: seven e minus three and one point two e minus two produced no descent inside three hundred fifty steps. three point five e minus three crossed near step three hundred fifty. two point five e minus three crossed near step three hundred. one point eight e minus three crossed near step two hundred seventy five. one point two e minus three and eight e minus four both slipped back to near step three hundred.

Observed in the warmup length sweep: a warmup shorter than roughly forty steps pinned the run near the marginal loss for the whole budget rather than merely slowing it, and three percent of a three hundred step budget is nine steps, which is how an earlier short probe read as a dead task when it was a collapsed warmup.

Observed in the per parameter group rate multipliers sweep: attention at three times the base rate stopped the descent entirely, at five point five one three at step three hundred against two point eight three for the uniform rule. attention at three tenths delayed it to four point seven two at step three hundred. embeddings at three times and the head at three times both delayed it. no multiplier beat uniform.

Observed in the blend share sweep: a blend of four tenths reached two point four zero six at step four hundred where one tenth reached two point one five nine, and the crossing moved later with the larger blend.

The frozen baseline recipe crosses at step 310 and the reference crosses at step 280, so solve.sh at the graded full scale scores 0.4838709677419355 in 195 and 228 seconds across 2 runs, against a bound of 432 seconds, with all eleven verifier outcomes true. Reproducibility: two graded runs from a cold corpus cache and cleared mount points returned the same graded step and the same score, and differed only in elapsed seconds, which is contention on a card three sibling lanes were sharing.

The two hundred eighty is the reference crossing and the three hundred ten is the baseline crossing, both read by the same grading rule from the same recorded series, so the reference earns an interior score by beating the baseline anchor by thirty steps and still falls sixty steps short of the full reward anchor. A submission that crosses at or before step two hundred forty eight scores one, and the reference does not.

The same call graph on the smoke scale runs on cpu in 22 seconds, where the baseline crosses at step 560, the reference crosses at step 540, and the score is 0.17857142857142858.

## Defeat mechanism

Published optimizer speedrun records reach their step counts with a complete training recipe, and a large share of the reported gain is carried by parts of that recipe this interface does not expose, specifically the weight initialization and the eval-time blending of weights. This task freezes the initialization inside the harness and evaluates the live parameters with no blending, so a record recalled and ported through build_optimizer arrives stripped of the components that carried its published figure and lands well short of it. The intended path is to reason about which part of a record survives the reduced surface and to derive a rule against that surface.

Status: UNMEASURED, design intent only. Evidence owed: an external signed pilot over frozen bytes against the pinned cohort.

## Coverage gaps

### single-machine-single-window

Every figure under measurement was taken in one window on one H100 that three sibling lanes were sharing. The step anchors are step counts and are portable across accelerators, so contention cannot move them, but the elapsed seconds are an upper bound from a contended card and no second machine has reproduced any of it.

Cap: HOLD:PILOT_REQUIRED. Closes when: the baseline anchor and the reference score are reproduced on a second H100 and the two readings are recorded side by side

### difficulty-unmeasured

The defeat mechanism is design intent. No pass probability, no self-solve claim and no difficulty tier is recorded for this bundle.

Cap: HOLD:PILOT_REQUIRED. Closes when: an external signed pilot over these frozen bytes reports a bound against the pinned cohort

### image-digest-inherited

The pinned base image digest is inherited from the verified track3 reference bundle rather than resolved by this lane, because the authoring environment has no registry egress. The bundle ships no layer on top of it.

Cap: HOLD:PILOT_REQUIRED. Closes when: the digest is re-resolved against the registry at delivery time

### canary-tripwire-not-planted

No per-bundle canary token set is planted in the private artifacts. The required layout this lane authored against does not carry a canary slot contract, and planting tokens without a bound normalization domain would make the content hash cyclic, so the tripwire is absent rather than approximated.

Cap: HOLD:PILOT_REQUIRED. Closes when: a canary normalization domain is bound at sign-off and the token slots are planted after the hash is computed and before the binding block is applied

### write-guard-bounded

The no_submission_writes checker observes write-mode opens through the builtins open path only, so it bounds that named path and does not claim that no write of any kind occurred.

Cap: HOLD:PILOT_REQUIRED. Closes when: the guard is widened to the os level write surface and its both-halves control is recorded

## Checker set

| Checker | Kind | Compiled test |
|---|---|---|
| frozen_batch_contract | INVARIANT | tests/test_output.py::test_frozen_batch_contract |
| frozen_init_digest | VALUE | tests/test_output.py::test_frozen_init_digest |
| frozen_architecture_signature | VALUE | tests/test_output.py::test_frozen_architecture_signature |
| frozen_corpus_digest | VALUE | tests/test_output.py::test_frozen_corpus_digest |
| telemetry_bound_to_submission | INVARIANT | tests/test_output.py::test_telemetry_bound_to_submission |
| no_submission_writes | ABSENCE | tests/test_output.py::test_no_submission_writes |
| optimizer_step_moves_parameters | EFFECT | tests/test_output.py::test_optimizer_step_moves_parameters |
| crossing_sustained | ORDERING | tests/test_output.py::test_crossing_sustained |
| report_matches_telemetry | DIVERGENCE | tests/test_output.py::test_report_matches_telemetry |
| steps_to_target_within_baseline | VALUE | tests/test_output.py::test_steps_to_target_within_baseline |
| no_verbatim_record_copy | ABSENCE | tests/test_output.py::test_no_verbatim_record_copy |
