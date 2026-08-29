# TRUTH: bia slot S06, gradient estimator under injected noise

GENERATED SECTION. DO NOT HAND-EDIT.

Source of truth: solution/grounding.yaml. Generator: solution/recompute.py.

## What is frozen and what is free

Frozen surface, in full, because a solver that edits any of it is not solving this task.

- the byte level decoder only transformer defined in environment/harness/model.py, including its weight initialization
- the real text corpus at environment/fixtures/corpus.bin, its split, and the frozen batch order in environment/harness/data.py
- the injected gradient noise process at environment/fixtures/noise_process.py, with frozen seed 20260821
- the AdamW update rule, its hyperparameters and its cosine schedule in environment/harness/train.py
- the rule of exactly one forward and one backward pass per optimizer step
- the verifier tree under tests/

Free surface, in full, because a task that leaves nothing free measures nothing.

- the gradient estimator submitted at submission/estimator.py, meaning the map from the sequence of corrupted observations to the update direction
- any variance reduction, filtering, buffering or reconstruction the estimator carries out, and any state the estimator keeps across steps

## The operating point

the consolidated spec fixes six hours per session and fifty attempts per session, so one graded attempt gets 7.2 minutes of single H100 time, and the upstream nanogpt operating point costs 84 minutes per graded run which is 11.7 times over that budget. The declared point is scaled-h100-7m2 with budget_hours 0.12 per attempt, max_timeout 6.0 hours across the refinement loop, and max_attempts 50. One graded attempt runs 4 training arms: two frozen seeds, and for each seed one baseline arm and one agent arm of 1500 steps at 8192 tokens per step on a 10.7 million parameter model.

Reference score at this operating point is MEASURED at 0.397321 on one H100, from the per seed block the graded run itself recorded: seed 0 crossed at agent step 825 against baseline step 950, seed 1 crossed at agent step 825 against baseline step 1000, and target steps was 600 for both. This bundle states no reference score it has not measured. A value offered in place of that measurement would be the exact defect this project names unverifiable-recorded-value.

## The reward

Raw reward is mean over seeds of (baseline_steps - agent_steps) / (baseline_steps - target_steps), and the emitted reward is min(max(raw, 0), 1). The target loss is the validation loss the baseline arm reaches at the frozen reference step, measured inside the same graded run rather than asserted by the author. Baseline steps is the first evaluation step at which the baseline arm reaches that target loss and stays at or below it. Target steps is ceil(reference_step * 0.60). the baseline and the target are both produced by the run itself, so this bundle declares no baseline number it cannot show a grader how to reproduce.

## Ordered golden path

Each step names the action, the state it establishes, the pressure it must survive, and the checker identifier it satisfies.

### Step 1, checker noise_realization_reproduces

Read instruction.md and the frozen fixture at environment/fixtures/noise_process.py, and establish that the corruption is deterministic in the frozen seed, the tensor identity and the step index rather than freshly random.

State established: the four injected components and their public schedules are known to the solver. Survives: the silent assumption that a noisy gradient is an independent zero mean sample.

### Step 2, checker submission_bound_to_run

Copy environment/starter/estimator.py to submission/estimator.py and run the harness once, establishing that the reflex answer of an exponential moving average with a norm gate produces a running submission and a measurable reward.

State established: a submission exists at submission/estimator.py and the harness stamps its digest on every agent arm record. Survives: an attempt that never produces a gradeable artifact.

### Step 3, checker score_matches_telemetry

Notice that the erasure and the two additive components interact, because on an erased coordinate the true gradient contributes nothing at all, so whatever is read there is the drift term plus, on a spike step, the spike term, and nothing else.

State established: the erased support is identified as a two column regression problem in the drift direction and the spike direction, both of which the fixture publishes for every step. Survives: the circularity of appearing to need the gradient magnitude in order to recover the gradient, which the erased support removes because it carries the additive components without the gradient.

### Step 4, checker one_backward_per_step

Solve that regression for the drift coefficient and the spike coefficient and subtract both fitted components from the whole observation, rather than clipping them away or averaging over them.

State established: the two additive components are removed to the accuracy of a fit made where they are the only thing present, so the reconstruction does not degrade as those components grow. Survives: a scheduled spike of twelve times the gradient norm dominating a single observation.

### Step 5, checker substrate_digest_constant

Undo the erasure rescale on the coordinates the mask kept, and write an honest zero on the coordinates it erased instead of restoring them from a staleness buffer.

State established: no per coordinate buffer is carried, and an erased coordinate is left for the frozen optimizer's own momentum to carry. Survives: the reflex that a rescaled zero must be replaced by something, which is measurably wrong here: on seed 0 at the scaled point, filling erased coordinates from a buffer of the last kept observation reached the target loss at step 950, exactly where the identity baseline reached it, while writing zero reached it at step 850, against a ceiling of 800 for an arm handed the uncorrupted gradient.

### Step 6, checker frozen_fixture_bytes_match_bound

Invert the rank one amplification exactly, using the frozen direction and the frozen coefficient the fixture publishes.

State established: the component that averaging can never remove is removed algebraically instead. Survives: a deterministic function of the gradient itself masquerading as noise.

### Step 7, checker no_true_gradient_access

Return the reconstructed direction for every tensor the harness asked about, with the same shapes, and never reach for the uncorrupted gradient surface the harness exposes as a tripwire.

State established: the frozen optimizer applies the reconstructed direction and the sentinel counter stays at zero. Survives: the temptation to read the gradient the task exists to hide.

### Step 8, checker arm_order_and_sustained_crossing

Let the verifier run both arms itself, in the order it fixes, and read the reward off the run record it wrote.

State established: for every seed the baseline arm completed and fixed the target loss before the agent arm started, and the agent crossing was never given back. Survives: a transient dip below the target followed by divergence.

## Checker reconciliation

The checker identifiers named by the golden path equal the committed checker set exactly, in both directions.

- substrate_digest_constant reduces to INVARIANT
- frozen_fixture_bytes_match_bound reduces to VALUE
- one_backward_per_step reduces to VALUE
- no_true_gradient_access reduces to ABSENCE
- noise_realization_reproduces reduces to DIVERGENCE
- arm_order_and_sustained_crossing reduces to ORDERING
- submission_bound_to_run reduces to EFFECT
- score_matches_telemetry reduces to VALUE

## Rejected routes

Each rejected route names the known wrong control that stands for it. The controls are the negative half of the feasibility bundle, and a route with no control would be an author's assertion rather than a measured rejection.

- Submit the starter exponential moving average unchanged. Control control_starter_ema. the amplification component is a deterministic function of the gradient, so an average of observations averages the amplification in rather than out, and the drift component is positively autocorrelated over tens of steps so a short window average tracks it instead of cancelling it.
- Clip the observation to a running norm and pass it through. Control control_norm_clip_only. clipping suppresses the scheduled spike and also suppresses the informative part of every step on which the spike does not fire, and it leaves the erasure and the amplification untouched.
- Edit or monkeypatch the noise fixture so the corruption is milder. Control control_edited_fixture. the verifier compares the fixture bytes against a bound digest and independently recomputes the noise realization, so an edited fixture closes the frozen_fixture_bytes_match_bound and noise_realization_reproduces checkers and the reward is exactly zero.
- Reach through the harness for the uncorrupted gradients. Control control_sentinel_touch. the harness exposes a tripwire object where uncorrupted gradients would sit, any read increments a counter written into the run record, and the no_true_gradient_access checker closes on a non zero counter.
- Hand write the run record so it reports a favourable steps to target. Control control_forged_telemetry. the verifier runs the harness itself and reads only the record that run produced, so a record written by a submission is never an input to the reward.

## The injected process and why it is frozen

The fixture at environment/fixtures/noise_process.py carries seed 20260821 and sha256 78477bbfa8b6a818dee25338484fb4a078735c1a11faa74f9175b5f8707e5c88. an unfrozen noise process makes the reward unmeasurable, because two graders would measure two different problems and steps to target would carry no meaning across attempts.

- anisotropic amplification along one frozen unit direction per tensor, with BETA equal to 3.0, which is neither zero mean nor independent of the gradient
- block cyclic erasure of four of sixteen frozen coordinate blocks held for four consecutive steps, rescaled by one over the keep probability
- additive drift along a slowly rotating frozen direction with correlation time 64 steps, scaled by the true gradient norm
- a heavy spike on a frozen sparse schedule of period 23 and phase 7, with multiplier 12

## Design intent, and what was measured

The intended defeat mechanism is that standard variance reduction assumes noise structure this injected process deliberately violates.

The amplification component is a deterministic function of the gradient, so it survives averaging rather than cancelling under it.

The drift component is positively autocorrelated over tens of steps, so a short window average tracks it instead of removing it.

The erasure is held across consecutive steps, so a rescaled zero is a stale reading rather than an unbiased sample of that coordinate.

The spike schedule is sparse enough that clipping trades a rare large error for a constant small one on every other step.

Those four statements are design intent and they are UNMEASURED as explanations of why a solver fails.

What was measured, on seed 0 at the scaled operating point, is the step cost of each treatment of the observation against the target loss the baseline arm fixes at the reference step. The identity baseline reached it at step 950. An arm handed the uncorrupted gradient reached it at step 800, which is the ceiling any estimator is bounded by, because the erasure is lossy and one observation per step does not determine a coordinate nobody read. The reference estimator reached it at step 850.

One measured finding is worth stating because it contradicts the reflex. Restoring erased coordinates from a staleness buffer of the last kept observation reached the target at step 950, which is exactly the baseline and no gain at all, while writing zero there reached it at step 850. The frozen optimizer already carries a momentum estimate for every coordinate, so an unwritten coordinate coasts on it, whereas a buffered value overwrites that estimate with a stale minibatch reading held for the whole erasure window.

A second measured finding bounds the design. Cosine similarity between the applied direction and the true gradient does not order these arms: the buffered variant held a cosine near 0.90 against the baseline's 0.45 and still bought nothing, so an estimator tuned to maximize agreement with the true gradient is tuned against the wrong quantity.

No pass probability, no self solve claim and no difficulty tier is asserted anywhere in this bundle, and the bundle ships at HOLD:PILOT_REQUIRED, which is the correct terminal state.

