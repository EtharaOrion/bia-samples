# Gradient estimator under injected noise

## Objective

You are given a frozen language model, a frozen real text corpus, a frozen optimizer and a frozen process that corrupts every gradient before you see it. Reach the target validation loss in fewer optimizer steps than the baseline arm needs, by designing a better estimator of the descent direction from the corrupted observations. You cannot change the model, the data, the optimizer or the noise. The only thing you own is the estimator.

## What is frozen

The following are substrate. A run that changes any of them scores exactly zero, and the verifier detects the change from its own recomputation rather than from your report.

| Frozen | Where it lives |
|---|---|
| Byte level decoder only transformer and its weight initialization | `environment/harness/model.py` |
| Real text corpus, its split, and the batch order | `environment/fixtures/corpus.bin` and `environment/harness/data.py` |
| Injected gradient noise process, seed 20260821 | `environment/fixtures/noise_process.py` |
| AdamW update rule, its hyperparameters and its cosine schedule | `environment/harness/train.py` |
| Exactly one forward and one backward pass per optimizer step | enforced and counted by the harness |
| The verifier tree | `tests/` |

## What is free

The estimator, and any variance reduction, filtering, buffering or reconstruction it performs, and any state it carries across steps. That is the whole free surface, and it is where the entire score comes from.

## What you submit

Write `submission/estimator.py`. It must define `build_estimator(meta)` returning an object that exposes `estimate(step, observed)`.

`meta` is a dict carrying `tensor_ids`, `shapes`, `numels`, `max_steps`, `device`, `point` and `seed`. `observed` is a dict mapping each tensor id to the corrupted gradient observation for that step, as a float32 tensor of the parameter's shape. `estimate` must return a dict with the same keys and the same shapes, holding the direction the frozen optimizer will apply. You get exactly one observation per step and there is no way to buy a second one.

Your estimator never receives the model, the data, the loss or the uncorrupted gradient, and it does not merely lack a reference to them. It runs in its own interpreter, started by the harness and holding none of that state. The channel between the two carries corrupted observations in and your update direction out, and nothing else.

A working starting point is at `environment/starter/estimator.py`. It is an exponential moving average with a norm gate, it runs, and it is deliberately suboptimal. Improving it is the objective. Submitting it unchanged is the starting line, not a solution.

## The injected process

The noise fixture is public. Read it. It is deterministic in the frozen seed, the parameter tensor identity and the step index, which is what makes steps to target reproducible across attempts and across graders at all. It applies four components in this order to each parameter tensor's true gradient: an anisotropic amplification along one frozen unit direction per tensor, a block cyclic erasure of a quarter of the coordinates held for several consecutive steps and rescaled by the keep probability, an additive drift along a slowly rotating frozen direction with a correlation time of tens of steps, and a heavy spike on a frozen sparse schedule.

Every component scales with the true gradient, so the corruption vanishes at a stationary point and the optimum of the frozen objective is unchanged. What the components destroy is the descent direction on the way there. Erasure is lossy, so no single observation determines the true gradient and the process is not invertible one step at a time however carefully you read it.

## How you are graded

The reward is one float on the closed interval from zero to one, higher is better, and it is not binary. It is `min(max(raw, 0), 1)`.

For each frozen seed the verifier runs two arms over the identical initialization, the identical batch order and the identical noise realization. The baseline arm runs the frozen identity estimator, which passes each observation through untouched. The validation loss the baseline arm reaches at the frozen reference step becomes the target loss for that seed. `baseline_steps` is the first evaluation step at which the baseline arm reaches that target loss and stays at or below it for every later evaluation. `agent_steps` is the same quantity for your estimator. `target_steps` is `ceil(reference_step * 0.60)`.

`raw` is the mean over seeds of `(baseline_steps - agent_steps) / (baseline_steps - target_steps)`. Matching the baseline scores zero. Reaching the target loss in sixty percent of the baseline's steps scores one. Failing to reach it at all scores zero for that seed. Nothing about the baseline is asserted by the author: it is measured inside your own graded run.

## Who runs the graded attempt

You do not. The verifier does, and the isolation is structural rather than a convention you are asked to respect.

- The verifier runs its own private copy of the harness, the corpus and the noise fixture, in a working directory that does not exist inside your container. The copies under `environment/` are there so you can read exactly what you are being measured against; editing them changes nothing about the graded run, and the verifier digests them at grade time, so an edit costs you the whole score.
- Your estimator runs in a separate interpreter from the harness. It holds no model, no data, no optimizer, no evaluation, no run record and no target loss. There is no module in that process whose rebinding changes a graded quantity, because every graded quantity is computed in a process your code never enters.
- The run record is held in memory and written to disk once, after every arm of every seed has finished. There is no window during which a graded number is on disk for your estimator to read, and telemetry you write yourself is never an input to the reward.

Knowing the target loss would not help you even if you derived it yourself from the public substrate, because the reward is measured from the loss your estimator actually drives the model to, not from any claim about it.

Eight checkers hard gate the reward. Any one of them failing sets the score to exactly zero, and the score record names which one closed it.

| Checker | Kind | What must hold |
|---|---|---|
| `substrate_digest_constant` | INVARIANT | the substrate digest holds constant across the whole run |
| `frozen_fixture_bytes_match_bound` | VALUE | the noise fixture and corpus bytes match their bound digests, and the agent visible harness and fixture trees are unedited |
| `one_backward_per_step` | VALUE | the backward pass count equals the step count |
| `no_true_gradient_access` | ABSENCE | the uncorrupted gradient tripwire is never read |
| `noise_realization_reproduces` | DIVERGENCE | the realization the run saw agrees with the one the verifier derives independently |
| `arm_order_and_sustained_crossing` | ORDERING | the baseline arm completes and fixes the target before the agent arm starts, and the crossing is never given back |
| `submission_bound_to_run` | EFFECT | the graded arm is bound to your submitted file |
| `score_matches_telemetry` | VALUE | the emitted reward equals the reward recomputed from the run record |

## Budget

One graded attempt is bounded at `budget_hours` of 0.12, which is 7.2 minutes of single H100 time. That is the six hour session budget divided by fifty attempts, and it is the number this task is designed against. The refinement loop across attempts terminates at `max_timeout` of 6.0 hours or at `max_attempts` of 50, whichever binds first, and the submission that counts is the best attempt of the session.

One graded attempt runs four training arms, a baseline arm and an agent arm for each of two frozen seeds. Budget for all four. An estimator that costs more per step than the model's own forward and backward pass will not finish, and a graded attempt that does not finish scores zero.

## What you may use

Network egress is available during environment setup only. There is no network during scored work, so download nothing at solve time and expect nothing to be fetchable. Everything you need is in the bundle. The noise fixture is public and reading it is intended. Anything you recall about variance reduction is yours to use, and the reason this task exists is that the injected process was chosen so the standard assumptions behind those techniques do not hold here.
